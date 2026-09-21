"""Run the one-layout ROI observation-contract repair probe."""
import argparse, hashlib, json, random, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image
from guard.active_vision.belief_branch_scene import BranchLayout, BranchingBeliefEnv, CAMERAS
from guard.active_vision.belief_branching import STATES, observation
from guard.active_vision.runtime import execute_route, physical_state, state_hash
from guard.json_io import canonical_dumps, write_json
SEED=66200; PAID=CAMERAS[1:]
def sha(b): return hashlib.sha256(b).hexdigest()
def env(state):
    random.seed(SEED); np.random.seed(SEED); e=BranchingBeliefEnv(hidden_state=state); e.reset(); return e
def capture(state,out):
    e=env(state)
    try:
        rows=[]
        for camera in CAMERAS:
            before=state_hash(physical_state(e)); image=e.capture_camera(camera); after=state_hash(physical_state(e))
            if before!=after: raise RuntimeError('query changed state')
            Image.fromarray(image).save(out/f'{camera}.png')
            # Registered mechanism ROI: central 96x96 cue board crop.
            roi=image[64:160,64:160]
            rows.append({'camera':camera,'state_hash':before,'image_sha256':sha(image.tobytes()),'roi_sha256':sha(roi.tobytes()),'roi_mean':roi.mean(axis=(0,1)).round(3).tolist()})
        return {'state_hash':rows[0]['state_hash'],'ledger':rows}
    finally: e.close()
def main():
    p=argparse.ArgumentParser(); p.add_argument('output',type=Path,nargs='?'); p.add_argument('--fingerprint-state'); p.add_argument('--fingerprint-output',type=Path); a=p.parse_args();
    if a.fingerprint_state:
        write_json(a.fingerprint_output,capture(a.fingerprint_state,a.fingerprint_output.parent)); return
    a.output.mkdir(parents=True); rows=[]
    for state in STATES:
        sd=a.output/state; sd.mkdir(); fp=capture(state,sd); rp=sd/'fresh_process.json'
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--fingerprint-state',state,'--fingerprint-output',str(rp)],check=True)
        replay=json.loads(rp.read_text()); routes=[]
        for name in ('left_route','center_route','right_route'):
            e=env(state)
            try:
                result=execute_route(e,name,max_steps=100,hold_steps=5)
                trajectory_hash=sha(canonical_dumps(result['records']).encode())
                clearances=[]
                for record in result['records']:
                    point=np.asarray(record['eef'])
                    for index,lane in enumerate((e.layout.lane_y,0.0,-e.layout.lane_y)):
                        if state[index]=='0': clearances.append(float(np.linalg.norm(point-np.asarray((e.layout.obstacle_x,lane,e.layout.target[2])))-0.085))
                result={k:v for k,v in result.items() if k!='records'}; result['trajectory_hash']=trajectory_hash; result['minimum_clearance_m']=max(0.0,min(clearances)) if clearances else 0.0; routes.append(result)
            finally: e.close()
        rows.append({'state':state,'observation_table':{camera:observation(state,camera) for camera in PAID},'fingerprint':fp,'fresh_process_replay_exact':canonical_dumps(fp)==canonical_dumps(replay),'routes':routes})
    write_json(a.output/'summary.json',{'schema_version':1,'seed':SEED,'roi':[64,64,160,160],'states':rows,'clearance_threshold_m':0.004,'checker_version':'roi-probe-v1'})
if __name__=='__main__': main()
