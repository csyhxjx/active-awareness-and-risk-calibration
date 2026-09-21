"""Run the frozen 12-layout exact-belief development experiment."""
import argparse, hashlib, json, os, random, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image
from guard.active_vision.belief_branch_scene import BranchLayout, BranchingBeliefEnv, CAMERAS, ALL_STATES
from guard.active_vision.belief_branching import STATES, all_fixed_results, observation, run_adaptive
from guard.active_vision.runtime import execute_route, physical_state, state_hash
from guard.json_io import canonical_dumps, write_json
SEED=66121
def sha(b): return hashlib.sha256(b).hexdigest()
def env(state, layout):
    random.seed(SEED); np.random.seed(SEED); e=BranchingBeliefEnv(layout=layout, hidden_state=state); e.reset(); return e
def capture(state, layout, out):
    e=env(state,layout)
    try:
        ledger=[]
        for camera in CAMERAS:
            before=state_hash(physical_state(e)); image=e.capture_camera(camera); after=state_hash(physical_state(e))
            if before != after: raise RuntimeError('state changed')
            ledger.append({'camera':camera,'state_hash':before,'image_sha256':sha(image.tobytes())})
            Image.fromarray(image).save(out/f'{camera}.png')
        return {'ledger':ledger,'state_hash':ledger[0]['state_hash']}
    finally: e.close()
def route(state, layout, name, expected):
    e=env(state,layout)
    try:
        if state_hash(physical_state(e)) != expected: raise RuntimeError('branch mismatch')
        return execute_route(e,name,max_steps=100,hold_steps=5)
    finally: e.close()
def clearance(result, state, layout):
    margins=[]
    for record in result['records']:
        point=np.asarray(record['eef'])
        for index,lane in enumerate((layout.lane_y,0.0,-layout.lane_y)):
            if state[index]=='0':
                center=np.asarray((layout.obstacle_x,lane,layout.target[2]))
                margins.append(float(np.linalg.norm(point-center)-0.085))
    return max(0.0,min(margins)) if margins else 0.0
def main():
    p=argparse.ArgumentParser(); p.add_argument('manifest',type=Path); p.add_argument('output',type=Path, nargs='?'); p.add_argument('--fingerprint-state'); p.add_argument('--fingerprint-output',type=Path); p.add_argument('--fingerprint-layout'); a=p.parse_args()
    if a.fingerprint_state:
        spec=next(item for item in json.loads(a.manifest.read_text())['layouts'] if item['layout_id']==a.fingerprint_layout); layout=BranchLayout(**{k:tuple(v) if k in ('start','target','color_permutation') else v for k,v in spec.items() if k not in ('states','main_states','control_states','observation_table')})
        write_json(a.fingerprint_output, capture(a.fingerprint_state, layout, a.fingerprint_output.parent)); return
    if a.output is None: p.error('output required')
    a.output.mkdir(parents=True)
    manifest=json.loads(a.manifest.read_text()); layouts=[]
    for spec in manifest['layouts']:
        layout=BranchLayout(**{k:tuple(v) if k in ('start','target','color_permutation') else v for k,v in spec.items() if k not in ('states','main_states','control_states')})
        root=a.output/layout.layout_id; root.mkdir(); rows=[]; route_map={}
        for state in ALL_STATES:
            sd=root/state; sd.mkdir(); fp=capture(state,layout,sd); routes=[]
            for name in ('left_route','center_route','right_route'):
                result=route(state,layout,name,fp['state_hash']); result['clearance_margin']=clearance(result,state,layout); route_map[(state,name)]=result; routes.append({k:v for k,v in result.items() if k!='records'})
            replay_path=sd/'fresh_process_fingerprint.json'
            subprocess.run([sys.executable, str(Path(__file__).resolve()), str(a.manifest), '--fingerprint-state', state, '--fingerprint-layout', layout.layout_id, '--fingerprint-output', str(replay_path)], check=True)
            rows.append({'state':state,'fingerprint':fp,'fresh_process_replay_exact':replay_path.exists() and canonical_dumps(fp)==canonical_dumps(json.loads(replay_path.read_text())),'routes':routes})
        adaptive=[]
        for state in STATES:
            trace=run_adaptive(state); action=trace[-1]['decision']['id']; result=route_map[(state,action)]
            adaptive.append({'state':state,'trace':trace,'terminal_action':action,'success':result['collision_free_success'],'collision':result['collision']})
        fixed=[]
        for result in all_fixed_results():
            trials=[]
            for trial in result['trials']:
                phys=route_map[(trial['state'],trial['terminal_action'])] if trial['terminal_action']!='stop' else {'collision':False,'collision_free_success':False}
                trials.append({**trial,'physical_outcome':{k:phys[k] for k in ('collision','collision_free_success')}})
            fixed.append({**{k:v for k,v in result.items() if k!='trials'},'trials':trials})
        write_json(root/'summary.json',{'layout_id':layout.layout_id,'spec':spec,'states':rows,'adaptive':adaptive,'fixed':fixed})
        layouts.append({'layout_id':layout.layout_id,'summary':str(root/'summary.json')})
    write_json(a.output/'summary.json',{'schema_version':1,'seed':SEED,'layouts':layouts})
if __name__=='__main__': main()
