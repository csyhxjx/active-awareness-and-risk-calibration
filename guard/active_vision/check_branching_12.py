import argparse,json
from pathlib import Path
from guard.active_vision.belief_branching import STATES
from guard.active_vision.belief_branch_scene import ALL_STATES
from guard.active_vision.belief_branching import CAMERAS, observation, STATES
from guard.json_io import write_json

def check(root):
    errors=[]; layout_results=[]
    for summary_ref in root['layouts']:
        summary=json.loads(Path(summary_ref['summary']).read_text()); rows={r['state']:r for r in summary['states']}
        if set(rows)!=set(ALL_STATES): errors.append(summary['layout_id']+':scope')
        v0={r['fingerprint']['ledger'][0]['image_sha256'] for r in rows.values()}
        if len(v0)!=1: errors.append(summary['layout_id']+':v0')
        for camera in CAMERAS:
            groups={}
            for state in STATES:
                ledger={x['camera']:x for x in rows[state]['fingerprint']['ledger']}
                groups.setdefault(observation(state,camera),set()).add(ledger[camera]['image_sha256'])
            if not all(len(values)==1 for values in groups.values()): errors.append(summary['layout_id']+':image_within')
            if len({next(iter(values)) for values in groups.values()}) != len(groups): errors.append(summary['layout_id']+':image_between')
        route_count=0
        for state,row in rows.items():
            if len({x['state_hash'] for x in row['fingerprint']['ledger']}) != 1: errors.append(summary['layout_id']+':query')
            for route in row['routes']:
                idx=('left_route','center_route','right_route').index(route['route']); expected=route['collision'] if state[idx]=='1' else route['collision_free_success']
                if not expected: errors.append(f"{summary['layout_id']}:{state}:{route['route']}")
                route_count+=1
        adaptive=sum(bool(x['success']) for x in summary['adaptive'])
        best=max(x['completion'] for x in summary['fixed'])
        fixed_utility=max(x['mean_utility'] for x in summary['fixed'])
        if adaptive != 6: errors.append(summary['layout_id']+':adaptive')
        layout_results.append({'layout_id':summary['layout_id'],'v0':len(v0)==1,'routes':route_count,'adaptive_main_completion':adaptive,'best_fixed_completion':best,'best_fixed_utility':fixed_utility,'strict_gain':adaptive>best,'fresh_process_replay_recorded':False})
    majority=sum(x['strict_gain'] for x in layout_results) > 6
    result={'schema_version':1,'layouts':layout_results,'majority_strict_gain':majority,'all_pass':not errors and majority,'errors':errors}
    return result
def main():
    p=argparse.ArgumentParser(); p.add_argument('summary',type=Path); p.add_argument('output',type=Path); a=p.parse_args(); result=check(json.loads(a.summary.read_text())); write_json(a.output,result); print(json.dumps(result,indent=2)); raise SystemExit(0 if result['all_pass'] else 1)
if __name__=='__main__': main()
