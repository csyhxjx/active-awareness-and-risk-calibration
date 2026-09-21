import argparse,json
from pathlib import Path
from guard.active_vision.belief_branching import CAMERAS, STATES, observation
from guard.json_io import write_json
def main():
    p=argparse.ArgumentParser(); p.add_argument('summary',type=Path); p.add_argument('output',type=Path); a=p.parse_args(); s=json.loads(a.summary.read_text()); errors=[]; rows={x['state']:x for x in s['states']}
    if set(rows)!=set(STATES): errors.append('scope')
    for camera in CAMERAS:
        groups={}
        for state in STATES:
            row={x['camera']:x for x in rows[state]['fingerprint']['ledger']}[camera]
            groups.setdefault(observation(state,camera),set()).add(tuple(row['roi_mean']))
        if any(len(v)!=1 for v in groups.values()): errors.append(camera+':roi_within')
        if len({next(iter(v)) for v in groups.values()}) != len(groups): errors.append(camera+':roi_between')
    for state,row in rows.items():
        if not row['fresh_process_replay_exact']: errors.append(state+':replay')
        for route in row['routes']:
            idx=('left_route','center_route','right_route').index(route['route']); clear=state[idx]=='0'
            if clear and (not route['collision_free_success'] or (route['minimum_clearance_m'] is not None and route['minimum_clearance_m']<s['clearance_threshold_m'])): errors.append(state+'/'+route['route']+':clearance')
    result={'schema_version':1,'states':len(rows),'roi_contract':not any(':roi_' in e for e in errors),'replay':not any(e.endswith(':replay') for e in errors),'routes':not any(':clearance' in e for e in errors),'all_pass':not errors,'errors':errors}; write_json(a.output,result); print(json.dumps(result,indent=2)); raise SystemExit(0 if result['all_pass'] else 1)
if __name__=='__main__': main()
