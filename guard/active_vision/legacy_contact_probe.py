"""MuJoCo 2.3.7-only contact neighborhood probe for acceptance attribution."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from guard.active_vision.continuous_nominal_calibration import make_environment, measure_replay, set_target_obstacle

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--manifest', type=Path, required=True); args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text())
    route = 'left_route'
    offsets = (-0.101, -0.100, -0.099, -0.081, -0.080, -0.079, 0.079, 0.080, 0.081)
    rows = []
    for offset in offsets:
        env = make_environment(route)
        try:
            set_target_obstacle(env, route, offset)
            result = measure_replay(env, manifest['nominals'][route], keep_steps=True)
        finally:
            env.close()
        rows.append({'route': route, 'offset_m': offset,
                     'minimum_clearance_m': result['minimum_clearance_m'],
                     'first_contact': result['first_contact'],
                     'collision_classes': result['collision_classes'],
                     'step_trace_sha256': result['step_trace_sha256']})
    print(json.dumps({'engine': '2.3.7', 'route': route, 'rows': rows}, sort_keys=True, separators=(',', ':')))

if __name__ == '__main__': main()
