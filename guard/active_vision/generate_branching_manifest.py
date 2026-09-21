"""Deterministically generate the frozen 12-layout development manifest."""
import argparse, json
from pathlib import Path
from guard.active_vision.belief_branch_scene import BranchLayout
from guard.active_vision.belief_branching import CAMERAS, OBSERVATION_TABLE, STATES
from guard.json_io import write_json

SEED = 66121
ALL_STATES = ("001","110","010","101","011","100","000","111")

def build():
    layouts=[]
    for index in range(12):
        layouts.append({
            "layout_id": f"branch_dev_{index:02d}",
            "start": [-0.103, 0.0, 1.01], "target": [0.20, 0.0, 1.01],
            "lane_y": round(0.16 + 0.006 * (index % 4), 6),
            "obstacle_x": round(0.045 + 0.008 * (index % 5), 6),
            "occluder_x": round(0.29 + 0.008 * (index % 3), 6),
            "cue_shift_x": round(0.012 * ((index % 3) - 1), 6),
            "cue_shift_y": round(0.012 * ((index % 4) - 1.5), 6),
            "camera_shift_x": round(0.01 * ((index % 5) - 2), 6),
            "camera_shift_y": round(0.01 * ((index % 3) - 1), 6),
            "color_permutation": [0, 1, 2],
            "observation_table": {state: {camera: OBSERVATION_TABLE[(state, camera)] for camera in CAMERAS} for state in STATES},
            "states": list(ALL_STATES),
            "main_states": list(ALL_STATES[:6]), "control_states": list(ALL_STATES[6:]),
        })
    return {"schema_version": 1, "seed": SEED, "layout_count": 12, "layouts": layouts}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("output", type=Path); args=parser.parse_args()
    write_json(args.output, build())
if __name__ == "__main__": main()
