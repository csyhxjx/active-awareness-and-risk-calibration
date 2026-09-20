"""Build the fresh train-only Phase 5E smoke manifest."""

import argparse
from pathlib import Path

import numpy as np

from guard.active_vision.build_manifest_v2 import FAMILIES
from guard.json_io import write_json

SEED = 55008
SCENE_SEED = 20260921


def build_manifest():
    rng = np.random.Generator(np.random.PCG64(SEED))
    rows = []
    for index in range(8):
        family_name, yaw, divider_y, side_azimuth = FAMILIES[index % 4]
        rows.append({
            "layout_id": f"av4_smoke_{index:03d}", "split": "train", "family": family_name,
            "manifest_seed": SEED, "start": [-0.103, 0.0, 1.01],
            "target": [float(rng.uniform(0.18, 0.22)), 0.0, 1.01],
            "lane_y": float(rng.uniform(0.16, 0.20)), "obstacle_x": float(rng.uniform(0.045, 0.085)),
            "occluder_x": float(rng.uniform(0.28, 0.32)), "mirror": 1 if index % 2 == 0 else -1,
            "occluder_yaw": float(yaw + rng.uniform(-0.02, 0.02)),
            "divider_y": float(divider_y + rng.uniform(-0.01, 0.01)),
            "side_camera_azimuth_deg": float(side_azimuth + rng.uniform(-2, 2)),
            "side_camera_fovy": float(rng.uniform(18, 26)),
            "high_camera_azimuth_deg": float(rng.uniform(-6, 6)),
            "high_camera_fovy": float(rng.uniform(26, 34)), "route_ribbons": True,
        })
    return {"schema_version": 4, "protocol": "active_vision_v4", "manifest_seed": SEED,
            "scene_seed": SCENE_SEED, "layouts": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    write_json(args.output, build_manifest())


if __name__ == "__main__":
    main()
