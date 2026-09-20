"""Build the deterministic, train-only Phase 5D smoke manifest."""

import argparse
from pathlib import Path

import numpy as np

from guard.active_vision.build_manifest_v2 import FAMILIES
from guard.json_io import write_json


SEED = 54040
SCENE_SEED = 20260920
SMOKE_FAMILIES = ("F0", "F1", "F2", "F3")


def build_manifest():
    rng = np.random.Generator(np.random.PCG64(SEED))
    rows = []
    for family_index, family_name in enumerate(SMOKE_FAMILIES):
        family, yaw, divider_y, side_azimuth = FAMILIES[family_index]
        for member in range(2):
            mirror = 1 if (member + family_index) % 2 == 0 else -1
            rows.append(
                {
                    "layout_id": f"av3_smoke_{len(rows):03d}",
                    "split": "train",
                    "family": family_name,
                    "family_member": member,
                    "manifest_seed": SEED,
                    "start": [-0.103, 0.0, 1.01],
                    "target": [float(rng.uniform(0.18, 0.22)), 0.0, 1.01],
                    "lane_y": float(rng.uniform(0.16, 0.20)),
                    "obstacle_x": float(rng.uniform(0.045, 0.085)),
                    "occluder_x": float(rng.uniform(0.28, 0.32)),
                    "mirror": mirror,
                    "occluder_yaw": float(yaw + rng.uniform(-0.02, 0.02)),
                    "divider_y": float(divider_y + rng.uniform(-0.01, 0.01)),
                    "side_camera_azimuth_deg": float(side_azimuth + rng.uniform(-2.0, 2.0)),
                    "side_camera_fovy": float(rng.uniform(18.0, 26.0)),
                    "high_camera_azimuth_deg": float(rng.uniform(-6.0, 6.0)),
                    "high_camera_fovy": float(rng.uniform(26.0, 34.0)),
                    "route_ribbons": True,
                }
            )
    return {
        "schema_version": 3,
        "protocol": "active_vision_v3",
        "rng": "numpy.PCG64",
        "manifest_seed": SEED,
        "scene_seed": SCENE_SEED,
        "families": list(SMOKE_FAMILIES),
        "layouts": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite manifest: {args.output}")
    write_json(args.output, build_manifest())


if __name__ == "__main__":
    main()
