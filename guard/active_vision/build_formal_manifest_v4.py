"""Build the frozen Phase 5E 60/20/40 grouped formal manifest."""

import argparse
from pathlib import Path

import numpy as np

from guard.active_vision.build_manifest_v2 import FAMILIES
from guard.json_io import write_json

SPLITS = {
    "train": (55060, [10] * 6, [(5, 5)] * 6),
    "validation": (56020, [4, 4, 3, 3, 3, 3], [(2, 2), (2, 2), (2, 1), (1, 2), (2, 1), (1, 2)]),
    "test": (57040, [6, 6, 7, 7, 7, 7], [(3, 3), (3, 3), (4, 3), (3, 4), (4, 3), (3, 4)]),
}


def build_manifest():
    rows = []
    for split, (seed, counts, mirror_counts) in SPLITS.items():
        rng = np.random.Generator(np.random.PCG64(seed)); split_index = 0
        for family_index, (family, yaw, divider_y, side_azimuth) in enumerate(FAMILIES):
            positive, negative = mirror_counts[family_index]; mirrors = [1] * positive + [-1] * negative; rng.shuffle(mirrors)
            if len(mirrors) != counts[family_index]: raise ValueError("family count")
            for family_member, mirror in enumerate(mirrors):
                rows.append({"layout_id": f"av4_{'val' if split == 'validation' else split}_{split_index:03d}",
                    "split": split, "family": family, "family_member": family_member, "manifest_seed": seed,
                    "start": [-0.103, 0.0, 1.01], "target": [float(rng.uniform(.18, .22)), 0., 1.01],
                    "lane_y": float(rng.uniform(.16, .20)), "obstacle_x": float(rng.uniform(.045, .085)),
                    "occluder_x": float(rng.uniform(.28, .32)), "mirror": mirror,
                    "occluder_yaw": float(yaw + rng.uniform(-.02, .02)), "divider_y": float(divider_y + rng.uniform(-.01, .01)),
                    "side_camera_azimuth_deg": float(side_azimuth + rng.uniform(-2, 2)), "side_camera_fovy": float(rng.uniform(18, 26)),
                    "high_camera_azimuth_deg": float(rng.uniform(-6, 6)), "high_camera_fovy": float(rng.uniform(26, 34)), "route_ribbons": True})
                split_index += 1
    return {"schema_version": 4, "protocol": "active_vision_v4", "rng": "numpy.PCG64", "scene_seed": 20260921, "layouts": rows}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("output", type=Path); args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    write_json(args.output, build_manifest())


if __name__ == "__main__": main()
