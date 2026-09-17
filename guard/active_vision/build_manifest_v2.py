"""Build the deterministic Phase 5B 60/20/40 grouped layout manifest."""

import argparse
from pathlib import Path

import numpy as np

from guard.json_io import write_json


FAMILIES = (
    ("F0", -0.10, -0.035, -8.0),
    ("F1", -0.10, +0.035, +8.0),
    ("F2", 0.00, -0.035, 0.0),
    ("F3", 0.00, +0.035, 0.0),
    ("F4", +0.10, -0.035, +8.0),
    ("F5", +0.10, +0.035, -8.0),
)
SPLITS = {
    "train": {"seed": 51060, "counts": [10, 10, 10, 10, 10, 10], "mirrors": [(5, 5)] * 6},
    "validation": {
        "seed": 52020,
        "counts": [4, 4, 3, 3, 3, 3],
        "mirrors": [(2, 2), (2, 2), (2, 1), (1, 2), (2, 1), (1, 2)],
    },
    "test": {
        "seed": 53040,
        "counts": [6, 6, 7, 7, 7, 7],
        "mirrors": [(3, 3), (3, 3), (4, 3), (3, 4), (4, 3), (3, 4)],
    },
}


def build_manifest():
    rows = []
    for split, spec in SPLITS.items():
        rng = np.random.Generator(np.random.PCG64(spec["seed"]))
        split_index = 0
        for family_index, (family, yaw, divider_y, side_azimuth) in enumerate(FAMILIES):
            positive, negative = spec["mirrors"][family_index]
            mirrors = [1] * positive + [-1] * negative
            if len(mirrors) != spec["counts"][family_index]:
                raise ValueError(f"family count mismatch: {split}/{family}")
            rng.shuffle(mirrors)
            for family_member, mirror in enumerate(mirrors):
                rows.append(
                    {
                        "layout_id": f"av2_{'val' if split == 'validation' else split}_{split_index:03d}",
                        "split": split,
                        "family": family,
                        "family_member": family_member,
                        "manifest_seed": spec["seed"],
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
                split_index += 1
    return {
        "schema_version": 2,
        "protocol": "active_vision_v2",
        "rng": "numpy.PCG64",
        "scene_seed": 20260917,
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
