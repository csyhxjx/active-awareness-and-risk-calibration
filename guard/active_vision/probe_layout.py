"""Run one-layout / four-hidden-state Phase 5A mechanism probe."""

import argparse
import hashlib
import random
from pathlib import Path

import numpy as np
from PIL import Image

from guard.active_vision.runtime import QueryBroker, execute_route
from guard.active_vision.scene import CAMERAS, DEV_LAYOUT, HIDDEN_STATES, ROUTES, OccludedRouteEnv
from guard.json_io import write_json


def image_sha(image):
    return hashlib.sha256(np.asarray(image).tobytes()).hexdigest()


def execute_fresh(hidden_state, route):
    random.seed(0)
    np.random.seed(0)
    env = OccludedRouteEnv(layout=DEV_LAYOUT, hidden_state=hidden_state)
    try:
        env.reset()
        return execute_route(env, route)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite probe: {args.output_root}")
    args.output_root.mkdir(parents=True)
    summary = {"layout": DEV_LAYOUT.layout_id, "states": []}
    for hidden_state in HIDDEN_STATES:
        state_dir = args.output_root / hidden_state
        state_dir.mkdir()
        env = OccludedRouteEnv(layout=DEV_LAYOUT, hidden_state=hidden_state)
        try:
            env.reset()
            broker = QueryBroker(env, budget=3)
            images = {"v0": broker.free_observation()}
            for camera in CAMERAS[1:]:
                images[camera] = broker.query(camera)
            for camera, image in images.items():
                Image.fromarray(image).save(state_dir / f"{camera}.png")
            route_results = [execute_fresh(hidden_state, route) for route in ROUTES]
            write_json(state_dir / "routes.json", route_results)
            write_json(state_dir / "queries.json", broker.ledger)
            summary["states"].append(
                {
                    "hidden_state": hidden_state,
                    "image_sha256": {camera: image_sha(image) for camera, image in images.items()},
                    "routes": [{key: value for key, value in row.items() if key != "records"} for row in route_results],
                }
            )
        finally:
            env.close()
    v0_hashes = {row["image_sha256"]["v0"] for row in summary["states"]}
    summary["v0_pixel_identical"] = len(v0_hashes) == 1
    write_json(args.output_root / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
