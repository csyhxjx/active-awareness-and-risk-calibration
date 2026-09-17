"""Collect the frozen 12-layout Phase 5A development pilot."""

import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from guard.active_vision.runtime import QueryBroker, execute_route, physical_state, state_hash
from guard.active_vision.scene import CAMERAS, HIDDEN_STATES, ROUTES, Layout, OccludedRouteEnv
from guard.json_io import canonical_dumps, write_json


def _sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _git_head():
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _fresh_env(layout, hidden_state, seed):
    random.seed(seed)
    np.random.seed(seed)
    env = OccludedRouteEnv(layout=layout, hidden_state=hidden_state)
    env.reset()
    return env


def _execute_fresh(layout, hidden_state, route, config, expected_hash):
    env = _fresh_env(layout, hidden_state, config["seed"])
    try:
        actual_hash = state_hash(physical_state(env))
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"branch reconstruction mismatch: {layout.layout_id}/{hidden_state}/{route} "
                f"expected={expected_hash} actual={actual_hash}"
            )
        result = execute_route(
            env,
            route,
            max_steps=config["max_steps"],
            hold_steps=config["hold_steps"],
        )
        result["branch_state_hash"] = actual_hash
        return result
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite pilot: {args.output_root}")
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    layouts = [Layout(**row) for row in config["layouts"]]
    if len(layouts) != 12 or len({layout.layout_id for layout in layouts}) != 12:
        raise ValueError("pilot requires exactly 12 uniquely named layouts")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    args.output_root.mkdir(parents=True)
    summary = {
        "version": config["version"],
        "git_head": _git_head(),
        "config_sha256": _sha_bytes(config_bytes),
        "canonical_candidate_trajectories": 0,
        "replay_audits": 0,
        "layouts": [],
    }
    for layout in layouts:
        layout_summary = {"layout": layout.layout_id, "states": [], "replay": None}
        for hidden_state in HIDDEN_STATES:
            state_dir = args.output_root / layout.layout_id / hidden_state
            (state_dir / "images").mkdir(parents=True)
            env = _fresh_env(layout, hidden_state, config["seed"])
            try:
                env.reset()
                broker = QueryBroker(env, budget=config["cache_query_budget"])
                images = {"v0": broker.free_observation()}
                for camera in CAMERAS[1:]:
                    images[camera] = broker.query(camera)
                image_hashes = {}
                for camera, image in images.items():
                    image_hashes[camera] = _sha_bytes(np.asarray(image).tobytes())
                    Image.fromarray(image).save(state_dir / "images" / f"{camera}.png")
                initial_hash = state_hash(physical_state(env))
                route_results = []
                for route in ROUTES:
                    result = _execute_fresh(layout, hidden_state, route, config, initial_hash)
                    route_results.append(result)
                    summary["canonical_candidate_trajectories"] += 1
                write_json(state_dir / "routes.json", route_results)
                write_json(state_dir / "queries.json", broker.ledger)
                public_routes = {
                    route: [point.tolist() for point in env.route_waypoints(route)] for route in ROUTES
                }
                state_summary = {
                    "hidden_state": hidden_state,
                    "image_sha256": image_hashes,
                    "public_routes": public_routes,
                    "routes": [
                        {key: value for key, value in row.items() if key != "records"}
                        for row in route_results
                    ],
                }
                layout_summary["states"].append(state_summary)
                if hidden_state == "10":
                    replay = _execute_fresh(layout, hidden_state, "left_route", config, initial_hash)
                    reference = route_results[0]
                    layout_summary["replay"] = {
                        "state": hidden_state,
                        "route": "left_route",
                        "reference_sha256": _sha_bytes(canonical_dumps(reference).encode()),
                        "replay_sha256": _sha_bytes(canonical_dumps(replay).encode()),
                        "exact": canonical_dumps(reference) == canonical_dumps(replay),
                    }
                    summary["replay_audits"] += 1
            finally:
                env.close()
        summary["layouts"].append(layout_summary)
        write_json(args.output_root / "summary.partial.json", summary)
    if summary["canonical_candidate_trajectories"] != 96:
        raise RuntimeError("canonical trajectory count is not 96")
    write_json(args.output_root / "summary.json", summary)
    (args.output_root / "summary.partial.json").unlink()
    print(f"collected {len(layouts)} layouts and 96 canonical trajectories")


if __name__ == "__main__":
    main()
