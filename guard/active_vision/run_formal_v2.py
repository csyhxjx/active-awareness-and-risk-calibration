"""Collect the opened train split for the frozen Phase 5B formal corpus."""

import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from guard.active_vision.run_pilot import _execute_fresh, _fresh_env
from guard.active_vision.runtime import QueryBroker, physical_state, state_hash
from guard.active_vision.scene import CAMERAS, HIDDEN_STATES, ROUTES, Layout
from guard.json_io import canonical_dumps, write_json


LAYOUT_KEYS = set(Layout.__dataclass_fields__)


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def git_revision():
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout.strip()
    )
    return head, dirty


def layout_from_row(row):
    return Layout(**{key: value for key, value in row.items() if key in LAYOUT_KEYS})


def require_open_split(split):
    if split != "train":
        raise PermissionError("prospective clarification A opens only the train split")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    args = parser.parse_args()
    require_open_split(args.split)
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite formal corpus: {args.output_root}")
    manifest_bytes = args.manifest.read_bytes()
    protocol_bytes = args.protocol.read_bytes()
    manifest = json.loads(manifest_bytes)
    rows = [row for row in manifest["layouts"] if row["split"] == args.split]
    if len(rows) != 60 or len({row["layout_id"] for row in rows}) != 60:
        raise ValueError("train split must contain exactly 60 unique layouts")
    head, dirty = git_revision()
    if dirty:
        raise RuntimeError("formal collection requires a clean Guard worktree")
    args.output_root.mkdir(parents=True)
    config = {"seed": manifest["scene_seed"], "max_steps": 100, "hold_steps": 5}
    summary = {
        "schema_version": 2,
        "protocol": manifest["protocol"],
        "split": args.split,
        "guard_head": head,
        "guard_dirty": dirty,
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "protocol_sha256": sha256_bytes(protocol_bytes),
        "canonical_candidate_trajectories": 0,
        "replay_audits": 0,
        "layouts": [],
    }
    for row in rows:
        layout = layout_from_row(row)
        layout_summary = {"layout": row, "states": [], "replay": None}
        for hidden_state in HIDDEN_STATES:
            state_dir = args.output_root / layout.layout_id / hidden_state
            (state_dir / "images").mkdir(parents=True)
            env = _fresh_env(layout, hidden_state, config["seed"])
            try:
                broker = QueryBroker(env, budget=3)
                images = {"v0": broker.free_observation()}
                for camera in CAMERAS[1:]:
                    images[camera] = broker.query(camera)
                image_hashes = {}
                for camera, image in images.items():
                    image_hashes[camera] = sha256_bytes(np.asarray(image).tobytes())
                    Image.fromarray(image).save(state_dir / "images" / f"{camera}.png")
                initial_hash = state_hash(physical_state(env))
                route_results = []
                for route in ROUTES:
                    result = _execute_fresh(layout, hidden_state, route, config, initial_hash)
                    route_results.append(result)
                    summary["canonical_candidate_trajectories"] += 1
                public_routes = {
                    route: [point.tolist() for point in env.route_waypoints(route)] for route in ROUTES
                }
                public_layout = {
                    key: value for key, value in row.items() if key not in {"manifest_seed"}
                }
                write_json(state_dir / "routes.json", route_results)
                write_json(state_dir / "queries.json", broker.ledger)
                write_json(
                    state_dir / "meta.json",
                    {
                        "layout_id": layout.layout_id,
                        "split": args.split,
                        "hidden_state": hidden_state,
                        "guard_head": head,
                        "manifest_sha256": summary["manifest_sha256"],
                        "protocol_sha256": summary["protocol_sha256"],
                        "branch_state_hash": initial_hash,
                    },
                )
                layout_summary["states"].append(
                    {
                        "hidden_state": hidden_state,
                        "image_sha256": image_hashes,
                        "public_layout": public_layout,
                        "public_routes": public_routes,
                        "routes": [
                            {key: value for key, value in result.items() if key != "records"}
                            for result in route_results
                        ],
                    }
                )
                if hidden_state == "10":
                    replay = _execute_fresh(layout, hidden_state, "left_route", config, initial_hash)
                    reference = route_results[0]
                    layout_summary["replay"] = {
                        "hidden_state": hidden_state,
                        "route": "left_route",
                        "reference_sha256": sha256_bytes(canonical_dumps(reference).encode()),
                        "replay_sha256": sha256_bytes(canonical_dumps(replay).encode()),
                        "exact": canonical_dumps(reference) == canonical_dumps(replay),
                    }
                    summary["replay_audits"] += 1
            finally:
                env.close()
        summary["layouts"].append(layout_summary)
        write_json(args.output_root / "summary.partial.json", summary)
    if summary["canonical_candidate_trajectories"] != 480 or summary["replay_audits"] != 60:
        raise RuntimeError("formal train scope does not match 480 trajectories / 60 replays")
    write_json(args.output_root / "summary.json", summary)
    (args.output_root / "summary.partial.json").unlink()
    print("collected train: 60 layouts, 240 scenes, 480 canonical trajectories, 60 replays")


if __name__ == "__main__":
    main()
