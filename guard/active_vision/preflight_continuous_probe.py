"""Physical preflight for one Phase 6C probe world."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path

import numpy as np

from guard.active_vision.continuous_belief import ROUTES
from guard.active_vision.continuous_physics import execute_continuous_route
from guard.active_vision.continuous_scene import ContinuousGeometryEnv
from guard.active_vision.phase6a7 import sha256_bytes, sha256_file
from guard.active_vision.runtime import physical_state, state_hash
from guard.json_io import write_json


PREFLIGHT_VERSION = "phase6c-preflight-v1"


def git_head():
    return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def classify(result):
    margin = result["clearance_margin_m"]
    if result["collision"] or margin <= -0.004: return "blocked"
    if abs(margin) <= 0.003: return "boundary"
    if result["collision_free_success"] and margin >= 0.008: return "safe"
    return "gap"


def run_world(world: dict, seed: int) -> dict:
    routes = []
    for route in ROUTES:
        random.seed(seed); np.random.seed(seed)
        env = ContinuousGeometryEnv(world["state"]); env.reset()
        try:
            branch_hash = state_hash(physical_state(env))
            result = execute_continuous_route(env, route)
        finally:
            env.close()
        observed = classify(result)
        expected = world["requested_strata"][route]
        routes.append({key: value for key, value in result.items() if key != "records"} | {
            "branch_state_hash": branch_hash, "observed_stratum": observed,
            "expected_stratum": expected, "stratum_match": observed == expected,
        })
    return {"world_id": world["world_id"], "world_index": world["world_index"], "routes": routes, "all_match": all(row["stratum_match"] for row in routes)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--world-id")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if args.world_id:
        if args.output is None or args.output_dir is not None:
            raise ValueError("single world requires --output only")
        worlds = [next(row for row in manifest["worlds"] if row["world_id"] == args.world_id)]
        outputs = [args.output]
    else:
        if args.output_dir is None or args.output is not None:
            raise ValueError("sharded mode requires --output-dir only")
        if not 0 <= args.shard_index < args.shard_count:
            raise ValueError("invalid shard")
        args.output_dir.mkdir(parents=True, exist_ok=False)
        worlds = [row for index, row in enumerate(manifest["worlds"]) if index % args.shard_count == args.shard_index]
        outputs = [args.output_dir / f"{row['world_id']}.json" for row in worlds]
    summaries = []
    for world, output in zip(worlds, outputs):
        result = run_world(world, manifest["generator_seed"])
        payload = {
            "schema_version": 1, "preflight_version": PREFLIGHT_VERSION, "git_head": git_head(),
            "manifest_sha256": sha256_file(args.manifest), "generator_seed": manifest["generator_seed"],
            **result,
        }
        write_json(output, payload)
        summaries.append({"world_id": result["world_id"], "all_match": result["all_match"]})
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
