"""Generate and statically audit the frozen seed-66421 Phase 6C probe."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from guard.active_vision.continuous_belief import CLEARANCE_THRESHOLD_M, ROUTES, route_clearances
from guard.active_vision.phase6a7 import sha256_file
from guard.json_io import write_json


PROTOCOL_ID = "continuous_active_vision_protocol_v1"
PROTOCOL_FILE = "continuous_active_vision_protocol_v1.md"
NAMESPACE = "continuous_active_vision_probe_v1"
GENERATOR_SEED = 66421
PF_SEED = 66422
BOOTSTRAP_SEED = 66423
STRATA = ("safe", "boundary", "blocked")
PATTERNS = (("safe", "boundary", "blocked"), ("boundary", "blocked", "safe"), ("blocked", "safe", "boundary"))


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def _stratified_values(rng, count=24):
    unit = np.empty((count, 10), dtype=np.float64)
    for column in range(10):
        unit[:, column] = (rng.permutation(count) + rng.random(count)) / count
    return unit


def build(protocol: Path) -> dict:
    rng = np.random.default_rng(GENERATOR_SEED)
    unit = _stratified_values(rng)
    worlds = []
    occurrences = {(route, stratum): 0 for route in ROUTES for stratum in STRATA}
    for index in range(24):
        pattern = PATTERNS[index % len(PATTERNS)]
        sign = 1.0 if index % 2 == 0 else -1.0
        g = sign * (0.980 + 0.015 * unit[index, 0])
        state = [g]
        target_margins = []
        for route_index, (route, stratum) in enumerate(zip(ROUTES, pattern)):
            source = 1 + route_index * 3
            x = 0.04 + 0.06 * unit[index, source]
            if stratum == "safe":
                width = 0.025 + 0.0005 * unit[index, source + 2]
                margin = 0.0080 + 0.0004 * unit[index, source + 1]
            elif stratum == "boundary":
                width = 0.025 + 0.005 * unit[index, source + 2]
                margin = -0.0025 + 0.005 * unit[index, source + 1]
            else:
                width = 0.035 + 0.020 * unit[index, source + 2]
                margin = -0.050 + 0.025 * unit[index, source + 1]
            displacement = sign * (width + 0.060 + CLEARANCE_THRESHOLD_M + margin)
            epsilon = displacement - 0.030 * g
            if not -0.070 <= epsilon <= 0.070:
                raise RuntimeError(f"conditional stratum leaves registered prior support: {index}/{route}")
            state.extend((x, displacement, width))
            target_margins.append(margin)
            occurrences[(route, stratum)] += 1
        actual = (route_clearances(np.asarray(state).reshape(1, -1))[0] - CLEARANCE_THRESHOLD_M).tolist()
        if not np.allclose(actual, target_margins, atol=1e-12):
            raise RuntimeError("stratum construction mismatch")
        worlds.append({
            "world_id": f"continuous_probe_{index:03d}", "world_index": index,
            "requested_strata": dict(zip(ROUTES, pattern)), "state": state,
            "analytic_clearance_margins": actual,
        })
    return {
        "schema_version": 1, "protocol_id": PROTOCOL_ID, "protocol_sha256": sha256_file(protocol),
        "generator_head": git_head(), "namespace": NAMESPACE, "generator_seed": GENERATOR_SEED,
        "pf_seed": PF_SEED, "bootstrap_seed": BOOTSTRAP_SEED,
        "layout": {"layout_id": "continuous_probe_layout_00", "lane_y": 0.27, "obstacle_center_z": 0.96},
        "world_distribution": "conditionally balanced mechanism-stress probe; not the unconditional registered prior",
        "formal_distribution_lock": "a future addendum must freeze stratum proportions before formal generation",
        "selection_uses_policy_results": False,
        "world_count": len(worlds), "route_count": len(worlds) * 3,
        "required_stratum_counts": {route: {stratum: 8 for stratum in STRATA} for route in ROUTES},
        "worlds": worlds,
    }


def audit(manifest: dict, protocol: Path) -> dict:
    errors = []
    worlds = manifest.get("worlds", [])
    if manifest.get("protocol_sha256") != sha256_file(protocol): errors.append("protocol_sha256")
    if manifest.get("generator_seed") != GENERATOR_SEED: errors.append("generator_seed")
    if len(worlds) != 24 or len({row["world_id"] for row in worlds}) != 24: errors.append("world_scope")
    if manifest.get("route_count") != 72: errors.append("route_count")
    if manifest.get("selection_uses_policy_results") is not False: errors.append("policy_filtering")
    counts = {route: {stratum: 0 for stratum in STRATA} for route in ROUTES}
    for world in worlds:
        state = np.asarray(world["state"], dtype=np.float64)
        if state.shape != (10,) or not np.isfinite(state).all(): errors.append("state_schema"); continue
        for route, stratum in world["requested_strata"].items(): counts[route][stratum] += 1
    if counts != manifest.get("required_stratum_counts"): errors.append("stratum_counts")
    if "not the unconditional" not in manifest.get("world_distribution", ""): errors.append("distribution_honesty")
    return {"all_pass": not errors, "errors": sorted(set(errors)), "counts": counts, "world_count": len(worlds), "route_count": 3 * len(worlds)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--protocol", type=Path, default=Path(PROTOCOL_FILE))
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    manifest = build(args.protocol)
    manifest["static_audit"] = audit(manifest, args.protocol)
    write_json(args.output, manifest)
    print(json.dumps(manifest["static_audit"], indent=2))
    raise SystemExit(0 if manifest["static_audit"]["all_pass"] else 1)


if __name__ == "__main__":
    main()
