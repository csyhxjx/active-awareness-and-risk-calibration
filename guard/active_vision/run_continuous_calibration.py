"""Run the prospective Phase 6C physical calibration fixture only."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from guard.active_vision.continuous_physics import execute_continuous_route
from guard.active_vision.continuous_scene import ContinuousGeometryEnv, point_to_obstacle_clearance
from guard.active_vision.continuous_belief import ROUTES


CALIBRATION_SEED = 66530
ROUTE_TOLERANCE_M = 0.020
CLEARANCE_TOLERANCE_M = 0.003


def run_route(state: list[float], route: str) -> dict:
    env = ContinuousGeometryEnv(np.asarray(state, dtype=np.float64))
    env.reset()
    try:
        result = execute_continuous_route(env, route)
    finally:
        env.close()
    return {key: value for key, value in result.items() if key != "records"}


def empty_state() -> list[float]:
    # Obstacles remain in the scene graph but are placed outside the reachable
    # workspace; this keeps the controller and collision geometry unchanged.
    return [0.0, 2.0, 0.0, 0.025, 2.0, 0.0, 0.025, 2.0, 0.0, 0.025]


def mirror_error(left: dict, right: dict) -> float:
    # Re-run with records so the geometry comparison is based on controller
    # paths, not waypoint definitions.
    errors = []
    for route, sign in (("left_route", 1.0), ("right_route", -1.0)):
        env = ContinuousGeometryEnv(np.asarray(empty_state(), dtype=np.float64))
        env.reset()
        try:
            result = execute_continuous_route(env, route)
            if route == "left_route":
                left_records = result["records"]
            else:
                right_records = result["records"]
        finally:
            env.close()
    for lrow, rrow in zip(left_records, right_records):
        lp = np.asarray(lrow["eef"], dtype=np.float64)
        rp = np.asarray(rrow["eef"], dtype=np.float64)
        errors.append(float(np.max(np.abs(lp - np.asarray([rp[0], -rp[1], rp[2]])))))
    return max(errors) if errors else float("inf")


def selected_fixture_states(manifest: dict) -> list[dict]:
    chosen = {}
    for world in manifest["worlds"]:
        for route in ROUTES:
            stratum = world["requested_strata"][route]
            chosen.setdefault((route, stratum), world)
    return [{"route": route, "stratum": stratum, "world": world} for (route, stratum), world in sorted(chosen.items())]


def fresh_replay(state: list[float], route: str, directory: Path) -> dict:
    state_path = directory / f"state_{route}.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    outputs = []
    for index in (0, 1):
        output = directory / f"replay_{route}_{index}.json"
        subprocess.run([
            sys.executable, "-m", "guard.active_vision.run_continuous_calibration",
            "--single-state", str(state_path), "--route", route, "--output", str(output),
        ], check=True, cwd=Path(__file__).parents[2], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        outputs.append(json.loads(output.read_text()))
    encoded = [json.dumps(item, sort_keys=True, separators=(",", ":")).encode() for item in outputs]
    return {"route": route, "match": encoded[0] == encoded[1], "hashes": [hashlib.sha256(item).hexdigest() for item in encoded]}


def run_fixture(manifest: dict, output: Path) -> dict:
    output.mkdir(parents=True)
    empty = {route: run_route(empty_state(), route) for route in ROUTES}
    mirror = mirror_error(empty["left_route"], empty["right_route"])
    rows = []
    fixtures = selected_fixture_states(manifest)
    for fixture in fixtures:
        result = run_route(fixture["world"]["state"], fixture["route"])
        analytic = float(fixture["world"]["analytic_clearance_margins"][ROUTES.index(fixture["route"])])
        rows.append({"route": fixture["route"], "expected_stratum": fixture["stratum"], "observed": result,
                     "analytic_margin_m": analytic, "delta_m": result["clearance_margin_m"] - analytic})
    by_route = defaultdict(list)
    for row in rows:
        by_route[row["route"]].append(row)
    strata = Counter((row["route"], row["expected_stratum"], classify(row["observed"])) for row in rows)
    replay = []
    for route in ROUTES:
        fixture = next(item for item in fixtures if item["route"] == route)
        replay.append(fresh_replay(fixture["world"]["state"], route, output))
    return {
        "schema_version": 1, "calibration_seed": CALIBRATION_SEED,
        "scene_constants": {"lane_centers_m": [0.27, 0.0, -0.27], "target_z_m": 1.01, "obstacle_center_z_m": 0.96, "obstacle_longitudinal_half_m": 0.035, "obstacle_lateral_half_range_m": [0.025, 0.055]},
        "empty_scene": {"routes": {key: {k: v for k, v in value.items() if k != "records"} for key, value in empty.items()}, "left_right_mirror_max_error_m": mirror},
        "fixture_rows": rows,
        "stratum_matrix": {f"{a}/{b}/{c}": count for (a, b, c), count in sorted(strata.items())},
        "fresh_process_replay": replay,
        "gates": {
            "empty_path_symmetry": bool(mirror <= ROUTE_TOLERANCE_M and all(value["collision_free_success"] for value in empty.values())),
            "three_physical_strata": all(any(row["route"] == route and classify(row["observed"]) == stratum for row in rows) for route in ROUTES for stratum in ("safe", "boundary", "blocked")),
            "analytic_physical_tolerance": all(abs(row["delta_m"]) <= CLEARANCE_TOLERANCE_M for row in rows),
            "no_undeclared_neighbor_or_self_contact": all(not row["observed"]["collision_classes"][key] for row in rows for key in ("static_environment", "self_collision")),
            "fresh_process_replay": all(row["match"] for row in replay),
        },
        "source": "read-only calibration fixture using archived 66421 candidate states; no new probe manifest generated",
    }


def classify(result: dict) -> str:
    margin = result["clearance_margin_m"]
    if result["collision"] or margin <= -0.004:
        return "blocked"
    if abs(margin) <= 0.003:
        return "boundary"
    if result["collision_free_success"] and margin >= 0.008:
        return "safe"
    return "gap"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest")
    parser.add_argument("--output")
    parser.add_argument("--single-state")
    parser.add_argument("--route")
    args = parser.parse_args()
    if args.single_state:
        result = run_route(json.loads(Path(args.single_state).read_text()), args.route)
        Path(args.output).write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
        return
    if not args.manifest or not args.output:
        raise SystemExit("--manifest and --output are required")
    manifest = json.loads(Path(args.manifest).read_text())
    result = run_fixture(manifest, Path(args.output))
    (Path(args.output) / "calibration.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["gates"], indent=2))
    raise SystemExit(0 if all(result["gates"].values()) else 1)


if __name__ == "__main__":
    main()
