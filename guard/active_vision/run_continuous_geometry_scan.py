"""Run the preregistered Phase 6C geometry diagnostic grid."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from guard.active_vision.continuous_belief import ROUTES
from guard.active_vision.continuous_geometry_diagnostic import (
    classify_physical,
    enable_proximity_contacts,
    execute_geometry_diagnostic,
)
from guard.active_vision.continuous_scene import ContinuousGeometryEnv
from guard.json_io import write_json


WIDTHS_M = (0.025, 0.040, 0.055)
OFFSET_MAGNITUDES_M = (0.065, 0.075, 0.085, 0.095)
OBSTACLE_X_M = 0.070
NAMESPACE = "continuous_geometry_diagnostic_v1"


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def diagnostic_state(route: str, width: float, offset: float) -> list[float]:
    state = [0.0]
    for candidate in ROUTES:
        if candidate == route:
            state.extend((OBSTACLE_X_M, float(offset), float(width)))
        else:
            state.extend((2.0, 0.0, 0.025))
    return state


def empty_state() -> list[float]:
    return [0.0, 2.0, 0.0, 0.025, 2.0, 0.0, 0.025, 2.0, 0.0, 0.025]


def run(state: list[float], route: str) -> dict:
    env = ContinuousGeometryEnv(np.asarray(state, dtype=np.float64))
    env.reset()
    try:
        enable_proximity_contacts(env)
        return execute_geometry_diagnostic(env, route)
    finally:
        env.close()


def path_difference(records: list[dict], empty_records: list[dict], minimum_step: int | None) -> dict:
    count = min(len(records), len(empty_records))
    values = [float(np.linalg.norm(np.asarray(records[i]["eef"]) - np.asarray(empty_records[i]["eef"]))) for i in range(count)]
    return {
        "matched_steps": count,
        "max_eef_difference_m": max(values) if values else None,
        "difference_at_minimum_step_m": values[minimum_step] if minimum_step is not None and minimum_step < count else None,
    }


def cases() -> list[dict]:
    rows = []
    index = 0
    for route in ROUTES:
        for width in WIDTHS_M:
            for sign in (-1.0, 1.0):
                for magnitude in OFFSET_MAGNITUDES_M:
                    rows.append({"case_index": index, "route": route, "width_m": width, "offset_m": sign * magnitude})
                    index += 1
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    empty = {route: run(empty_state(), route) for route in ROUTES}
    write_json(args.output_dir / "empty_routes.json", {
        "schema_version": 1,
        "git_head": git_head(),
        "routes": {route: {key: value for key, value in result.items() if key != "records"} for route, result in empty.items()},
    })
    selected = [row for row in cases() if row["case_index"] % args.shard_count == args.shard_index]
    for case in selected:
        state = diagnostic_state(case["route"], case["width_m"], case["offset_m"])
        result = run(state, case["route"])
        minimum_step = None if result["minimum_pair"] is None else result["minimum_pair"]["step"]
        payload = {
            "schema_version": 1,
            "namespace": NAMESPACE,
            "git_head": git_head(),
            **case,
            "state": state,
            "physical_stratum": classify_physical(result),
            "path_difference_from_empty": path_difference(result["records"], empty[case["route"]]["records"], minimum_step),
            "result": result,
        }
        write_json(args.output_dir / f"case_{case['case_index']:03d}.json", payload)
    print(json.dumps({"shard": args.shard_index, "cases": len(selected)}, indent=2))


if __name__ == "__main__":
    main()
