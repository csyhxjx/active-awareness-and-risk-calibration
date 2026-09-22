"""Fixture-only nominal trajectory recording and fixed geometry scan."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from guard.active_vision.continuous_nominal_calibration import (
    WIDTHS_M, empty_state, load_nominal, make_environment, measure_replay,
    record_nominal, scan_cases, set_target_obstacle,
)
from guard.active_vision.continuous_scene import ContinuousGeometryEnv
from guard.active_vision.phase6a7 import sha256_file
from guard.json_io import write_json


PROTOCOL = Path(__file__).parents[2] / "continuous_nominal_trajectory_amendment_v1.md"
NAMESPACE = "continuous_nominal_calibration_v1"


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROTOCOL.parent, capture_output=True,
                          text=True, check=True).stdout.strip()


def record(route: str, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    env = ContinuousGeometryEnv(empty_state())
    env.reset()
    try:
        nominal = record_nominal(env, route)
        if not nominal["reached"]:
            raise RuntimeError(f"nominal route {route} did not reach goal")
    finally:
        env.close()
    write_json(destination, {**nominal, "protocol_sha256": sha256_file(PROTOCOL), "source_head": git_head(),
                             "empty_obstacle_state": empty_state()})


def measure_case(nominal: dict, route: str, width: float, offset: float,
                 intervals: int, keep_steps: bool) -> dict:
    env = make_environment(route, width)
    try:
        set_target_obstacle(env, route, offset)
        return measure_replay(env, nominal, intervals=intervals, keep_steps=keep_steps)
    finally:
        env.close()


def scan(route: str, nominal_path: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    nominal = load_nominal(nominal_path)
    if nominal["route"] != route or not nominal["reached"]:
        raise ValueError("wrong or incomplete nominal route")
    check_env = make_environment(route)
    try:
        distances = []
        for offset in (0.085, 0.090, 0.095):
            set_target_obstacle(check_env, route, offset)
            measured = measure_replay(check_env, nominal, keep_steps=False)["minimum_clearance_m"]
            if measured is None:
                raise RuntimeError("proximity query disappeared in continuity check")
            distances.append(measured)
        if max(abs(left - right) for left, right in zip(distances, distances[1:])) >= 0.010:
            raise RuntimeError("proximity query is discontinuous in continuity check")
    finally:
        check_env.close()
    output.mkdir(parents=True)
    current_width = None
    env = None
    try:
        for index, case in enumerate(scan_cases(route)):
            width = case["width_m"]
            if width != current_width:
                if env is not None:
                    env.close()
                env = make_environment(route, width)
                current_width = width
            set_target_obstacle(env, route, case["offset_m"])
            result = measure_replay(env, nominal)
            write_json(output / f"case_{index:03d}.json", {
                "schema_version": 1, "namespace": NAMESPACE, "protocol_sha256": sha256_file(PROTOCOL),
                "source_head": git_head(), "nominal_file_sha256": sha256_file(nominal_path),
                "case_index": index, **case, "result": result,
            })
    finally:
        if env is not None:
            env.close()
    print(json.dumps({"route": route, "count": len(scan_cases(route)), "output": str(output)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-route", choices=("left_route", "center_route", "right_route"))
    parser.add_argument("--scan-route", choices=("left_route", "center_route", "right_route"))
    parser.add_argument("--nominal", type=Path)
    parser.add_argument("--width", type=float, choices=WIDTHS_M)
    parser.add_argument("--offset", type=float)
    parser.add_argument("--intervals", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.record_route:
        record(args.record_route, args.output)
    elif args.scan_route:
        if args.nominal is None:
            raise ValueError("scan needs --nominal")
        if args.width is None and args.offset is None:
            scan(args.scan_route, args.nominal, args.output)
        elif args.width is not None and args.offset is not None:
            if args.output.exists():
                raise FileExistsError(args.output)
            result = measure_case(load_nominal(args.nominal), args.scan_route, args.width,
                                  args.offset, args.intervals, keep_steps=True)
            write_json(args.output, {"route": args.scan_route, "width_m": args.width,
                                      "offset_m": args.offset, "result": result})
        else:
            raise ValueError("specify both width and offset or neither")
    else:
        raise ValueError("--record-route or --scan-route required")


if __name__ == "__main__":
    main()
