"""Export, gate, and run the isolated Phase 6C nominal calibration backend."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from guard.active_vision.continuous_nominal_backend_v2 import (
    ROUTES, backend_request, export_scene, run_worker, sha256_file,
)
from guard.active_vision.continuous_nominal_calibration import (
    OFFSET_MAGNITUDES_M, WIDTHS_M, make_environment, measure_replay,
    scan_cases, set_target_obstacle,
)
from guard.json_io import write_json


def _requests(manifest, route, offsets=(0.085, 0.090, 0.095), width=0.025, intervals=8):
    nominal = manifest["nominals"][route]
    return [backend_request(route, width, offset, nominal["qpos"], intervals) for offset in offsets]


def gate(export_dir: Path, backend_python: str) -> dict:
    manifest = json.loads((export_dir / "export_manifest.json").read_text())
    if manifest["xml_sha256"] != sha256_file(export_dir / "scene.xml"):
        raise RuntimeError("export hash identity failed")
    import hashlib
    for route, nominal in manifest["nominals"].items():
        qpos_hash = hashlib.sha256(json.dumps(nominal["qpos"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        if qpos_hash != nominal.get("qpos_sha256"):
            raise RuntimeError(f"nominal qpos hash identity failed: {route}")
    all_requests = []
    for route in ROUTES:
        all_requests.extend(_requests(manifest, route))
    first = run_worker(export_dir, all_requests, backend_python)
    second = run_worker(export_dir, all_requests, backend_python)
    if first != second:
        raise RuntimeError("subprocess replay is not byte-identical")
    results = [json.loads(line) for line in first.splitlines()]
    continuity = {}
    for route in ROUTES:
        values = [r["minimum_clearance_m"] for r in results if r["route"] == route]
        if any(v is None or not np.isfinite(v) for v in values):
            raise RuntimeError(f"non-finite continuity distance: {route}")
        deltas = [abs(a - b) for a, b in zip(values, values[1:])]
        continuity[route] = {"distances_m": values, "adjacent_delta_m": deltas, "pass": max(deltas, default=0) <= 0.010}
        if not continuity[route]["pass"]:
            raise RuntimeError(f"distance continuity failed: {route}")

    # Compare signs only where legacy 2.3.7 reports an actual stable contact.
    signs = []
    for route in ROUTES:
        # These offsets are deep, repeatable penetrations in the legacy
        # contact stream; the adjacent .085/.090/.095 continuity probe is
        # intentionally kept separate from the sign check.
        for offset in (-0.080, 0.080):
            env = make_environment(route)
            try:
                set_target_obstacle(env, route, offset)
                old = measure_replay(env, manifest["nominals"][route], keep_steps=False)
            finally:
                env.close()
            if old["first_contact"] is None:
                continue
            req = backend_request(route, 0.025, offset, manifest["nominals"][route]["qpos"])
            fresh = json.loads(run_worker(export_dir, [req], backend_python).decode())
            old_sign = old["minimum_clearance_m"] < 0
            new_sign = fresh["minimum_clearance_m"] < 0
            signs.append({"route": route, "offset_m": offset, "old_negative": old_sign, "new_negative": new_sign})
            if old_sign != new_sign:
                raise RuntimeError(f"stable-contact sign disagreement: {route} {offset}")
    if not signs:
        raise RuntimeError("no stable legacy contact case available for sign gate")
    if any(r["mujoco_version"] != "3.13.0" or not r["native_ccd"] for r in results):
        raise RuntimeError("backend provenance/version gate failed")
    report = {
        "schema_version": 2, "backend": "mujoco_native_ccd", "backend_version": results[0]["mujoco_version"],
        "native_ccd": results[0]["native_ccd"], "export_xml_sha256": manifest["xml_sha256"],
        "continuity": continuity, "stable_contact_signs": signs,
        "subprocess_replay_sha256": __import__("hashlib").sha256(first).hexdigest(),
        "gates": {"export_hash": True, "continuity": True, "subprocess_replay": True,
                  "stable_contact_sign": True, "version_native_ccd": True},
    }
    write_json(export_dir / "backend_gates.json", report)
    return report


def run_grid(export_dir: Path, backend_python: str, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((export_dir / "export_manifest.json").read_text())
    gates = json.loads((export_dir / "backend_gates.json").read_text())
    if not all(gates["gates"].values()):
        raise RuntimeError("backend gates are incomplete")
    output.mkdir(parents=True)
    cases = []
    for route in ROUTES:
        for case in scan_cases(route):
            cases.append(backend_request(route, case["width_m"], case["offset_m"], manifest["nominals"][route]["qpos"], 8))
    raw = run_worker(export_dir, cases, backend_python)
    for index, line in enumerate(raw.splitlines()):
        result = json.loads(line)
        write_json(output / f"case_{index:03d}.json", {"schema_version": 2, "namespace": "continuous_nominal_calibration_v2",
            "case_index": index, "route": result["route"], "width_m": result["width_m"], "offset_m": result["offset_m"],
            "nominal_trajectory_sha256": manifest["nominals"][result["route"]]["trajectory_sha256"], "result": result})
    if len(raw.splitlines()) != 828:
        raise RuntimeError("calibration grid did not produce exactly 828 cases")


def summarize_grid(export_dir: Path, backend_python: str, grid: Path, report_path: Path) -> None:
    manifest = json.loads((export_dir / "export_manifest.json").read_text())
    records = [json.loads(path.read_text()) for path in sorted(grid.glob("case_*.json"))]
    if len(records) != 828:
        raise RuntimeError("summary requires complete 828-case grid")
    strata = {}
    refine = []
    for record in records:
        clearance = record["result"]["minimum_clearance_m"]
        if clearance is None:
            raise RuntimeError("right-censored distance cannot be classified")
        margin = clearance - 0.004
        if clearance <= 0:
            stratum = "blocked"
        elif margin >= 0.008:
            stratum = "safe"
        elif abs(margin) <= 0.003:
            stratum = "boundary"
        else:
            stratum = "gap"
        record["stratum"] = stratum
        strata.setdefault(record["route"], {}).setdefault(stratum, 0)
        strata[record["route"]][stratum] += 1
        if abs(margin) <= 0.010:
            refine.append(record)
    requests = [
        backend_request(r["route"], r["width_m"], r["offset_m"], manifest["nominals"][r["route"]]["qpos"], 16)
        for r in refine
    ]
    refined = [json.loads(line) for line in run_worker(export_dir, requests, backend_python).splitlines()]
    errors = []
    flips = []
    for r, high in zip(refine, refined):
        low = r["result"]["minimum_clearance_m"]
        hi = high["minimum_clearance_m"]
        errors.append(abs(low - hi))
        if (low - 0.004) * (hi - 0.004) < 0:
            flips.append({"case_index": r["case_index"], "c8": low, "c16": hi})
    report = {
        "schema_version": 2, "namespace": "continuous_nominal_calibration_v2", "case_count": len(records),
        "strata": strata, "refinement_case_count": len(refine), "max_abs_c8_c16_error_m": max(errors, default=0.0),
        "wrong_side_count": len(flips), "wrong_side_cases": flips,
        "gates": {"all_routes_have_safe_boundary_blocked": all(all(k in strata.get(route, {}) for k in ("safe", "boundary", "blocked")) for route in ROUTES),
                  "c8_c16_max_error": max(errors, default=0.0) <= 0.0005, "c8_c16_zero_side_flip": not flips},
        "cases": records,
    }
    write_json(report_path, report)
    if not all(report["gates"].values()):
        raise RuntimeError(f"calibration gates failed: {report['gates']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--backend-python", required=True)
    parser.add_argument("--export-scene", action="store_true")
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--run-grid", type=Path)
    parser.add_argument("--summarize-grid", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.export_scene:
        export_scene(args.export)
    if args.gate:
        gate(args.export, args.backend_python)
    if args.run_grid is not None:
        run_grid(args.export, args.backend_python, args.run_grid)
    if args.summarize_grid is not None:
        if args.report is None:
            raise ValueError("--summarize-grid requires --report")
        summarize_grid(args.export, args.backend_python, args.summarize_grid, args.report)


if __name__ == "__main__":
    main()
