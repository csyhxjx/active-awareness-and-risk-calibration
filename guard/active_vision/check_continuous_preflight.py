"""Archive and gate the Phase 6C physical preflight before any policy run."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from guard.active_vision.continuous_belief import ROUTES
from guard.active_vision.phase6a7 import sha256_file
from guard.json_io import write_json


CHECKER_VERSION = "phase6c-preflight-checker-v1"


def check(manifest_path: Path, result_dirs: list[Path]) -> dict:
    manifest = json.loads(manifest_path.read_text())
    paths = sorted(path for directory in result_dirs for path in directory.glob("continuous_probe_*.json"))
    rows = [json.loads(path.read_text()) for path in paths]
    errors = []
    if len(rows) != 24 or len({row["world_id"] for row in rows}) != 24:
        errors.append("preflight_world_scope")
    if any(row.get("manifest_sha256") != sha256_file(manifest_path) for row in rows):
        errors.append("manifest_hash")
    if any(len(row.get("routes", [])) != 3 for row in rows):
        errors.append("route_scope")
    route_rows = [route for row in rows for route in row.get("routes", [])]
    if len(route_rows) != 72:
        errors.append("route_count")
    matrix = Counter((row["expected_stratum"], row["observed_stratum"]) for row in route_rows)
    mismatches = [
        {"world_id": world["world_id"], **{
            key: route[key] for key in (
                "route", "expected_stratum", "observed_stratum", "minimum_clearance_m",
                "clearance_margin_m", "collision", "reached", "trajectory_sha256",
            )
        }}
        for world in rows for route in world.get("routes", []) if not route["stratum_match"]
    ]
    c0 = not errors and not mismatches
    return {
        "schema_version": 1, "checker_version": CHECKER_VERSION,
        "manifest_sha256": sha256_file(manifest_path),
        "preflight_result_sha256": {path.name: sha256_file(path) for path in paths},
        "scope": {"worlds": len(rows), "routes": len(route_rows), "fully_matching_worlds": sum(row.get("all_match", False) for row in rows)},
        "stratum_matrix": {f"{expected}->{observed}": count for (expected, observed), count in sorted(matrix.items())},
        "mismatches": mismatches, "errors": errors,
        "gates": {
            "C0_physical_truth": "PASS" if c0 else "FAIL",
            "C1_observation_integrity": "NOT_RUN",
            "C2_genuine_branching": "NOT_RUN",
            "C3_fixed_impossibility": "NOT_RUN",
            "C4_finite_inference": "NOT_RUN",
            "C5_permissions": "NOT_RUN",
            "C6_reference_stability": "NOT_RUN",
        },
        "all_pass": False,
        "terminal_action": "archive_failure_root_and_stop_before_policy_runner",
        "distribution_warning": "conditionally balanced probe; not an estimate of unconditional population performance",
    }


def report(result: dict) -> str:
    lines = [
        "# Phase 6C physical preflight", "",
        f"C0: **{result['gates']['C0_physical_truth']}**. Policy runner: **NOT RUN**.", "",
        f"Scope: {result['scope']['worlds']} worlds, {result['scope']['routes']} routes; "
        f"{result['scope']['fully_matching_worlds']} worlds match all requested strata.", "",
        "## Stratum matrix", "",
    ]
    lines.extend(f"- `{key}`: {value}" for key, value in result["stratum_matrix"].items())
    lines.extend([
        "", f"Mismatched routes: `{len(result['mismatches'])}`.", "",
        "This is an infrastructure/task-generation failure, not a MAP, PF, QMC, adaptive-view, or fixed-policy result.",
        "The balanced probe distribution is conditional and cannot be interpreted as unconditional population performance.", "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, action="append", required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    if args.archive.exists():
        raise FileExistsError(args.archive)
    args.archive.mkdir(parents=True)
    result = check(args.manifest, args.result_dir)
    raw = args.archive / "preflight_results"
    raw.mkdir()
    for directory in args.result_dir:
        for path in directory.glob("continuous_probe_*.json"):
            shutil.copyfile(path, raw / path.name)
    shutil.copyfile(args.manifest, args.archive / "candidate_manifest.json")
    write_json(args.archive / "preflight_check.json", result)
    (args.archive / "preflight_report.md").write_text(report(result), encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["all_pass"] else 1)


if __name__ == "__main__":
    main()
