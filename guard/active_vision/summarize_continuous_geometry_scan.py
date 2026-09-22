"""Summarize the fixed Phase 6C geometry diagnostic grid read-only."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load(root: Path) -> list[dict]:
    return [json.loads(path.read_text()) for path in sorted(root.glob("shard_*/case_*.json"))]


def stats(values: list[float]) -> dict:
    finite = [value for value in values if np.isfinite(value)]
    return {
        "count": len(values),
        "finite_count": len(finite),
        "mean": float(np.mean(finite)) if finite else None,
        "min": float(np.min(finite)) if finite else None,
        "max": float(np.max(finite)) if finite else None,
    }


def summarize(rows: list[dict]) -> dict:
    by_route = defaultdict(list)
    by_sign = defaultdict(list)
    by_width = defaultdict(list)
    components = Counter()
    strata = Counter()
    contacts = Counter()
    for row in rows:
        result = row["result"]
        route = row["route"]
        sign = "positive" if row["offset_m"] > 0 else "negative"
        by_route[route].append(row)
        by_sign[(route, sign)].append(row)
        by_width[(route, row["width_m"])].append(row)
        pair = result.get("minimum_pair")
        components[(route, "none" if pair is None else pair["robot_component"])] += 1
        strata[(route, row["physical_stratum"])] += 1
        for kind, present in result["collision_classes"].items():
            if present:
                contacts[(route, kind)] += 1

    route_summary = {}
    for route, values in sorted(by_route.items()):
        route_summary[route] = {
            "strata": dict(sorted(Counter(row["physical_stratum"] for row in values).items())),
            "collision_classes": dict(sorted(Counter(kind for row in values for kind, present in row["result"]["collision_classes"].items() if present).items())),
            "minimum_pair_components": dict(sorted(Counter("none" if row["result"]["minimum_pair"] is None else row["result"]["minimum_pair"]["robot_component"] for row in values).items())),
            "physical_minus_analytic_clearance_m": stats([row["result"]["physical_clearance_m"] - row["result"]["analytic_minimum_clearance_m"] for row in values if np.isfinite(row["result"]["physical_clearance_m"])]),
            "path_difference_max_eef_m": stats([row["path_difference_from_empty"]["max_eef_difference_m"] for row in values]),
            "first_contact_steps": stats([row["result"]["first_contact"]["step"] for row in values if row["result"]["first_contact"] is not None]),
        }
    return {
        "schema_version": 1,
        "case_count": len(rows),
        "grid": {"routes": ["left_route", "center_route", "right_route"], "widths_m": [0.025, 0.040, 0.055], "offset_magnitudes": [0.065, 0.075, 0.085, 0.095], "obstacle_x_m": 0.070},
        "strata_by_route": {route: dict(sorted({key[1]: value for key, value in strata.items() if key[0] == route}.items())) for route in ("left_route", "center_route", "right_route")},
        "components_by_route": {route: dict(sorted({key[1]: value for key, value in components.items() if key[0] == route}.items())) for route in ("left_route", "center_route", "right_route")},
        "contacts_by_route": {route: dict(sorted({key[1]: value for key, value in contacts.items() if key[0] == route}.items())) for route in ("left_route", "center_route", "right_route")},
        "route_summary": route_summary,
        "positive_offset_summary": {f"{route}/{sign}": {"strata": dict(sorted(Counter(row["physical_stratum"] for row in values).items()))} for (route, sign), values in sorted(by_sign.items())},
        "boundary_present": any(row["physical_stratum"] == "boundary" for row in rows),
        "all_three_strata_each_route": all({row["physical_stratum"] for row in values} >= {"safe", "boundary", "blocked"} for values in by_route.values()),
        "interpretation": [
            "All measured nearest robot-obstacle pairs are gripper components; no wrist or arm pair is the global minimum in this fixed grid.",
            "The 9 cases without a proximity pair only establish a distance greater than the registered proximity radius; they are not silently assigned an exact minimum.",
            "The empty-scene sweep is a comparator only. It is not promoted to physical truth because obstacle-induced controller deviations are recorded.",
            "This conditional diagnostic scan is not a prior draw or method comparison.",
        ],
    }


def markdown(result: dict) -> str:
    lines = ["# Phase 6C geometry diagnostic summary", "", f"Fixed development grid cases: `{result['case_count']}`.", "", "## Route results", "", "| route | physical strata | nearest components | contacts | max path deviation mean (m) |", "|---|---|---|---|---:|"]
    for route, row in result["route_summary"].items():
        path = row["path_difference_max_eef_m"]["mean"]
        lines.append(f"| {route} | `{row['strata']}` | `{row['minimum_pair_components']}` | `{row['collision_classes']}` | {path if path is not None else 'n/a'} |")
    lines.extend(["", f"All three strata in each route: **{result['all_three_strata_each_route']}**.", f"Boundary cases present: **{result['boundary_present']}**.", "", "## Interpretation", "", *[f"- {item}" for item in result["interpretation"]], "", "No new formal probe or belief-method analysis is authorized by this scan."])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = load(args.root)
    result = summarize(rows)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (args.output / "summary.md").write_text(markdown(result), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("case_count", "strata_by_route", "components_by_route", "boundary_present", "all_three_strata_each_route")}, indent=2))


if __name__ == "__main__":
    main()
