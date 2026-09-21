"""Read-only attribution of the archived seed-66421 C0 failure."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(manifest_path: Path, results_dir: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads(manifest_path.read_text())
    rows = [json.loads(path.read_text()) for path in sorted(results_dir.glob("continuous_probe_*.json"))]
    by_id = {row["world_id"]: row for row in rows}
    routes = []
    for world in manifest["worlds"]:
        result = by_id[world["world_id"]]
        analytic = dict(zip(("left_route", "center_route", "right_route"), world["analytic_clearance_margins"]))
        for route in result["routes"]:
            route = dict(route)
            route["analytic_margin_m"] = float(analytic[route["route"]])
            route["physical_margin_m"] = float(route["clearance_margin_m"])
            route["margin_delta_m"] = route["physical_margin_m"] - route["analytic_margin_m"]
            route["world_id"] = world["world_id"]
            routes.append(route)
    return manifest, routes


def _mean_sd(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean_m": None, "sd_m": None, "min_m": None, "max_m": None}
    array = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "mean_m": float(array.mean()), "sd_m": float(array.std(ddof=0)),
            "min_m": float(array.min()), "max_m": float(array.max())}


def attribute(manifest: dict, routes: list[dict]) -> dict:
    by_route = defaultdict(list)
    by_expected = defaultdict(list)
    contacts = defaultdict(int)
    for route in routes:
        by_route[route["route"]].append(route["margin_delta_m"])
        by_expected[route["expected_stratum"]].append(route["margin_delta_m"])
        classes = route["collision_classes"]
        for name, present in classes.items():
            if present:
                contacts[name] += 1
        for event in route.get("first_contacts", []):
            contacts[f"first_contact:{event['class']}"] += 1
            geom_pair = sorted((event.get("geom1", ""), event.get("geom2", "")))
            if any("continuous_obstacle_" in geom for geom in geom_pair):
                own = route["route"].replace("_route", "")
                other = next((name for name in ("left", "center", "right") if name in geom_pair[0] + geom_pair[1]), "unknown")
                if other != own:
                    contacts["adjacent_obstacle_contact"] += 1
    mismatch = [route for route in routes if not route["stratum_match"]]
    return {
        "schema_version": 1,
        "source": "immutable seed-66421 C0 archive; read-only attribution",
        "world_count": len({route["world_id"] for route in routes}),
        "route_count": len(routes),
        "physical_margin_minus_analytic_margin": {
            "overall": _mean_sd([route["margin_delta_m"] for route in routes]),
            "by_route": {key: _mean_sd(value) for key, value in sorted(by_route.items())},
            "by_expected_stratum": {key: _mean_sd(value) for key, value in sorted(by_expected.items())},
        },
        "collision_classes": dict(sorted(contacts.items())),
        "mismatch_count": len(mismatch),
        "mismatches_by_route": {key: sum(row["route"] == key for row in mismatch) for key in ("left_route", "center_route", "right_route")},
        "mismatches_by_expected_stratum": {key: sum(row["expected_stratum"] == key for row in mismatch) for key in ("safe", "boundary", "blocked")},
        "mismatch_rows": [
            {key: row[key] for key in ("world_id", "route", "expected_stratum", "observed_stratum", "analytic_margin_m", "physical_margin_m", "margin_delta_m", "collision_classes", "first_contacts")}
            for row in mismatch
        ],
        "limitations": [
            "The archived preflight summaries contain trajectory hashes and first contacts, not stepwise records; route-convergence attribution is therefore limited to route-level and contact-level evidence.",
            "No adaptive, fixed, MAP, PF, QMC, or observation result is included or inferred.",
            "A positive analytic/physical discrepancy is diagnostic evidence about the approximation and/or controller sweep, not a relabeling permission.",
        ],
    }


def markdown(result: dict) -> str:
    lines = [
        "# Phase 6C seed-66421 C0 failure attribution", "",
        "This is a read-only analysis of the immutable preflight archive. It does not modify the manifest, trajectories, labels, or failure report.", "",
        f"Scope: `{result['world_count']}` worlds, `{result['route_count']}` routes; mismatches: `{result['mismatch_count']}`.", "",
        "## Margin discrepancy", "",
        "`delta = physical controller margin - analytic margin`.", "",
    ]
    overall = result["physical_margin_minus_analytic_margin"]["overall"]
    lines.append(f"Overall mean delta `{overall['mean_m']:.6f} m`, SD `{overall['sd_m']:.6f} m`, range `[{overall['min_m']:.6f}, {overall['max_m']:.6f}] m`.")
    lines.extend(["", "| group | n | mean delta (m) | sd (m) | min (m) | max (m) |", "|---|---:|---:|---:|---:|---:|"])
    for family in ("by_route", "by_expected_stratum"):
        for key, stats in result["physical_margin_minus_analytic_margin"][family].items():
            lines.append(f"| {family}:{key} | {stats['count']} | {stats['mean_m']:.6f} | {stats['sd_m']:.6f} | {stats['min_m']:.6f} | {stats['max_m']:.6f} |")
    lines.extend(["", "## Contact and mismatch summary", "", f"Collision/contact counts: `{json.dumps(result['collision_classes'], sort_keys=True)}`.", ""])
    lines.append(f"Mismatches by route: `{json.dumps(result['mismatches_by_route'], sort_keys=True)}`.")
    lines.append(f"Mismatches by expected stratum: `{json.dumps(result['mismatches_by_expected_stratum'], sort_keys=True)}`.")
    lines.extend(["", "## Interpretation boundary", "", *[f"- {item}" for item in result["limitations"]]])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest, routes = load(args.manifest, args.results)
    result = attribute(manifest, routes)
    args.output.mkdir(parents=True)
    (args.output / "attribution.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (args.output / "attribution.md").write_text(markdown(result), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("world_count", "route_count", "mismatch_count", "mismatches_by_route", "mismatches_by_expected_stratum", "collision_classes")}, indent=2))


if __name__ == "__main__":
    main()
