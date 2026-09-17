"""Evaluate Phase 5B train corpus against prospective gates T0-T5."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from guard.active_vision.scene import CAMERAS, HIDDEN_STATES, ROUTES
from guard.json_io import write_json


def route_result(rows, route):
    return next(row for row in rows if row["route"] == route)


def verifier(image, route):
    image = np.asarray(image)
    red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    if route == "left_route":
        obstacle = (red > 170) & (green < 150) & (blue < 150)
        ribbon = (red > 170) & (green > 150) & (blue < 100)
    elif route == "right_route":
        obstacle = (blue > 150) & (red < 150) & (green < 140)
        ribbon = (red < 100) & (green > 170) & (blue > 150)
    else:
        raise ValueError(route)
    obstacle_pixels = int(obstacle.sum())
    ribbon_pixels = int(ribbon.sum())
    if obstacle_pixels >= 100:
        verdict = "blocked"
    elif ribbon_pixels >= 100:
        verdict = "clear"
    else:
        verdict = "unobserved"
    return {"verdict": verdict, "obstacle_pixels": obstacle_pixels, "ribbon_pixels": ribbon_pixels}


def finite(value):
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus_root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summary = json.loads((args.corpus_root / "summary.json").read_text())
    manifest_bytes = args.manifest.read_bytes()
    protocol_bytes = args.protocol.read_bytes()
    manifest = json.loads(manifest_bytes)
    train_rows = [row for row in manifest["layouts"] if row["split"] == "train"]
    expected_ids = {row["layout_id"] for row in train_rows}
    observed_ids = {row["layout"]["layout_id"] for row in summary["layouts"]}
    layout_checks = []
    hard_ok = True
    physical_count = paid_count = left_unique = right_unique = 0
    for layout in summary["layouts"]:
        row = layout["layout"]
        layout_id = row["layout_id"]
        states = {state["hidden_state"]: state for state in layout["states"]}
        files_ok = set(states) == set(HIDDEN_STATES)
        query_ok = public_ok = finite_ok = True
        v0_hashes = set()
        public_payloads = set()
        route_files = 0
        images = {}
        for hidden_state in HIDDEN_STATES:
            state_dir = args.corpus_root / layout_id / hidden_state
            required = [
                state_dir / "routes.json",
                state_dir / "queries.json",
                state_dir / "meta.json",
                *[state_dir / "images" / f"{camera}.png" for camera in CAMERAS],
            ]
            files_ok &= all(path.is_file() for path in required)
            if not files_ok:
                continue
            route_files += 1
            queries = json.loads((state_dir / "queries.json").read_text())
            query_ok &= len(queries) == 4 and len({item["state_hash"] for item in queries}) == 1
            meta = json.loads((state_dir / "meta.json").read_text())
            query_ok &= meta["branch_state_hash"] == queries[0]["state_hash"]
            query_ok &= (
                meta["guard_head"] == summary["guard_head"]
                and meta["manifest_sha256"] == summary["manifest_sha256"]
                and meta["protocol_sha256"] == summary["protocol_sha256"]
            )
            query_ok &= all(
                route["branch_state_hash"] == meta["branch_state_hash"]
                for route in states[hidden_state]["routes"]
            )
            route_rows = json.loads((state_dir / "routes.json").read_text())
            finite_ok &= finite(route_rows) and finite(queries)
            state = states[hidden_state]
            expected_summaries = [
                {key: value for key, value in result.items() if key != "records"}
                for result in route_rows
            ]
            files_ok &= expected_summaries == state["routes"]
            v0_hashes.add(state["image_sha256"]["v0"])
            public_payloads.add(
                json.dumps(
                    {"layout": state["public_layout"], "routes": state["public_routes"]},
                    sort_keys=True,
                )
            )
            loaded = {
                camera: Image.open(state_dir / "images" / f"{camera}.png").copy()
                for camera in CAMERAS
            }
            files_ok &= all(
                sha256_bytes(np.asarray(loaded[camera]).tobytes()) == state["image_sha256"][camera]
                for camera in CAMERAS
            )
            images[hidden_state] = {camera: loaded[camera] for camera in CAMERAS[1:]}
        public_ok &= len(v0_hashes) == 1 and len(public_payloads) == 1
        physical_ok = True
        for index, route in enumerate(ROUTES):
            for hidden_state in HIDDEN_STATES:
                outcome = route_result(states[hidden_state]["routes"], route)
                blocked = hidden_state[index] == "1"
                physical_ok &= outcome["collision"] if blocked else outcome["collision_free_success"]
        informative = {route: {} for route in ROUTES}
        for index, route in enumerate(ROUTES):
            for camera in CAMERAS[1:]:
                informative[route][camera] = all(
                    verifier(images[state][camera], route)["verdict"]
                    == ("blocked" if state[index] == "1" else "clear")
                    for state in HIDDEN_STATES
                )
        paid_ok = all(any(informative[route].values()) for route in ROUTES)
        left_only = informative["left_route"]["v_left"] and not (
            informative["left_route"]["v_right"] or informative["left_route"]["v_high"]
        )
        right_only = informative["right_route"]["v_right"] and not (
            informative["right_route"]["v_left"] or informative["right_route"]["v_high"]
        )
        replay_ok = bool(layout["replay"] and layout["replay"]["exact"])
        layout_hard = files_ok and route_files == 4 and query_ok and public_ok and finite_ok and replay_ok
        hard_ok &= layout_hard
        physical_count += int(physical_ok)
        paid_count += int(paid_ok)
        left_unique += int(left_only)
        right_unique += int(right_only)
        layout_checks.append(
            {
                "layout_id": layout_id,
                "hard_integrity": layout_hard,
                "physical_mechanism": physical_ok,
                "paid_view_mechanism": paid_ok,
                "left_unique": left_only,
                "right_unique": right_only,
                "informative": informative,
            }
        )
    no_other_split = not any(
        path.name.startswith(("av2_val_", "av2_test_")) for path in args.corpus_root.iterdir()
    )
    t0 = (
        len(summary["layouts"]) == 60
        and observed_ids == expected_ids
        and summary["canonical_candidate_trajectories"] == 480
        and summary["replay_audits"] == 60
        and no_other_split
    )
    provenance = not summary["guard_dirty"] and all(
        layout["replay"]["exact"] for layout in summary["layouts"]
    ) and summary["manifest_sha256"] == sha256_bytes(manifest_bytes) and summary[
        "protocol_sha256"
    ] == sha256_bytes(protocol_bytes)
    gates = {
        "T0": {"pass": t0, "layouts": len(summary["layouts"]), "trajectories": summary["canonical_candidate_trajectories"]},
        "T1": {"pass": provenance and hard_ok, "query_and_replay_provenance": provenance},
        "T2": {"pass": hard_ok, "hard_integrity_layouts": sum(row["hard_integrity"] for row in layout_checks), "required": 60},
        "T3": {"pass": physical_count >= 54, "physical_layouts": physical_count, "required": 54},
        "T4": {
            "pass": paid_count >= 54 and left_unique >= 20 and right_unique >= 20,
            "paid_view_layouts": paid_count,
            "left_unique_layouts": left_unique,
            "right_unique_layouts": right_unique,
            "required_paid": 54,
            "required_unique_each": 20,
        },
        "T5": {"pass": observed_ids == expected_ids, "retained_layouts": len(observed_ids), "required": 60},
    }
    result = {
        "schema_version": 2,
        "split": "train",
        "gates": gates,
        "all_pass": all(item["pass"] for item in gates.values()),
        "failed_physical_layouts": [row["layout_id"] for row in layout_checks if not row["physical_mechanism"]],
        "failed_paid_view_layouts": [row["layout_id"] for row in layout_checks if not row["paid_view_mechanism"]],
        "layouts": layout_checks,
    }
    write_json(args.output, result)
    print(json.dumps(gates, indent=2))
    if not result["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
