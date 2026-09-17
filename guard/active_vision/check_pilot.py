"""Check Phase 5A P0-P5 against the frozen development pilot."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from guard.active_vision.scene import HIDDEN_STATES, ROUTES
from guard.json_io import write_json


def obstacle_visible(image, route):
    image = np.asarray(image)
    if route == "left_route":
        mask = (image[:, :, 0] > 170) & (image[:, :, 1] < 150) & (image[:, :, 2] < 150)
    elif route == "right_route":
        mask = (image[:, :, 2] > 150) & (image[:, :, 0] < 150) & (image[:, :, 1] < 180)
    else:
        raise ValueError(route)
    return int(mask.sum()) >= 100


def _route(rows, route):
    return next(row for row in rows if row["route"] == route)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summary = json.loads((args.pilot_root / "summary.json").read_text())
    checks = []
    p1_layouts = p2_layouts = p3_layouts = 0
    left_unique = right_unique = paired_examples = 0
    replay_exact = True
    query_invariant = True
    for layout in summary["layouts"]:
        layout_id = layout["layout"]
        states = {row["hidden_state"]: row for row in layout["states"]}
        replay_exact &= bool(layout["replay"]["exact"])
        p1_ok = True
        for state in HIDDEN_STATES:
            query_rows = json.loads((args.pilot_root / layout_id / state / "queries.json").read_text())
            query_invariant &= len(query_rows) == 4 and len({row["state_hash"] for row in query_rows}) == 1
            for index, route in enumerate(ROUTES):
                blocked = state[index] == "1"
                outcome = _route(states[state]["routes"], route)
                p1_ok &= outcome["collision"] if blocked else outcome["collision_free_success"]
        p1_layouts += int(p1_ok)
        v0 = {states[state]["image_sha256"]["v0"] for state in HIDDEN_STATES}
        public = {
            json.dumps(states[state]["public_routes"], sort_keys=True) for state in HIDDEN_STATES
        }
        p2_ok = len(v0) == 1 and len(public) == 1
        p2_layouts += int(p2_ok)
        images = {
            state: {
                camera: Image.open(args.pilot_root / layout_id / state / "images" / f"{camera}.png").copy()
                for camera in ("v_left", "v_right", "v_high")
            }
            for state in HIDDEN_STATES
        }
        p3_ok = all(
            obstacle_visible(images[state][camera], route) == (state[index] == "1")
            for state in HIDDEN_STATES
            for index, (route, camera) in enumerate(
                (("left_route", "v_left"), ("right_route", "v_right"))
            )
        )
        p3_layouts += int(p3_ok)
        left_high_informative = all(
            obstacle_visible(images[state]["v_high"], "left_route") == (state[0] == "1")
            for state in HIDDEN_STATES
        )
        right_high_informative = all(
            obstacle_visible(images[state]["v_high"], "right_route") == (state[1] == "1")
            for state in HIDDEN_STATES
        )
        left_other_insufficient = (
            states["00"]["image_sha256"]["v_right"] == states["10"]["image_sha256"]["v_right"]
            and states["01"]["image_sha256"]["v_right"] == states["11"]["image_sha256"]["v_right"]
            and not left_high_informative
        )
        right_other_insufficient = (
            states["00"]["image_sha256"]["v_left"] == states["01"]["image_sha256"]["v_left"]
            and states["10"]["image_sha256"]["v_left"] == states["11"]["image_sha256"]["v_left"]
            and not right_high_informative
        )
        left_unique += int(p3_ok and left_other_insufficient)
        right_unique += int(p3_ok and right_other_insufficient)
        for route, camera, clear_state, blocked_state in (
            ("left_route", "v_left", "00", "10"),
            ("right_route", "v_right", "00", "01"),
        ):
            clear = _route(states[clear_state]["routes"], route)
            blocked = _route(states[blocked_state]["routes"], route)
            if (
                states[clear_state]["image_sha256"]["v0"] == states[blocked_state]["image_sha256"]["v0"]
                and not obstacle_visible(images[clear_state][camera], route)
                and obstacle_visible(images[blocked_state][camera], route)
                and clear["collision_free_success"]
                and blocked["collision"]
            ):
                paired_examples += 1
        checks.append({"layout": layout_id, "p1": p1_ok, "p2": p2_ok, "p3": p3_ok})
    p0 = query_invariant and replay_exact and summary["canonical_candidate_trajectories"] == 96
    gates = {
        "P0": {"pass": p0, "query_invariant": query_invariant, "replays_exact": replay_exact},
        "P1": {"pass": p1_layouts >= 10, "passing_layouts": p1_layouts, "required": 10},
        "P2": {"pass": p2_layouts == 12, "passing_layouts": p2_layouts, "required": 12},
        "P3": {"pass": p3_layouts >= 10, "passing_layouts": p3_layouts, "required": 10},
        "P4": {
            "pass": left_unique >= 4 and right_unique >= 4,
            "left_unique_layouts": left_unique,
            "right_unique_layouts": right_unique,
            "required_each": 4,
        },
        "P5": {"pass": paired_examples >= 6, "paired_examples": paired_examples, "required": 6},
    }
    result = {"gates": gates, "layouts": checks, "all_pass": all(row["pass"] for row in gates.values())}
    write_json(args.output, result)
    print(json.dumps(gates, indent=2))
    if not result["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
