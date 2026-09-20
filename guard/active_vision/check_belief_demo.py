"""Check the Phase 6A2 one-layout closed-loop demo gates."""

import argparse
import json
from pathlib import Path

from guard.active_vision.belief_scene import HIDDEN_STATES, ROUTES
from guard.json_io import write_json


def check(summary):
    errors = []
    states = {row["hidden_state"]: row for row in summary.get("states", [])}
    if set(states) != set(HIDDEN_STATES):
        errors.append("hidden_state_scope")
    v0 = {row["image_sha256"]["v0"] for row in states.values()}
    if len(v0) != 1:
        errors.append("v0_leakage")
    physical = 0
    for hidden_state, row in states.items():
        if len(row["query_ledger"]) != 5 or len({item["state_hash"] for item in row["query_ledger"]}) != 1:
            errors.append(f"{hidden_state}: query_provenance")
        routes = {result["route"]: result for result in row["routes"]}
        for index, route in enumerate(ROUTES):
            expected = bool(routes[route]["collision"]) if hidden_state[index] == "1" else bool(routes[route]["collision_free_success"])
            if not expected:
                errors.append(f"{hidden_state}/{route}: physical")
            else:
                physical += 1
    adaptive = summary["adaptive"]
    fixed = summary["fixed"]
    adaptive_queries = [row["decision"]["id"] for row in adaptive["trace"] if row["decision"]["kind"] == "query"]
    if adaptive_queries != ["q_left", "q_right"] or adaptive["terminal_action"] != "right_route":
        errors.append("adaptive_trace")
    if not adaptive["physical_outcome"] or not adaptive["physical_outcome"]["collision_free_success"]:
        errors.append("adaptive_physical")
    fixed_queries = [row["decision"]["id"] for row in fixed["trace"] if row["decision"]["kind"] == "query"]
    if fixed_queries != ["q_left", "q_front"] or fixed["terminal_action"] != "stop":
        errors.append("fixed_trace")
    if not summary.get("adaptive_replay_exact") or not summary.get("fixed_replay_exact"):
        errors.append("replay")
    result = {"schema_version": 1, "states": len(states), "canonical_routes": physical,
              "v0_pixel_identical": len(v0) == 1, "adaptive_switch_success": "adaptive_trace" not in errors and "adaptive_physical" not in errors,
              "fixed_matched_budget_stop": "fixed_trace" not in errors, "all_pass": not errors, "errors": errors}
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("summary", type=Path); parser.add_argument("output", type=Path)
    args = parser.parse_args(); result = check(json.loads(args.summary.read_text())); write_json(args.output, result)
    print(json.dumps(result, indent=2)); raise SystemExit(0 if result["all_pass"] else 1)


if __name__ == "__main__": main()
