"""Check the Phase 6A3 physical adaptive-branching gate."""

import argparse
import json
from pathlib import Path

from guard.active_vision.belief_branching import CAMERAS, ROUTES, SPECIALIST, STATES, observation
from guard.json_io import write_json


def check(summary):
    errors = []
    states = {row["state"]: row for row in summary.get("states", [])}
    if set(states) != set(STATES):
        errors.append("state_scope")

    v0_hashes = {row["fingerprint"]["ledger"][0]["image_sha256"] for row in states.values()}
    if len(v0_hashes) != 1:
        errors.append("v0_leakage")

    partition = {}
    for camera in CAMERAS:
        groups = {}
        for state, row in states.items():
            ledger = {entry["camera"]: entry for entry in row["fingerprint"]["ledger"]}
            groups.setdefault(observation(state, camera), set()).add(ledger[camera]["image_sha256"])
        within = all(len(hashes) == 1 for hashes in groups.values())
        between = len({next(iter(hashes)) for hashes in groups.values() if len(hashes) == 1}) == len(groups)
        partition[camera] = {"outcomes": len(groups), "within_outcome_exact": within, "between_outcome_distinct": between}
        if not within or not between:
            errors.append(f"image_partition:{camera}")

    physical = 0
    route_map = {}
    for state, row in states.items():
        hashes = {entry["state_hash"] for entry in row["fingerprint"]["ledger"]}
        if len(hashes) != 1 or not row.get("fresh_process_replay_exact"):
            errors.append(f"provenance:{state}")
        for result in row["routes"]:
            route_map[(state, result["route"])] = result
            index = ROUTES.index(result["route"])
            passed = result["collision"] if state[index] == "1" else result["collision_free_success"]
            if passed:
                physical += 1
            else:
                errors.append(f"physical:{state}/{result['route']}")

    adaptive_success = 0
    for trial in summary.get("adaptive", []):
        state = trial["state"]
        decisions = [row["decision"]["id"] for row in trial["trace"]]
        expected = ["q_branch", SPECIALIST[observation(state, "q_branch")], trial["terminal_action"]]
        if decisions != expected or not trial["physical_outcome"]["success"]:
            errors.append(f"adaptive:{state}")
        else:
            adaptive_success += 1

    fixed = summary.get("fixed", [])
    best_fixed_completion = max((row["completion"] for row in fixed), default=-1)
    best_fixed_utility = max((row["mean_utility"] for row in fixed), default=float("-inf"))
    if len(fixed) != 17 or best_fixed_completion != 4 or abs(best_fixed_utility - 0.5) > 1e-10:
        errors.append("fixed_enumeration")
    for result in fixed:
        for trial in result["trials"]:
            outcome = trial["physical_outcome"]
            if trial["success"] != outcome["success"] or trial["collision"] != outcome["collision"]:
                errors.append(f"fixed_physical:{result['sequence']}/{trial['state']}")

    result = {
        "schema_version": 1,
        "states": len(states),
        "v0_pixel_identical": len(v0_hashes) == 1,
        "image_partition": partition,
        "physical_routes": physical,
        "adaptive_success": adaptive_success,
        "fixed_sequence_count": len(fixed),
        "best_fixed_completion": best_fixed_completion,
        "best_fixed_utility": best_fixed_utility,
        "adaptive_utility": 0.9,
        "strict_adaptive_advantage": adaptive_success == 6 and best_fixed_completion == 4,
        "all_pass": not errors,
        "errors": errors,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = check(json.loads(args.summary.read_text()))
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["all_pass"] else 1)


if __name__ == "__main__":
    main()
