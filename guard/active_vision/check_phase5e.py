"""Check the registered Phase 5E train-only smoke gates."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from guard.active_vision.phase5e import CHOICES, replay_tuple, validate_five_tuple
from guard.json_io import write_json


def check(payload):
    trials = payload.get("trials", [])
    errors = []
    valid_routes = correct = stops = recall = physical = 0
    paired = defaultdict(list)
    ablation = defaultdict(lambda: [0, 0, 0])
    for index, row in enumerate(trials):
        record = row.get("record", {})
        valid, reason = validate_five_tuple(record)
        if not valid: errors.append(f"trial {index}: {reason}"); continue
        if not replay_tuple(record, row.get("replay")): errors.append(f"trial {index}: replay")
        proposal = record["proposer_output"]["proposal"]
        if proposal in CHOICES[:2]: valid_routes += 1
        if proposal == "stop": stops += 1
        if proposal not in CHOICES: recall += 1
        if proposal == row["preference"]: correct += 1
        paired[(row["layout_id"], row["preference"])].append(record["proposer_output"])
        blocked = row["hidden_state"][0 if proposal == "left_route" else 1] == "1" if proposal in CHOICES[:2] else False
        outcome = record["physical_outcome"]
        if proposal in CHOICES[:2] and (bool(outcome.get("collision")) if blocked else bool(outcome.get("collision_free_success"))): physical += 1
        for variant, result in record["proposer_output"]["ablations"].items():
            ablation[variant][0] += result["proposal"] == row["preference"]
            ablation[variant][1] += result["proposal"] in CHOICES[:2]
            ablation[variant][2] += 1
    hidden_exact = all(len({json.dumps(item, sort_keys=True) for item in values}) == 1 for values in paired.values())
    count = len(trials)
    result = {"schema_version": 1, "trials": count, "valid_non_stop": valid_routes,
              "coverage": valid_routes / count if count else 0, "preference_accuracy": correct / count if count else 0,
              "proposer_stops": stops, "stop_rate": stops / count if count else 0,
              "candidate_recall_failures": recall, "hidden_state_outputs_exact": hidden_exact,
              "physical_transfer_passes": physical,
              "ablations": {key: {"accuracy": a / n, "coverage": c / n} for key, (a, c, n) in ablation.items()},
              "errors": errors}
    result["all_pass"] = count == 64 and not errors and result["coverage"] >= .90 and result["preference_accuracy"] >= .95 and hidden_exact and result["stop_rate"] <= .05 and recall == 0 and physical == count
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("payload", type=Path); parser.add_argument("output", type=Path)
    args = parser.parse_args(); result = check(json.loads(args.payload.read_text())); write_json(args.output, result)
    print(json.dumps(result, indent=2)); raise SystemExit(0 if result["all_pass"] else 1)


if __name__ == "__main__": main()
