"""Integrity checker for Phase 5D smoke tuple/replay artifacts."""

import argparse
import json
import math
from pathlib import Path

from guard.active_vision.phase5d import derive_failures, replay_equal, validate_trial_tuple
from guard.json_io import write_json


def finite(value):
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def check(payload):
    errors = []
    trials = payload.get("trials", [])
    for index, row in enumerate(trials):
        valid, reason = validate_trial_tuple(row.get("record", {}))
        if not valid:
            errors.append(f"trial {index}: {reason}")
        if not finite(row):
            errors.append(f"trial {index}: non_finite")
        replay = row.get("replay")
        if replay is None or not replay_equal(row["record"], replay):
            errors.append(f"trial {index}: replay_mismatch")
        if valid:
            try:
                row["derived_failures"] = derive_failures(row["record"])
            except ValueError as exc:
                errors.append(f"trial {index}: {exc}")
    mapping = [row["record"]["mapped_candidate"].get("mapped_candidate") for row in trials if "record" in row]
    valid_routes = sum(candidate in ("left_route", "right_route") for candidate in mapping)
    coverage = valid_routes / len(trials) if trials else 0.0
    recall_failures = sum(row.get("derived_failures", {}).get("candidate_recall_failure", False) for row in trials)
    transfer_rows = [
        row for row in trials
        if row.get("record", {}).get("mapped_candidate", {}).get("mapped_candidate") in ("left_route", "right_route")
    ]
    transfers = [row["record"]["physical_outcome"].get("mapped_route_transfer") for row in transfer_rows]
    transfer_ok = bool(transfers)
    for row, transfer in zip(transfer_rows, transfers):
        if not isinstance(transfer, dict):
            transfer_ok = False
            continue
        route = row["record"]["mapped_candidate"]["mapped_candidate"]
        blocked = row["hidden_state"][0 if route == "left_route" else 1] == "1"
        expected = bool(transfer.get("collision")) if blocked else bool(transfer.get("collision_free_success"))
        transfer_ok &= expected
    result = {
        "schema_version": 1,
        "trials": len(trials),
        "valid_non_stop_route_mappings": valid_routes,
        "non_stop_mapping_rate": coverage,
        "candidate_recall_failures": recall_failures,
        "mapped_route_transfer_trials": len(transfers),
        "route_transfer_pass": transfer_ok,
        "all_replays_exact": not any("replay_mismatch" in error for error in errors),
        "all_pass": not errors and coverage >= 0.60 and recall_failures == 0 and transfer_ok,
        "errors": errors,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("payload", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = check(json.loads(args.payload.read_text()))
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
    if not result["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
