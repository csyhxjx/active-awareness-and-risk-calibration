"""Read-only numerical audit of the frozen Phase 5D OpenVLA chunks."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def audit(payload):
    rows = payload["trials"]
    chunks = []
    per_trial = []
    by_instruction = defaultdict(list)
    by_v0_instruction = defaultdict(list)
    stop_reasons = Counter()
    for row in rows:
        record = row["record"]
        proposer = record["proposer_output"]
        chunk = np.asarray(proposer["action_chunk"], dtype=np.float64)
        mapped = record["mapped_candidate"]
        chunks.append(chunk)
        by_instruction[proposer["instruction"]].append(chunk)
        by_v0_instruction[(proposer["input_image_sha256"], proposer["instruction"])].append(chunk)
        if mapped.get("mapped_candidate") == "stop":
            stop_reasons[record["selector_decision"].get("stop_reason", "missing")] += 1
        per_trial.append({
            "layout_id": row["layout_id"], "hidden_state": row["hidden_state"],
            "preference": row["preference"], "chunk_min": float(chunk.min()),
            "chunk_max": float(chunk.max()), "chunk_mean": float(chunk.mean()),
            "chunk_std": float(chunk.std()), "terminal_position": mapped["terminal_position"],
            "distance_left": mapped["distances"]["left_route"],
            "distance_right": mapped["distances"]["right_route"],
            "stop_reason": record["selector_decision"].get("stop_reason"),
        })
    stack = np.stack(chunks)
    instruction_change = {}
    for instruction, values in by_instruction.items():
        instruction_change[instruction] = {
            "count": len(values),
            "mean_chunk_hash": _hash(np.mean(values, axis=0).tolist()),
        }
    same_v0_hidden = []
    for (v0, instruction), values in by_v0_instruction.items():
        hashes = [_hash(value.tolist()) for value in values]
        same_v0_hidden.append({"v0_sha256": v0, "instruction": instruction,
                               "count": len(values), "unique_chunk_hashes": len(set(hashes))})
    terminal = np.asarray([row["record"]["mapped_candidate"]["terminal_position"] for row in rows])
    distances = {
        route: [row["record"]["mapped_candidate"]["distances"][route] for row in rows]
        for route in ("left_route", "right_route")
    }
    return {
        "schema_version": 1,
        "source_trials": len(rows),
        "chunk_shape": list(stack.shape[1:]),
        "chunk_global": {
            "min": float(stack.min()), "max": float(stack.max()),
            "mean": float(stack.mean()), "std": float(stack.std()),
            "near_zero_fraction_abs_le_0.1": float(np.mean(np.abs(stack) <= 0.1)),
            "near_zero_fraction_abs_le_0.2": float(np.mean(np.abs(stack) <= 0.2)),
        },
        "chunk_per_dimension": {
            str(i): {"min": float(stack[:, :, i].min()), "max": float(stack[:, :, i].max()),
                     "mean": float(stack[:, :, i].mean()), "std": float(stack[:, :, i].std())}
            for i in range(stack.shape[2])
        },
        "per_trial": per_trial,
        "terminal_position": {"min": terminal.min(axis=0).tolist(), "max": terminal.max(axis=0).tolist(),
                               "mean": terminal.mean(axis=0).tolist(), "std": terminal.std(axis=0).tolist()},
        "route_distance": {route: {"min": float(min(values)), "max": float(max(values)),
                                    "mean": float(np.mean(values)), "std": float(np.std(values))}
                           for route, values in distances.items()},
        "stop_reasons": dict(stop_reasons),
        "instruction_effect": instruction_change,
        "same_v0_different_hidden_state": same_v0_hidden,
        "interpretation": {
            "projection_frozen": True,
            "five_d_conclusion_unchanged": "non-stop coverage remains 0/64 under frozen projection",
            "audit_scope": "distinguish delta semantic mismatch from task OOD collapse; no relabeling or projection change",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = audit(json.loads(args.payload.read_text()))
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
