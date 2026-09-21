"""Generate the policy-independent Phase 6A7 candidate plan."""

import argparse
import subprocess
from pathlib import Path

from guard.active_vision.phase6a7 import (
    ALL_STATES,
    CLEARANCE_THRESHOLD_M,
    FIXED_SEQUENCES,
    PROTOCOL_FILE,
    PROTOCOL_ID,
    ROI,
    SEED,
    candidate_spec,
    observation_table,
    sha256_file,
)
from guard.json_io import write_json


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def build(protocol: Path, candidate_count: int = 36) -> dict:
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": sha256_file(protocol),
        "generator_head": git_head(),
        "seed": SEED,
        "selection_rule": "first 12 candidates passing frozen physical preflight in candidate_index order",
        "selection_inputs": ["public layout geometry", "official-controller physical preflight"],
        "policy_result_filtering": False,
        "requested_layouts": 12,
        "states_per_layout": len(ALL_STATES),
        "routes_per_scene": 3,
        "roi": list(ROI),
        "clearance_threshold_m": CLEARANCE_THRESHOLD_M,
        "color_semantics": "global_fixed_phase6a6",
        "observation_table": observation_table(),
        "fixed_sequences": [list(sequence) for sequence in FIXED_SEQUENCES],
        "candidates": [candidate_spec(index) for index in range(candidate_count)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--protocol", type=Path, default=Path(PROTOCOL_FILE))
    parser.add_argument("--candidate-count", type=int, default=36)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    write_json(args.output, build(args.protocol, args.candidate_count))


if __name__ == "__main__":
    main()

