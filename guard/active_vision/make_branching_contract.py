"""Write the complete Phase 6A3 belief and observation contract."""

import argparse
from pathlib import Path

from guard.active_vision.belief_branching import (
    CAMERAS,
    FAMILIES,
    OBSERVATION_TABLE,
    PRIOR,
    SPECIALIST,
    STATES,
    all_fixed_results,
    observation,
    plan,
    posterior,
    run_adaptive,
    support,
)
from guard.json_io import write_json


def build_contract():
    branch_posteriors = []
    terminal_posteriors = []
    for family, states in FAMILIES.items():
        belief = posterior(PRIOR, "q_branch", family)
        branch_posteriors.append({
            "outcome": family, "support": list(support(belief)), "probabilities": list(belief),
            "terminal_before_followup": plan(belief, (), 0), "next_query": plan(belief, (SPECIALIST[family],), 1),
        })
        for state in states:
            camera = SPECIALIST[family]
            singleton = posterior(belief, camera, observation(state, camera))
            terminal_posteriors.append({
                "state": state, "history": [["q_branch", family], [camera, observation(state, camera)]],
                "support": list(support(singleton)), "probabilities": list(singleton),
                "terminal": plan(singleton, (), 0),
            })
    fixed = all_fixed_results()
    return {
        "schema_version": 1,
        "states": list(STATES),
        "prior": {state: probability for state, probability in zip(STATES, PRIOR)},
        "cameras": list(CAMERAS),
        "observation_table": {
            state: {camera: OBSERVATION_TABLE[(state, camera)] for camera in CAMERAS} for state in STATES
        },
        "branch_posteriors": branch_posteriors,
        "terminal_posteriors": terminal_posteriors,
        "adaptive_traces": {state: run_adaptive(state) for state in STATES},
        "fixed_sequences": [{key: value for key, value in row.items() if key != "trials"} for row in fixed],
        "best_fixed_utility": max(row["mean_utility"] for row in fixed),
        "adaptive_utility": plan(PRIOR, CAMERAS, 2)["expected_value"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_json(args.output, build_contract())


if __name__ == "__main__":
    main()
