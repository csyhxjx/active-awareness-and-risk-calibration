"""Fresh-process deterministic replay for the Phase 6C PF contract."""

from __future__ import annotations

import argparse

from guard.active_vision.continuous_belief import ParticleFilter, make_observation, sample_prior
from guard.json_io import canonical_dumps


def replay(seed: int) -> dict:
    truth = sample_prior(1, seed + 17)[0]
    observations = [make_observation(truth, "q_branch", seed + 31), make_observation(truth, "q_a", seed + 47)]
    pf = ParticleFilter.create(seed)
    updates = [pf.update(observation) for observation in observations]
    return {"seed": seed, "observations": observations, "updates": updates}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=66422)
    args = parser.parse_args()
    print(canonical_dumps(replay(args.seed), sort_keys=True))


if __name__ == "__main__":
    main()
