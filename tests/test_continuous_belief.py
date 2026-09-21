import json
import subprocess
import sys
import unittest

import numpy as np

from guard.active_vision.continuous_belief import (
    ContinuousContractError,
    ParticleFilter,
    decide,
    effective_sample_size,
    make_observation,
    normalize_log_weights,
    reference_belief,
    route_risks,
    sample_prior,
    systematic_resample,
)
from guard.active_vision.continuous_broker import ContinuousDepthBroker
from guard.active_vision.replay_continuous_contract import replay


class ContinuousBeliefContractTest(unittest.TestCase):
    def test_qmc_reference_uses_prior_and_observations_not_truth(self):
        truth = sample_prior(1, 91)[0]
        observation = make_observation(truth, "q_a", 92)
        particles, weights = reference_belief(93, [observation], count=2 ** 17)
        self.assertEqual(particles.shape, (2 ** 17, 10))
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertFalse(any(key in observation for key in ("truth", "state", "clearance")))

    def test_non_finite_log_weights_hard_fail(self):
        with self.assertRaises(ContinuousContractError):
            normalize_log_weights(np.array([0.0, np.nan]))
        with self.assertRaises(ContinuousContractError):
            normalize_log_weights(np.array([-np.inf, -np.inf]))

    def test_ess_and_systematic_resampling_are_deterministic(self):
        weights = np.array([0.97, 0.01, 0.01, 0.01])
        self.assertLess(effective_sample_size(weights), 2.0)
        self.assertTrue(np.array_equal(systematic_resample(weights, 5), systematic_resample(weights, 5)))

    def test_pf_resamples_below_frozen_threshold(self):
        truth = sample_prior(1, 101)[0]
        pf = ParticleFilter.create(102)
        row = pf.update(make_observation(truth, "q_a", 103))
        self.assertTrue(row["resampled"])
        self.assertEqual(len(pf.particles), 2048)

    def test_broker_rejects_metadata_and_enforces_b2(self):
        truth = sample_prior(1, 111)[0]

        def supplier(camera):
            return {"observation": make_observation(truth, camera, 112), "state_before": "x", "state_after": "x", "feature_sha256": "h"}

        broker = ContinuousDepthBroker(supplier)
        broker.query("q_branch")
        broker.query("q_a")
        with self.assertRaises(PermissionError):
            broker.query("q_b")
        bad = ContinuousDepthBroker(lambda camera: supplier(camera) | {"hidden_state": "leak"})
        with self.assertRaises(ValueError):
            bad.query("q_branch")

    def test_common_decision_rule_respects_budget(self):
        particles = sample_prior(2048, 121)
        weights = np.full(2048, 1.0 / 2048)
        decision = decide(particles, weights, remaining_budget=2)
        self.assertIn(decision["kind"], {"query", "terminal"})
        exhausted = decide(particles, weights, remaining_budget=0)
        self.assertEqual(exhausted["kind"], "terminal")

    def test_fresh_process_replay_is_byte_equivalent(self):
        expected = replay(66422)
        result = subprocess.run(
            [sys.executable, "-m", "guard.active_vision.replay_continuous_contract", "--seed", "66422"],
            check=True, capture_output=True, text=True,
        )
        self.assertEqual(expected, json.loads(result.stdout))


if __name__ == "__main__":
    unittest.main()
