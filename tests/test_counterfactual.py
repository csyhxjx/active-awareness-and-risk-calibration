import tempfile
import unittest
from pathlib import Path

import numpy as np

from guard.counterfactual.run_counterfactual import (
    compare_constraint_record,
    perturb_action,
    rng_sha256,
    rng_snapshot,
    state_sha256,
)
from guard.counterfactual.make_pilot_stats import build_labels
from guard.json_io import write_json


class CounterfactualTest(unittest.TestCase):
    def setUp(self):
        self.actions = {
            10: np.array([0.8, -0.9, 0.2, 0, 0, 0, 1.0]),
            11: np.array([0.5, 0.5, 0.1, 0, 0, 0, 1.0]),
            12: np.array([0.2, 0.2, 0.0, 0, 0, 0, -1.0]),
        }

    def test_registered_perturbations(self):
        np.testing.assert_array_equal(perturb_action("A", self.actions, 10), self.actions[10])
        self.assertEqual(perturb_action("B", self.actions, 10)[2], -0.3)
        self.assertEqual(perturb_action("C", self.actions, 10)[2], -0.8)
        self.assertEqual(perturb_action("D", self.actions, 10)[6], -1.0)
        np.testing.assert_allclose(perturb_action("E", self.actions, 10)[:2], [1.0, -1.0])

    def test_snapshot_hashes_survive_json_roundtrip(self):
        state = np.array([1.0, 2.0], dtype=np.float64)
        snapshot = rng_snapshot()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rng.json"
            write_json(path, snapshot)
            import json

            restored = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(rng_sha256(snapshot), rng_sha256(restored))
        self.assertEqual(state_sha256(state), state_sha256(state.copy()))

    def test_parity_enforces_registered_absolute_tolerance(self):
        names = ("workspace", "gripper_env", "self_collision", "object_drop", "non_finite")
        expected = {name: {"violated": False, "margin": 0.1} for name in names}
        actual = {name: dict(value) for name, value in expected.items()}
        actual["workspace"]["margin"] += 5e-8
        self.assertLess(compare_constraint_record(actual, expected, context="fixture"), 1e-7)
        actual["workspace"]["margin"] += 2e-7
        with self.assertRaises(ValueError):
            compare_constraint_record(actual, expected, context="fixture")

    def test_counterfactual_labels_keep_arm_identity(self):
        records = []
        names = ("workspace", "gripper_env", "self_collision", "object_drop", "non_finite")
        for arm_step, margin in enumerate((0.1, -0.1, -0.2)):
            records.append(
                {
                    "step_index": arm_step + 40,
                    "arm_step": arm_step,
                    **{name: {"violated": margin < 0, "margin": margin} for name in names},
                }
            )
        arm = {
            "dir": Path("fixture"),
            "meta": {
                "state_id": "task_00_init_001",
                "parent_state": "task_00_init_001",
                "task_id": 0,
                "init_index": 1,
                "split": "train",
                "arm_id": "C",
                "branch_step": 40,
                "horizon": 3,
            },
            "constraints": records,
        }
        labels = build_labels([arm], {"run_id": "fixture"}, sustained_k=2)
        self.assertEqual(len(labels), 5)
        self.assertTrue(all(label["arm_id"] == "C" for label in labels))
        self.assertTrue(labels[0]["branch"]["sustained_violation"])


if __name__ == "__main__":
    unittest.main()
