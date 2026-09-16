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
from guard.counterfactual.run_counterfactual_v2 import select_branches, v2_action
from guard.counterfactual.make_smoke_stats_v2 import evaluate_smoke
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

    def test_v2_targeted_actions(self):
        baseline = np.array([0.2, -0.3, 0.1, 0, 0, 0, 1.0])
        workspace = {"action_index": 1, "direction": -1}
        self.assertAlmostEqual(v2_action("C04", baseline, workspace)[2], -0.3)
        self.assertAlmostEqual(v2_action("C10", baseline, workspace)[2], -0.9)
        self.assertEqual(v2_action("D_force", baseline, workspace)[6], -1.0)
        self.assertEqual(v2_action("E_push", baseline, workspace)[1], -1.0)

    def test_v2_branch_selection_requires_carry_and_full_horizon(self):
        trace = []
        for step in range(101):
            trace.append(
                {
                    "step_index": step,
                    "target_name": "akita_black_bowl_1",
                    "target_z": 0.8 + (0.05 if step == 35 else 0.03 if step >= 25 else 0.0),
                    "gripper": 1.0 if step >= 20 else -1.0,
                    "lateral_terms": [
                        {"axis": "x", "action_index": 0, "direction": -1, "margin": 0.2},
                        {"axis": "x", "action_index": 0, "direction": 1, "margin": 0.1},
                        {"axis": "y", "action_index": 1, "direction": -1, "margin": 0.15},
                        {"axis": "y", "action_index": 1, "direction": 1, "margin": 0.3},
                    ],
                }
            )
        branches = select_branches(trace, "task_07_init_023", horizon=30, max_action_step=100)
        self.assertEqual(branches["drop"]["step"], 35)
        self.assertEqual(branches["workspace"]["step"], 10)
        self.assertEqual(branches["workspace"]["direction"], 1)

    def test_v2_smoke_gate_blocks_missing_drop_family(self):
        labels = []
        for state in ("task_07_init_023", "task_04_init_035", "task_08_init_003"):
            for arm in ("A_grip", "A_drop", "A_work", "C04", "C06", "C08", "C10", "D_force", "E_push"):
                for constraint in ("workspace", "gripper_env", "self_collision", "object_drop", "non_finite"):
                    violation = (arm in {"C06", "C08", "C10"} and state == "task_07_init_023" and constraint == "gripper_env") or (arm == "E_push" and constraint == "workspace")
                    labels.append(
                        {
                            "state_id": state,
                            "arm_id": arm,
                            "constraint": constraint,
                            "branch": {"any_violation": violation},
                        }
                    )
        gates, _ = evaluate_smoke(labels, True, True)
        self.assertEqual([passed for _, passed, _ in gates], [True, False, True, True, True])


if __name__ == "__main__":
    unittest.main()
