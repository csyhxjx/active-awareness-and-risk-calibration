import json
import tempfile
import unittest
from pathlib import Path

from guard.json_io import canonical_dumps
from guard.labeling.label_collection import label_episode


CONSTRAINTS = ("workspace", "gripper_env", "self_collision", "object_drop", "non_finite")


def make_records(values):
    records = []
    for step, margin in enumerate(values):
        records.append(
            {
                "step_index": step,
                **{name: {"violated": bool(margin < 0), "margin": margin} for name in CONSTRAINTS},
            }
        )
    return records


def make_meta(step_count=6):
    return {
        "state_id": "task_00_init_001",
        "task_id": 0,
        "task_name": "fixture",
        "init_index": 1,
        "state_hash": "a" * 16,
        "split": "train",
        "success": True,
        "termination_reason": "success",
        "constraint_step_count": step_count,
        "image_window": {"start": 2, "end": 230},
    }


class LabelingTest(unittest.TestCase):
    def test_wait_policy_minimum_and_sustained_k(self):
        records = make_records([0.2, -0.1, -0.2, -0.3, -0.4, 0.5])
        label = label_episode(make_meta(), records, "workspace", 3, {"run_id": "fixture"})
        self.assertEqual(label["wait"]["min_margin"], -0.1)
        self.assertEqual(label["policy"]["min_margin"], -0.4)
        self.assertFalse(label["wait"]["sustained_violation"])
        self.assertTrue(label["policy"]["sustained_violation"])
        self.assertEqual(label["policy"]["sustained_onset_step"], 2)

    def test_k_is_configurable(self):
        records = make_records([0.1, -0.1, -0.2, 0.2, 0.3, 0.4])
        label = label_episode(make_meta(), records, "workspace", 2, {"run_id": "fixture"})
        self.assertEqual(label["episode"]["sustained_onset_step"], 1)

    def test_nan_is_hard_failure(self):
        records = make_records([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        records[3]["workspace"]["margin"] = float("nan")
        with self.assertRaises(ValueError):
            label_episode(make_meta(), records, "workspace", 3, {"run_id": "fixture"})

    def test_json_fixture_has_no_numpy_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.jsonl"
            path.write_text(canonical_dumps({"margin": 0.1}) + "\n", encoding="utf-8")
            self.assertNotIn("__numpy__", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
