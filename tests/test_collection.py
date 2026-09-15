import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

from guard.collection import EpisodeCollector, initial_state_hash, load_selected_states, validate_episode_dir


class CollectionTest(unittest.TestCase):
    def test_state_hash_ignores_global_json_dumps_patch(self):
        state = np.array([0.0, 1.25, -2.5])
        expected = initial_state_hash(state)
        with mock.patch.object(json, "dumps", return_value='{"__numpy__": "patched"}'):
            self.assertEqual(initial_state_hash(state), expected)

    def test_manifest_selection_and_episode_integrity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = {
                "tasks": [
                    {
                        "task_id": 0,
                        "task_name": "task_zero",
                        "train": [{"init_index": 3, "state_hash": "abc"}],
                        "calibration": [{"init_index": 4, "state_hash": "def"}],
                        "test": [],
                    }
                ]
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            manifest_hash, states = load_selected_states(manifest_path, "train,task_00_init_004")
            self.assertEqual(len(manifest_hash), 64)
            self.assertEqual([state["init_index"] for state in states], [3, 4])

            collector = EpisodeCollector(root / "out", states[0], 2, 3)
            image = np.zeros((4, 4, 3), dtype=np.uint8)
            for step in range(3):
                collector.log_constraint({"t": step, "workspace": {"violated": False, "margin": 1.0}})
                if step:
                    collector.log_chunk(step - 1, step, [[0.0] * 7])
                    collector.log_step(step - 1, 0, step, [0.0] * 7)
                    collector.save_images(step, image, image)
            collector.finish({"success": True, "termination_reason": "success"})
            result = validate_episode_dir(collector.episode_dir)
            self.assertEqual(result["step_count"], 3)
            self.assertGreater(result["bytes"], 0)


if __name__ == "__main__":
    unittest.main()
