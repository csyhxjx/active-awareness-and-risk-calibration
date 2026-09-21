import unittest
from pathlib import Path

import numpy as np

from guard.active_vision.generate_phase6a7_manifest import build
from guard.active_vision.check_phase6a7 import check_candidate_plan
from guard.active_vision.phase6a7 import (
    ALL_STATES,
    FIXED_SEQUENCES,
    PAID_CAMERAS,
    candidate_spec,
    detect_roi,
    observation_outcome,
)


class Phase6A7FixtureTest(unittest.TestCase):
    def test_candidate_generation_is_deterministic_and_fixed_color(self):
        self.assertEqual(candidate_spec(7), candidate_spec(7))
        self.assertNotEqual(candidate_spec(7), candidate_spec(8))
        self.assertEqual(candidate_spec(7)["color_permutation"], [0, 1, 2])

    def test_complete_eight_state_observation_contract(self):
        table = {
            state: {camera: observation_outcome(state, camera) for camera in PAID_CAMERAS}
            for state in ALL_STATES
        }
        self.assertEqual(len(table), 8)
        self.assertTrue(all(len(row) == 4 for row in table.values()))
        self.assertEqual(table["000"]["q_branch"], "family_a")
        self.assertEqual(table["111"]["q_branch"], "family_c")

    def test_all_seventeen_fixed_sequences_are_unique(self):
        self.assertEqual(len(FIXED_SEQUENCES), 17)
        self.assertEqual(len(set(FIXED_SEQUENCES)), 17)

    def test_roi_detector_uses_registered_color_semantics(self):
        examples = {
            ("q_branch", (240, 90, 80)): "family_a",
            ("q_branch", (80, 230, 100)): "family_b",
            ("q_branch", (80, 120, 240)): "family_c",
            ("q_a", (220, 190, 70)): "family_a:001",
            ("q_a", (175, 80, 200)): "family_a:110",
            ("q_a", (130, 130, 130)): "not_applicable",
        }
        for (camera, rgb), expected in examples.items():
            roi = np.full((8, 8, 3), rgb, dtype=np.uint8)
            self.assertEqual(detect_roi(camera, roi), expected)

    def test_candidate_plan_declares_no_policy_filtering(self):
        class Protocol:
            def read_bytes(self):
                return b"protocol"

        plan = build(Protocol(), candidate_count=12)
        self.assertFalse(plan["policy_result_filtering"])
        self.assertEqual(plan["requested_layouts"], 12)
        self.assertEqual(len(plan["fixed_sequences"]), 17)
        result = check_candidate_plan(plan, plan["protocol_sha256"])
        self.assertTrue(result["all_pass"], result["errors"])

    def test_runner_replay_uses_importable_module_entrypoint(self):
        source = Path("guard/active_vision/run_phase6a7.py").read_text()
        self.assertIn('"-m",\n            "guard.active_vision.run_phase6a7"', source)

    def test_preflight_has_explicit_admitted_output(self):
        source = Path("guard/active_vision/preflight_phase6a7.py").read_text()
        self.assertIn('parser.add_argument("--admitted-output", type=Path)', source)
        self.assertIn('"preflight_results_dir": "preflight_results"', source)


if __name__ == "__main__":
    unittest.main()
