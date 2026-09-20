import json
import subprocess
import sys
import unittest
import numpy as np

from guard.active_vision.phase5e import (
    contract_fixture_logits,
    logits_to_proposal,
    selector_decision,
    validate_five_tuple,
    validate_proposer_input,
)
from guard.active_vision.build_manifest_v4 import build_manifest
from guard.active_vision.build_formal_manifest_v4 import build_manifest as build_formal_manifest
from guard.active_vision.scene import Layout
from guard.active_vision.train_proposer_v4 import feature


class Phase5EContractTest(unittest.TestCase):
    def test_ties_and_stop(self):
        self.assertEqual(logits_to_proposal([1, 1, 1]), "left_route")
        self.assertEqual(logits_to_proposal([0, 2, 2]), "right_route")
        self.assertEqual(logits_to_proposal([0, 0, 3]), "stop")

    def test_invalid_logits(self):
        with self.assertRaises(ValueError):
            logits_to_proposal([0, float("nan"), 0])
        with self.assertRaises(ValueError):
            logits_to_proposal([0, 0])
        with self.assertRaises(ValueError):
            logits_to_proposal([0, float("inf"), 0])
        with self.assertRaises(ValueError):
            logits_to_proposal([0, "bad", 0])

    def test_budget_semantics(self):
        self.assertEqual(selector_decision("left_route", "blocked", 0)["selected_candidate"], "left_route")
        self.assertEqual(selector_decision("left_route", "blocked", 1)["selected_candidate"], None)
        self.assertEqual(selector_decision("left_route", "clear", 1)["selected_candidate"], "left_route")
        self.assertEqual(selector_decision("stop", "unobserved", 1)["stop_reason"], "proposer_stop")

    def test_tuple_contract(self):
        record = {"proposer_output": {"proposal": "left_route"}, "mapped_candidate": {},
                  "selector_decision": {}, "executed_action": {}, "physical_outcome": {}}
        self.assertEqual(validate_five_tuple(record), (True, None))
        self.assertEqual(validate_five_tuple({}), (False, "tuple_keys"))

    def test_public_input_boundary(self):
        value = {"v0_rgb_sha256": "x", "instruction": "prefer left route", "public_route_metadata": {}}
        self.assertTrue(validate_proposer_input(value))
        self.assertEqual(contract_fixture_logits(value), [2.0, 0.0, -1.0])
        with self.assertRaises(ValueError):
            validate_proposer_input({**value, "hidden_state": "00"})
        with self.assertRaises(ValueError):
            validate_proposer_input({**value, "public_route_metadata": {"oracle": 1}})

    def test_fresh_process_pure_decision(self):
        code = "from guard.active_vision.phase5e import logits_to_proposal; print(logits_to_proposal([.1,.2,.2]))"
        result = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
        self.assertEqual(result, "right_route")

    def test_smoke_manifest_is_fresh_and_grouped(self):
        manifest = build_manifest()
        self.assertEqual(manifest["manifest_seed"], 55008)
        self.assertEqual(len(manifest["layouts"]), 8)
        self.assertEqual({row["split"] for row in manifest["layouts"]}, {"train"})
        self.assertTrue(all(row["layout_id"].startswith("av4_smoke_") for row in manifest["layouts"]))

    def test_ablation_feature_dimensions(self):
        row = build_manifest()["layouts"][0]
        layout = Layout(**{key: value for key, value in row.items() if key in Layout.__dataclass_fields__})
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        self.assertEqual(feature("instruction_only", "prefer left route", image, layout).shape, (3,))
        self.assertEqual(feature("v0_instruction", "prefer left route", image, layout).shape, (9,))
        self.assertEqual(feature("v0_instruction_geometry", "prefer left route", image, layout).shape, (27,))

    def test_formal_manifest_grouped_scope(self):
        rows = build_formal_manifest()["layouts"]
        self.assertEqual(sum(row["split"] == "train" for row in rows), 60)
        self.assertEqual(sum(row["split"] == "validation" for row in rows), 20)
        self.assertEqual(sum(row["split"] == "test" for row in rows), 40)
        self.assertEqual({row["manifest_seed"] for row in rows if row["split"] == "test"}, {57040})
        self.assertEqual(len({row["layout_id"] for row in rows}), 120)


if __name__ == "__main__":
    unittest.main()
