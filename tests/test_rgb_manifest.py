import json
import tempfile
import unittest
from pathlib import Path

from guard.active_vision.generate_rgb_manifest import audit, build
from guard.active_vision.phase6a7 import ALL_STATES, FIXED_SEQUENCES


class RGBManifestTest(unittest.TestCase):
    def test_group_counts_and_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            protocol = Path(directory) / "protocol.md"
            phase6a7 = Path(directory) / "phase6a7.json"
            protocol.write_text("rgb protocol")
            phase6a7.write_text(json.dumps({"layouts": [{"layout_id": "roi_candidate_000"}]}))
            manifest = build(protocol, phase6a7)
            self.assertEqual(manifest["group_counts"], {"train": 24, "validation": 12, "sealed_test": 12})
            self.assertEqual(manifest["scene_count"], 48 * len(ALL_STATES))
            self.assertEqual(manifest["route_count"], 48 * len(ALL_STATES) * 3)
            self.assertEqual(len({tuple(row["start"]) + (row["lane_y"], row["obstacle_x"], row["occluder_x"]) for row in manifest["groups"]}), 48)
            self.assertEqual(len(manifest["fixed_sequences"]), 17)
            self.assertFalse(manifest["policy_result_filtering"])
            result = audit(manifest, protocol, phase6a7)
            self.assertTrue(result["all_pass"], result["errors"])

    def test_split_groups_are_unique(self):
        with tempfile.TemporaryDirectory() as directory:
            protocol = Path(directory) / "protocol.md"
            phase6a7 = Path(directory) / "phase6a7.json"
            protocol.write_text("rgb protocol")
            phase6a7.write_text(json.dumps({"layouts": []}))
            manifest = build(protocol, phase6a7)
            ids = [row["layout_group_id"] for row in manifest["groups"]]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(len(set(tuple(row) for row in manifest["fixed_sequences"])), len(FIXED_SEQUENCES))


if __name__ == "__main__":
    unittest.main()
