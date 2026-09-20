import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import torch

from guard.active_vision.runtime import QueryBroker
from guard.active_vision.build_manifest_v2 import build_manifest
from guard.active_vision.check_formal_v2 import verifier
from guard.active_vision.evaluate_formal_v2 import geometric_coverage, holm_adjust
from guard.active_vision.check_phase5d import check as check_phase5d
from guard.active_vision.phase5d import derive_failures, project_chunk, replay_equal
from guard.active_vision.run_formal_v2 import require_open_split
from guard.active_vision.train_selector_v2 import Selector, geometry_features, prepare_geometry
from guard.active_vision.check_pilot import obstacle_visible
from guard.active_vision.scene import DEV_LAYOUT, OccludedRouteEnv, _look_at_quat, route_waypoints


class ActiveVisionTest(unittest.TestCase):
    def test_camera_quaternion_is_unit(self):
        quat = _look_at_quat([1, 0, 1], [0, 0, 0])
        self.assertAlmostEqual(float(np.linalg.norm(quat)), 1.0)

    def test_routes_are_hidden_state_independent(self):
        left = OccludedRouteEnv.route_waypoints(Mock(layout=DEV_LAYOUT), "left_route")
        right = OccludedRouteEnv.route_waypoints(Mock(layout=DEV_LAYOUT), "right_route")
        self.assertEqual(len(left), 3)
        np.testing.assert_allclose(left[0][[0, 2]], right[0][[0, 2]])
        self.assertEqual(left[0][1], -right[0][1])

    def test_query_broker_rejects_free_and_over_budget_queries(self):
        env = Mock()
        env.capture_camera.return_value = np.zeros((4, 4, 3), dtype=np.uint8)
        env.sim.data.qpos = np.zeros(2)
        env.sim.data.qvel = np.zeros(2)
        env.sim.data.act = np.zeros(0)
        env.sim.data.ctrl = np.zeros(1)
        env.sim.data.time = 0.0
        env.robots = []
        broker = QueryBroker(env, budget=1)
        with self.assertRaises(PermissionError):
            broker.query("v0")
        broker.query("v_left")
        with self.assertRaises(PermissionError):
            broker.query("v_right")

    def test_pilot_config_has_exact_registered_scope(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / "configs/active_vision_v1/pilot.json").read_text())
        self.assertEqual(len(config["layouts"]), 12)
        self.assertEqual(len({row["layout_id"] for row in config["layouts"]}), 12)
        self.assertEqual(12 * 4 * 2, 96)

    def test_rgb_obstacle_detector_is_route_specific(self):
        image = np.full((32, 32, 3), 255, dtype=np.uint8)
        image[4:20, 4:20] = [220, 30, 25]
        self.assertTrue(obstacle_visible(image, "left_route"))
        self.assertFalse(obstacle_visible(image, "right_route"))

    def test_v2_manifest_scope_balance_and_determinism(self):
        first = build_manifest()
        second = build_manifest()
        self.assertEqual(first, second)
        counts = {
            split: sum(row["split"] == split for row in first["layouts"])
            for split in ("train", "validation", "test")
        }
        self.assertEqual(counts, {"train": 60, "validation": 20, "test": 40})
        self.assertEqual(len({row["layout_id"] for row in first["layouts"]}), 120)
        for split in counts:
            rows = [row for row in first["layouts"] if row["split"] == split]
            self.assertEqual(sum(row["mirror"] for row in rows), 0)
        root = Path(__file__).resolve().parents[1]
        development = json.loads((root / "configs/active_vision_v1/pilot.json").read_text())["layouts"]
        keys = ("target", "lane_y", "obstacle_x", "occluder_x", "mirror", "occluder_yaw")
        formal_geometries = {tuple(json.dumps(row[key], sort_keys=True) for key in keys) for row in first["layouts"]}
        development_geometries = {
            tuple(json.dumps(row[key], sort_keys=True) for key in keys) for row in development
        }
        self.assertTrue(formal_geometries.isdisjoint(development_geometries))

    def test_v2_runner_split_opening(self):
        require_open_split("train")
        with self.assertRaises(PermissionError):
            require_open_split("validation")
        with self.assertRaises(PermissionError):
            require_open_split("test")
        with tempfile.TemporaryDirectory() as directory:
            gate = Path(directory) / "gate.json"
            gate.write_text(
                json.dumps(
                    {
                        "split": "train",
                        "all_pass": True,
                        "gates": {
                            "T0": {"pass": True, "layouts": 60},
                            "T1": {"pass": True, "query_and_replay_provenance": True},
                            "T2": {"pass": True, "hard_integrity_layouts": 60},
                            "T3": {"pass": True, "physical_layouts": 60},
                            "T4": {"pass": True, "paid_view_layouts": 60},
                            "T5": {"pass": True, "retained_layouts": 60},
                        },
                    }
                )
            )
            require_open_split("validation", gate)
            with self.assertRaises(PermissionError):
                require_open_split("test", gate)
            freeze = Path(directory) / "freeze.json"
            freeze.write_text(json.dumps({"status": "frozen", "best_fixed_camera": "v_left", "files": {}}))
            with self.assertRaises(PermissionError):
                require_open_split("test", freeze=freeze)

    def test_v2_rgb_verifier_has_three_states(self):
        blank = np.full((32, 32, 3), 255, dtype=np.uint8)
        self.assertEqual(verifier(blank, "left_route")["verdict"], "unobserved")
        ribbon = blank.copy()
        ribbon[4:20, 4:20] = [220, 200, 20]
        self.assertEqual(verifier(ribbon, "left_route")["verdict"], "clear")
        ribbon[4:20, 4:20] = [220, 30, 25]
        self.assertEqual(verifier(ribbon, "left_route")["verdict"], "blocked")

    def test_v2_selector_geometry_and_architecture(self):
        row = build_manifest()["layouts"][0]
        features = geometry_features(row, "left_route", "v_left")
        self.assertEqual(features.shape, (31,))
        model = Selector(len(features))
        result = model(torch.zeros(2, 3, 224, 224), torch.zeros(2, len(features)))
        self.assertEqual(tuple(result.shape), (2,))

    def test_v2_candidate_agnostic_encoder_input_is_zeroed_after_normalization(self):
        geometry = np.arange(31, dtype=np.float32) + 1.0
        mean = np.arange(31, dtype=np.float32) + 0.5
        scale = np.full(31, 0.25, dtype=np.float32)
        aware = prepare_geometry(geometry.copy(), mean, scale, True)
        agnostic = prepare_geometry(geometry.copy(), mean, scale, False)
        np.testing.assert_allclose(aware, 2.0)
        np.testing.assert_array_equal(agnostic[:17], 0.0)
        np.testing.assert_allclose(agnostic[17:], 2.0)

    def test_v2_geometric_coverage_is_public_and_bounded(self):
        row = build_manifest()["layouts"][0]
        values = [geometric_coverage(row, "left_route", camera) for camera in ("v_left", "v_right", "v_high")]
        self.assertTrue(all(0.0 <= value <= 1.0 for value in values))
        self.assertGreater(max(values), min(values))

    def test_holm_adjustment_is_monotone_in_sorted_order(self):
        adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.02})
        self.assertEqual(adjusted, {"a": 0.03, "c": 0.04, "b": 0.04})

    def test_phase5d_projection_and_stop_contract(self):
        row = build_manifest()["layouts"][0]
        from guard.active_vision.run_formal_v2 import layout_from_row
        layout = layout_from_row(row)
        left = np.zeros((8, 7), dtype=np.float64)
        left[-1, :3] = np.asarray(route_waypoints(layout, "left_route")[1])
        left[-1, :3] = (left[-1, :3] - np.array([0.0, 0.0, 1.00])) / np.array([0.20, 0.20, 0.05])
        result = project_chunk(layout, left)
        self.assertTrue(result["valid"])
        self.assertEqual(result["mapped_candidate"], "left_route")
        stop = left.copy()
        stop[-1, :3] = [-1.0, -1.0, -1.0]
        self.assertEqual(project_chunk(layout, stop)["mapped_candidate"], "stop")
        invalid = left.copy()
        invalid[0, 0] = np.nan
        self.assertFalse(project_chunk(layout, invalid)["valid"])

    def test_phase5d_tuple_failures_and_exact_replay(self):
        record = {
            "proposer_output": {"chunk_hash": "x"},
            "mapped_candidate": {"valid": True, "mapped_candidate": "left_route"},
            "selector_decision": {"correct": False, "unresolved": False},
            "executed_action": {"route": "left_route"},
            "physical_outcome": {"collision": True, "timeout": False},
        }
        self.assertTrue(replay_equal(record, json.loads(json.dumps(record))))
        self.assertTrue(derive_failures(record)["selector_failure"])
        payload = {"trials": [{"record": record, "replay": record}]}
        result = check_phase5d(payload)
        self.assertTrue(result["all_pass"])
        self.assertEqual(result["candidate_recall_failures"], 0)

if __name__ == "__main__":
    unittest.main()
