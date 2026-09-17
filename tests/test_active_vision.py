import json
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from guard.active_vision.runtime import QueryBroker
from guard.active_vision.check_pilot import obstacle_visible
from guard.active_vision.scene import DEV_LAYOUT, OccludedRouteEnv, _look_at_quat


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

if __name__ == "__main__":
    unittest.main()
