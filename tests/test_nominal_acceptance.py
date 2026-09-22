import unittest

from guard.active_vision.attribute_sign_disagreement import validate_pair_row

from guard.active_vision.nominal_acceptance_contract import (
    ROUTES, LIMIT, digest, expected_grid, validate_output, validate_result,
)


class NominalAcceptanceContractTest(unittest.TestCase):
    def test_missing_pair_cannot_be_positive_or_comparable(self):
        row = {"distance_m": None, "sign": "unknown", "reason": "no_contact_record"}
        validate_pair_row(row)
        with self.assertRaises(ValueError):
            validate_pair_row({"distance_m": None, "sign": "positive", "reason": "no_contact_record"})
        with self.assertRaises(ValueError):
            validate_pair_row({"distance_m": 0.25, "sign": "unknown", "reason": "no_contact_record"})

    def test_frozen_grid_is_exactly_828_and_unique(self):
        grid = expected_grid()
        self.assertEqual(len(grid), 828)
        self.assertEqual(len(set(grid)), 828)
        for route in ROUTES:
            self.assertEqual(sum(row[0] == route for row in grid), 276)

    def test_truncated_worker_output_is_rejected(self):
        q = {"route": "left_route", "width_m": 0.025, "offset_m": -0.04,
             "intervals": 8, "qpos": [[0.0] * 9]}
        with self.assertRaises(ValueError):
            validate_output(b"{}\n", [q, q], {"robot_geom_names": [], "obstacle_geom_names": {"left_route": []}})

    def test_wrong_request_identity_is_rejected(self):
        q = {"route": "left_route", "width_m": 0.025, "offset_m": -0.04,
             "intervals": 1, "qpos": [[0.0] * 9]}
        m = {"robot_geom_names": ["robot"], "obstacle_geom_names": {"left_route": ["obstacle"]}}
        step = {"step": 0, "sample": 0, "fraction": 0.0, "distance_m": 0.25,
                "robot_geom": "robot", "obstacle_geom": "obstacle", "fromto": None}
        result = {"route": "center_route", "width_m": .025, "offset_m": -.04, "intervals": 1,
                  "mujoco_version": "3.13.0", "native_ccd": True, "steps": [step],
                  "minimum": step, "right_censored": True, "minimum_clearance_m": None,
                  "clearance_lower_bound_m": LIMIT, "qpos_sha256": digest(q["qpos"])}
        with self.assertRaises(ValueError):
            validate_result(result, q, m, enhanced=True)

    def test_right_censor_is_lower_bound_not_exact_distance(self):
        q = {"route": "left_route", "width_m": 0.025, "offset_m": -0.04,
             "intervals": 1, "qpos": [[0.0] * 9]}
        m = {"robot_geom_names": ["robot"], "obstacle_geom_names": {"left_route": ["obstacle"]}}
        step = {"step": 0, "sample": 0, "fraction": 0.0, "distance_m": LIMIT,
                "robot_geom": "robot", "obstacle_geom": "obstacle", "fromto": None}
        result = {"route": q["route"], "width_m": q["width_m"], "offset_m": q["offset_m"],
                  "intervals": 1, "mujoco_version": "3.13.0", "native_ccd": True,
                  "steps": [step], "minimum": step, "right_censored": True,
                  "minimum_clearance_m": None, "clearance_lower_bound_m": LIMIT,
                  "qpos_sha256": digest(q["qpos"])}
        validate_result(result, q, m, enhanced=True)
        result["minimum_clearance_m"] = LIMIT
        with self.assertRaises(ValueError):
            validate_result(result, q, m, enhanced=True)


if __name__ == "__main__":
    unittest.main()
