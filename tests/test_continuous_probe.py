import unittest
import json
import tempfile
from pathlib import Path

from guard.active_vision.generate_continuous_probe import audit, build
from guard.active_vision.check_continuous_preflight import check
from guard.active_vision.continuous_geometry_diagnostic import classify_physical, robot_component
from guard.active_vision.run_continuous_geometry_scan import cases
from guard.active_vision.continuous_nominal_calibration import (
    OFFSET_MAGNITUDES_M, digest, empty_state, load_nominal, scan_cases, make_environment,
    measure_replay, record_nominal, set_target_obstacle,
)
from guard.active_vision.continuous_scene import ContinuousGeometryEnv
from guard.active_vision.run_continuous_probe import require_preflight


class ContinuousProbeManifestTest(unittest.TestCase):
    def test_balanced_probe_is_explicitly_not_population_performance(self):
        class Protocol:
            def read_bytes(self): return b"protocol"
        manifest = build(Protocol())
        result = audit(manifest, Protocol())
        self.assertTrue(result["all_pass"], result["errors"])
        self.assertEqual(result["world_count"], 24)
        self.assertEqual(result["route_count"], 72)
        for counts in result["counts"].values():
            self.assertEqual(counts, {"safe": 8, "boundary": 8, "blocked": 8})
        self.assertIn("not the unconditional", manifest["world_distribution"])

    def test_runner_refuses_failed_c0(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "check.json"
            path.write_text(json.dumps({"gates": {"C0_physical_truth": "FAIL"}}))
            with self.assertRaises(RuntimeError):
                require_preflight(path)

    def test_checker_keeps_later_gates_not_run_after_c0_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text("{}")
            result = check(manifest, [root])
            self.assertEqual(result["gates"]["C0_physical_truth"], "FAIL")
            self.assertTrue(all(
                status == "NOT_RUN"
                for gate, status in result["gates"].items()
                if gate != "C0_physical_truth"
            ))
            self.assertFalse(result["all_pass"])

    def test_geometry_diagnostic_grid_is_frozen(self):
        rows = cases()
        self.assertEqual(len(rows), 72)
        self.assertEqual(len({row["case_index"] for row in rows}), 72)
        self.assertEqual({row["route"] for row in rows}, {"left_route", "center_route", "right_route"})

    def test_geometry_component_and_stratum_rules(self):
        self.assertEqual(robot_component("gripper0_finger1_collision"), "gripper")
        self.assertEqual(robot_component("robot0_link7_collision"), "wrist")
        self.assertEqual(robot_component("robot0_link3_collision"), "arm")
        base = {"collision": False, "reached": True}
        self.assertEqual(classify_physical(base | {"physical_margin_m": 0.01}), "safe")
        self.assertEqual(classify_physical(base | {"physical_margin_m": 0.0}), "boundary")
        self.assertEqual(classify_physical(base | {"physical_margin_m": -0.005}), "blocked")

    def test_nominal_grid_and_trajectory_hash(self):
        self.assertEqual(len(scan_cases("left_route")), 276)
        self.assertEqual((OFFSET_MAGNITUDES_M[0], OFFSET_MAGNITUDES_M[-1]), (0.04, 0.13))
        self.assertEqual([empty_state()[i] for i in (1, 4, 7)], [2.0] * 3)
        record = {"route": "left_route", "qpos": [[0.0], [1.0]], "eef": [[0.0], [1.0]],
                  "control_steps": 1, "controller": "OSC_POSE", "reached": True}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nominal.json"
            path.write_text(json.dumps(record | {"trajectory_sha256": digest(record),
                                                 "protocol_sha256": "metadata"}))
            self.assertEqual(load_nominal(path)["trajectory_sha256"], digest(record))
            record["qpos"][1] = [2.0]
            path.write_text(json.dumps(record | {"trajectory_sha256": digest(record) + "wrong"}))
            with self.assertRaises(ValueError):
                load_nominal(path)

    def test_compiled_proximity_and_obstacle_independent_replay(self):
        source = ContinuousGeometryEnv(empty_state())
        source.reset()
        try:
            nominal = record_nominal(source, "left_route")
        finally:
            source.close()
        self.assertTrue(nominal["reached"])
        env = make_environment("left_route")
        try:
            distances = []
            for offset in (0.085, 0.090, 0.095):
                set_target_obstacle(env, "left_route", offset)
                result = measure_replay(env, nominal, keep_steps=False)
                self.assertEqual(result["nominal_trajectory_sha256"], nominal["trajectory_sha256"])
                self.assertIsNotNone(result["minimum_clearance_m"])
                distances.append(result["minimum_clearance_m"])
            self.assertLess(max(abs(a - b) for a, b in zip(distances, distances[1:])), 0.010)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
