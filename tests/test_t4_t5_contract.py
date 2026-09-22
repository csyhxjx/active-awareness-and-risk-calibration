import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class T4T5ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / "configs/continuous_active_vision_probe_v2.json").open() as handle:
            cls.config = json.load(handle)

    def test_frozen_identity_and_quota(self):
        config = self.config
        self.assertEqual(config["namespace"], "continuous_active_vision_probe_v2")
        self.assertEqual(
            [config[key] for key in ("generator_seed", "pf_seed", "qmc_seed", "bootstrap_seed")],
            [66701, 66702, 66703, 66704],
        )
        self.assertEqual(config["sample_size"], {
            "retained_worlds": 24,
            "worlds_per_route": 8,
            "routes": 3,
            "counterfactual_route_trajectories": 72,
        })
        self.assertEqual(config["strata"]["quota"], {"safe": 8, "boundary": 8, "blocked": 8})

    def test_geometry_scope_rejects_unsupported_extensions(self):
        support = self.config["latent_support"]
        self.assertEqual(support["active_obstacle_count"], 1)
        self.assertEqual(support["obstacle_x_m"], 0.07)
        self.assertEqual(support["other_obstacle_x_m"], 2.0)
        self.assertEqual(
            set(support["unsupported_extensions"]),
            {"longitudinal_variation", "multiple_active_obstacles", "new_controller", "new_route_spline"},
        )

    def test_common_kernel_and_compute_stop_gate(self):
        methods = self.config["methods"]
        self.assertEqual(methods["common_risk_kernel"], "exact_mujoco_3_13_native_ccd_16_interval")
        self.assertEqual(methods["ordinary_pf"]["particles"], 2048)
        self.assertEqual(methods["reference_qmc"]["counts"], [131072, 262144])
        self.assertEqual(self.config["stop_rules"]["common_kernel_cost_fail"].split(";")[0], "stop before runner")
        self.assertEqual(self.config["status"], "frozen_prospective_config_only; no probe authorized")

    def test_observation_and_metrics_are_closed(self):
        observation = self.config["observation"]
        self.assertEqual(observation["budget"], 2)
        self.assertTrue(observation["state_preserving"])
        self.assertIn("clearance", observation["forbidden"])
        self.assertIn("collision", observation["forbidden"])
        self.assertIn("qmc_convergence", self.config["metrics"])
        self.assertEqual(self.config["risk_threshold"], 0.05)


if __name__ == "__main__":
    unittest.main()
