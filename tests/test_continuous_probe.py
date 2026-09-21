import unittest
import json
import tempfile
from pathlib import Path

from guard.active_vision.generate_continuous_probe import audit, build
from guard.active_vision.check_continuous_preflight import check
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


if __name__ == "__main__":
    unittest.main()
