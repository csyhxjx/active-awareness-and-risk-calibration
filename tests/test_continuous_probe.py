import unittest

from guard.active_vision.generate_continuous_probe import audit, build


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


if __name__ == "__main__":
    unittest.main()
