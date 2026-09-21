import tempfile
import unittest
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from guard.active_vision.rgb_broker import (
    OracleObservationSource,
    PurchasedROI,
    PurchasedViewBroker,
    RGBObservationSource,
)
from guard.active_vision.rgb_detector import (
    DETECTOR_VERSION,
    DetectorConfig,
    DetectorContractError,
    RGBDetector,
    preprocessing_hash,
)
from guard.active_vision.rgb_data import load_training_samples
from guard.active_vision.freeze_rgb_validation import run_noisy_adaptive
from guard.active_vision.run_rgb_sealed_test import adaptive_with_source, fixed_with_source


COLORS = {
    "q_branch": {"family_a": (230, 31, 26), "family_b": (26, 191, 51), "family_c": (26, 71, 230)},
    "q_a": {"family_a:001": (250, 199, 13), "family_a:110": (184, 31, 219), "not_applicable": (107, 107, 107)},
    "q_b": {"family_b:010": (250, 199, 13), "family_b:101": (184, 31, 219), "not_applicable": (107, 107, 107)},
    "q_c": {"family_c:011": (250, 199, 13), "family_c:100": (184, 31, 219), "not_applicable": (107, 107, 107)},
}


def image(rgb):
    return np.full((12, 13, 3), rgb, dtype=np.uint8)


def trained_detector():
    samples = []
    for camera, classes in COLORS.items():
        for label, rgb in classes.items():
            samples.append({"request": {"camera_id": camera, "roi_rgb": image(rgb)}, "target_symbol": label})
    return RGBDetector.fit(samples, DetectorConfig(abstain_confidence=0.0))


class RGBDetectorContractTest(unittest.TestCase):
    def test_rejects_every_metadata_side_channel(self):
        detector = trained_detector()
        for key in ("filename", "layout_id", "hidden_state", "route_result", "collision", "outcome"):
            request = {"camera_id": "q_branch", "roi_rgb": image((230, 31, 26)), key: "leak"}
            with self.assertRaises(DetectorContractError):
                detector.predict(request)

    def test_output_is_only_registered_symbol_or_likelihood(self):
        detector = trained_detector()
        result = detector.predict({"camera_id": "q_branch", "roi_rgb": image((230, 31, 26))})
        self.assertEqual(result["predicted_outcome"], "family_a")
        self.assertEqual(set(result), {
            "camera_id", "predicted_outcome", "confidence", "calibrated_probabilities",
            "model_hash", "detector_version",
        })
        self.assertEqual(result["detector_version"], DETECTOR_VERSION)
        self.assertAlmostEqual(sum(result["calibrated_probabilities"].values()), 1.0)

    def test_model_preprocessing_and_replay_hashes_are_stable(self):
        detector = trained_detector()
        request = {"camera_id": "q_a", "roi_rgb": image((250, 199, 13))}
        self.assertEqual(detector.predict(request), detector.predict(request))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            detector.save(path)
            restored = RGBDetector.load(path)
            self.assertEqual(detector.model_hash, restored.model_hash)
            self.assertEqual(detector.config_hash, restored.config_hash)
            self.assertEqual(preprocessing_hash(), restored.payload()["preprocessing_sha256"])
            self.assertEqual(detector.predict(request), restored.predict(request))

    def test_validation_calibration_returns_new_hashed_model(self):
        detector = trained_detector()
        calibrated = detector.with_calibration(0.005, 0.8)
        self.assertNotEqual(detector.model_hash, calibrated.model_hash)
        self.assertEqual(calibrated.config.distance_temperature, 0.005)
        self.assertEqual(calibrated.config.abstain_confidence, 0.8)

    def test_fresh_process_replay_is_exact(self):
        detector = trained_detector()
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.json"
            roi = Path(directory) / "roi.png"
            detector.save(model)
            Image = __import__("PIL.Image", fromlist=["Image"])
            pixels = image((230, 31, 26))
            Image.fromarray(pixels).save(roi)
            expected = detector.predict({"camera_id": "q_branch", "roi_rgb": pixels})
            result = subprocess.run(
                [sys.executable, "-m", "guard.active_vision.replay_rgb_detector", "--model", str(model),
                 "--camera", "q_branch", "--roi", str(roi)],
                check=True, capture_output=True, text=True,
            )
            self.assertEqual(expected, json.loads(result.stdout))

    def test_oracle_rgb_and_fixed_can_use_the_same_b2_broker_contract(self):
        detector = trained_detector()

        def supplier(camera):
            rgb = next(iter(COLORS[camera].values()))
            return PurchasedROI(camera, image(rgb), "state-hash", "state-hash")

        rgb_broker = PurchasedViewBroker(supplier)
        rgb = RGBObservationSource(rgb_broker, detector)
        rgb.query("q_branch")
        rgb.query("q_a")
        self.assertEqual(rgb_broker.remaining, 0)
        with self.assertRaises(RuntimeError):
            rgb.query("q_b")

        oracle_broker = PurchasedViewBroker(supplier)
        oracle = OracleObservationSource(oracle_broker, {"q_branch": "family_a", "q_a": "family_a:001"})
        oracle.query("q_branch")
        oracle.query("q_a")
        self.assertEqual(oracle_broker.remaining, 0)

    def test_broker_rejects_state_mutation_and_repeat_purchase(self):
        good = PurchasedViewBroker(lambda camera: PurchasedROI(camera, image((1, 2, 3)), "x", "x"))
        good.purchase("q_branch")
        with self.assertRaises(RuntimeError):
            good.purchase("q_branch")
        bad = PurchasedViewBroker(lambda camera: PurchasedROI(camera, image((1, 2, 3)), "x", "y"))
        with self.assertRaises(RuntimeError):
            bad.purchase("q_branch")

    def test_training_loader_rejects_non_train_index(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            path.write_text('{"split":"validation","records":[]}')
            with self.assertRaises(DetectorContractError):
                load_training_samples([path], Path(directory))

    def test_training_loader_resolves_paths_relative_to_each_shard_index(self):
        with tempfile.TemporaryDirectory() as directory:
            shard = Path(directory) / "shard_0"
            (shard / "images").mkdir(parents=True)
            Image = __import__("PIL.Image", fromlist=["Image"])
            Image.fromarray(image((1, 2, 3))).save(shard / "images" / "roi.png")
            index = shard / "index.json"
            index.write_text(
                '{"split":"train","records":[{"roi_path":"images/roi.png",'
                '"camera_id":"q_branch","target_symbol":"family_a"}]}'
            )
            samples = load_training_samples([index], Path(directory) / "unused")
            self.assertEqual(samples[0]["request"]["roi_rgb"].shape, (12, 13, 3))

    def test_unobserved_consumes_budget_without_becoming_clear(self):
        trace = run_noisy_adaptive({
            "q_branch": "unobserved", "q_a": "unobserved",
            "q_b": "unobserved", "q_c": "unobserved",
        })
        self.assertEqual(trace[-1]["decision"]["id"], "stop")
        self.assertEqual(sum(row["decision"]["kind"] == "query" for row in trace), 2)

    def test_sealed_policy_uses_broker_budget_for_oracle_and_rgb(self):
        detector = trained_detector()
        table = {"q_branch": "family_a", "q_a": "family_a:001", "q_b": "not_applicable", "q_c": "not_applicable"}

        def supplier(camera):
            rgb = next(iter(COLORS[camera].values()))
            return PurchasedROI(camera, image(rgb), "same", "same")

        oracle_broker = PurchasedViewBroker(supplier)
        oracle_trace = adaptive_with_source(OracleObservationSource(oracle_broker, table))
        self.assertLessEqual(len(oracle_broker.ledger), 2)
        self.assertEqual(oracle_trace[-1]["decision"]["id"], "left_route")
        fixed_broker = PurchasedViewBroker(supplier)
        fixed_with_source(RGBObservationSource(fixed_broker, detector), ("q_a", "q_b"))
        self.assertLessEqual(len(fixed_broker.ledger), 2)


if __name__ == "__main__":
    unittest.main()
