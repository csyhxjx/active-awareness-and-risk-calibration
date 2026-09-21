"""Fit the frozen Phase 6B detector using train split data only."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from guard.active_vision.rgb_data import load_training_samples
from guard.active_vision.rgb_detector import DetectorConfig, RGBDetector, preprocessing_hash
from guard.active_vision.phase6a7 import sha256_file
from guard.json_io import write_json


TRAINER_VERSION = "phase6b-rgb-trainer-v1"


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def train(index_paths: list[Path], data_root: Path, model_path: Path, freeze_path: Path) -> dict:
    config = DetectorConfig()
    samples = load_training_samples(index_paths, data_root)
    detector = RGBDetector.fit(samples, config)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    detector.save(model_path)
    record = {
        "schema_version": 1,
        "stage": "train_only",
        "trainer_version": TRAINER_VERSION,
        "git_head": git_head(),
        "train_index_sha256": {str(path): sha256_file(path) for path in index_paths},
        "train_sample_count": len(samples),
        "architecture": config.architecture,
        "optimizer": config.optimizer,
        "epochs": config.epochs,
        "seed": config.seed,
        "output_classes": detector.payload()["classes_by_camera"],
        "abstain_rule": "predicted_outcome=unobserved iff max_probability < abstain_confidence",
        "abstain_confidence": config.abstain_confidence,
        "checkpoint_rule": config.checkpoint_rule,
        "preprocessing_sha256": preprocessing_hash(),
        "config_sha256": detector.config_hash,
        "model_sha256": sha256_file(model_path),
        "model_content_hash": detector.model_hash,
        "validation_opened": False,
        "sealed_test_opened": False,
    }
    write_json(freeze_path, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, action="append", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    args = parser.parse_args()
    result = train(args.index, args.data_root, args.model, args.freeze)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
