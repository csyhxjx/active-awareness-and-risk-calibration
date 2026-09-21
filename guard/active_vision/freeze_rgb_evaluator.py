"""Create the final Phase 6B evaluator authorization before sealed test."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from guard.active_vision.phase6a7 import sha256_file
from guard.active_vision.rgb_detector import RGBDetector, preprocessing_hash, sha256_canonical
from guard.json_io import write_json


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def freeze(manifest_path: Path, validation_path: Path, model_path: Path, output: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    validation = json.loads(validation_path.read_text())
    model = RGBDetector.load(model_path)
    if validation["sealed_test_opened"] is not False:
        raise RuntimeError("validation record already opened sealed test")
    if sha256_file(model_path) != validation["model_sha256"] or model.model_hash != validation["model_content_hash"]:
        raise RuntimeError("validation model hash mismatch")
    source_root = Path(__file__).resolve().parent
    source_files = (
        "rgb_detector.py", "rgb_broker.py", "run_rgb_sealed_test.py",
        "freeze_rgb_evaluator.py", "preflight_phase6a7.py",
    )
    record = {
        "schema_version": 1, "stage": "sealed_evaluator_frozen", "git_head": git_head(),
        "manifest_path": str(manifest_path), "manifest_sha256": sha256_file(manifest_path),
        "validation_freeze_sha256": sha256_file(validation_path),
        "frozen_model_path": str(model_path), "model_sha256": sha256_file(model_path),
        "model_content_hash": model.model_hash, "config_sha256": model.config_hash,
        "preprocessing_sha256": preprocessing_hash(),
        "selected_abstain_confidence": validation["selected_abstain_confidence"],
        "selected_distance_temperature": validation["selected_distance_temperature"],
        "selected_fixed_b2_sequence": validation["selected_fixed_b2_sequence"],
        "budget": 2, "test_split": "sealed_test", "test_seed": manifest["seed"],
        "methods": ["oracle_roi_adaptive", "rgb_detector_adaptive", "best_fixed_b2_rgb"],
        "noisy_baseline": "omitted: validation confusion matrix has no off-diagonal mass",
        "source_sha256": {name: sha256_file(source_root / name) for name in source_files},
        "evaluator_contract_sha256": sha256_canonical({
            "manifest": sha256_file(manifest_path), "model": model.model_hash,
            "fixed": validation["selected_fixed_b2_sequence"], "budget": 2,
            "methods": ["oracle_roi_adaptive", "rgb_detector_adaptive", "best_fixed_b2_rgb"],
        }),
        "sealed_test_authorized": True, "sealed_test_opened": False,
    }
    write_json(output, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = freeze(args.manifest, args.validation, args.model, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
