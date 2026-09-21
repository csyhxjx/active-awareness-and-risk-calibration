"""Validation-only calibration and evaluator freeze for Phase 6B."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

from guard.active_vision.belief_branching import CAMERAS, PRIOR, plan, posterior, support
from guard.active_vision.phase6a7 import run_adaptive_from_observations, sha256_file
from guard.active_vision.rgb_data import load_labeled_samples
from guard.active_vision.rgb_detector import RGBDetector, preprocessing_hash, sha256_canonical
from guard.json_io import write_json


FREEZER_VERSION = "phase6b-validation-freezer-v1"
THRESHOLD_CANDIDATES = (0.0, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95)


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def run_noisy_adaptive(observations: dict[str, str]) -> list[dict]:
    belief = PRIOR
    cameras = list(CAMERAS)
    remaining = 2
    trace = []
    while True:
        decision = plan(belief, tuple(cameras), remaining)
        row = {"support": support(belief), "decision": decision}
        trace.append(row)
        if decision["kind"] == "terminal":
            return trace
        camera = decision["id"]
        outcome = observations[camera]
        row["observation"] = outcome
        cameras.remove(camera)
        remaining -= 1
        if outcome == "unobserved":
            row["update"] = "no_evidence"
            continue
        try:
            belief = posterior(belief, camera, outcome)
        except ValueError:
            row["update"] = "contradiction_stop"
            trace.append({"support": support(belief), "decision": {"kind": "terminal", "id": "stop", "reason": "contradiction"}})
            return trace


def expected_calibration_error(rows: list[tuple[str, dict]], bins: int = 10) -> float:
    total = len(rows)
    value = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        bucket = [(target, pred) for target, pred in rows if low <= pred["confidence"] <= high and (index == bins - 1 or pred["confidence"] < high)]
        if not bucket:
            continue
        accuracy = sum(target == pred["predicted_outcome"] for target, pred in bucket) / len(bucket)
        confidence = sum(pred["confidence"] for _, pred in bucket) / len(bucket)
        value += len(bucket) / total * abs(accuracy - confidence)
    return value


def freeze(index_paths: list[Path], data_root: Path, train_model: Path, manifest_path: Path, output_model: Path, output_record: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    samples = load_labeled_samples(index_paths, data_root, "validation")
    base = RGBDetector.load(train_model)
    candidates = []
    for threshold in THRESHOLD_CANDIDATES:
        detector = base.with_abstain_confidence(threshold)
        rows = [(sample["target_symbol"], detector.predict(sample["request"])) for sample in samples]
        accuracy = sum(target == prediction["predicted_outcome"] for target, prediction in rows) / len(rows)
        candidates.append((accuracy, threshold, detector, rows))
    accuracy, threshold, detector, rows = max(candidates, key=lambda row: (row[0], row[1]))
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for target, prediction in rows:
        confusion[target][prediction["predicted_outcome"]] += 1

    scene_predictions = {}
    sample_cursor = 0
    for index_path in index_paths:
        index = json.loads(index_path.read_text())
        for record in index["records"]:
            prediction = rows[sample_cursor][1]
            sample_cursor += 1
            key = (record["layout_group_id"], record["hidden_state"])
            scene_predictions.setdefault(key, {})[record["camera_id"]] = prediction["predicted_outcome"]
    main_states = set(manifest["main_states"])
    decision_rows = []
    for (layout_id, state), observations in sorted(scene_predictions.items()):
        if state not in main_states:
            continue
        oracle_trace = run_adaptive_from_observations(manifest["observation_table"][state])
        rgb_trace = run_noisy_adaptive(observations)
        oracle_action = oracle_trace[-1]["decision"]["id"]
        rgb_action = rgb_trace[-1]["decision"]["id"]
        decision_rows.append({
            "layout_group_id": layout_id,
            "hidden_state": state,
            "oracle_action": oracle_action,
            "rgb_action": rgb_action,
            "error": oracle_action != rgb_action,
            "rgb_queries": sum(row["decision"]["kind"] == "query" for row in rgb_trace),
        })

    output_model.parent.mkdir(parents=True, exist_ok=True)
    detector.save(output_model)
    record = {
        "schema_version": 1,
        "stage": "validation_frozen",
        "freezer_version": FREEZER_VERSION,
        "git_head": git_head(),
        "manifest_sha256": sha256_file(manifest_path),
        "train_model_sha256": sha256_file(train_model),
        "validation_index_sha256": {str(path): sha256_file(path) for path in index_paths},
        "validation_sample_count": len(samples),
        "checkpoint_selection": "single train checkpoint",
        "threshold_candidates": list(THRESHOLD_CANDIDATES),
        "threshold_selection": "max ROI accuracy, then highest threshold",
        "selected_abstain_confidence": threshold,
        "roi_accuracy": accuracy,
        "expected_calibration_error_10bin": expected_calibration_error(rows),
        "observation_confusion_matrix": {key: dict(value) for key, value in sorted(confusion.items())},
        "adaptive_decision_error_rate": sum(row["error"] for row in decision_rows) / len(decision_rows),
        "adaptive_decision_rows": decision_rows,
        "unobserved_update_rule": "belief unchanged; budget consumed",
        "contradiction_rule": "stop",
        "preprocessing_sha256": preprocessing_hash(),
        "config_sha256": detector.config_hash,
        "model_content_hash": detector.model_hash,
        "model_sha256": sha256_file(output_model),
        "evaluator_inputs_sha256": sha256_canonical({
            "model_content_hash": detector.model_hash,
            "preprocessing_sha256": preprocessing_hash(),
            "threshold": threshold,
            "unobserved_rule": "belief_unchanged",
            "contradiction_rule": "stop",
            "budget": 2,
        }),
        "sealed_test_opened": False,
    }
    write_json(output_record, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, action="append", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-record", type=Path, required=True)
    args = parser.parse_args()
    result = freeze(args.index, args.data_root, args.train_model, args.manifest, args.output_model, args.output_record)
    print(json.dumps({key: result[key] for key in ("roi_accuracy", "expected_calibration_error_10bin", "adaptive_decision_error_rate", "selected_abstain_confidence", "model_content_hash")}, indent=2))


if __name__ == "__main__":
    main()
