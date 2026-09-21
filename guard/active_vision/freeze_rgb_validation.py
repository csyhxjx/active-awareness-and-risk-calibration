"""Validation-only calibration and evaluator freeze for Phase 6B."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from guard.active_vision.belief_branching import CAMERAS, PRIOR, ROUTE_INDEX, ROUTES, plan, posterior, support, terminal_decision
from guard.active_vision.phase6a7 import FIXED_SEQUENCES, run_adaptive_from_observations, sha256_file
from guard.active_vision.rgb_data import load_labeled_samples
from guard.active_vision.rgb_detector import RGBDetector, preprocessing_hash, sha256_canonical
from guard.json_io import canonical_dumps, write_json


FREEZER_VERSION = "phase6b-validation-freezer-v1"
THRESHOLD_CANDIDATES = (0.0, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
TEMPERATURE_CANDIDATES = (0.001, 0.002, 0.005, 0.010, 0.020, 0.040, 0.080, 0.160)


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


def run_noisy_fixed(observations: dict[str, str], sequence: tuple[str, ...]) -> list[dict]:
    belief = PRIOR
    trace = []
    for camera in sequence:
        decision = terminal_decision(belief)
        if decision["id"] != "stop":
            break
        outcome = observations[camera]
        row = {"support": support(belief), "camera": camera, "observation": outcome}
        trace.append(row)
        if outcome == "unobserved":
            row["update"] = "no_evidence"
            continue
        try:
            belief = posterior(belief, camera, outcome)
        except ValueError:
            row["update"] = "contradiction_stop"
            return trace + [{"support": support(belief), "decision": {"kind": "terminal", "id": "stop", "reason": "contradiction"}}]
    return trace + [{"support": support(belief), "decision": terminal_decision(belief)}]


def decision_outcome(state: str, trace: list[dict]) -> dict:
    action = trace[-1]["decision"]["id"]
    collision = action in ROUTES and state[ROUTE_INDEX[action]] == "1"
    completion = action in ROUTES and not collision
    queries = sum(("camera" in row) or row.get("decision", {}).get("kind") == "query" for row in trace[:-1])
    utility = (1.0 if completion else -4.0 if collision else -0.25) - 0.05 * queries
    return {"action": action, "completion": completion, "collision": collision, "stop": action == "stop", "queries": queries, "utility": utility}


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
    temperature_rows = []
    for temperature in TEMPERATURE_CANDIDATES:
        detector = base.with_calibration(temperature)
        rows = [(sample["target_symbol"], detector.predict(sample["request"])) for sample in samples]
        nll = -sum(math.log(max(prediction["calibrated_probabilities"].get(target, 1e-300), 1e-300)) for target, prediction in rows) / len(rows)
        temperature_rows.append((nll, temperature))
    selected_nll, selected_temperature = min(temperature_rows)
    calibrated_base = base.with_calibration(selected_temperature)
    candidates = []
    for threshold in THRESHOLD_CANDIDATES:
        detector = calibrated_base.with_abstain_confidence(threshold)
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

    fixed_candidates = []
    for sequence_index, sequence in enumerate(FIXED_SEQUENCES):
        outcomes = []
        for (layout_id, state), observations in sorted(scene_predictions.items()):
            if state in main_states:
                outcomes.append(decision_outcome(state, run_noisy_fixed(observations, sequence)))
        fixed_candidates.append({
            "sequence": list(sequence),
            "sequence_index": sequence_index,
            "completion": sum(row["completion"] for row in outcomes),
            "collision": sum(row["collision"] for row in outcomes),
            "stop": sum(row["stop"] for row in outcomes),
            "mean_queries": sum(row["queries"] for row in outcomes) / len(outcomes),
            "mean_utility": sum(row["utility"] for row in outcomes) / len(outcomes),
        })
    selected_fixed = max(fixed_candidates, key=lambda row: (
        row["mean_utility"], row["completion"], -row["collision"], -row["mean_queries"], -row["sequence_index"]
    ))

    output_model.parent.mkdir(parents=True, exist_ok=True)
    detector.save(output_model)
    first_index = json.loads(index_paths[0].read_text())
    first_record = first_index["records"][0]
    replay_roi = data_root / first_record["roi_path"]
    if not replay_roi.exists():
        replay_roi = index_paths[0].parent / first_record["roi_path"]
    replay_request = samples[0]["request"]
    replay_expected = detector.predict(replay_request)
    replay_process = subprocess.run(
        [sys.executable, "-m", "guard.active_vision.replay_rgb_detector", "--model", str(output_model),
         "--camera", first_record["camera_id"], "--roi", str(replay_roi)],
        check=True, capture_output=True, text=True,
    )
    replay_actual = json.loads(replay_process.stdout)
    replay_exact = canonical_dumps(replay_expected, sort_keys=True) == canonical_dumps(replay_actual, sort_keys=True)
    if not replay_exact:
        raise RuntimeError("fresh-process detector replay mismatch")
    source_root = Path(__file__).resolve().parent
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
        "temperature_candidates": list(TEMPERATURE_CANDIDATES),
        "temperature_selection": "minimum validation negative log likelihood",
        "selected_distance_temperature": selected_temperature,
        "validation_negative_log_likelihood": selected_nll,
        "threshold_candidates": list(THRESHOLD_CANDIDATES),
        "threshold_selection": "max ROI accuracy, then highest threshold",
        "selected_abstain_confidence": threshold,
        "roi_accuracy": accuracy,
        "expected_calibration_error_10bin": expected_calibration_error(rows),
        "observation_confusion_matrix": {key: dict(value) for key, value in sorted(confusion.items())},
        "adaptive_decision_error_rate": sum(row["error"] for row in decision_rows) / len(decision_rows),
        "adaptive_decision_rows": decision_rows,
        "fixed_sequence_selection_rule": "max mean utility, completion, min collision, min queries, earliest frozen sequence index",
        "fixed_sequence_candidates": fixed_candidates,
        "selected_fixed_b2_sequence": selected_fixed["sequence"],
        "selected_fixed_validation_metrics": selected_fixed,
        "unobserved_update_rule": "belief unchanged; budget consumed",
        "contradiction_rule": "stop",
        "preprocessing_sha256": preprocessing_hash(),
        "config_sha256": detector.config_hash,
        "model_content_hash": detector.model_hash,
        "model_sha256": sha256_file(output_model),
        "fresh_process_replay_exact": replay_exact,
        "fresh_process_replay": replay_actual,
        "evaluator_source_sha256": {
            name: sha256_file(source_root / name)
            for name in ("rgb_detector.py", "rgb_broker.py", "freeze_rgb_validation.py", "replay_rgb_detector.py")
        },
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
