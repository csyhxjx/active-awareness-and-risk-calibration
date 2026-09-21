"""Integrity checks and three-layer report for the one-shot Phase 6B test."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from guard.active_vision.belief_branching import ROUTE_INDEX, ROUTES
from guard.active_vision.phase6a7 import sha256_file
from guard.active_vision.freeze_rgb_validation import expected_calibration_error
from guard.json_io import write_json


CHECKER_VERSION = "phase6b-sealed-checker-v1"


def aggregate(result_paths: list[Path], manifest_path: Path, freeze_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    freeze = json.loads(freeze_path.read_text())
    shards = [json.loads(path.read_text()) for path in result_paths]
    errors = []
    scenes = [scene for shard in shards for scene in shard["scenes"]]
    expected_groups = {row["layout_group_id"] for row in manifest["groups"] if row["split"] == "sealed_test"}
    groups = {scene["layout_group_id"] for scene in scenes}
    if groups != expected_groups or len(scenes) != 96:
        errors.append("sealed_scope")
    if len({(scene["layout_group_id"], scene["hidden_state"]) for scene in scenes}) != 96:
        errors.append("scene_uniqueness")
    for shard in shards:
        if shard["manifest_sha256"] != freeze["manifest_sha256"]:
            errors.append("manifest_hash")
        if shard["evaluator_freeze_sha256"] != sha256_file(freeze_path):
            errors.append("evaluator_freeze_hash")
        if shard["model_sha256"] != freeze["model_sha256"] or shard["preprocessing_sha256"] != freeze["preprocessing_sha256"]:
            errors.append("model_or_preprocessing_hash")
    source_root = Path(__file__).resolve().parent
    for name, expected in freeze["source_sha256"].items():
        if sha256_file(source_root / name) != expected:
            errors.append("frozen_evaluator_source_hash")

    detector_rows = []
    hidden_confusion = defaultdict(lambda: defaultdict(int))
    planner_errors = 0
    planner_count = 0
    method_rows = defaultdict(list)
    route_count = 0
    physical_errors = []
    main_states = set(manifest["main_states"])
    controls = set(manifest["control_states"])
    table = manifest["observation_table"]
    for scene, result_path in [(scene, path) for path, shard in zip(result_paths, shards) for scene in shard["scenes"]]:
        state = scene["hidden_state"]
        predictions = scene["all_view_predictions_diagnostic"]
        view_by_camera = {row["camera_id"]: row for row in scene["fingerprint"]["views"]}
        scene_dir = result_path.parent / scene["layout_group_id"] / state
        for camera, view in view_by_camera.items():
            full_path = scene_dir / f"{camera}_full.png"
            roi_path = scene_dir / f"{camera}_roi.png"
            if not full_path.exists() or sha256_file(full_path) != view["full_image_sha256"]:
                errors.append("full_image_hash")
            if not roi_path.exists() or sha256_file(roi_path) != view["roi_file_sha256"]:
                errors.append("roi_image_hash")
        for camera, prediction in predictions.items():
            target = table[state][camera]
            detector_rows.append((target, prediction))
        signature = {camera: prediction["predicted_outcome"] for camera, prediction in predictions.items()}
        matches = [candidate for candidate in manifest["main_states"] if table[candidate] == signature]
        predicted_state = matches[0] if len(matches) == 1 else "ambiguous"
        hidden_confusion[state][predicted_state] += 1
        methods = {row["method"]: row for row in scene["methods"]}
        if set(methods) != set(freeze["methods"]):
            errors.append("method_rows")
        for method, row in methods.items():
            if row["queries"] > 2 or len(row["broker_ledger"]) != row["queries"]:
                errors.append("broker_budget")
            if any(entry["roi_sha256"] != view_by_camera[entry["camera_id"]]["roi_pixel_sha256"] for entry in row["broker_ledger"]):
                errors.append("broker_roi_hash")
            method_rows[(method, "main" if state in main_states else "control")].append(row)
        if state in main_states:
            planner_count += 1
            planner_errors += methods["rgb_detector_adaptive"]["action"] != methods["oracle_roi_adaptive"]["action"]
        for route, route_result in scene["routes"].items():
            route_count += 1
            blocked = state[ROUTE_INDEX[route]] == "1"
            if blocked and not route_result["collision"]:
                physical_errors.append(f"blocked_without_collision:{scene['layout_group_id']}:{state}:{route}")
            if not blocked and not route_result["collision_free_success"]:
                physical_errors.append(f"clear_without_success:{scene['layout_group_id']}:{state}:{route}")
            trajectory = result_path.parent / scene["layout_group_id"] / state / route_result["trajectory_file"]
            if not trajectory.exists() or sha256_file(trajectory) != route_result["trajectory_file_sha256"]:
                errors.append("trajectory_hash")
    if route_count != 288:
        errors.append("route_count")
    if physical_errors:
        errors.append("route_physics")

    confusion = defaultdict(lambda: defaultdict(int))
    for target, prediction in detector_rows:
        confusion[target][prediction["predicted_outcome"]] += 1
    roi_accuracy = sum(target == prediction["predicted_outcome"] for target, prediction in detector_rows) / len(detector_rows)

    task_metrics = {}
    for (method, scope), rows in sorted(method_rows.items()):
        task_metrics[f"{method}:{scope}"] = {
            "n": len(rows),
            "completion": sum(row["completion"] for row in rows) / len(rows),
            "collision": sum(row["collision"] for row in rows) / len(rows),
            "stop": sum(row["stop"] for row in rows) / len(rows),
            "mean_queries": sum(row["queries"] for row in rows) / len(rows),
            "mean_utility": sum(row["utility"] for row in rows) / len(rows),
        }
    return {
        "schema_version": 1, "checker_version": CHECKER_VERSION, "all_hard_gates_pass": not errors,
        "errors": sorted(set(errors)), "physical_errors": physical_errors,
        "scope": {"layout_groups": len(groups), "scenes": len(scenes), "candidate_routes": route_count},
        "detector_layer": {
            "roi_samples": len(detector_rows), "roi_accuracy": roi_accuracy,
            "expected_calibration_error_10bin": expected_calibration_error(detector_rows),
            "observation_confusion_matrix": {key: dict(value) for key, value in sorted(confusion.items())},
            "hidden_state_confusion_matrix": {key: dict(value) for key, value in sorted(hidden_confusion.items())},
            "control_alias_note": "000 aliases 110 and 111 aliases 100 under the registered observation table",
        },
        "planner_layer": {
            "main_trials": planner_count, "adaptive_decision_errors": planner_errors,
            "adaptive_decision_error_rate": planner_errors / planner_count,
        },
        "task_layer": task_metrics,
        "hashes": {
            "manifest_sha256": sha256_file(manifest_path), "evaluator_freeze_sha256": sha256_file(freeze_path),
            "checker_source_sha256": sha256_file(Path(__file__)),
            "runner_git_heads": sorted({shard["git_head"] for shard in shards}),
            "result_sha256": {str(path): sha256_file(path) for path in result_paths},
        },
    }


def report(result: dict) -> str:
    lines = [
        "# Phase 6B sealed RGB result", "",
        f"Hard gates: **{'PASS' if result['all_hard_gates_pass'] else 'FAIL'}**. ",
        f"Scope: {result['scope']['layout_groups']} layout groups, {result['scope']['scenes']} scenes, "
        f"{result['scope']['candidate_routes']} physical candidate trajectories.", "",
        "## Detector layer", "",
        f"- ROI accuracy: `{result['detector_layer']['roi_accuracy']:.6f}` over `{result['detector_layer']['roi_samples']}` paid-view crops.",
        f"- 10-bin ECE: `{result['detector_layer']['expected_calibration_error_10bin']:.9f}`.",
        "- The full hidden-state confusion retains the frozen structural aliases: `000 -> 110` and `111 -> 100`; controls are not in the main planner denominator.", "",
        "## Planner layer", "",
        f"- RGB-vs-oracle adaptive decision error: `{result['planner_layer']['adaptive_decision_error_rate']:.6f}` "
        f"(`{result['planner_layer']['adaptive_decision_errors']}/{result['planner_layer']['main_trials']}`).", "",
        "## Task layer", "",
        "| Method | Scope | n | Completion | Collision | Stop | Mean queries | Mean utility |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, row in result["task_layer"].items():
        method, scope = key.split(":")
        lines.append(
            f"| {method} | {scope} | {row['n']} | {row['completion']:.6f} | {row['collision']:.6f} | "
            f"{row['stop']:.6f} | {row['mean_queries']:.6f} | {row['mean_utility']:.6f} |"
        )
    lines.extend(["", "The result is limited to the controlled cue-ROI mechanism. It is not evidence of natural obstacle understanding.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.result, args.manifest, args.freeze)
    write_json(args.output, result)
    args.report.write_text(report(result), encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["all_hard_gates_pass"] else 1)


if __name__ == "__main__":
    main()
