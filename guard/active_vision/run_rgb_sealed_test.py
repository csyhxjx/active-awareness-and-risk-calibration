"""One-shot sealed Phase 6B evaluator over frozen test layout groups."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from guard.active_vision.belief_branching import CAMERAS, PRIOR, ROUTES, plan, posterior, support, terminal_decision
from guard.active_vision.collect_rgb_train import fresh_env
from guard.active_vision.phase6a7 import ROI, roi_array, sha256_bytes, sha256_file, trajectory_sha256
from guard.active_vision.preflight_phase6a7 import execute_audited_route
from guard.active_vision.rgb_broker import OracleObservationSource, PurchasedROI, PurchasedViewBroker, RGBObservationSource
from guard.active_vision.rgb_detector import RGBDetector, preprocessing_hash
from guard.active_vision.runtime import physical_state, state_hash
from guard.json_io import write_json


RUNNER_VERSION = "phase6b-sealed-evaluator-v1"


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def adaptive_with_source(source) -> list[dict]:
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
        prediction = source.query(camera)
        outcome = prediction["predicted_outcome"]
        row["observation"] = outcome
        row["confidence"] = prediction.get("confidence")
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


def fixed_with_source(source, sequence: tuple[str, ...]) -> list[dict]:
    belief = PRIOR
    trace = []
    for camera in sequence:
        if terminal_decision(belief)["id"] != "stop":
            break
        prediction = source.query(camera)
        outcome = prediction["predicted_outcome"]
        row = {"support": support(belief), "camera": camera, "observation": outcome, "confidence": prediction.get("confidence")}
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


def method_outcome(method: str, state: str, trace: list[dict], broker: PurchasedViewBroker, route_map: dict) -> dict:
    action = trace[-1]["decision"]["id"]
    if action == "stop":
        success, collision = False, False
    else:
        physical = route_map[action]
        success, collision = physical["collision_free_success"], physical["collision"]
    queries = len(broker.ledger)
    utility = (1.0 if success else -4.0 if collision else -0.25) - 0.05 * queries
    return {
        "method": method, "state": state, "trace": trace, "broker_ledger": broker.ledger,
        "action": action, "completion": success, "collision": collision, "stop": action == "stop",
        "queries": queries, "utility": utility,
    }


def capture_scene(spec: dict, state: str, manifest: dict, output: Path) -> tuple[dict, dict[str, np.ndarray]]:
    env = fresh_env(spec, state, manifest["seed"])
    cache = {}
    rows = []
    try:
        for camera in CAMERAS:
            before = state_hash(physical_state(env))
            image = env.capture_camera(camera)
            after = state_hash(physical_state(env))
            if before != after:
                raise RuntimeError("camera query changed simulator state")
            roi = roi_array(image)
            cache[camera] = roi
            Image.fromarray(image).save(output / f"{camera}_full.png")
            Image.fromarray(roi).save(output / f"{camera}_roi.png")
            rows.append({
                "camera_id": camera, "state_fingerprint": before,
                "full_image_sha256": sha256_file(output / f"{camera}_full.png"),
                "roi_file_sha256": sha256_file(output / f"{camera}_roi.png"),
                "roi_pixel_sha256": sha256_bytes(roi.tobytes()), "roi_bounds": list(ROI),
            })
    finally:
        env.close()
    return {"views": rows}, cache


def run_routes(spec: dict, state: str, manifest: dict, output: Path) -> dict:
    results = {}
    for route in ROUTES:
        env = fresh_env(spec, state, manifest["seed"])
        try:
            result = execute_audited_route(env, state, route)
        finally:
            env.close()
        trajectory_path = output / f"{route}_trajectory.json"
        write_json(trajectory_path, result["records"])
        results[route] = {key: value for key, value in result.items() if key != "records"} | {
            "trajectory_sha256": trajectory_sha256(result["records"]),
            "trajectory_file_sha256": sha256_file(trajectory_path),
            "trajectory_file": trajectory_path.name,
        }
    return results


def source_for(cache, table, detector, fingerprint, oracle=False):
    broker = PurchasedViewBroker(lambda camera: PurchasedROI(camera, cache[camera], fingerprint, fingerprint))
    source = OracleObservationSource(broker, table) if oracle else RGBObservationSource(broker, detector)
    return broker, source


def run_shard(manifest_path: Path, freeze_path: Path, output: Path, shard_index: int, shard_count: int) -> dict:
    manifest = json.loads(manifest_path.read_text())
    freeze = json.loads(freeze_path.read_text())
    detector_path = Path(freeze["frozen_model_path"])
    detector = RGBDetector.load(detector_path)
    if sha256_file(detector_path) != freeze["model_sha256"] or detector.model_hash != freeze["model_content_hash"]:
        raise RuntimeError("frozen detector hash mismatch")
    if preprocessing_hash() != freeze["preprocessing_sha256"]:
        raise RuntimeError("preprocessing hash mismatch")
    groups = [row for row in manifest["groups"] if row["split"] == "sealed_test"]
    groups = [row for index, row in enumerate(groups) if index % shard_count == shard_index]
    output.mkdir(parents=True, exist_ok=False)
    scenes = []
    selected_fixed = tuple(freeze["selected_fixed_b2_sequence"])
    for spec in groups:
        for state in manifest["states"]:
            scene_dir = output / spec["layout_group_id"] / state
            scene_dir.mkdir(parents=True)
            fingerprint, cache = capture_scene(spec, state, manifest, scene_dir)
            routes = run_routes(spec, state, manifest, scene_dir)
            scene_state_hash = fingerprint["views"][0]["state_fingerprint"]
            oracle_broker, oracle_source = source_for(cache, manifest["observation_table"][state], detector, scene_state_hash, oracle=True)
            rgb_broker, rgb_source = source_for(cache, manifest["observation_table"][state], detector, scene_state_hash)
            fixed_broker, fixed_source = source_for(cache, manifest["observation_table"][state], detector, scene_state_hash)
            oracle_trace = adaptive_with_source(oracle_source)
            rgb_trace = adaptive_with_source(rgb_source)
            fixed_trace = fixed_with_source(fixed_source, selected_fixed)
            all_predictions = {
                camera: detector.predict({"camera_id": camera, "roi_rgb": cache[camera]})
                for camera in CAMERAS
            }
            scenes.append({
                "layout_group_id": spec["layout_group_id"], "hidden_state": state,
                "fingerprint": fingerprint, "routes": routes, "all_view_predictions_diagnostic": all_predictions,
                "methods": [
                    method_outcome("oracle_roi_adaptive", state, oracle_trace, oracle_broker, routes),
                    method_outcome("rgb_detector_adaptive", state, rgb_trace, rgb_broker, routes),
                    method_outcome("best_fixed_b2_rgb", state, fixed_trace, fixed_broker, routes),
                ],
            })
    result = {
        "schema_version": 1, "runner_version": RUNNER_VERSION, "git_head": git_head(),
        "manifest_sha256": sha256_file(manifest_path), "evaluator_freeze_sha256": sha256_file(freeze_path),
        "model_sha256": freeze["model_sha256"], "model_content_hash": freeze["model_content_hash"],
        "preprocessing_sha256": freeze["preprocessing_sha256"], "seed": manifest["seed"],
        "split": "sealed_test", "shard_index": shard_index, "shard_count": shard_count,
        "layout_group_count": len(groups), "scene_count": len(scenes), "scenes": scenes,
    }
    write_json(output / "results.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    result = run_shard(args.manifest, args.freeze, args.output, args.shard_index, args.shard_count)
    print(json.dumps({"layout_group_count": result["layout_group_count"], "scene_count": result["scene_count"]}, indent=2))


if __name__ == "__main__":
    main()
