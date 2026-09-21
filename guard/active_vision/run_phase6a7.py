"""Run immutable Phase 6A7 layout shards and assemble their summaries."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from guard.active_vision.belief_branch_scene import BranchingBeliefEnv
from guard.active_vision.belief_branching import ROUTE_INDEX, ROUTES, STATES
from guard.active_vision.phase6a7 import (
    ALL_CAMERAS,
    ALL_STATES,
    CHECKER_VERSION,
    FIXED_SEQUENCES,
    PROTOCOL_ID,
    ROI,
    SEED,
    allowed_roi_geom,
    detect_roi,
    evaluate_fixed_from_observations,
    layout_from_spec,
    roi_array,
    run_adaptive_from_observations,
    segmentation_geom_names,
    sha256_bytes,
)
from guard.active_vision.preflight_phase6a7 import execute_audited_route
from guard.active_vision.runtime import physical_state, state_hash
from guard.json_io import canonical_dumps, write_json


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def fresh_env(state: str, spec: dict) -> BranchingBeliefEnv:
    random.seed(SEED)
    np.random.seed(SEED)
    env = BranchingBeliefEnv(layout=layout_from_spec(spec), hidden_state=state)
    env.reset()
    return env


def capture_fingerprint(state: str, spec: dict, output_dir: Path | None = None) -> dict:
    env = fresh_env(state, spec)
    try:
        ledger = []
        for camera in ALL_CAMERAS:
            before = state_hash(physical_state(env))
            image = env.capture_camera(camera)
            segmentation = env.sim.render(
                camera_name=camera, width=224, height=224, segmentation=True
            )[::-1]
            after = state_hash(physical_state(env))
            if before != after:
                raise RuntimeError(f"camera query changed state: {state}/{camera}")
            roi = roi_array(image)
            roi_segmentation = roi_array(segmentation)
            paid = camera != "v0"
            row = {
                "camera": camera,
                "charged": paid,
                "state_hash": before,
                "image_sha256": sha256_bytes(image.tobytes()),
                "roi_sha256": sha256_bytes(roi.tobytes()),
                "roi_bounds": list(ROI),
                "roi_geom_names": segmentation_geom_names(env, roi_segmentation),
            }
            if paid:
                row.update({
                    "detector_output": detect_roi(camera, roi),
                    "expected_outcome": spec["observation_table"][state][camera],
                    "allowed_roi_geom": allowed_roi_geom(camera),
                })
            ledger.append(row)
            if output_dir is not None:
                Image.fromarray(image).save(output_dir / f"{camera}.png")
                Image.fromarray(roi).save(output_dir / f"{camera}_roi.png")
        return {"state_hash": ledger[0]["state_hash"], "ledger": ledger}
    finally:
        env.close()


def run_route(state: str, spec: dict, route: str, expected_hash: str, output: Path) -> dict:
    env = fresh_env(state, spec)
    try:
        actual_hash = state_hash(physical_state(env))
        if actual_hash != expected_hash:
            raise RuntimeError(f"route branch mismatch: {state}/{route}")
        result = execute_audited_route(env, state, route)
    finally:
        env.close()
    write_json(output, result["records"])
    return {key: value for key, value in result.items() if key != "records"} | {
        "trajectory_file": output.name,
        "branch_state_hash": actual_hash,
    }


def physical_outcome(route_map: dict, state: str, action: str) -> dict:
    if action == "stop":
        return {"action": action, "success": False, "collision": False}
    result = route_map[(state, action)]
    return {
        "action": action,
        "success": result["collision_free_success"],
        "collision": result["collision"],
    }


def run_layout(spec: dict, manifest: dict, manifest_sha256: str, protocol_sha256: str, output: Path, shard_id: str) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    scene_rows = []
    route_map = {}
    head = git_head()
    for state in ALL_STATES:
        state_dir = output / state
        state_dir.mkdir()
        fingerprint = capture_fingerprint(state, spec, state_dir)
        replay_path = state_dir / "fresh_process_fingerprint.json"
        command = [
            sys.executable,
            "-m",
            "guard.active_vision.run_phase6a7",
            "--fingerprint",
            "--manifest",
            str(manifest["_path"]),
            "--layout-id",
            spec["layout_id"],
            "--state",
            state,
            "--output",
            str(replay_path),
        ]
        subprocess.run(command, check=True, env=os.environ.copy())
        replay = json.loads(replay_path.read_text())
        routes = []
        for route in ROUTES:
            route_result = run_route(
                state, spec, route, fingerprint["state_hash"], state_dir / f"{route}_trajectory.json"
            )
            route_map[(state, route)] = route_result
            routes.append(route_result)
        observations = {
            row["camera"]: row["detector_output"]
            for row in fingerprint["ledger"]
            if row["camera"] != "v0"
        }
        scene_rows.append({
            "protocol_id": PROTOCOL_ID,
            "git_head": head,
            "manifest_sha256": manifest_sha256,
            "protocol_sha256": protocol_sha256,
            "checker_version": CHECKER_VERSION,
            "seed": SEED,
            "shard_id": shard_id,
            "layout_id": spec["layout_id"],
            "state": state,
            "fingerprint": fingerprint,
            "fresh_process_fingerprint_sha256": sha256_bytes(replay_path.read_bytes()),
            "fresh_process_replay_exact": canonical_dumps(fingerprint, sort_keys=True)
            == canonical_dumps(replay, sort_keys=True),
            "routes": routes,
        })
    adaptive = []
    for state in STATES:
        scene = next(row for row in scene_rows if row["state"] == state)
        observations = {
            row["camera"]: row["detector_output"]
            for row in scene["fingerprint"]["ledger"]
            if row["camera"] != "v0"
        }
        trace = run_adaptive_from_observations(observations)
        action = trace[-1]["decision"]["id"]
        adaptive.append({
            "state": state,
            "trace": trace,
            "terminal_action": action,
            "physical_outcome": physical_outcome(route_map, state, action),
        })
    fixed = []
    for sequence in FIXED_SEQUENCES:
        trials = []
        for state in STATES:
            scene = next(row for row in scene_rows if row["state"] == state)
            observations = {
                row["camera"]: row["detector_output"]
                for row in scene["fingerprint"]["ledger"]
                if row["camera"] != "v0"
            }
            trial = evaluate_fixed_from_observations(state, observations, sequence)
            outcome = physical_outcome(route_map, state, trial["terminal_action"])
            query_count = len(trial["trace"])
            utility = (
                1.0 if outcome["success"] else -4.0 if outcome["collision"] else -0.25
            ) - 0.05 * query_count
            trials.append(trial | {"physical_outcome": outcome, "utility": utility})
        fixed.append({
            "sequence": list(sequence),
            "trials": trials,
            "completion": sum(row["physical_outcome"]["success"] for row in trials),
            "collision": sum(row["physical_outcome"]["collision"] for row in trials),
            "stop": sum(row["terminal_action"] == "stop" for row in trials),
            "mean_queries": sum(len(row["trace"]) for row in trials) / len(trials),
            "mean_utility": sum(row["utility"] for row in trials) / len(trials),
        })
    summary = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "git_head": head,
        "manifest_sha256": manifest_sha256,
        "protocol_sha256": protocol_sha256,
        "checker_version": CHECKER_VERSION,
        "seed": SEED,
        "shard_id": shard_id,
        "layout_id": spec["layout_id"],
        "spec": spec,
        "scenes": scene_rows,
        "adaptive": adaptive,
        "fixed": fixed,
    }
    write_json(output / "summary.json", summary)
    return {"layout_id": spec["layout_id"], "summary": str((output / "summary.json").resolve())}


def assemble(manifest_path: Path, shard_dirs: list[Path], output: Path) -> None:
    manifest_sha = sha256_bytes(manifest_path.read_bytes())
    layouts = []
    shard_ids = []
    for shard_dir in shard_dirs:
        shard = json.loads((shard_dir / "shard_summary.json").read_text())
        if shard["manifest_sha256"] != manifest_sha:
            raise RuntimeError(f"manifest mismatch in {shard_dir}")
        shard_ids.append(shard["shard_id"])
        layouts.extend(shard["layouts"])
    ids = [row["layout_id"] for row in layouts]
    expected = [row["layout_id"] for row in json.loads(manifest_path.read_text())["layouts"]]
    if sorted(ids) != sorted(expected) or len(ids) != len(set(ids)):
        raise RuntimeError("shards do not form an exact disjoint manifest partition")
    write_json(output, {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "manifest_sha256": manifest_sha,
        "shard_ids": shard_ids,
        "layouts": sorted(layouts, key=lambda row: row["layout_id"]),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layout-ids")
    parser.add_argument("--layout-id")
    parser.add_argument("--state")
    parser.add_argument("--shard-id", default="single")
    parser.add_argument("--fingerprint", action="store_true")
    parser.add_argument("--assemble", nargs="*", type=Path)
    args = parser.parse_args()
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest["_path"] = str(args.manifest.resolve())
    specs = {row["layout_id"]: row for row in manifest["layouts"]}
    if args.fingerprint:
        if not args.layout_id or not args.state:
            parser.error("fingerprint mode requires --layout-id and --state")
        write_json(args.output, capture_fingerprint(args.state, specs[args.layout_id]))
        return
    if args.assemble is not None:
        assemble(args.manifest, args.assemble, args.output)
        return
    if args.protocol is None or not args.layout_ids:
        parser.error("run mode requires --protocol and --layout-ids")
    ids = args.layout_ids.split(",")
    if len(ids) != len(set(ids)) or any(layout_id not in specs for layout_id in ids):
        parser.error("layout ids must be unique manifest members")
    args.output.mkdir(parents=True, exist_ok=False)
    layout_refs = []
    for layout_id in ids:
        layout_refs.append(run_layout(
            specs[layout_id], manifest, sha256_bytes(manifest_bytes),
            sha256_bytes(args.protocol.read_bytes()), args.output / layout_id, args.shard_id,
        ))
    write_json(args.output / "shard_summary.json", {
        "schema_version": 1,
        "shard_id": args.shard_id,
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "layouts": layout_refs,
    })


if __name__ == "__main__":
    main()
