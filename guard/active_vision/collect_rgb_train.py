"""Collect only the frozen Phase 6B train ROI dataset."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from guard.active_vision.belief_branch_scene import BranchingBeliefEnv
from guard.active_vision.phase6a7 import ROI, allowed_roi_geom, roi_array, segmentation_geom_names, sha256_bytes, sha256_file
from guard.active_vision.rgb_data import layout_from_rgb_spec
from guard.active_vision.runtime import physical_state, state_hash
from guard.json_io import write_json


COLLECTOR_VERSION = "phase6b-rgb-train-collector-v1"


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def fresh_env(spec: dict, state: str, seed: int) -> BranchingBeliefEnv:
    random.seed(seed)
    np.random.seed(seed)
    env = BranchingBeliefEnv(layout=layout_from_rgb_spec(spec), hidden_state=state)
    env.reset()
    return env


def collect(manifest_path: Path, output: Path, shard_index: int, shard_count: int, max_groups: int | None) -> dict:
    manifest = json.loads(manifest_path.read_text())
    groups = [row for row in manifest["groups"] if row["split"] == "train"]
    groups = [row for index, row in enumerate(groups) if index % shard_count == shard_index]
    if max_groups is not None:
        groups = groups[:max_groups]
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for spec in groups:
        for state in manifest["states"]:
            env = fresh_env(spec, state, manifest["seed"])
            try:
                for camera in ("q_branch", "q_a", "q_b", "q_c"):
                    before = state_hash(physical_state(env))
                    image = env.capture_camera(camera)
                    segmentation = env.sim.render(camera_name=camera, width=224, height=224, segmentation=True)[::-1]
                    after = state_hash(physical_state(env))
                    if before != after:
                        raise RuntimeError(f"camera query changed state: {spec['layout_group_id']}/{state}/{camera}")
                    roi = roi_array(image)
                    roi_segmentation = roi_array(segmentation)
                    geom_names = segmentation_geom_names(env, roi_segmentation)
                    expected_geom = allowed_roi_geom(camera)
                    unexpected = [name for name in geom_names if name not in {expected_geom, "type_0:0"}]
                    if unexpected or expected_geom not in geom_names:
                        raise RuntimeError(f"ROI contract failed: {spec['layout_group_id']}/{state}/{camera}: {geom_names}")
                    relative = Path("images") / spec["layout_group_id"] / state
                    directory = output / relative
                    directory.mkdir(parents=True, exist_ok=True)
                    image_path = relative / f"{camera}_full.png"
                    roi_path = relative / f"{camera}_roi.png"
                    Image.fromarray(image).save(output / image_path)
                    Image.fromarray(roi).save(output / roi_path)
                    records.append({
                        "layout_group_id": spec["layout_group_id"],
                        "hidden_state": state,
                        "camera_id": camera,
                        "target_symbol": manifest["observation_table"][state][camera],
                        "full_image_path": str(image_path),
                        "roi_path": str(roi_path),
                        "full_image_sha256": sha256_file(output / image_path),
                        "roi_file_sha256": sha256_file(output / roi_path),
                        "roi_pixel_sha256": sha256_bytes(roi.tobytes()),
                        "state_fingerprint": before,
                        "roi_geom_names": geom_names,
                    })
            finally:
                env.close()
    index = {
        "schema_version": 1,
        "collector_version": COLLECTOR_VERSION,
        "git_head": git_head(),
        "manifest_sha256": sha256_file(manifest_path),
        "protocol_sha256": manifest["protocol_sha256"],
        "split": "train",
        "seed": manifest["seed"],
        "shard_index": shard_index,
        "shard_count": shard_count,
        "layout_group_count": len(groups),
        "record_count": len(records),
        "roi": list(ROI),
        "model_input_fields": ["camera_id", "roi_rgb"],
        "records": records,
    }
    write_json(output / "index.json", index)
    return index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/belief_active_vision_rgb_v1/manifest.json"))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--max-groups", type=int)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    result = collect(args.manifest, args.output, args.shard_index, args.shard_count, args.max_groups)
    print(json.dumps({key: result[key] for key in ("split", "layout_group_count", "record_count")}, indent=2))


if __name__ == "__main__":
    main()
