"""Generate and audit the independent Phase 6B RGB grouped manifests."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path

from guard.active_vision.phase6a7 import ALL_STATES, FIXED_SEQUENCES, ROI, observation_table, sha256_file
from guard.json_io import write_json


PROTOCOL_ID = "belief_active_vision_rgb_protocol_v1"
PROTOCOL_FILE = "belief_active_vision_rgb_protocol_v1.md"
SEED = 66301
N_GROUPS = {"train": 24, "validation": 12, "sealed_test": 12}
N_STATES = len(ALL_STATES)
N_ROUTES = 3
SPLIT_OFFSETS = {"train": 0, "validation": N_GROUPS["train"], "sealed_test": N_GROUPS["train"] + N_GROUPS["validation"]}


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def six_digit(index: int) -> str:
    return f"{index:03d}"


def rgb_layout_spec(split: str, index: int) -> dict:
    global_index = SPLIT_OFFSETS[split] + index
    rng = random.Random(SEED + global_index * 100003)
    return {
        "layout_group_id": f"rgb_{split}_{six_digit(index)}",
        "split": split,
        "group_index": index,
        "global_index": global_index,
        "start": [-0.103, 0.0, 1.01],
        "target": [0.20, 0.0, 1.01],
        "lane_y": round(rng.uniform(0.178, 0.190), 6),
        "obstacle_x": round(rng.uniform(0.058, 0.070), 6),
        "occluder_x": round(rng.uniform(0.296, 0.306), 6),
        "cue_shift_x": round(rng.uniform(-0.006, 0.006), 6),
        "cue_shift_y": round(rng.uniform(-0.006, 0.006), 6),
        "camera_shift_x": round(rng.uniform(-0.004, 0.004), 6),
        "camera_shift_y": round(rng.uniform(-0.004, 0.004), 6),
        "color_permutation": [0, 1, 2],
    }


def build(protocol: Path, phase6a7_manifest: Path) -> dict:
    groups = [
        rgb_layout_spec(split, index)
        for split in ("train", "validation", "sealed_test")
        for index in range(N_GROUPS[split])
    ]
    ids = {row["layout_group_id"] for row in groups}
    excluded = json.loads(phase6a7_manifest.read_text())
    excluded_ids = {row["layout_id"] for row in excluded["layouts"]}
    if ids & excluded_ids:
        raise RuntimeError("RGB namespace overlaps Phase 6A7 layout ids")
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": sha256_file(protocol),
        "generator_head": git_head(),
        "seed": SEED,
        "namespace": "belief_active_vision_rgb_v1",
        "phase6a7_excluded_manifest_sha256": sha256_file(phase6a7_manifest),
        "phase6a7_excluded_layout_ids": sorted(excluded_ids),
        "policy_result_filtering": False,
        "split_rule": "layout_group; all mirrors and derived scenes remain in one split",
        "roi": list(ROI),
        "color_semantics": "global_fixed_phase6a6",
        "states": list(ALL_STATES),
        "main_states": list(ALL_STATES[:6]),
        "control_states": list(ALL_STATES[6:]),
        "observation_table": observation_table(),
        "fixed_sequences": [list(sequence) for sequence in FIXED_SEQUENCES],
        "group_counts": N_GROUPS,
        "scene_count": len(groups) * N_STATES,
        "route_count": len(groups) * N_STATES * N_ROUTES,
        "groups": groups,
    }


def audit(manifest: dict, protocol: Path, phase6a7_manifest: Path) -> dict:
    errors = []
    expected = sum(N_GROUPS.values())
    groups = manifest.get("groups", [])
    ids = [row.get("layout_group_id") for row in groups]
    if manifest.get("protocol_id") != PROTOCOL_ID:
        errors.append("protocol_id")
    if manifest.get("protocol_sha256") != sha256_file(protocol):
        errors.append("protocol_sha256")
    if manifest.get("seed") != SEED:
        errors.append("seed")
    if len(groups) != expected or len(ids) != len(set(ids)):
        errors.append("group_scope_or_uniqueness")
    if manifest.get("scene_count") != expected * N_STATES:
        errors.append("scene_count")
    if manifest.get("route_count") != expected * N_STATES * N_ROUTES:
        errors.append("route_count")
    if manifest.get("policy_result_filtering") is not False:
        errors.append("policy_result_filtering")
    if manifest.get("roi") != list(ROI) or manifest.get("color_semantics") != "global_fixed_phase6a6":
        errors.append("observation_contract")
    if set(manifest.get("observation_table", {})) != set(ALL_STATES):
        errors.append("observation_table")
    if len({tuple(row) for row in manifest.get("fixed_sequences", [])}) != 17:
        errors.append("fixed_sequences")
    expected_counts = {name: N_GROUPS[name] for name in N_GROUPS}
    actual_counts = {name: sum(row.get("split") == name for row in groups) for name in N_GROUPS}
    if actual_counts != expected_counts:
        errors.append("split_counts")
    if any(row.get("color_permutation") != [0, 1, 2] for row in groups):
        errors.append("color_permutation")
    excluded = json.loads(phase6a7_manifest.read_text())
    excluded_ids = {row["layout_id"] for row in excluded["layouts"]}
    if set(ids) & excluded_ids:
        errors.append("phase6a7_overlap")
    if manifest.get("phase6a7_excluded_manifest_sha256") != sha256_file(phase6a7_manifest):
        errors.append("phase6a7_exclusion_hash")
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "stage": "rgb_manifest_static_audit",
        "all_pass": not errors,
        "errors": errors,
        "group_counts": actual_counts,
        "scene_count": manifest.get("scene_count"),
        "route_count": manifest.get("route_count"),
        "phase6a7_overlap": sorted(set(ids) & excluded_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--protocol", type=Path, default=Path(PROTOCOL_FILE))
    parser.add_argument(
        "--phase6a7-manifest",
        type=Path,
        default=Path("artifacts/belief_active_vision_v4_12/frozen/manifest.json"),
    )
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = build(args.protocol, args.phase6a7_manifest)
    if args.audit:
        result = audit(manifest, args.protocol, args.phase6a7_manifest)
        manifest["static_audit"] = result
    write_json(args.output, manifest)
    if args.audit:
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result["all_pass"] else 1)


if __name__ == "__main__":
    main()
