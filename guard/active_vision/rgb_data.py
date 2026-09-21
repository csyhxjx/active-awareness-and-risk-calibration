"""Phase 6B RGB layout and dataset helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from guard.active_vision.belief_branch_scene import BranchLayout
from guard.active_vision.rgb_detector import DetectorContractError


def layout_from_rgb_spec(spec: dict) -> BranchLayout:
    return BranchLayout(
        layout_id=spec["layout_group_id"],
        start=tuple(spec["start"]),
        target=tuple(spec["target"]),
        lane_y=spec["lane_y"],
        obstacle_x=spec["obstacle_x"],
        occluder_x=spec["occluder_x"],
        cue_shift_x=spec["cue_shift_x"],
        cue_shift_y=spec["cue_shift_y"],
        camera_shift_x=spec["camera_shift_x"],
        camera_shift_y=spec["camera_shift_y"],
        color_permutation=tuple(spec["color_permutation"]),
    )


def load_labeled_samples(index_paths: list[Path], root: Path, required_split: str) -> list[dict]:
    """Load pixels while stripping all storage/provenance metadata from model input."""
    samples = []
    for index_path in index_paths:
        payload = json.loads(index_path.read_text())
        if payload.get("split") != required_split:
            raise DetectorContractError(f"loader requires {required_split} split")
        for record in payload["records"]:
            image_path = root / record["roi_path"]
            if not image_path.exists():
                image_path = index_path.parent / record["roi_path"]
            roi = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
            samples.append({
                "request": {"camera_id": record["camera_id"], "roi_rgb": roi},
                "target_symbol": record["target_symbol"],
            })
    return samples


def load_training_samples(index_paths: list[Path], root: Path) -> list[dict]:
    return load_labeled_samples(index_paths, root, "train")
