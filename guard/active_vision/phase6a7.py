"""Frozen constants and pure helpers for the Phase 6A7 development run."""

from __future__ import annotations

import hashlib
import itertools
import random
from pathlib import Path

import numpy as np

from guard.active_vision.belief_branch_scene import BranchLayout
from guard.active_vision.belief_branching import (
    CAMERAS,
    FAMILIES,
    PRIOR,
    ROUTES,
    SPECIALIST,
    STATES,
    plan,
    posterior,
    support,
    terminal_decision,
)
from guard.json_io import canonical_dumps


PROTOCOL_ID = "belief_active_vision_protocol_v4_12layout_addendum"
PROTOCOL_FILE = "belief_active_vision_protocol_v4_12layout_addendum.md"
SEED = 66212
CHECKER_VERSION = "phase6a7-v1"
ROI = (64, 64, 160, 160)
CLEARANCE_THRESHOLD_M = 0.004
ALL_STATES = STATES + ("000", "111")
PAID_CAMERAS = CAMERAS
ALL_CAMERAS = ("v0",) + PAID_CAMERAS
FIXED_SEQUENCES = tuple(
    sequence
    for length in range(3)
    for sequence in itertools.permutations(PAID_CAMERAS, length)
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def candidate_spec(candidate_index: int) -> dict:
    """Return a deterministic public layout without using policy outcomes."""
    rng = random.Random(SEED + int(candidate_index) * 104729)
    return {
        "layout_id": f"roi_candidate_{candidate_index:03d}",
        "candidate_index": int(candidate_index),
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


def layout_from_spec(spec: dict) -> BranchLayout:
    fields = {
        key: tuple(value) if key in {"start", "target", "color_permutation"} else value
        for key, value in spec.items()
        if key not in {
            "candidate_index", "states", "main_states", "control_states",
            "observation_table", "roi", "preflight_result_sha256",
        }
    }
    return BranchLayout(**fields)


def observation_outcome(state: str, camera: str) -> str:
    """Complete eight-state table matching the fixed cue-color contract."""
    if state in STATES:
        from guard.active_vision.belief_branching import observation

        return observation(state, camera)
    if state == "000":
        family = "family_a"
        alias = "110"
    elif state == "111":
        family = "family_c"
        alias = "100"
    else:
        raise ValueError(state)
    if camera == "q_branch":
        return family
    if camera == SPECIALIST[family]:
        return f"{family}:{alias}"
    if camera in PAID_CAMERAS:
        return "not_applicable"
    raise ValueError(camera)


def observation_table() -> dict:
    return {
        state: {camera: observation_outcome(state, camera) for camera in PAID_CAMERAS}
        for state in ALL_STATES
    }


def detect_roi(camera: str, roi_rgb: np.ndarray) -> str:
    """Map only registered cue pixels to the frozen symbolic observation."""
    mean = np.asarray(roi_rgb, dtype=np.float64).mean(axis=(0, 1))
    red, green, blue = mean.tolist()
    if camera == "q_branch":
        return ("family_a", "family_b", "family_c")[int(np.argmax(mean))]
    family_by_camera = {value: key for key, value in SPECIALIST.items()}
    if camera not in family_by_camera:
        raise ValueError(camera)
    family = family_by_camera[camera]
    if max(mean) - min(mean) < 35.0:
        return "not_applicable"
    first_state, second_state = FAMILIES[family]
    state = first_state if green > blue else second_state
    return f"{family}:{state}"


def roi_array(image: np.ndarray) -> np.ndarray:
    y0, x0, y1, x1 = ROI
    return np.ascontiguousarray(image[y0:y1, x0:x1])


def allowed_roi_geom(camera: str) -> str:
    return f"branch_cue_{camera}_g0_vis"


def segmentation_geom_names(env, segmentation: np.ndarray) -> list[str]:
    names = set()
    for object_type, object_id in np.unique(segmentation.reshape(-1, 2), axis=0):
        if int(object_type) != 5:
            names.add(f"type_{int(object_type)}:{int(object_id)}")
            continue
        name = env.sim.model.geom_id2name(int(object_id))
        names.add(name if name is not None else f"geom:{int(object_id)}")
    return sorted(names)


def run_adaptive_from_observations(observations: dict[str, str]) -> list[dict]:
    belief = PRIOR
    cameras = list(PAID_CAMERAS)
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
        belief = posterior(belief, camera, outcome)
        cameras.remove(camera)
        remaining -= 1


def evaluate_fixed_from_observations(state: str, observations: dict[str, str], sequence: tuple[str, ...]) -> dict:
    belief = PRIOR
    trace = []
    for camera in sequence:
        decision = terminal_decision(belief)
        if decision["id"] != "stop":
            break
        outcome = observations[camera]
        trace.append({"support": support(belief), "camera": camera, "observation": outcome})
        belief = posterior(belief, camera, outcome)
    terminal = terminal_decision(belief)
    return {
        "state": state,
        "sequence": list(sequence),
        "trace": trace,
        "terminal_action": terminal["id"],
    }


def trajectory_sha256(records: list[dict]) -> str:
    return sha256_bytes(canonical_dumps(records, sort_keys=True).encode("utf-8"))

