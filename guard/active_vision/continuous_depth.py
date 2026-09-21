"""Registered metric-depth feature extraction for Phase 6C."""

from __future__ import annotations

import hashlib

import numpy as np
from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix

from guard.active_vision.continuous_belief import (
    BRANCH_QUANTIZATION_M,
    CAMERAS,
    ROBOT_ENVELOPE_M,
    SPECIALIST,
    validate_observation,
)
from guard.active_vision.continuous_scene import LANES
from guard.json_io import canonical_dumps


DEPTH_FEATURE_VERSION = "phase6c-depth-rays-v1"


def depth_to_world(sim, camera: str, depth: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    intrinsic = get_camera_intrinsic_matrix(sim, camera, height, width)
    extrinsic = get_camera_extrinsic_matrix(sim, camera)
    rows, cols = np.indices((height, width))
    z = depth.reshape(-1)
    camera_points = np.column_stack((
        (cols.reshape(-1) - intrinsic[0, 2]) / intrinsic[0, 0] * z,
        (rows.reshape(-1) - intrinsic[1, 2]) / intrinsic[1, 1] * z,
        z,
        np.ones_like(z),
    ))
    return (extrinsic @ camera_points.T).T[:, :3]


def extract_depth_observation(env, camera: str, depth: np.ndarray) -> dict:
    if camera not in CAMERAS:
        raise ValueError(camera)
    points = depth_to_world(env.sim, camera, np.asarray(depth, dtype=np.float64))
    # Public spatial gate only. Segmentation and geom ids are deliberately absent.
    gate = (
        (points[:, 0] >= 0.005) & (points[:, 0] <= 0.135)
        & (points[:, 2] >= 0.885) & (points[:, 2] <= 1.035)
    )
    points = points[gate]
    values = np.zeros(3, dtype=np.float64)
    mask = np.zeros(3, dtype=bool)
    allowed = range(3) if camera == "q_branch" else (SPECIALIST[camera],)
    for route_index in allowed:
        local = points[np.abs(points[:, 1] - LANES[route_index]) <= 0.13]
        if len(local) < 12:
            continue
        lateral_surface = float(np.min(np.abs(local[:, 1] - LANES[route_index])))
        values[route_index] = lateral_surface - ROBOT_ENVELOPE_M
        mask[route_index] = True
    if camera == "q_branch":
        values[mask] = np.round(values[mask] / BRANCH_QUANTIZATION_M) * BRANCH_QUANTIZATION_M
    observation = {"camera_id": camera, "values": values.tolist(), "mask": mask.tolist()}
    validate_observation(observation)
    return observation


def feature_hash(observation: dict) -> str:
    return hashlib.sha256(canonical_dumps(observation, sort_keys=True).encode("utf-8")).hexdigest()
