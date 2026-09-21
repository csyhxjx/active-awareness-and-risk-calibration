"""One-layout natural-geometry scene for the Phase 6C probe."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from robosuite.controllers import load_controller_config
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BallObject, BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.camera_utils import get_real_depth_map

from guard.active_vision.continuous_belief import CAMERAS, ROBOT_ENVELOPE_M, ROUTES
from guard.active_vision.scene import _look_at_quat, _set_static_pose


ALL_CAMERAS = ("v0",) + CAMERAS
LANES = (0.27, 0.0, -0.27)


@dataclass(frozen=True)
class ContinuousLayout:
    layout_id: str = "continuous_probe_layout_00"
    start: tuple = (-0.103, 0.0, 1.01)
    target: tuple = (0.20, 0.0, 1.01)
    lane_y: float = 0.27


def state_components(state):
    value = np.asarray(state, dtype=np.float64)
    if value.shape != (10,) or not np.isfinite(value).all():
        raise ValueError("continuous state must be finite length ten")
    return [(value[1 + 3 * index], value[2 + 3 * index], value[3 + 3 * index]) for index in range(3)]


def route_waypoints(layout, route):
    if route not in ROUTES:
        raise ValueError(route)
    lane = (layout.lane_y, 0.0, -layout.lane_y)[ROUTES.index(route)]
    return (
        np.array([-0.02, lane, layout.target[2]]),
        np.array([0.14, lane, layout.target[2]]),
        np.array(layout.target),
    )


def camera_specs(layout):
    return {
        "v0": ([0.58, 0.0, 1.20], [0.04, 0.0, 1.0], 42),
        "q_branch": ([0.02, 0.0, 1.72], [0.065, 0.0, 1.00], 38),
        "q_a": ([0.04, 0.48, 1.32], [0.065, layout.lane_y, 1.00], 25),
        "q_b": ([0.34, 0.0, 1.30], [0.065, 0.0, 1.00], 25),
        "q_c": ([0.04, -0.48, 1.32], [0.065, -layout.lane_y, 1.00], 25),
    }


class ContinuousGeometryEnv(SingleArmEnv):
    def __init__(self, state, layout=ContinuousLayout(), **kwargs):
        self.continuous_state = np.asarray(state, dtype=np.float64).copy()
        state_components(self.continuous_state)
        self.layout = layout
        self.table_full_size = (0.8, 0.8, 0.05)
        self.table_friction = (1.0, 5e-3, 1e-4)
        self.table_offset = np.array((0.0, 0.0, 0.8))
        super().__init__(
            robots="Panda", controller_configs=load_controller_config(default_controller="OSC_POSE"),
            gripper_types="default", initialization_noise=None, use_camera_obs=False,
            has_renderer=False, has_offscreen_renderer=True, render_gpu_device_id=-1,
            control_freq=20, horizon=140, ignore_done=True, hard_reset=False, **kwargs,
        )

    def reward(self, action=None):
        return 0.0

    def _check_success(self):
        return np.linalg.norm(self.eef_position - np.asarray(self.layout.target)) <= 0.02

    @property
    def eef_position(self):
        return self.sim.data.site_xpos[self.robots[0].eef_site_id].copy()

    def _load_model(self):
        super()._load_model()
        base = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(base)
        arena = TableArena(table_full_size=self.table_full_size, table_friction=self.table_friction, table_offset=self.table_offset)
        arena.set_origin([0, 0, 0])
        for light in arena.worldbody.findall("light"):
            light.set("castshadow", "false")
        for name, (position, target, fovy) in camera_specs(self.layout).items():
            arena.set_camera(name, pos=position, quat=_look_at_quat(position, target), camera_attribs={"fovy": str(fovy)})
        self.route_obstacles = []
        neutral_colors = ([0.46, 0.49, 0.52, 1.0], [0.50, 0.47, 0.45, 1.0], [0.44, 0.48, 0.46, 1.0])
        names = ("left", "center", "right")
        for index, ((x, displacement, half_width), lane, name) in enumerate(zip(state_components(self.continuous_state), LANES, names)):
            obstacle = BoxObject(
                name=f"continuous_obstacle_{name}", size=[0.035, float(half_width), 0.065],
                rgba=neutral_colors[index], joints=None, obj_type="all",
            )
            _set_static_pose(obstacle, [float(x), lane + float(displacement), self.layout.target[2] - 0.05])
            self.route_obstacles.append(obstacle)
        self.v0_occluder = BoxObject(
            name="continuous_v0_occluder", size=[0.025, 0.34, 0.18],
            rgba=[0.18, 0.19, 0.20, 1.0], joints=None, obj_type="visual",
        )
        _set_static_pose(self.v0_occluder, [0.285, 0.0, 1.04])
        self.target_marker = BallObject(name="continuous_target", size=[0.018], rgba=[0.15, 0.65, 0.25, 1], joints=None, obj_type="visual")
        _set_static_pose(self.target_marker, self.layout.target)
        self.model = ManipulationTask(
            mujoco_arena=arena, mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[*self.route_obstacles, self.v0_occluder, self.target_marker],
        )

    def _setup_references(self):
        super()._setup_references()
        names = ("left", "center", "right")
        self.obstacle_geom_ids = {
            name: {self.sim.model.geom_name2id(geom) for geom in obstacle.contact_geoms}
            for name, obstacle in zip(names, self.route_obstacles)
        }
        robot_names = set(self.robots[0].robot_model.contact_geoms)
        robot_names.update(self.robots[0].gripper.contact_geoms)
        self.robot_geom_ids = {self.sim.model.geom_name2id(name) for name in robot_names}

    def _reset_internal(self):
        super()._reset_internal()

    def capture_camera(self, camera, resolution=224):
        if camera not in ALL_CAMERAS:
            raise ValueError(camera)
        return np.ascontiguousarray(self.sim.render(camera_name=camera, width=resolution, height=resolution)[::-1], dtype=np.uint8)

    def capture_rgb_depth(self, camera, resolution=224):
        if camera not in ALL_CAMERAS:
            raise ValueError(camera)
        rgb, raw_depth = self.sim.render(camera_name=camera, width=resolution, height=resolution, depth=True)
        rgb = np.ascontiguousarray(rgb[::-1], dtype=np.uint8)
        raw_depth = np.ascontiguousarray(raw_depth[::-1], dtype=np.float64)
        return rgb, get_real_depth_map(self.sim, raw_depth)

    def route_waypoints(self, route):
        return route_waypoints(self.layout, route)


def point_to_obstacle_clearance(point: np.ndarray, state: np.ndarray, route_index: int) -> float:
    x, displacement, half_width = state_components(state)[route_index]
    center = np.array([x, LANES[route_index] + displacement, 0.96], dtype=np.float64)
    half = np.array([0.035, half_width, 0.065], dtype=np.float64)
    outside = np.maximum(np.abs(point - center) - half, 0.0)
    outside_distance = float(np.linalg.norm(outside))
    inside = np.all(np.abs(point - center) <= half)
    signed = -float(np.min(half - np.abs(point - center))) if inside else outside_distance
    return signed - ROBOT_ENVELOPE_M
