"""Independent three-route scene for the Phase 6A2 closed-loop demo."""

from dataclasses import dataclass
from itertools import product

import numpy as np
from robosuite.controllers import load_controller_config
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BallObject, BoxObject
from robosuite.models.tasks import ManipulationTask

from guard.active_vision.scene import _look_at_quat, _set_static_pose


ROUTES = ("left_route", "center_route", "right_route")
PAID_CAMERAS = ("q_left", "q_right", "q_front", "q_high")
CAMERAS = ("v0",) + PAID_CAMERAS
HIDDEN_STATES = tuple("".join(bits) for bits in product("01", repeat=3))


@dataclass(frozen=True)
class BeliefLayout:
    layout_id: str = "belief_demo_00"
    start: tuple = (-0.103, 0.0, 1.01)
    target: tuple = (0.20, 0.0, 1.01)
    lane_y: float = 0.18
    obstacle_x: float = 0.065
    occluder_x: float = 0.30


DEMO_LAYOUT = BeliefLayout()
DEMO_VISIBILITY = {
    "q_left": ("left_route",),
    "q_right": ("right_route",),
    "q_front": (),
    "q_high": (),
}


def route_waypoints(layout, route):
    if route not in ROUTES:
        raise ValueError(route)
    lane = {"left_route": layout.lane_y, "center_route": 0.0, "right_route": -layout.lane_y}[route]
    return (
        np.array([-0.02, lane, layout.target[2]]),
        np.array([0.14, lane, layout.target[2]]),
        np.array(layout.target),
    )


def camera_specs(layout):
    return {
        "v0": ([0.62, 0.0, 1.22], [0.04, 0.0, 1.0], 45),
        "q_left": ([0.04, 0.43, 1.48], [layout.obstacle_x, layout.lane_y, 1.01], 24),
        "q_right": ([0.04, -0.43, 1.48], [layout.obstacle_x, -layout.lane_y, 1.01], 24),
        "q_front": ([0.39, 0.0, 1.12], [layout.obstacle_x, 0.0, 1.01], 26),
        "q_high": ([0.02, 0.0, 1.78], [layout.obstacle_x, 0.0, 1.01], 28),
    }


class ThreeRouteBeliefEnv(SingleArmEnv):
    def __init__(self, layout=DEMO_LAYOUT, hidden_state="000", **kwargs):
        if hidden_state not in HIDDEN_STATES:
            raise ValueError(hidden_state)
        self.layout = layout
        self.hidden_state = hidden_state
        self.table_full_size = (0.8, 0.8, 0.05)
        self.table_friction = (1.0, 5e-3, 1e-4)
        self.table_offset = np.array((0.0, 0.0, 0.8))
        super().__init__(
            robots="Panda",
            controller_configs=load_controller_config(default_controller="OSC_POSE"),
            gripper_types="default",
            initialization_noise=None,
            use_camera_obs=False,
            has_renderer=False,
            has_offscreen_renderer=True,
            render_gpu_device_id=-1,
            control_freq=20,
            horizon=140,
            ignore_done=True,
            hard_reset=False,
            **kwargs,
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

        colors = ([0.9, 0.12, 0.1, 1], [0.2, 0.8, 0.2, 1], [0.1, 0.3, 0.9, 1])
        lanes = (self.layout.lane_y, 0.0, -self.layout.lane_y)
        names = ("left", "center", "right")
        self.route_obstacles = []
        for index, (name, lane, color) in enumerate(zip(names, lanes, colors)):
            obstacle = BoxObject(name=f"belief_obstacle_{name}", size=[0.045, 0.045, 0.065], rgba=color, joints=None, obj_type="all")
            z = self.layout.target[2] if self.hidden_state[index] == "1" else -5.0
            _set_static_pose(obstacle, [self.layout.obstacle_x, lane, z])
            self.route_obstacles.append(obstacle)
        self.occluder = BoxObject(name="belief_occluder", size=[0.025, 0.34, 0.20], rgba=[0.16, 0.17, 0.19, 1], joints=None, obj_type="all")
        _set_static_pose(self.occluder, [self.layout.occluder_x, 0.0, 1.02])
        self.target_marker = BallObject(name="belief_target", size=[0.018], rgba=[0.1, 0.8, 0.25, 1], joints=None, obj_type="visual")
        _set_static_pose(self.target_marker, self.layout.target)
        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[*self.route_obstacles, self.occluder, self.target_marker],
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
        if camera not in CAMERAS:
            raise ValueError(camera)
        return np.ascontiguousarray(self.sim.render(camera_name=camera, width=resolution, height=resolution)[::-1], dtype=np.uint8)

    def route_waypoints(self, route):
        return route_waypoints(self.layout, route)
