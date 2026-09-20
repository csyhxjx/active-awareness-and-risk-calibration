"""Physical cue-board scene for the Phase 6A3 adaptive-branching gate."""

from dataclasses import dataclass

import numpy as np
from robosuite.controllers import load_controller_config
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BallObject, BoxObject
from robosuite.models.tasks import ManipulationTask

from guard.active_vision.belief_branching import CAMERAS as PAID_CAMERAS
from guard.active_vision.belief_branching import STATE_FAMILY, STATES
from guard.active_vision.belief_scene import ROUTES, route_waypoints
from guard.active_vision.scene import _look_at_quat, _set_static_pose


CAMERAS = ("v0",) + PAID_CAMERAS
FAMILY_COLORS = {
    "family_a": [0.90, 0.12, 0.10, 1.0],
    "family_b": [0.10, 0.75, 0.20, 1.0],
    "family_c": [0.10, 0.28, 0.90, 1.0],
}
FIRST_COLOR = [0.98, 0.78, 0.05, 1.0]
SECOND_COLOR = [0.72, 0.12, 0.86, 1.0]
INACTIVE_COLOR = [0.42, 0.42, 0.42, 1.0]


@dataclass(frozen=True)
class BranchLayout:
    layout_id: str = "branch_demo_00"
    start: tuple = (-0.103, 0.0, 1.01)
    target: tuple = (0.20, 0.0, 1.01)
    lane_y: float = 0.18
    obstacle_x: float = 0.065
    occluder_x: float = 0.30


DEMO_LAYOUT = BranchLayout()
CUE_POSITIONS = {
    "q_branch": (0.12, 0.0, 1.17),
    "q_a": (-0.25, 0.28, 1.08),
    "q_b": (-0.05, 0.0, 1.22),
    "q_c": (-0.25, -0.28, 1.08),
}


def camera_specs():
    return {
        "v0": ([0.62, 0.0, 1.22], [0.04, 0.0, 1.0], 45),
        "q_branch": ([0.12, 0.0, 1.50], CUE_POSITIONS["q_branch"], 15),
        "q_a": ([-0.25, 0.58, 1.08], CUE_POSITIONS["q_a"], 15),
        "q_b": ([-0.45, 0.0, 1.22], CUE_POSITIONS["q_b"], 15),
        "q_c": ([-0.25, -0.58, 1.08], CUE_POSITIONS["q_c"], 15),
    }


def cue_colors(state):
    family = STATE_FAMILY[state]
    colors = {"q_branch": FAMILY_COLORS[family]}
    family_states = {
        "family_a": ("001", "110"),
        "family_b": ("010", "101"),
        "family_c": ("011", "100"),
    }
    for candidate, camera in (("family_a", "q_a"), ("family_b", "q_b"), ("family_c", "q_c")):
        if family != candidate:
            colors[camera] = INACTIVE_COLOR
        else:
            colors[camera] = FIRST_COLOR if state == family_states[candidate][0] else SECOND_COLOR
    return colors


class BranchingBeliefEnv(SingleArmEnv):
    def __init__(self, layout=DEMO_LAYOUT, hidden_state="001", **kwargs):
        if hidden_state not in STATES:
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
        for name, (position, target, fovy) in camera_specs().items():
            arena.set_camera(name, pos=position, quat=_look_at_quat(position, target), camera_attribs={"fovy": str(fovy)})

        route_colors = ([0.9, 0.12, 0.1, 1], [0.2, 0.8, 0.2, 1], [0.1, 0.3, 0.9, 1])
        lanes = (self.layout.lane_y, 0.0, -self.layout.lane_y)
        names = ("left", "center", "right")
        self.route_obstacles = []
        for index, (name, lane, color) in enumerate(zip(names, lanes, route_colors)):
            obstacle = BoxObject(name=f"branch_obstacle_{name}", size=[0.045, 0.045, 0.065], rgba=color, joints=None, obj_type="all")
            z = self.layout.target[2] if self.hidden_state[index] == "1" else -5.0
            _set_static_pose(obstacle, [self.layout.obstacle_x, lane, z])
            self.route_obstacles.append(obstacle)

        self.occluder = BoxObject(name="branch_occluder", size=[0.025, 0.38, 0.25], rgba=[0.16, 0.17, 0.19, 1], joints=None, obj_type="all")
        _set_static_pose(self.occluder, [self.layout.occluder_x, 0.0, 1.03])

        colors = cue_colors(self.hidden_state)
        self.cues = []
        for camera in PAID_CAMERAS:
            cue = BoxObject(name=f"branch_cue_{camera}", size=[0.045, 0.045, 0.045], rgba=colors[camera], joints=None, obj_type="visual")
            _set_static_pose(cue, CUE_POSITIONS[camera])
            self.cues.append(cue)

        self.target_marker = BallObject(name="branch_target", size=[0.018], rgba=[0.1, 0.8, 0.25, 1], joints=None, obj_type="visual")
        _set_static_pose(self.target_marker, self.layout.target)
        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[*self.route_obstacles, self.occluder, *self.cues, self.target_marker],
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
