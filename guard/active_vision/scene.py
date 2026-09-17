"""Independent Panda route scene for the Phase 5A feasibility study."""

from dataclasses import dataclass

import numpy as np
from robosuite.controllers import load_controller_config
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BallObject, BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import array_to_string
from robosuite.utils.transform_utils import mat2quat


CAMERAS = ("v0", "v_left", "v_right", "v_high")
HIDDEN_STATES = ("00", "10", "01", "11")
ROUTES = ("left_route", "right_route")


@dataclass(frozen=True)
class Layout:
    layout_id: str
    start: tuple
    target: tuple
    lane_y: float
    obstacle_x: float
    occluder_x: float
    mirror: int = 1
    occluder_yaw: float = 0.0


DEV_LAYOUT = Layout(
    layout_id="dev_00",
    start=(-0.103, 0.0, 1.01),
    target=(0.20, 0.0, 1.01),
    lane_y=0.18,
    obstacle_x=0.065,
    occluder_x=0.30,
)


def _look_at_quat(position, target):
    position = np.asarray(position, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    backward = position - target
    backward /= np.linalg.norm(backward)
    right = np.cross(np.array([0.0, 0.0, 1.0]), backward)
    if np.linalg.norm(right) < 1e-8:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    xyzw = mat2quat(np.column_stack((right, up, backward)))
    quat = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)
    return quat / np.linalg.norm(quat)


def _set_static_pose(obj, position, quat=None):
    obj.get_obj().set("pos", array_to_string(position))
    if quat is not None:
        obj.get_obj().set("quat", array_to_string(quat))


class OccludedRouteEnv(SingleArmEnv):
    """Panda reach scene whose hidden state only changes two route obstacles."""

    def __init__(self, layout=DEV_LAYOUT, hidden_state="00", **kwargs):
        if hidden_state not in HIDDEN_STATES:
            raise ValueError(f"invalid hidden state: {hidden_state}")
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
        arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        arena.set_origin([0, 0, 0])
        for light in arena.worldbody.findall("light"):
            light.set("castshadow", "false")

        camera_poses = {
            "v0": ([0.62, 0.0, 1.22], [0.04, 0.0, 1.0], 45),
            "v_left": (
                [self.layout.obstacle_x, self.layout.mirror * self.layout.lane_y, 1.62],
                [self.layout.obstacle_x, self.layout.mirror * self.layout.lane_y, 1.0],
                20,
            ),
            "v_right": (
                [self.layout.obstacle_x, -self.layout.mirror * self.layout.lane_y, 1.62],
                [self.layout.obstacle_x, -self.layout.mirror * self.layout.lane_y, 1.0],
                20,
            ),
            "v_high": (
                [self.layout.obstacle_x, 0.78, 1.45],
                [self.layout.obstacle_x, 0.12, 1.0],
                30,
            ),
        }
        for name, (position, target, fovy) in camera_poses.items():
            arena.set_camera(
                name,
                pos=position,
                quat=_look_at_quat(position, target),
                camera_attribs={"fovy": str(fovy)},
            )

        left_present = self.hidden_state[0] == "1"
        right_present = self.hidden_state[1] == "1"
        present_z = self.layout.target[2]
        absent_z = -5.0
        self.left_obstacle = BoxObject(
            name="route_obstacle_left",
            size=[0.045, 0.045, 0.065],
            rgba=[0.9, 0.12, 0.1, 1],
            joints=None,
            obj_type="all",
        )
        self.right_obstacle = BoxObject(
            name="route_obstacle_right",
            size=[0.045, 0.045, 0.065],
            rgba=[0.1, 0.3, 0.9, 1],
            joints=None,
            obj_type="all",
        )
        _set_static_pose(
            self.left_obstacle,
            [self.layout.obstacle_x, self.layout.mirror * self.layout.lane_y, present_z if left_present else absent_z],
        )
        _set_static_pose(
            self.right_obstacle,
            [self.layout.obstacle_x, -self.layout.mirror * self.layout.lane_y, present_z if right_present else absent_z],
        )
        self.occluder = BoxObject(
            name="route_occluder",
            size=[0.025, 0.34, 0.20],
            rgba=[0.16, 0.17, 0.19, 1],
            joints=None,
            obj_type="all",
        )
        yaw = self.layout.occluder_yaw
        occluder_quat = [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]
        _set_static_pose(self.occluder, [self.layout.occluder_x, 0.0, 1.02], quat=occluder_quat)
        self.query_divider = BoxObject(
            name="query_divider",
            size=[0.065, 0.025, 0.18],
            rgba=[0.16, 0.17, 0.19, 1],
            joints=None,
            obj_type="visual",
        )
        _set_static_pose(self.query_divider, [self.layout.obstacle_x, 0.0, 1.02])
        self.target_marker = BallObject(
            name="route_target",
            size=[0.018],
            rgba=[0.1, 0.8, 0.25, 1],
            joints=None,
            obj_type="visual",
        )
        _set_static_pose(self.target_marker, self.layout.target)
        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[
                self.left_obstacle,
                self.right_obstacle,
                self.occluder,
                self.query_divider,
                self.target_marker,
            ],
        )

    def _setup_references(self):
        super()._setup_references()
        self.obstacle_geom_ids = {
            "left": {self.sim.model.geom_name2id(name) for name in self.left_obstacle.contact_geoms},
            "right": {self.sim.model.geom_name2id(name) for name in self.right_obstacle.contact_geoms},
        }
        robot_names = set(self.robots[0].robot_model.contact_geoms)
        robot_names.update(self.robots[0].gripper.contact_geoms)
        self.robot_geom_ids = {self.sim.model.geom_name2id(name) for name in robot_names}

    def _reset_internal(self):
        super()._reset_internal()

    def capture_camera(self, camera, resolution=224):
        if camera not in CAMERAS:
            raise ValueError(f"unknown camera: {camera}")
        return np.ascontiguousarray(
            self.sim.render(camera_name=camera, width=resolution, height=resolution)[::-1], dtype=np.uint8
        )

    def route_waypoints(self, route):
        if route not in ROUTES:
            raise ValueError(f"unknown route: {route}")
        sign = 1 if route == "left_route" else -1
        lane = sign * self.layout.mirror * self.layout.lane_y
        return (
            np.array([-0.02, lane, self.layout.target[2]]),
            np.array([0.14, lane, self.layout.target[2]]),
            np.array(self.layout.target),
        )
