"""Obstacle-independent joint replay for Phase 6C calibration, not policy evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import mujoco

from guard.active_vision.continuous_belief import CLEARANCE_THRESHOLD_M, ROUTES
from guard.active_vision.continuous_geometry_diagnostic import (
    PROXIMITY_DISTANCE_M, nearest_robot_obstacle,
)
from guard.active_vision.continuous_scene import ContinuousGeometryEnv


WIDTHS_M = (0.025, 0.040, 0.055)
OFFSET_MAGNITUDES_M = tuple(round(0.040 + 0.002 * index, 3) for index in range(46))
OBSTACLE_X_M = 0.070
INTERVALS = 8
REFINEMENT_INTERVALS = 16
ROUTE_NAMES = ("left", "center", "right")


class NominalProximityEnv(ContinuousGeometryEnv):
    """Compile proximity margins into collision geoms before broadphase setup."""

    def _load_model(self):
        super()._load_model()
        names = set(self.robots[0].robot_model.contact_geoms)
        names.update(self.robots[0].gripper.contact_geoms)
        for obstacle in self.route_obstacles:
            names.update(obstacle.contact_geoms)
        found = set()
        for geom in self.model.worldbody.findall(".//geom"):
            name = geom.get("name")
            if name in names:
                geom.set("margin", str(PROXIMITY_DISTANCE_M))
                geom.set("gap", "0")
                found.add(name)
        if found != names:
            raise RuntimeError(f"proximity XML missing collision geoms: {sorted(names - found)}")


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def parked_state(route: str, width: float = 0.025, offset: float = 0.0) -> list[float]:
    state = [0.0]
    for candidate in ROUTES:
        state.extend((OBSTACLE_X_M, offset, width) if candidate == route else (2.0, 0.0, 0.025))
    return state


def empty_state() -> list[float]:
    return [0.0] + [value for _ in ROUTES for value in (2.0, 0.0, 0.025)]


def record_nominal(env: ContinuousGeometryEnv, route: str, max_steps: int = 100, hold_steps: int = 5) -> dict:
    """Use the existing controller once in an obstacle-parked scene."""
    waypoints = env.route_waypoints(route)
    target = np.asarray(env.layout.target)
    index = 0
    held = 0
    qpos = [np.asarray(env.sim.data.qpos, dtype=np.float64).tolist()]
    eef = [env.eef_position.tolist()]
    for _ in range(max_steps):
        error = waypoints[index] - env.eef_position
        if np.linalg.norm(error) < 0.018 and index < len(waypoints) - 1:
            index += 1
            error = waypoints[index] - env.eef_position
        action = np.zeros(7, dtype=np.float64)
        action[:3] = np.clip(error * 12.0, -0.65, 0.65)
        action[6] = -1.0
        env.step(action)
        qpos.append(np.asarray(env.sim.data.qpos, dtype=np.float64).tolist())
        eef.append(env.eef_position.tolist())
        held = held + 1 if np.linalg.norm(env.eef_position - target) <= 0.02 else 0
        if held >= hold_steps:
            break
    payload = {"route": route, "controller": "OSC_POSE", "qpos": qpos, "eef": eef,
               "reached": held >= hold_steps, "control_steps": len(qpos) - 1}
    payload["trajectory_sha256"] = digest(payload)
    return payload


def geometry_signature(env: ContinuousGeometryEnv) -> str:
    model = env.sim.model
    ids = sorted(env.robot_geom_ids | set().union(*env.obstacle_geom_ids.values()))
    return digest([{
        "name": model.geom_id2name(geom_id), "type": int(model.geom_type[geom_id]),
        "size": model.geom_size[geom_id].tolist(), "body": model.body_id2name(int(model.geom_bodyid[geom_id])),
        "body_pos": model.body_pos[int(model.geom_bodyid[geom_id])].tolist(),
    } for geom_id in ids])


def set_target_obstacle(env: ContinuousGeometryEnv, route: str, offset_m: float) -> None:
    lane = (env.layout.lane_y, 0.0, -env.layout.lane_y)[ROUTES.index(route)]
    for geom_id in env.obstacle_geom_ids[ROUTE_NAMES[ROUTES.index(route)]]:
        body_id = int(env.sim.model.geom_bodyid[geom_id])
        env.sim.model.body_pos[body_id, 1] = lane + offset_m
    env.continuous_state[2 + 3 * ROUTES.index(route)] = offset_m


def _penetrating_contact_classes(env: ContinuousGeometryEnv) -> dict[str, bool]:
    obstacle_ids = set().union(*env.obstacle_geom_ids.values())
    classes = {"route_obstacle": False, "self_collision": False, "static_environment": False}
    for contact in env.sim.data.contact[:env.sim.data.ncon]:
        if float(contact.dist) >= 0:
            continue
        pair = {int(contact.geom1), int(contact.geom2)}
        if not pair & env.robot_geom_ids:
            continue
        if pair <= env.robot_geom_ids:
            classes["self_collision"] = True
        elif pair & obstacle_ids:
            classes["route_obstacle"] = True
        else:
            classes["static_environment"] = True
    return classes


def measure_replay(env: ContinuousGeometryEnv, nominal: dict, intervals: int = INTERVALS,
                   *, keep_steps: bool = True) -> dict:
    if intervals < 1 or len(nominal["qpos"]) < 2:
        raise ValueError("nonempty nominal trajectory and positive intervals required")
    poses = np.asarray(nominal["qpos"], dtype=np.float64)
    if poses.shape[1] != env.sim.model.nq or not np.isfinite(poses).all():
        raise ValueError("joint trajectory does not match simulator")
    minimum = None
    first_contact = None
    classes = {"route_obstacle": False, "self_collision": False, "static_environment": False}
    steps = []
    # The initial pose is sampled once; later endpoints belong to each segment.
    for step in range(len(poses)):
        start = 0 if step == 0 else 1
        step_minimum = None
        for sample in range(start, intervals + 1 if step else 1):
            fraction = sample / intervals if step else 0.0
            pose = (1.0 - fraction) * poses[step - 1] + fraction * poses[step] if step else poses[0]
            env.sim.data.qpos[:] = pose
            env.sim.data.qvel[:] = 0
            env.sim.forward()
            found = nearest_robot_obstacle(env)
            hit = _penetrating_contact_classes(env)
            for kind, present in hit.items():
                classes[kind] = classes[kind] or present
            where = {"step": step, "fraction": fraction, "qpos": pose.tolist(),
                     "eef": env.eef_position.tolist()}
            if found is not None:
                candidate = {**where, **found}
                if step_minimum is None or candidate["distance_m"] < step_minimum["distance_m"]:
                    step_minimum = candidate
                if minimum is None or candidate["distance_m"] < minimum["distance_m"]:
                    minimum = candidate
                if first_contact is None and found["distance_m"] < 0:
                    first_contact = candidate
        if keep_steps:
            steps.append({"step": step, "nominal_qpos": poses[step].tolist(),
                          "nearest": step_minimum})
    clearance = None if minimum is None else minimum["distance_m"]
    collision = any(classes.values())
    margin = None if clearance is None else clearance - CLEARANCE_THRESHOLD_M
    if collision or (margin is not None and margin <= -0.004):
        stratum = "blocked"
    elif not nominal["reached"]:
        stratum = "incomplete"
    elif clearance is None or margin >= 0.008:
        stratum = "safe"
    elif abs(margin) <= 0.003:
        stratum = "boundary"
    else:
        stratum = "gap"
    result = {
        "route": nominal["route"], "nominal_trajectory_sha256": nominal["trajectory_sha256"],
        "geom_model_sha256": geometry_signature(env), "intervals": intervals,
        "proximity_radius_m": PROXIMITY_DISTANCE_M, "minimum": minimum,
        "minimum_clearance_m": clearance, "clearance_lower_bound_m": PROXIMITY_DISTANCE_M if clearance is None else None,
        "first_contact": first_contact, "collision_classes": classes,
        "reached": nominal["reached"], "stratum": stratum,
        "first_discrete_contact_step": None if first_contact is None else first_contact["step"],
    }
    if keep_steps:
        result["steps"] = steps
        result["step_trace_sha256"] = digest(steps)
    return result


def make_environment(route: str, width_m: float = 0.025) -> ContinuousGeometryEnv:
    if mujoco.__version__ != "2.3.7":
        raise RuntimeError("nominal calibration contact inclusion frozen for MuJoCo 2.3.7")
    env = NominalProximityEnv(parked_state(route, width_m, 0.0))
    env.reset()
    ids = env.robot_geom_ids | set().union(*env.obstacle_geom_ids.values())
    for geom_id in ids:
        if env.sim.model.geom_margin[geom_id] != PROXIMITY_DISTANCE_M or env.sim.model.geom_gap[geom_id] != 0:
            env.close()
            raise RuntimeError("compiled proximity parameters do not match protocol")
    return env


def scan_cases(route: str) -> list[dict]:
    if route not in ROUTES:
        raise ValueError(route)
    return [{"route": route, "width_m": width, "offset_m": sign * magnitude}
            for width in WIDTHS_M for sign in (-1, 1) for magnitude in OFFSET_MAGNITUDES_M]


def load_nominal(path: Path) -> dict:
    value = json.loads(path.read_text())
    for key in ("protocol_sha256", "source_head", "empty_obstacle_state"):
        value.pop(key, None)
    hash_value = value.pop("trajectory_sha256")
    if digest(value) != hash_value:
        raise ValueError("nominal trajectory hash mismatch")
    value["trajectory_sha256"] = hash_value
    return value
