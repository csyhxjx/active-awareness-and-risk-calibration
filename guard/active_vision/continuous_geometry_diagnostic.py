"""Full collision-geometry diagnostics for Phase 6C development fixtures."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from guard.active_vision.continuous_belief import CLEARANCE_THRESHOLD_M, ROUTES
from guard.active_vision.continuous_physics import _contacts
from guard.active_vision.continuous_scene import point_to_obstacle_clearance


PROXIMITY_DISTANCE_M = 0.25


def enable_proximity_contacts(env, distance: float = PROXIMITY_DISTANCE_M) -> None:
    """Emit positive-distance contacts without activating contact forces."""
    ids = env.robot_geom_ids | set().union(*env.obstacle_geom_ids.values())
    for geom_id in ids:
        env.sim.model.geom_margin[geom_id] = float(distance)
        env.sim.model.geom_gap[geom_id] = float(distance)


def robot_component(name: str) -> str:
    lowered = name.lower()
    if any(token in lowered for token in ("gripper", "finger", "hand")):
        return "gripper"
    if any(token in lowered for token in ("link6", "link7", "wrist")):
        return "wrist"
    return "arm"


def nearest_robot_obstacle(env) -> dict | None:
    obstacle_owner = {
        geom_id: route
        for route, name in zip(ROUTES, ("left", "center", "right"))
        for geom_id in env.obstacle_geom_ids[name]
    }
    nearest = None
    for index in range(env.sim.data.ncon):
        contact = env.sim.data.contact[index]
        first, second = int(contact.geom1), int(contact.geom2)
        if first in env.robot_geom_ids and second in obstacle_owner:
            robot_id, obstacle_id = first, second
        elif second in env.robot_geom_ids and first in obstacle_owner:
            robot_id, obstacle_id = second, first
        else:
            continue
        distance = float(contact.dist)
        row = {
            "distance_m": distance,
            "robot_geom": env.sim.model.geom_id2name(robot_id),
            "robot_component": robot_component(env.sim.model.geom_id2name(robot_id)),
            "obstacle_geom": env.sim.model.geom_id2name(obstacle_id),
            "obstacle_route": obstacle_owner[obstacle_id],
        }
        if nearest is None or distance < nearest["distance_m"]:
            nearest = row
    return nearest


def _records_hash(records: list[dict]) -> str:
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def execute_geometry_diagnostic(env, route: str, max_steps: int = 100, hold_steps: int = 5) -> dict:
    route_index = ROUTES.index(route)
    waypoints = env.route_waypoints(route)
    waypoint_index = 0
    reached_run = 0
    records = []
    collision_classes = {"route_obstacle": False, "static_environment": False, "self_collision": False}
    first_contact = None
    minimum = None
    for step in range(max_steps):
        target = waypoints[waypoint_index]
        error = target - env.eef_position
        if np.linalg.norm(error) < 0.018 and waypoint_index < len(waypoints) - 1:
            waypoint_index += 1
            target = waypoints[waypoint_index]
            error = target - env.eef_position
        action = np.zeros(7, dtype=np.float64)
        action[:3] = np.clip(error * 12.0, -0.65, 0.65)
        action[6] = -1.0
        env.step(action)
        classes, _ = _contacts(env)
        for key, value in classes.items():
            collision_classes[key] = collision_classes[key] or value
        nearest = nearest_robot_obstacle(env)
        if nearest is not None and (minimum is None or nearest["distance_m"] < minimum["distance_m"]):
            minimum = {"step": step, **nearest}
        if nearest is not None and nearest["distance_m"] < 0.0 and first_contact is None:
            first_contact = {"step": step, **nearest}
        eef = env.eef_position.copy()
        target_distance = float(np.linalg.norm(eef - np.asarray(env.layout.target)))
        reached_run = reached_run + 1 if target_distance <= 0.02 else 0
        records.append({
            "step": step,
            "qpos": np.asarray(env.sim.data.qpos, dtype=np.float64).tolist(),
            "eef": eef.tolist(),
            "waypoint_index": waypoint_index,
            "target_distance_m": target_distance,
            "nearest": nearest,
            "analytic_eef_clearance_m": float(point_to_obstacle_clearance(eef, env.continuous_state, route_index)),
            "collision_classes": classes,
        })
        if reached_run >= hold_steps:
            break
    reached = reached_run >= hold_steps
    physical_clearance = float("inf") if minimum is None else float(minimum["distance_m"])
    physical_margin = physical_clearance - CLEARANCE_THRESHOLD_M
    collision = any(collision_classes.values())
    return {
        "route": route,
        "steps": len(records),
        "reached": reached,
        "collision": collision,
        "collision_classes": collision_classes,
        "minimum_pair": minimum,
        "first_contact": first_contact,
        "physical_clearance_m": physical_clearance,
        "physical_margin_m": physical_margin,
        "analytic_minimum_clearance_m": min(row["analytic_eef_clearance_m"] for row in records),
        "records_sha256": _records_hash(records),
        "records": records,
    }


def classify_physical(result: dict) -> str:
    margin = float(result["physical_margin_m"])
    if result["collision"] or margin <= -0.004:
        return "blocked"
    if abs(margin) <= 0.003:
        return "boundary"
    if result["reached"] and margin >= 0.008:
        return "safe"
    return "gap"
