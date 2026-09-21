"""Official-controller preflight helpers for Phase 6C."""

from __future__ import annotations

import numpy as np

from guard.active_vision.continuous_belief import CLEARANCE_THRESHOLD_M, ROUTES
from guard.active_vision.continuous_scene import point_to_obstacle_clearance
from guard.active_vision.phase6a7 import trajectory_sha256


def _contacts(env):
    obstacle_ids = set().union(*env.obstacle_geom_ids.values())
    classes = {"route_obstacle": False, "static_environment": False, "self_collision": False}
    events = []
    for index in range(env.sim.data.ncon):
        contact = env.sim.data.contact[index]
        if float(contact.dist) >= 0:
            continue
        first, second = int(contact.geom1), int(contact.geom2)
        pair = {first, second}
        if not pair & env.robot_geom_ids:
            continue
        if pair <= env.robot_geom_ids:
            kind = "self_collision"
        elif pair & obstacle_ids:
            kind = "route_obstacle"
        else:
            kind = "static_environment"
        classes[kind] = True
        if not events:
            events.append({"class": kind, "geom1": env.sim.model.geom_id2name(first), "geom2": env.sim.model.geom_id2name(second), "penetration": -float(contact.dist)})
    return classes, events


def execute_continuous_route(env, route: str, max_steps=100, hold_steps=5) -> dict:
    route_index = ROUTES.index(route)
    waypoints = env.route_waypoints(route)
    waypoint_index = 0
    reached_run = 0
    records = []
    collisions = {"route_obstacle": False, "static_environment": False, "self_collision": False}
    first_contacts = []
    minimum_clearance = float("inf")
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
        step_classes, events = _contacts(env)
        for key, value in step_classes.items():
            collisions[key] = collisions[key] or value
        if events and not first_contacts:
            first_contacts = events
        point = env.eef_position.copy()
        clearance = point_to_obstacle_clearance(point, env.continuous_state, route_index)
        minimum_clearance = min(minimum_clearance, clearance)
        target_distance = float(np.linalg.norm(point - np.asarray(env.layout.target)))
        reached_run = reached_run + 1 if target_distance <= 0.02 else 0
        records.append({
            "step": step, "eef": point.tolist(), "waypoint_index": waypoint_index,
            "target_distance": target_distance, "clearance_m": clearance,
            "collision_classes": dict(step_classes),
        })
        if reached_run >= hold_steps:
            break
    reached = reached_run >= hold_steps
    collision = any(collisions.values())
    feasible = reached and not collision and minimum_clearance >= CLEARANCE_THRESHOLD_M
    return {
        "route": route, "steps": len(records), "reached": reached, "collision": collision,
        "collision_classes": collisions, "first_contacts": first_contacts,
        "minimum_clearance_m": minimum_clearance, "clearance_margin_m": minimum_clearance - CLEARANCE_THRESHOLD_M,
        "collision_free_success": reached and not collision, "feasible": feasible,
        "trajectory_sha256": trajectory_sha256(records), "records": records,
    }
