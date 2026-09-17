"""Fixed controller, collision oracle, and observation broker for Phase 5A."""

import hashlib
import random
import time

import numpy as np

from guard.active_vision.scene import CAMERAS
from guard.json_io import canonical_dumps


def _numeric_attributes(value):
    snapshot = {}
    for key, item in vars(value).items():
        if isinstance(item, np.ndarray):
            snapshot[key] = item.copy()
        elif isinstance(item, (bool, int, float, str, type(None), np.generic)):
            snapshot[key] = item.item() if isinstance(item, np.generic) else item
        elif isinstance(item, (tuple, list)) and all(
            isinstance(part, (bool, int, float, str, type(None), np.generic)) for part in item
        ):
            snapshot[key] = [part.item() if isinstance(part, np.generic) else part for part in item]
    return snapshot


def controller_state(env):
    if not getattr(env, "robots", None):
        return {}
    controller = getattr(env.robots[0], "controller", None)
    if controller is None:
        return {}
    snapshot = _numeric_attributes(controller)
    for name in ("interpolator_pos", "interpolator_ori"):
        value = getattr(controller, name, None)
        if value is not None:
            snapshot[name] = _numeric_attributes(value)
    return snapshot


def physical_state(env):
    data = env.sim.data
    return {
        "qpos": data.qpos.copy(),
        "qvel": data.qvel.copy(),
        "act": data.act.copy() if data.act is not None else np.zeros(0),
        "ctrl": data.ctrl.copy(),
        "time": float(data.time),
        "controller": controller_state(env),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
    }


def state_hash(snapshot):
    digest = hashlib.sha256()
    for key in ("qpos", "qvel", "act", "ctrl"):
        value = np.asarray(snapshot[key])
        digest.update(key.encode() + b"\0" + value.tobytes() + b"\0")
    digest.update(repr(snapshot["time"]).encode() + b"\0")
    digest.update(canonical_dumps(snapshot["controller"], sort_keys=True).encode() + b"\0")
    digest.update(canonical_dumps(snapshot["python_rng"]).encode() + b"\0")
    digest.update(canonical_dumps(snapshot["numpy_rng"]).encode())
    return digest.hexdigest()


class QueryBroker:
    def __init__(self, env, budget, free_camera="v0"):
        self.env = env
        self.budget = int(budget)
        self.free_camera = free_camera
        self.ledger = []

    def free_observation(self):
        return self._render(self.free_camera, charged=False)

    def query(self, camera):
        if camera == self.free_camera or camera not in CAMERAS:
            raise PermissionError(f"camera is not a paid query: {camera}")
        if len([row for row in self.ledger if row["charged"]]) >= self.budget:
            raise PermissionError("observation budget exhausted")
        return self._render(camera, charged=True)

    def _render(self, camera, charged):
        before = state_hash(physical_state(self.env))
        start = time.perf_counter()
        image = self.env.capture_camera(camera)
        latency = time.perf_counter() - start
        after = state_hash(physical_state(self.env))
        if before != after:
            raise RuntimeError(f"camera query changed simulator state: {camera}")
        self.ledger.append(
            {"camera": camera, "charged": bool(charged), "latency_seconds": latency, "state_hash": before}
        )
        return image


def contact_event(env):
    data = env.sim.data
    deepest = 0.0
    hit = None
    for index in range(data.ncon):
        contact = data.contact[index]
        pair = {int(contact.geom1), int(contact.geom2)}
        if not (pair & env.robot_geom_ids):
            continue
        for side, obstacle_ids in env.obstacle_geom_ids.items():
            if pair & obstacle_ids:
                depth = max(0.0, -float(contact.dist))
                if hit is None or depth > deepest:
                    hit = side
                    deepest = depth
    return hit, deepest


def execute_route(env, route, max_steps=100, hold_steps=5):
    waypoints = env.route_waypoints(route)
    waypoint_index = 0
    collision = False
    first_contact = None
    deepest = 0.0
    contact_run = 0
    max_contact_run = 0
    reached_run = 0
    records = []
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
        side, depth = contact_event(env)
        if side is not None:
            collision = True
            first_contact = step if first_contact is None else first_contact
            contact_run += 1
            max_contact_run = max(max_contact_run, contact_run)
            deepest = max(deepest, depth)
        else:
            contact_run = 0
        target_distance = float(np.linalg.norm(env.eef_position - np.asarray(env.layout.target)))
        reached_run = reached_run + 1 if target_distance <= 0.02 else 0
        records.append(
            {
                "step": step,
                "eef": env.eef_position.tolist(),
                "waypoint_index": waypoint_index,
                "target_distance": target_distance,
                "obstacle_contact": side,
                "penetration": depth,
            }
        )
        if reached_run >= hold_steps:
            break
    reached = reached_run >= hold_steps
    return {
        "route": route,
        "steps": len(records),
        "reached": reached,
        "collision": collision,
        "collision_free_success": reached and not collision,
        "first_contact": first_contact,
        "deepest_penetration": deepest,
        "max_contact_run": max_contact_run,
        "records": records,
    }
