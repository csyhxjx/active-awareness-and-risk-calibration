"""Physical candidate preflight and deterministic admission for Phase 6A7."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path

import numpy as np

from guard.active_vision.belief_branch_scene import BranchingBeliefEnv
from guard.active_vision.belief_branching import ROUTES
from guard.active_vision.phase6a7 import (
    ALL_STATES,
    CHECKER_VERSION,
    CLEARANCE_THRESHOLD_M,
    PROTOCOL_ID,
    SEED,
    layout_from_spec,
    sha256_bytes,
    trajectory_sha256,
)
from guard.active_vision.runtime import physical_state, state_hash
from guard.json_io import canonical_dumps, write_json


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def fresh_env(state: str, spec: dict) -> BranchingBeliefEnv:
    random.seed(SEED)
    np.random.seed(SEED)
    env = BranchingBeliefEnv(layout=layout_from_spec(spec), hidden_state=state)
    env.reset()
    return env


def _contacts(env) -> tuple[list[dict], dict]:
    active_obstacle_ids = set().union(*env.obstacle_geom_ids.values())
    events = []
    classes = {"route_obstacle": False, "static_environment": False, "self_collision": False}
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
        elif pair & active_obstacle_ids:
            kind = "route_obstacle"
        else:
            kind = "static_environment"
        classes[kind] = True
        events.append({
            "class": kind,
            "geom1": env.sim.model.geom_id2name(first),
            "geom2": env.sim.model.geom_id2name(second),
            "penetration": -float(contact.dist),
        })
    return events, classes


def execute_audited_route(env, state: str, route: str, max_steps: int = 100, hold_steps: int = 5) -> dict:
    waypoints = env.route_waypoints(route)
    waypoint_index = 0
    reached_run = 0
    records = []
    collisions = {"route_obstacle": False, "static_environment": False, "self_collision": False}
    first_contacts = []
    minimum_clearance = None
    nearest_obstacle = None
    lanes = (env.layout.lane_y, 0.0, -env.layout.lane_y)
    obstacle_names = ("left", "center", "right")
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
        events, step_classes = _contacts(env)
        for key, value in step_classes.items():
            collisions[key] = collisions[key] or value
        if events and not first_contacts:
            first_contacts = events
        point = env.eef_position.copy()
        for index, (name, lane) in enumerate(zip(obstacle_names, lanes)):
            if state[index] != "1":
                continue
            center = np.asarray((env.layout.obstacle_x, lane, env.layout.target[2]))
            clearance = float(np.linalg.norm(point - center) - 0.085)
            if minimum_clearance is None or clearance < minimum_clearance:
                minimum_clearance = clearance
                nearest_obstacle = f"branch_obstacle_{name}_g0"
        target_distance = float(np.linalg.norm(point - np.asarray(env.layout.target)))
        reached_run = reached_run + 1 if target_distance <= 0.02 else 0
        records.append({
            "step": step,
            "eef": point.tolist(),
            "waypoint_index": waypoint_index,
            "target_distance": target_distance,
            "collision_classes": dict(step_classes),
        })
        if reached_run >= hold_steps:
            break
    reached = reached_run >= hold_steps
    any_collision = any(collisions.values())
    return {
        "route": route,
        "steps": len(records),
        "trajectory_length_m": float(sum(
            np.linalg.norm(np.asarray(second["eef"]) - np.asarray(first["eef"]))
            for first, second in zip(records[:-1], records[1:])
        )),
        "trajectory_sha256": trajectory_sha256(records),
        "reached": reached,
        "collision": any_collision,
        "collision_classes": collisions,
        "first_contacts": first_contacts,
        "collision_free_success": reached and not any_collision,
        "minimum_clearance_m": minimum_clearance,
        "nearest_active_obstacle": nearest_obstacle,
        "clearance_metric": "eef_center_to_obstacle_bounding_sphere_r0.085m",
        "records": records,
    }


def assess_candidate(spec: dict) -> dict:
    rows = []
    errors = []
    for state in ALL_STATES:
        routes = []
        for route_index, route in enumerate(ROUTES):
            env = fresh_env(state, spec)
            try:
                branch_hash = state_hash(physical_state(env))
                result = execute_audited_route(env, state, route)
            finally:
                env.close()
            clear = state[route_index] == "0"
            route_errors = []
            if clear:
                if not result["collision_free_success"]:
                    route_errors.append("clear_route_not_collision_free_complete")
                if result["collision_classes"]["static_environment"]:
                    route_errors.append("static_environment_collision")
                if result["collision_classes"]["self_collision"]:
                    route_errors.append("self_collision")
                if (
                    result["minimum_clearance_m"] is not None
                    and result["minimum_clearance_m"] < CLEARANCE_THRESHOLD_M
                ):
                    route_errors.append("clearance_below_threshold")
            elif not result["collision_classes"]["route_obstacle"]:
                route_errors.append("blocked_route_did_not_hit_obstacle")
            if route_errors:
                errors.extend(f"{state}/{route}:{error}" for error in route_errors)
            routes.append({
                key: value for key, value in result.items() if key != "records"
            } | {"expected_clear": clear, "branch_state_hash": branch_hash, "errors": route_errors})
        rows.append({"state": state, "routes": routes})
    return {
        "schema_version": 1,
        "candidate_index": spec["candidate_index"],
        "layout_id": spec["layout_id"],
        "spec": spec,
        "seed": SEED,
        "checker_version": CHECKER_VERSION,
        "preflight_head": git_head(),
        "accepted": not errors,
        "errors": errors,
        "states": rows,
    }


def assemble(candidate_plan: Path, results_dir: Path, output_dir: Path) -> None:
    plan_bytes = candidate_plan.read_bytes()
    plan = json.loads(plan_bytes)
    result_paths = sorted(results_dir.glob("candidate_*.json"))
    results = [json.loads(path.read_text()) for path in result_paths]
    by_index = {row["candidate_index"]: (row, path) for row, path in zip(results, result_paths)}
    ledger = []
    accepted = []
    rejection_count = 0
    for spec in plan["candidates"]:
        index = spec["candidate_index"]
        if index not in by_index:
            if len(accepted) < plan["requested_layouts"]:
                raise RuntimeError(f"missing preflight result before admission completed: {index}")
            break
        row, path = by_index[index]
        result_hash = sha256_bytes(path.read_bytes())
        ledger.append({
            "candidate_index": index,
            "layout_id": spec["layout_id"],
            "accepted": row["accepted"],
            "errors": row["errors"],
            "result_sha256": result_hash,
            "cumulative_rejections": rejection_count + (0 if row["accepted"] else 1),
        })
        if row["accepted"]:
            admitted = dict(spec)
            admitted.update({
                "states": list(ALL_STATES),
                "main_states": list(ALL_STATES[:6]),
                "control_states": list(ALL_STATES[6:]),
                "observation_table": plan["observation_table"],
                "roi": plan["roi"],
                "preflight_result_sha256": result_hash,
            })
            accepted.append(admitted)
            if len(accepted) == plan["requested_layouts"]:
                break
        else:
            rejection_count += 1
    if len(accepted) != plan["requested_layouts"]:
        raise RuntimeError(f"only {len(accepted)} candidates passed preflight")
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "preflight_ledger.json", {
        "schema_version": 1,
        "candidate_plan_sha256": sha256_bytes(plan_bytes),
        "selection_rule": plan["selection_rule"],
        "policy_result_filtering": False,
        "rejection_count": rejection_count,
        "entries": ledger,
    })
    write_json(output_dir / "manifest.json", {
        key: value for key, value in plan.items() if key not in {"candidates", "requested_layouts"}
    } | {
        "schema_version": 1,
        "layout_count": len(accepted),
        "scene_count": len(accepted) * len(ALL_STATES),
        "route_count": len(accepted) * len(ALL_STATES) * len(ROUTES),
        "preflight_head": git_head(),
        "preflight_ledger_sha256": sha256_file(output_dir / "preflight_ledger.json"),
        "layouts": accepted,
    })


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_plan", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--candidate-index", type=int)
    parser.add_argument("--assemble", action="store_true")
    args = parser.parse_args()
    if args.assemble:
        assemble(args.candidate_plan, args.output, args.output.parent / "admitted")
        return
    if args.candidate_index is None:
        parser.error("--candidate-index is required unless --assemble is used")
    if args.output.exists():
        raise FileExistsError(args.output)
    plan = json.loads(args.candidate_plan.read_text())
    spec = next(row for row in plan["candidates"] if row["candidate_index"] == args.candidate_index)
    write_json(args.output, assess_candidate(spec))


if __name__ == "__main__":
    main()

