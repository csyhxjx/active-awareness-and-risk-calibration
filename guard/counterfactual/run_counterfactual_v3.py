"""Run the pre-registered Phase 4C edge-drop mechanism smoke."""

import argparse
import os
import random
from pathlib import Path

import numpy as np

from guard.collection import initial_state_hash, load_selected_states
from guard.constraints.libero_constraints import LiberoConstraintMonitor
from guard.counterfactual.run_counterfactual import (
    PARITY_ATOL,
    compare_constraint_record,
    git_revision,
    load_parent_episode,
    make_env,
    reconstruct_branch,
    replay_predecessors,
    restore_rng,
    rng_sha256,
    rng_snapshot,
    save_image_pair,
    sha256_file,
    state_sha256,
)
from guard.json_io import append_jsonl, write_json


SMOKE_STATES = ("task_07_init_023", "task_04_init_035", "task_07_init_021")
V3_ARMS = ("A_edge", "F_edge")
HORIZON = 60
TARGET_NAME = "akita_black_bowl_1"


def target_body(env):
    name = env.obj_of_interest[0]
    if name != TARGET_NAME:
        raise ValueError(f"unexpected payload: {name}")
    return name, env.env.obj_body_id[name]


def edge_terms(x, y, bounds):
    x0, x1, y0, y1 = bounds
    return (
        {"axis": "x", "action_index": 0, "direction": -1, "bound": x0, "distance": x - x0},
        {"axis": "x", "action_index": 0, "direction": 1, "bound": x1, "distance": x1 - x},
        {"axis": "y", "action_index": 1, "direction": -1, "bound": y0, "distance": y - y0},
        {"axis": "y", "action_index": 1, "direction": 1, "bound": y1, "distance": y1 - y},
    )


def select_edge_branch(trace, horizon, max_action_step):
    if horizon != HORIZON:
        raise ValueError(f"v3 horizon must be {HORIZON}")
    close_steps = []
    previous = -1.0
    for row in trace:
        command = row["gripper"]
        if command > 0 and previous <= 0:
            close_steps.append(row["step_index"])
        previous = command
    if not close_steps:
        raise ValueError("no open-to-closed gripper transition")
    first_close = close_steps[0]
    z0 = next(row["target_z"] for row in trace if row["step_index"] == 9)
    candidates = [
        row
        for row in trace
        if row["step_index"] >= first_close
        and row["gripper"] > 0
        and row["target_z"] - z0 >= 0.02
        and row["step_index"] + horizon <= max_action_step
    ]
    if not candidates:
        raise ValueError("no elevated carry observation with 60 following actions")
    carry = min(candidates, key=lambda row: (-row["target_z"], row["step_index"]))
    terms = edge_terms(carry["target_x"], carry["target_y"], carry["table_xy"])
    if any(not np.isfinite(term["distance"]) or term["distance"] < 0 for term in terms):
        raise ValueError("payload center is outside table projection at carry observation")
    edge = min(enumerate(terms), key=lambda item: (item[1]["distance"], item[0]))[1]
    return {
        "observation_step": carry["step_index"],
        "step": carry["step_index"] + 1,
        "source": "post_action_max_payload_elevation_during_closed_gripper_carry",
        "target_name": carry["target_name"],
        "target_xyz": [carry["target_x"], carry["target_y"], carry["target_z"]],
        "target_z0": z0,
        "target_elevation": carry["target_z"] - z0,
        "first_close_step": first_close,
        "table_xy": list(carry["table_xy"]),
        "table_z": carry["table_z"],
        "drop_line_z": carry["drop_line_z"],
        "edge": edge,
    }


def discover_edge_branch(env, initial_state, parent, horizon):
    env.reset()
    env.set_init_state(initial_state)
    monitor = LiberoConstraintMonitor(env)
    monitor.episode_reset()
    name, body = target_body(env)
    trace = []
    max_error = 0.0
    previous_gripper = -1.0
    for step_index in range(parent["meta"]["constraint_step_count"]):
        action = (
            np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float64)
            if step_index < 10
            else parent["actions"][step_index]
        )
        env.step(action.tolist())
        record = monitor.check(step_index)
        max_error = max(
            max_error,
            compare_constraint_record(record, parent["constraints"][step_index], context=f"discovery step={step_index}"),
        )
        xyz = env.env.sim.data.body_xpos[body].copy()
        previous_gripper = float(action[6]) if step_index >= 10 else -1.0
        trace.append(
            {
                "step_index": step_index,
                "target_name": name,
                "target_x": float(xyz[0]),
                "target_y": float(xyz[1]),
                "target_z": float(xyz[2]),
                "gripper": previous_gripper,
                "table_xy": list(monitor.table_xy),
                "table_z": float(monitor.table_z),
                "drop_line_z": float(monitor.table_z - monitor.cfg["obj_drop_below"]),
            }
        )
    branch = select_edge_branch(trace, horizon, max(parent["actions"]))
    return branch, trace, max_error


def crossed_edge(coordinate, edge):
    return coordinate < edge["bound"] if edge["direction"] < 0 else coordinate > edge["bound"]


def edge_action(arm_id, baseline, edge, release):
    action = np.asarray(baseline, dtype=np.float64).copy()
    if arm_id == "A_edge":
        return action
    if arm_id != "F_edge":
        raise ValueError(f"unknown v3 arm: {arm_id}")
    action[edge["action_index"]] = float(edge["direction"])
    action[6] = -1.0 if release else 1.0
    return np.clip(action, -1.0, 1.0)


def validate_f_append(output_root, selected):
    root = Path(output_root)
    actual_dirs = {path.name for path in root.iterdir() if path.is_dir()} if root.exists() else set()
    extras = actual_dirs - set(SMOKE_STATES)
    if extras:
        raise ValueError(f"unregistered state directories in F root: {sorted(extras)}")
    completed = tuple(state for state in SMOKE_STATES if state in actual_dirs)
    if completed != SMOKE_STATES[: len(completed)]:
        raise ValueError(f"F root is not a registered prefix: {completed}")
    if len(completed) == len(SMOKE_STATES):
        raise ValueError("F root already contains all registered states")
    expected = (SMOKE_STATES[len(completed)],)
    if tuple(selected) != expected:
        raise ValueError(f"F_edge must append exactly the next registered state: expected={expected}")


def run_state(state, *, arm_id, task_suite, collection_root, output_root, protocol_path, predecessors, repo_root):
    if state["split"] != "train" or state["state_id"] not in SMOKE_STATES:
        raise ValueError(f"v3 smoke permits only registered train states: {state['state_id']}")
    parent = load_parent_episode(collection_root, state["state_id"])
    for key in ("state_id", "task_id", "init_index", "state_hash", "split"):
        if parent["meta"][key] != state[key]:
            raise ValueError(f"parent metadata mismatch: state={state['state_id']} key={key}")
    initial_states = task_suite.get_task_init_states(state["task_id"])
    initial_state = initial_states[state["init_index"]]
    if initial_state_hash(initial_state) != state["state_hash"]:
        raise ValueError(f"manifest state hash mismatch: {state['state_id']}")

    state_dir = Path(output_root) / state["state_id"]
    if state_dir.exists():
        raise FileExistsError(f"refusing to overwrite v3 state: {state_dir}")
    state_dir.mkdir(parents=True)
    task = task_suite.get_task(state["task_id"])
    try:
        random.seed(7)
        np.random.seed(7)
        discovery_env = make_env(task)
        try:
            replayed = replay_predecessors(discovery_env, initial_states, collection_root, predecessors)
            branch, trace, discovery_error = discover_edge_branch(discovery_env, initial_state, parent, HORIZON)
        finally:
            discovery_env.close()
        write_json(
            state_dir / "branch_discovery.json",
            {
                "schema_version": 1,
                **state,
                "horizon": HORIZON,
                "predecessor_states": replayed,
                "discovery_parity_max_abs_error": discovery_error,
                "branch": branch,
                "trace": trace,
            },
        )

        random.seed(7)
        np.random.seed(7)
        env = make_env(task)
        try:
            low, high = (np.asarray(value, dtype=np.float64) for value in env.env.action_spec)
            if not np.array_equal(low, -np.ones(7)) or not np.array_equal(high, np.ones(7)):
                raise ValueError(f"unexpected action bounds: low={low.tolist()} high={high.tolist()}")
            replayed = replay_predecessors(env, initial_states, collection_root, predecessors)
            _, branch_state, snapshot, replay_error = reconstruct_branch(env, initial_state, parent, branch["step"])
            branch_hash = state_sha256(branch_state)
            snapshot_hash = rng_sha256(snapshot)
            _, branch_body = target_body(env)
            branch_target_xyz = env.env.sim.data.body_xpos[branch_body].copy()
            branch_target_error = float(np.max(np.abs(branch_target_xyz - np.asarray(branch["target_xyz"]))))
            if branch_target_error > PARITY_ATOL:
                raise ValueError(
                    f"branch target mismatch: state={state['state_id']} error={branch_target_error}"
                )
            np.save(state_dir / "branch_state.npy", branch_state, allow_pickle=False)
            write_json(state_dir / "rng_snapshot.json", snapshot)
            restore_rng(snapshot)

            monitor = LiberoConstraintMonitor(env)
            monitor.episode_reset()
            _, body = target_body(env)
            arm_dir = state_dir / f"arm_{arm_id}"
            (arm_dir / "images" / "full").mkdir(parents=True)
            (arm_dir / "images" / "wrist").mkdir(parents=True)
            crossed = False
            crossing_step = None
            release_start_step = None
            nominal_error = 0.0
            target_min_drop_margin = float("inf")
            for arm_step, step_index in enumerate(range(branch["step"], branch["step"] + HORIZON)):
                release = crossed
                stage = "nominal" if arm_id == "A_edge" else ("push_release" if release else "push_hold")
                baseline = parent["actions"][step_index]
                action = edge_action(arm_id, baseline, branch["edge"], release)
                obs, _, done, _ = env.step(action.tolist())
                record = monitor.check(step_index)
                record["step_index"] = step_index
                record["arm_step"] = arm_step
                append_jsonl(arm_dir / "constraints.jsonl", record)
                xyz = env.env.sim.data.body_xpos[body].copy()
                target_drop_margin = float(xyz[2] - branch["drop_line_z"])
                target_min_drop_margin = min(target_min_drop_margin, target_drop_margin)
                coordinate = float(xyz[branch["edge"]["action_index"]])
                crossed_after = crossed_edge(coordinate, branch["edge"])
                if arm_id == "F_edge" and not crossed and crossed_after:
                    crossed = True
                    crossing_step = step_index
                    if arm_step + 1 < HORIZON:
                        release_start_step = step_index + 1
                append_jsonl(
                    arm_dir / "actions.jsonl",
                    {
                        "state_id": state["state_id"],
                        "parent_state": state["state_id"],
                        "arm_id": arm_id,
                        "step_index": step_index,
                        "arm_step": arm_step,
                        "stage": stage,
                        "baseline_env_action": baseline.tolist(),
                        "env_action": action.tolist(),
                        "target_xyz": [float(value) for value in xyz],
                        "target_drop_margin": target_drop_margin,
                        "edge_crossed_after_action": bool(crossed_after),
                        "done": bool(done),
                    },
                )
                save_image_pair(arm_dir, step_index, obs)
                if arm_id == "A_edge":
                    nominal_error = max(
                        nominal_error,
                        compare_constraint_record(
                            record,
                            parent["constraints"][step_index],
                            context=f"arm=A_edge state={state['state_id']} step={step_index}",
                        ),
                    )

            head, dirty = git_revision(repo_root)
            write_json(
                arm_dir / "meta.json",
                {
                    "schema_version": 3,
                    **state,
                    "parent_state": state["state_id"],
                    "arm_id": arm_id,
                    "branch": branch,
                    "horizon": HORIZON,
                    "edge_crossed": bool(crossed) if arm_id == "F_edge" else None,
                    "crossing_step": crossing_step,
                    "release_triggered": release_start_step is not None if arm_id == "F_edge" else None,
                    "release_start_step": release_start_step,
                    "target_min_drop_margin": target_min_drop_margin,
                    "branch_target_xyz": [float(value) for value in branch_target_xyz],
                    "branch_target_xyz_max_abs_error": branch_target_error,
                    "branch_state_sha256": branch_hash,
                    "rng_snapshot_sha256": snapshot_hash,
                    "protocol_sha256": sha256_file(protocol_path),
                    "guard_head": head,
                    "guard_dirty": dirty,
                    "parent_collection": Path(collection_root).name,
                    "parent_collection_guard_head": parent["meta"]["guard_head"],
                    "parent_manifest_sha256": parent["meta"]["manifest_sha256"],
                    "parent_actions_sha256": parent["actions_sha256"],
                    "parent_constraints_sha256": parent["constraints_sha256"],
                    "predecessor_states": replayed,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "parity_atol": PARITY_ATOL,
                    "discovery_parity_max_abs_error": discovery_error,
                    "replay_max_abs_error": replay_error,
                    "nominal_parity_passed": arm_id == "A_edge",
                    "nominal_max_abs_error": nominal_error if arm_id == "A_edge" else None,
                },
            )
        finally:
            env.close()
    except Exception as exc:
        write_json(
            state_dir / "FAILED.json",
            {"state_id": state["state_id"], "arm_id": arm_id, "error_type": type(exc).__name__, "error": str(exc)},
        )
        raise


def main():
    from libero.libero import benchmark

    repo_root = Path(__file__).resolve().parents[2]
    project_root = repo_root.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", type=Path, default=project_root / "guard_workspace/collections/phase3_full70_fe76a6c")
    parser.add_argument("--manifest", type=Path, default=repo_root / "data/pilot_v0/manifest.json")
    parser.add_argument("--protocol", type=Path, default=repo_root / "pilot_protocol_v3.md")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state-list", required=True)
    parser.add_argument("--arm", choices=V3_ARMS, required=True)
    args = parser.parse_args()
    _, dirty = git_revision(repo_root)
    if dirty:
        raise RuntimeError("v3 execution requires a clean Guard worktree")
    _, states = load_selected_states(args.manifest, args.state_list)
    selected = tuple(state["state_id"] for state in states)
    if args.arm == "A_edge" and selected != SMOKE_STATES:
        raise ValueError("A_edge gate requires all registered smoke states in order")
    if args.arm == "F_edge":
        validate_f_append(args.output_root, selected)
    _, train_states = load_selected_states(args.manifest, "train")
    args.output_root.mkdir(parents=True, exist_ok=True)
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    for state in states:
        same_task = [candidate for candidate in train_states if candidate["task_id"] == state["task_id"]]
        position = [candidate["state_id"] for candidate in same_task].index(state["state_id"])
        run_state(
            state,
            arm_id=args.arm,
            task_suite=suite,
            collection_root=args.collection_root,
            output_root=args.output_root,
            protocol_path=args.protocol,
            predecessors=same_task[:position],
            repo_root=repo_root,
        )
        print(f"PASS {state['state_id']} arm={args.arm}", flush=True)


if __name__ == "__main__":
    main()
