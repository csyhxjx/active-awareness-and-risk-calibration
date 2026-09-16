"""Run the pre-registered Phase 4B targeted counterfactual smoke."""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np

from guard.collection import initial_state_hash, load_selected_states
from guard.constraints.libero_constraints import LiberoConstraintMonitor
from guard.counterfactual.run_counterfactual import (
    CONSTRAINTS,
    PARITY_ATOL,
    PILOT_BRANCH_STEPS,
    compare_constraint_record,
    git_revision,
    load_parent_episode,
    make_env,
    replay_predecessors,
    restore_rng,
    rng_sha256,
    rng_snapshot,
    save_image_pair,
    sha256_file,
    state_sha256,
)
from guard.json_io import append_jsonl, write_json


V2_ARMS = ("A_grip", "C04", "C06", "C08", "C10", "A_drop", "D_force", "A_work", "E_push")
ARM_BRANCH = {
    "A_grip": "grip",
    "C04": "grip",
    "C06": "grip",
    "C08": "grip",
    "C10": "grip",
    "A_drop": "drop",
    "D_force": "drop",
    "A_work": "workspace",
    "E_push": "workspace",
}
PRESSURE_DOSE = {"C04": 0.4, "C06": 0.6, "C08": 0.8, "C10": 1.0}
NOMINAL_ARMS = {"A_grip", "A_drop", "A_work"}
SMOKE_STATES = ("task_07_init_023", "task_04_init_035", "task_08_init_003")


def _target_body(env):
    target_name = env.obj_of_interest[0]
    if target_name != "akita_black_bowl_1":
        raise ValueError(f"unexpected payload: {target_name}")
    return target_name, env.env.obj_body_id[target_name]


def select_branches(trace, state_id, horizon, max_action_step):
    last_branch = max_action_step - horizon + 1
    grip_step = PILOT_BRANCH_STEPS[state_id]
    if grip_step > last_branch:
        raise ValueError(f"grip branch lacks full horizon: {state_id}")

    close_steps = [row["step_index"] for row in trace if row.get("gripper", -1) > 0]
    if not close_steps:
        raise ValueError(f"no closed-gripper phase: {state_id}")
    first_close = close_steps[0]
    z0 = next(row["target_z"] for row in trace if row["step_index"] == 9)
    carry = [
        row
        for row in trace
        if first_close <= row["step_index"] <= last_branch
        and row.get("gripper", -1) > 0
        and row["target_z"] - z0 >= 0.02
    ]
    if not carry:
        raise ValueError(f"no eligible carry branch: {state_id}")
    drop_row = min(carry, key=lambda row: (-row["target_z"], row["step_index"]))

    lateral = []
    for row in trace:
        if 10 <= row["step_index"] <= last_branch:
            for order, term in enumerate(row["lateral_terms"]):
                lateral.append((term["margin"], row["step_index"], order, term))
    if not lateral:
        raise ValueError(f"no eligible workspace branch: {state_id}")
    workspace_margin, workspace_step, _, workspace_term = min(lateral)
    return {
        "grip": {"step": grip_step, "source": "phase4a_gripper_env_policy_argmin"},
        "drop": {
            "step": drop_row["step_index"],
            "source": "max_payload_elevation_during_closed_gripper_carry",
            "target_name": drop_row["target_name"],
            "target_z0": z0,
            "target_z": drop_row["target_z"],
            "target_elevation": drop_row["target_z"] - z0,
            "first_close_step": first_close,
        },
        "workspace": {
            "step": workspace_step,
            "source": "minimum_lateral_workspace_margin",
            "axis": workspace_term["axis"],
            "action_index": workspace_term["action_index"],
            "direction": workspace_term["direction"],
            "margin": workspace_margin,
        },
    }


def discover_branches(env, initial_state, parent, state_id, horizon):
    env.reset()
    env.set_init_state(initial_state)
    monitor = LiberoConstraintMonitor(env)
    monitor.episode_reset()
    target_name, target_body = _target_body(env)
    trace = []
    max_error = 0.0
    step_count = parent["meta"]["constraint_step_count"]
    for step_index in range(step_count):
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
        x, y, _ = record["eef"]
        x0, x1, y0, y1 = monitor.table_xy
        xy_margin = monitor.cfg["xy_margin"]
        trace.append(
            {
                "step_index": step_index,
                "target_name": target_name,
                "target_z": float(env.env.sim.data.body_xpos[target_body][2]),
                "gripper": float(action[6]) if step_index >= 10 else -1.0,
                "lateral_terms": [
                    {"axis": "x", "action_index": 0, "direction": -1, "margin": float(x - (x0 - xy_margin))},
                    {"axis": "x", "action_index": 0, "direction": 1, "margin": float((x1 + xy_margin) - x)},
                    {"axis": "y", "action_index": 1, "direction": -1, "margin": float(y - (y0 - xy_margin))},
                    {"axis": "y", "action_index": 1, "direction": 1, "margin": float((y1 + xy_margin) - y)},
                ],
            }
        )
    branches = select_branches(trace, state_id, horizon, max(parent["actions"]))
    return branches, trace, max_error


def reconstruct_to_branch(env, initial_state, parent, branch_step):
    env.reset()
    env.set_init_state(initial_state)
    monitor = LiberoConstraintMonitor(env)
    monitor.episode_reset()
    max_error = 0.0
    for step_index in range(branch_step):
        action = (
            np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float64)
            if step_index < 10
            else parent["actions"][step_index]
        )
        env.step(action.tolist())
        actual = monitor.check(step_index)
        max_error = max(
            max_error,
            compare_constraint_record(actual, parent["constraints"][step_index], context=f"replay step={step_index}"),
        )
    return np.asarray(env.get_sim_state()).copy(), rng_snapshot(), max_error


def v2_action(arm_id, baseline, workspace_branch):
    action = np.asarray(baseline, dtype=np.float64).copy()
    if arm_id in NOMINAL_ARMS:
        return action
    if arm_id in PRESSURE_DOSE:
        action[2] -= PRESSURE_DOSE[arm_id]
    elif arm_id == "D_force":
        action[6] = -1.0
    elif arm_id == "E_push":
        action[workspace_branch["action_index"]] = float(workspace_branch["direction"])
    else:
        raise ValueError(f"unknown v2 arm: {arm_id}")
    return np.clip(action, -1.0, 1.0)


def run_state(state, *, task_suite, collection_root, output_root, protocol_path, horizon, predecessors, repo_root):
    if state["split"] != "train" or state["state_id"] not in SMOKE_STATES:
        raise ValueError(f"v2 smoke permits only registered train states: {state['state_id']}")
    parent = load_parent_episode(collection_root, state["state_id"])
    initial_states = task_suite.get_task_init_states(state["task_id"])
    initial_state = initial_states[state["init_index"]]
    if initial_state_hash(initial_state) != state["state_hash"]:
        raise ValueError(f"manifest state hash mismatch: {state['state_id']}")

    state_dir = Path(output_root) / state["state_id"]
    if state_dir.exists():
        raise FileExistsError(f"refusing to overwrite v2 state: {state_dir}")
    state_dir.mkdir(parents=True)
    task = task_suite.get_task(state["task_id"])
    try:
        random.seed(7)
        np.random.seed(7)
        discovery_env = make_env(task)
        try:
            replayed = replay_predecessors(discovery_env, initial_states, collection_root, predecessors)
            branches, trace, discovery_error = discover_branches(
                discovery_env, initial_state, parent, state["state_id"], horizon
            )
        finally:
            discovery_env.close()
        write_json(
            state_dir / "branch_discovery.json",
            {
                "schema_version": 1,
                **state,
                "horizon": horizon,
                "predecessor_states": replayed,
                "discovery_parity_max_abs_error": discovery_error,
                "branches": branches,
                "trace": trace,
            },
        )

        branch_reference = {}
        for arm_id in V2_ARMS:
            branch_name = ARM_BRANCH[arm_id]
            branch_step = branches[branch_name]["step"]
            random.seed(7)
            np.random.seed(7)
            env = make_env(task)
            try:
                replayed = replay_predecessors(env, initial_states, collection_root, predecessors)
                branch_state, snapshot, replay_error = reconstruct_to_branch(env, initial_state, parent, branch_step)
                branch_hash = state_sha256(branch_state)
                snapshot_hash = rng_sha256(snapshot)
                if branch_name not in branch_reference:
                    branch_reference[branch_name] = (branch_hash, snapshot_hash)
                    np.save(state_dir / f"branch_{branch_name}_state.npy", branch_state, allow_pickle=False)
                    write_json(state_dir / f"branch_{branch_name}_rng.json", snapshot)
                elif branch_reference[branch_name] != (branch_hash, snapshot_hash):
                    raise ValueError(f"branch hash mismatch: state={state['state_id']} branch={branch_name} arm={arm_id}")
                restore_rng(snapshot)
                monitor = LiberoConstraintMonitor(env)
                monitor.episode_reset()
                nominal_error = 0.0
                arm_dir = state_dir / f"arm_{arm_id}"
                (arm_dir / "images" / "full").mkdir(parents=True)
                (arm_dir / "images" / "wrist").mkdir(parents=True)
                for arm_step, step_index in enumerate(range(branch_step, branch_step + horizon)):
                    baseline = parent["actions"][step_index]
                    action = v2_action(arm_id, baseline, branches["workspace"])
                    obs, _, done, _ = env.step(action.tolist())
                    record = monitor.check(step_index)
                    record["step_index"] = step_index
                    record["arm_step"] = arm_step
                    append_jsonl(arm_dir / "constraints.jsonl", record)
                    append_jsonl(
                        arm_dir / "actions.jsonl",
                        {
                            "state_id": state["state_id"],
                            "parent_state": state["state_id"],
                            "arm_id": arm_id,
                            "branch_type": branch_name,
                            "step_index": step_index,
                            "arm_step": arm_step,
                            "baseline_env_action": baseline.tolist(),
                            "env_action": action.tolist(),
                            "done": bool(done),
                        },
                    )
                    save_image_pair(arm_dir, step_index, obs)
                    if arm_id in NOMINAL_ARMS:
                        nominal_error = max(
                            nominal_error,
                            compare_constraint_record(
                                record,
                                parent["constraints"][step_index],
                                context=f"arm={arm_id} state={state['state_id']} step={step_index}",
                            ),
                        )
                head, dirty = git_revision(repo_root)
                write_json(
                    arm_dir / "meta.json",
                    {
                        "schema_version": 2,
                        **state,
                        "parent_state": state["state_id"],
                        "arm_id": arm_id,
                        "branch_type": branch_name,
                        "branch": branches[branch_name],
                        "horizon": horizon,
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
                        "replay_max_abs_error": replay_error,
                        "nominal_parity_passed": arm_id in NOMINAL_ARMS,
                        "nominal_max_abs_error": nominal_error if arm_id in NOMINAL_ARMS else None,
                    },
                )
            finally:
                env.close()
    except Exception as exc:
        write_json(
            state_dir / "FAILED.json",
            {"state_id": state["state_id"], "error_type": type(exc).__name__, "error": str(exc)},
        )
        raise


def main():
    from libero.libero import benchmark

    repo_root = Path(__file__).resolve().parents[2]
    project_root = repo_root.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", type=Path, default=project_root / "guard_workspace/collections/phase3_full70_fe76a6c")
    parser.add_argument("--manifest", type=Path, default=repo_root / "data/pilot_v0/manifest.json")
    parser.add_argument("--protocol", type=Path, default=repo_root / "pilot_protocol_v2.md")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state-list", required=True)
    parser.add_argument("--horizon", type=int, default=30)
    args = parser.parse_args()
    if args.horizon != 30:
        raise ValueError("v2 horizon is exactly 30")
    _, dirty = git_revision(repo_root)
    if dirty:
        raise RuntimeError("v2 execution requires a clean Guard worktree")
    _, states = load_selected_states(args.manifest, args.state_list)
    if tuple(state["state_id"] for state in states) != tuple(
        state_id for state_id in SMOKE_STATES if state_id in {state["state_id"] for state in states}
    ):
        raise ValueError("v2 smoke state order or membership differs from pre-registration")
    _, train_states = load_selected_states(args.manifest, "train")
    args.output_root.mkdir(parents=True, exist_ok=True)
    task_suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    for state in states:
        same_task = [candidate for candidate in train_states if candidate["task_id"] == state["task_id"]]
        position = [candidate["state_id"] for candidate in same_task].index(state["state_id"])
        run_state(
            state,
            task_suite=task_suite,
            collection_root=args.collection_root,
            output_root=args.output_root,
            protocol_path=args.protocol,
            horizon=args.horizon,
            predecessors=same_task[:position],
            repo_root=repo_root,
        )
        print(f"PASS {state['state_id']} arms={','.join(V2_ARMS)}", flush=True)


if __name__ == "__main__":
    main()

