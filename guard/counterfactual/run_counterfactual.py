"""Replay archived LIBERO actions and run pre-registered counterfactual arms."""

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from guard.collection import initial_state_hash, load_selected_states
from guard.constraints.libero_constraints import LiberoConstraintMonitor
from guard.json_io import append_jsonl, canonical_dumps, write_json


ARMS = ("A", "B", "C", "D", "E")
CONSTRAINTS = ("workspace", "gripper_env", "self_collision", "object_drop", "non_finite")
PILOT_BRANCH_STEPS = {
    "task_07_init_023": 48,
    "task_07_init_021": 45,
    "task_04_init_035": 57,
    "task_08_init_003": 60,
    "task_05_init_034": 10,
    "task_07_init_043": 41,
    "task_04_init_032": 61,
    "task_01_init_042": 10,
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_sha256(state):
    array = np.asarray(state)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii") + b"\0")
    digest.update(canonical_dumps(list(array.shape)).encode("ascii") + b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def rng_snapshot():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }


def rng_sha256(snapshot):
    return hashlib.sha256(canonical_dumps(snapshot, sort_keys=True).encode("utf-8")).hexdigest()


def restore_rng(snapshot):
    python_state = snapshot["python"]
    numpy_state = snapshot["numpy"]
    random.setstate(_as_tuple(python_state))
    np.random.set_state(
        (
            numpy_state[0],
            np.asarray(numpy_state[1], dtype=np.uint32),
            int(numpy_state[2]),
            int(numpy_state[3]),
            float(numpy_state[4]),
        )
    )


def _as_tuple(value):
    if isinstance(value, list):
        return tuple(_as_tuple(item) for item in value)
    return value


def load_jsonl(path):
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
    return records


def load_parent_episode(collection_root, state_id):
    episode_dir = Path(collection_root) / state_id
    meta_path = episode_dir / "meta.json"
    actions_path = episode_dir / "actions.jsonl"
    constraints_path = episode_dir / "constraints.jsonl"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    action_rows = load_jsonl(actions_path)
    constraint_rows = load_jsonl(constraints_path)
    actions = {row["step_index"]: np.asarray(row["env_action"], dtype=np.float64) for row in action_rows if row["type"] == "step"}
    constraints = {row["step_index"]: row for row in constraint_rows}
    if len(actions) != sum(row["type"] == "step" for row in action_rows):
        raise ValueError(f"duplicate parent action steps: {state_id}")
    if len(constraints) != len(constraint_rows):
        raise ValueError(f"duplicate parent constraint steps: {state_id}")
    return {
        "dir": episode_dir,
        "meta": meta,
        "actions": actions,
        "constraints": constraints,
        "actions_sha256": sha256_file(actions_path),
        "constraints_sha256": sha256_file(constraints_path),
    }


def perturb_action(arm_id, actions, step_index):
    action = np.asarray(actions[step_index], dtype=np.float64).copy()
    if arm_id == "A":
        return action
    if arm_id == "B":
        action[2] -= 0.50
    elif arm_id == "C":
        action[2] -= 1.00
    elif arm_id == "D":
        future = [actions[future_step][6] for future_step in (step_index + 1, step_index + 2) if future_step in actions]
        if any(command < 0 for command in future):
            action[6] = -1.0
    elif arm_id == "E":
        action[:2] *= 1.3
    else:
        raise ValueError(f"unknown arm: {arm_id}")
    return np.clip(action, -1.0, 1.0)


def compare_constraint_record(actual, expected, *, context):
    for name in CONSTRAINTS:
        if bool(actual[name]["violated"]) != bool(expected[name]["violated"]):
            raise ValueError(f"parity violation flag mismatch: {context} constraint={name}")
        actual_margin = float(actual[name]["margin"])
        expected_margin = float(expected[name]["margin"])
        if not math.isfinite(actual_margin) or not math.isfinite(expected_margin):
            raise ValueError(f"non-finite parity margin: {context} constraint={name}")
        if actual_margin != expected_margin:
            raise ValueError(
                f"parity margin mismatch: {context} constraint={name} "
                f"actual={actual_margin!r} expected={expected_margin!r}"
            )


def make_env(task, resolution=224):
    bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(0)
    return env


def reconstruct_branch(env, initial_state, parent, branch_step):
    random.seed(7)
    np.random.seed(7)
    env.seed(0)
    env.reset()
    obs = env.set_init_state(initial_state)
    monitor = LiberoConstraintMonitor(env)
    monitor.episode_reset()
    for step_index in range(branch_step):
        action = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float64) if step_index < 10 else parent["actions"][step_index]
        obs, _, _, _ = env.step(action.tolist())
        actual = monitor.check(step_index)
        compare_constraint_record(actual, parent["constraints"][step_index], context=f"replay step={step_index}")
    state = np.asarray(env.get_sim_state()).copy()
    snapshot = rng_snapshot()
    return obs, state, snapshot


def save_image_pair(arm_dir, step_index, obs):
    full = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1], dtype=np.uint8)
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1], dtype=np.uint8)
    if full.shape != (224, 224, 3) or wrist.shape != (224, 224, 3):
        raise ValueError(f"unexpected image shapes: full={full.shape} wrist={wrist.shape}")
    name = f"step_{step_index:05d}.png"
    Image.fromarray(full).save(arm_dir / "images" / "full" / name)
    Image.fromarray(wrist).save(arm_dir / "images" / "wrist" / name)


def git_revision(repo_root):
    head = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    return head, dirty


def run_state(state, *, task_suite, collection_root, output_root, protocol_path, horizon, arms, repo_root):
    if state["split"] != "train":
        raise ValueError(f"Phase 4A forbids non-train state: {state['state_id']} split={state['split']}")
    if state["state_id"] not in PILOT_BRANCH_STEPS:
        raise ValueError(f"state is not pre-registered: {state['state_id']}")
    if horizon != 30:
        raise ValueError("pre-registered horizon is exactly 30")

    parent = load_parent_episode(collection_root, state["state_id"])
    for key in ("state_id", "task_id", "init_index", "state_hash", "split"):
        if parent["meta"][key] != state[key]:
            raise ValueError(f"parent metadata mismatch: state={state['state_id']} key={key}")
    branch_step = PILOT_BRANCH_STEPS[state["state_id"]]
    required_steps = range(branch_step, branch_step + horizon)
    missing_actions = [step for step in required_steps if step not in parent["actions"]]
    missing_constraints = [step for step in required_steps if step not in parent["constraints"]]
    if missing_actions or missing_constraints:
        raise ValueError(f"parent horizon missing: actions={missing_actions} constraints={missing_constraints}")

    initial_states = task_suite.get_task_init_states(state["task_id"])
    initial_state = initial_states[state["init_index"]]
    if initial_state_hash(initial_state) != state["state_hash"]:
        raise ValueError(f"manifest state hash mismatch: {state['state_id']}")

    state_dir = Path(output_root) / state["state_id"]
    if state_dir.exists():
        raise FileExistsError(f"refusing to overwrite counterfactual state: {state_dir}")
    state_dir.mkdir(parents=True)
    task = task_suite.get_task(state["task_id"])
    env = make_env(task)
    try:
        low, high = (np.asarray(value, dtype=np.float64) for value in env.env.action_spec)
        if not np.array_equal(low, -np.ones(7)) or not np.array_equal(high, np.ones(7)):
            raise ValueError(f"unexpected action bounds: low={low.tolist()} high={high.tolist()}")

        reference_state = None
        reference_state_hash = None
        reference_rng = None
        reference_rng_hash = None
        for arm_id in arms:
            _, branch_state, snapshot = reconstruct_branch(env, initial_state, parent, branch_step)
            branch_hash = state_sha256(branch_state)
            snapshot_hash = rng_sha256(snapshot)
            if reference_state is None:
                reference_state = branch_state.copy()
                reference_state_hash = branch_hash
                reference_rng = snapshot
                reference_rng_hash = snapshot_hash
                np.save(state_dir / "branch_state.npy", reference_state, allow_pickle=False)
                write_json(state_dir / "rng_snapshot.json", reference_rng)
            elif branch_hash != reference_state_hash or snapshot_hash != reference_rng_hash:
                raise ValueError(
                    f"branch reconstruction mismatch: state={state['state_id']} arm={arm_id} "
                    f"state_hash={branch_hash} rng_hash={snapshot_hash}"
                )

            # Replay reconstruction restores controller internals. Reapply the exact
            # frozen simulator/RNG snapshot immediately before branching.
            env.set_state(reference_state)
            env.env.sim.forward()
            restore_rng(reference_rng)
            monitor = LiberoConstraintMonitor(env)
            monitor.episode_reset()

            arm_dir = state_dir / f"arm_{arm_id}"
            (arm_dir / "images" / "full").mkdir(parents=True)
            (arm_dir / "images" / "wrist").mkdir(parents=True)
            for arm_step, step_index in enumerate(required_steps):
                baseline = parent["actions"][step_index]
                action = perturb_action(arm_id, parent["actions"], step_index)
                obs, _, done, _ = env.step(action.tolist())
                record = monitor.check(step_index)
                record["step_index"] = int(step_index)
                record["arm_step"] = int(arm_step)
                append_jsonl(arm_dir / "constraints.jsonl", record)
                append_jsonl(
                    arm_dir / "actions.jsonl",
                    {
                        "state_id": state["state_id"],
                        "parent_state": state["state_id"],
                        "arm_id": arm_id,
                        "step_index": int(step_index),
                        "arm_step": int(arm_step),
                        "baseline_env_action": baseline.tolist(),
                        "env_action": action.tolist(),
                        "done": bool(done),
                    },
                )
                save_image_pair(arm_dir, step_index, obs)
                if arm_id == "A":
                    compare_constraint_record(
                        record,
                        parent["constraints"][step_index],
                        context=f"arm=A state={state['state_id']} step={step_index}",
                    )

            write_json(
                arm_dir / "meta.json",
                {
                    "schema_version": 1,
                    **state,
                    "parent_state": state["state_id"],
                    "arm_id": arm_id,
                    "branch_step": branch_step,
                    "horizon": horizon,
                    "step_index_semantics": "post-action full/wrist images and post-action constraint share one index",
                    "branch_state_sha256": reference_state_hash,
                    "rng_snapshot_sha256": reference_rng_hash,
                    "protocol_sha256": sha256_file(protocol_path),
                    "guard_head": git_revision(repo_root)[0],
                    "guard_dirty": git_revision(repo_root)[1],
                    "parent_collection": Path(collection_root).name,
                    "parent_collection_guard_head": parent["meta"]["guard_head"],
                    "parent_manifest_sha256": parent["meta"]["manifest_sha256"],
                    "parent_actions_sha256": parent["actions_sha256"],
                    "parent_constraints_sha256": parent["constraints_sha256"],
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "arm_a_exact_parity": arm_id == "A",
                },
            )
    except Exception:
        write_json(
            state_dir / "FAILED.json",
            {"state_id": state["state_id"], "branch_step": branch_step},
        )
        raise
    finally:
        env.close()


def main():
    repo_root = Path(__file__).resolve().parents[2]
    project_root = repo_root.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", type=Path, default=project_root / "guard_workspace/collections/phase3_full70_fe76a6c")
    parser.add_argument("--manifest", type=Path, default=repo_root / "data/pilot_v0/manifest.json")
    parser.add_argument("--protocol", type=Path, default=repo_root / "pilot_protocol.md")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state-list", required=True)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--horizon", type=int, default=30)
    args = parser.parse_args()

    head, dirty = git_revision(repo_root)
    if dirty:
        raise RuntimeError("counterfactual execution requires a clean Guard worktree")
    _, states = load_selected_states(args.manifest, args.state_list)
    arms = tuple(part.strip().upper() for part in args.arms.split(",") if part.strip())
    if not arms or any(arm not in ARMS for arm in arms) or len(arms) != len(set(arms)):
        raise ValueError(f"invalid arm selection: {arms}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    task_suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    for state in states:
        run_state(
            state,
            task_suite=task_suite,
            collection_root=args.collection_root,
            output_root=args.output_root,
            protocol_path=args.protocol,
            horizon=args.horizon,
            arms=arms,
            repo_root=repo_root,
        )
        print(f"PASS {state['state_id']} arms={','.join(arms)} guard_head={head}", flush=True)


if __name__ == "__main__":
    main()

