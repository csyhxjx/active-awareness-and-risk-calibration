"""Validate Phase 4C A-gate and edge-drop smoke artifacts."""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image

from guard.counterfactual.run_counterfactual import CONSTRAINTS, rng_sha256, state_sha256
from guard.counterfactual.run_counterfactual_v3 import HORIZON, SMOKE_STATES, crossed_edge
from guard.json_io import canonical_dumps


IMAGE_NAME = re.compile(r"step_(\d{5})\.png$")
SIGNED_CONSTRAINTS = ("workspace", "gripper_env", "object_drop", "non_finite")


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def validate_state(state_dir, arm_id):
    state_dir = Path(state_dir)
    if (state_dir / "FAILED.json").exists():
        raise ValueError(f"failed marker present: {state_dir}")
    discovery = json.loads((state_dir / "branch_discovery.json").read_text(encoding="utf-8"))
    branch = discovery["branch"]
    arm_dir = state_dir / f"arm_{arm_id}"
    meta = json.loads((arm_dir / "meta.json").read_text(encoding="utf-8"))
    actions = jsonl(arm_dir / "actions.jsonl")
    constraints = jsonl(arm_dir / "constraints.jsonl")
    if meta["branch"] != branch:
        raise ValueError(f"branch metadata mismatch: {arm_dir}")
    expected = list(range(branch["step"], branch["step"] + HORIZON))
    if len(actions) != HORIZON or len(constraints) != HORIZON:
        raise ValueError(f"horizon mismatch: {arm_dir}")
    if [row["step_index"] for row in actions] != expected or [row["step_index"] for row in constraints] != expected:
        raise ValueError(f"step coverage mismatch: {arm_dir}")
    if (
        meta["guard_dirty"]
        or meta["replay_max_abs_error"] > meta["parity_atol"]
        or meta["discovery_parity_max_abs_error"] > meta["parity_atol"]
        or meta["branch_target_xyz_max_abs_error"] > meta["parity_atol"]
    ):
        raise ValueError(f"dirty or replay parity failure: {arm_dir}")
    state_hash = state_sha256(np.load(state_dir / "branch_state.npy", allow_pickle=False))
    rng_hash = rng_sha256(json.loads((state_dir / "rng_snapshot.json").read_text(encoding="utf-8")))
    if (meta["branch_state_sha256"], meta["rng_snapshot_sha256"]) != (state_hash, rng_hash):
        raise ValueError(f"branch provenance mismatch: {arm_dir}")
    if arm_id == "A_edge" and (
        not meta["nominal_parity_passed"] or meta["nominal_max_abs_error"] > meta["parity_atol"]
    ):
        raise ValueError(f"nominal parity failure: {arm_dir}")

    release_seen = False
    crossing_step = None
    target_drop_margins = []
    for action, record in zip(actions, constraints):
        if action["arm_step"] != record["arm_step"]:
            raise ValueError(f"arm-step mismatch: {arm_dir}")
        for name in CONSTRAINTS:
            margin = record[name]["margin"]
            if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not math.isfinite(margin):
                raise ValueError(f"invalid margin: {arm_dir} constraint={name}")
            if name in SIGNED_CONSTRAINTS and bool(record[name]["violated"]) != (margin < 0):
                raise ValueError(f"signed oracle mismatch: {arm_dir} constraint={name}")
        baseline = np.asarray(action["baseline_env_action"])
        actual = np.asarray(action["env_action"])
        calculated_drop_margin = action["target_xyz"][2] - branch["drop_line_z"]
        if action["target_drop_margin"] != calculated_drop_margin:
            raise ValueError(f"target drop margin mismatch: {arm_dir}")
        target_drop_margins.append(calculated_drop_margin)
        if arm_id == "A_edge":
            if not np.array_equal(actual, baseline) or action["stage"] != "nominal":
                raise ValueError(f"A action mismatch: {arm_dir}")
        else:
            edge = branch["edge"]
            if actual[edge["action_index"]] != edge["direction"]:
                raise ValueError(f"edge direction mismatch: {arm_dir}")
            expected_stage = "push_release" if release_seen else "push_hold"
            expected_gripper = -1.0 if release_seen else 1.0
            if action["stage"] != expected_stage or actual[6] != expected_gripper:
                raise ValueError(f"stage transition mismatch: {arm_dir}")
            coordinate = action["target_xyz"][edge["action_index"]]
            calculated_crossing = crossed_edge(coordinate, edge)
            if bool(action["edge_crossed_after_action"]) != calculated_crossing:
                raise ValueError(f"edge crossing mismatch: {arm_dir}")
            if crossing_step is None and calculated_crossing:
                crossing_step = action["step_index"]
                release_seen = action["arm_step"] + 1 < HORIZON

    if arm_id == "F_edge":
        if meta["crossing_step"] != crossing_step:
            raise ValueError(f"crossing metadata mismatch: {arm_dir}")
        expected_release = crossing_step + 1 if crossing_step is not None and crossing_step < expected[-1] else None
        if meta["release_start_step"] != expected_release or meta["release_triggered"] != (expected_release is not None):
            raise ValueError(f"release metadata mismatch: {arm_dir}")
        if meta["edge_crossed"] != (crossing_step is not None):
            raise ValueError(f"edge metadata mismatch: {arm_dir}")
    if meta["target_min_drop_margin"] != min(target_drop_margins):
        raise ValueError(f"target minimum drop margin mismatch: {arm_dir}")

    for view in ("full", "wrist"):
        paths = sorted((arm_dir / "images" / view).glob("step_*.png"))
        steps = []
        for path in paths:
            match = IMAGE_NAME.fullmatch(path.name)
            if match is None:
                raise ValueError(f"invalid image name: {path}")
            with Image.open(path) as image:
                image.verify()
            steps.append(int(match.group(1)))
        if steps != expected:
            raise ValueError(f"image coverage mismatch: {arm_dir} view={view}")
    violations = {
        name: any(bool(row[name]["violated"]) for row in constraints)
        for name in CONSTRAINTS
    }
    return {
        "state_id": state_dir.name,
        "arm_id": arm_id,
        "branch_step": branch["step"],
        "branch_state_sha256": state_hash,
        "rng_snapshot_sha256": rng_hash,
        "parity_max_abs_error": meta["nominal_max_abs_error"] if arm_id == "A_edge" else None,
        "edge_crossed": meta["edge_crossed"],
        "crossing_step": meta["crossing_step"],
        "release_triggered": meta["release_triggered"],
        "release_start_step": meta["release_start_step"],
        "target_min_drop_margin": meta["target_min_drop_margin"],
        "violations": violations,
    }


def validate_root(root, arm_id):
    root = Path(root)
    extras = sorted(path.name for path in root.iterdir() if path.is_dir() and path.name not in SMOKE_STATES)
    if extras:
        raise ValueError(f"unregistered state directories: {extras}")
    dirs = [root / state for state in SMOKE_STATES if (root / state).is_dir()]
    if not dirs:
        raise ValueError(f"no registered states in {root}")
    actual = tuple(path.name for path in dirs)
    if actual != SMOKE_STATES[: len(actual)]:
        raise ValueError(f"state root is not a registered prefix: {actual}")
    return [validate_state(path, arm_id) for path in dirs]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--arm", choices=("A_edge", "F_edge"))
    parser.add_argument("--a-root", type=Path)
    parser.add_argument("--f-root", type=Path)
    args = parser.parse_args()
    if args.root is not None:
        if args.arm is None or args.a_root is not None or args.f_root is not None:
            parser.error("--root requires --arm and excludes cross-root options")
        result = {"arm": args.arm, "states": validate_root(args.root, args.arm)}
    else:
        if args.a_root is None or args.f_root is None or args.arm is not None:
            parser.error("cross-root validation requires --a-root and --f-root")
        a_states = validate_root(args.a_root, "A_edge")
        f_states = validate_root(args.f_root, "F_edge")
        a_by_state = {row["state_id"]: row for row in a_states}
        for row in f_states:
            reference = a_by_state.get(row["state_id"])
            if reference is None or (row["branch_state_sha256"], row["rng_snapshot_sha256"]) != (
                reference["branch_state_sha256"], reference["rng_snapshot_sha256"]
            ):
                raise ValueError(f"A/F branch provenance mismatch: {row['state_id']}")
        result = {"a_states": a_states, "f_states": f_states, "branch_provenance_match": True}
    print(canonical_dumps(result, indent=2))


if __name__ == "__main__":
    main()
