"""Validate Phase 4A counterfactual branch artifacts."""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image

from guard.counterfactual.run_counterfactual import ARMS, CONSTRAINTS, rng_sha256, state_sha256
from guard.json_io import canonical_dumps


IMAGE_NAME = re.compile(r"step_(\d{5})\.png$")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def validate_arm(arm_dir, expected_state_hash, expected_rng_hash):
    arm_dir = Path(arm_dir)
    meta = json.loads((arm_dir / "meta.json").read_text(encoding="utf-8"))
    actions = read_jsonl(arm_dir / "actions.jsonl")
    constraints = read_jsonl(arm_dir / "constraints.jsonl")
    expected_steps = list(range(meta["branch_step"], meta["branch_step"] + meta["horizon"]))
    if [row["step_index"] for row in actions] != expected_steps:
        raise ValueError(f"non-contiguous actions: {arm_dir}")
    if [row["step_index"] for row in constraints] != expected_steps:
        raise ValueError(f"non-contiguous constraints: {arm_dir}")
    if [row["arm_step"] for row in constraints] != list(range(meta["horizon"])):
        raise ValueError(f"non-contiguous arm steps: {arm_dir}")
    for row in constraints:
        for name in CONSTRAINTS:
            value = row[name]["margin"]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"invalid margin: {arm_dir} step={row['step_index']} constraint={name}")
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
        if steps != expected_steps:
            raise ValueError(f"image coverage mismatch: {arm_dir} view={view}")
    if meta["branch_state_sha256"] != expected_state_hash or meta["rng_snapshot_sha256"] != expected_rng_hash:
        raise ValueError(f"branch provenance mismatch: {arm_dir}")
    if meta["guard_dirty"]:
        raise ValueError(f"dirty Guard execution: {arm_dir}")
    if meta["arm_id"] == "A" and not meta["arm_a_exact_parity"]:
        raise ValueError(f"arm A parity not certified: {arm_dir}")
    return {"arm_id": meta["arm_id"], "steps": len(actions)}


def validate_state(state_dir, expected_arms=ARMS):
    state_dir = Path(state_dir)
    failed = state_dir / "FAILED.json"
    if failed.exists():
        raise ValueError(f"failed marker present: {failed}")
    state = np.load(state_dir / "branch_state.npy", allow_pickle=False)
    snapshot = json.loads((state_dir / "rng_snapshot.json").read_text(encoding="utf-8"))
    state_hash = state_sha256(state)
    rng_hash = rng_sha256(snapshot)
    results = []
    for arm_id in expected_arms:
        results.append(validate_arm(state_dir / f"arm_{arm_id}", state_hash, rng_hash))
    return {"state_id": state_dir.name, "branch_state_sha256": state_hash, "rng_snapshot_sha256": rng_hash, "arms": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("state_ids", nargs="*")
    parser.add_argument("--arms", default=",".join(ARMS))
    args = parser.parse_args()
    arms = tuple(part.strip().upper() for part in args.arms.split(",") if part.strip())
    state_dirs = [args.output_root / state_id for state_id in args.state_ids] if args.state_ids else sorted(path for path in args.output_root.iterdir() if path.is_dir())
    result = {"states": [validate_state(path, arms) for path in state_dirs]}
    result["sha256"] = hashlib.sha256(canonical_dumps(result, sort_keys=True).encode("utf-8")).hexdigest()
    print(canonical_dumps(result, indent=2))


if __name__ == "__main__":
    main()

