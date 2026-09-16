"""Validate Phase 4B targeted smoke artifacts."""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image

from guard.counterfactual.run_counterfactual import CONSTRAINTS, rng_sha256, state_sha256
from guard.counterfactual.run_counterfactual_v2 import ARM_BRANCH, NOMINAL_ARMS, V2_ARMS
from guard.json_io import canonical_dumps


IMAGE_NAME = re.compile(r"step_(\d{5})\.png$")


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def validate_state(state_dir):
    state_dir = Path(state_dir)
    if (state_dir / "FAILED.json").exists():
        raise ValueError(f"failed marker present: {state_dir}")
    discovery = json.loads((state_dir / "branch_discovery.json").read_text(encoding="utf-8"))
    references = {}
    for branch_name in ("grip", "drop", "workspace"):
        references[branch_name] = (
            state_sha256(np.load(state_dir / f"branch_{branch_name}_state.npy", allow_pickle=False)),
            rng_sha256(json.loads((state_dir / f"branch_{branch_name}_rng.json").read_text(encoding="utf-8"))),
        )
    arms = []
    for arm_id in V2_ARMS:
        arm_dir = state_dir / f"arm_{arm_id}"
        meta = json.loads((arm_dir / "meta.json").read_text(encoding="utf-8"))
        actions = _jsonl(arm_dir / "actions.jsonl")
        constraints = _jsonl(arm_dir / "constraints.jsonl")
        branch_name = ARM_BRANCH[arm_id]
        start = discovery["branches"][branch_name]["step"]
        expected = list(range(start, start + 30))
        if [row["step_index"] for row in actions] != expected or [row["step_index"] for row in constraints] != expected:
            raise ValueError(f"step coverage mismatch: {arm_dir}")
        if (meta["branch_state_sha256"], meta["rng_snapshot_sha256"]) != references[branch_name]:
            raise ValueError(f"branch provenance mismatch: {arm_dir}")
        if meta["guard_dirty"] or meta["replay_max_abs_error"] > meta["parity_atol"]:
            raise ValueError(f"dirty or replay parity failure: {arm_dir}")
        if arm_id in NOMINAL_ARMS and (
            not meta["nominal_parity_passed"] or meta["nominal_max_abs_error"] > meta["parity_atol"]
        ):
            raise ValueError(f"nominal parity failure: {arm_dir}")
        for row in constraints:
            for name in CONSTRAINTS:
                margin = row[name]["margin"]
                if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not math.isfinite(margin):
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
            if steps != expected:
                raise ValueError(f"image coverage mismatch: {arm_dir} view={view}")
        arms.append({"arm_id": arm_id, "branch_type": branch_name, "steps": len(actions)})
    return {"state_id": state_dir.name, "branches": discovery["branches"], "arms": arms}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    states = sorted(path for path in args.output_root.iterdir() if path.is_dir())
    result = {"states": [validate_state(path) for path in states]}
    print(canonical_dumps(result, indent=2))


if __name__ == "__main__":
    main()

