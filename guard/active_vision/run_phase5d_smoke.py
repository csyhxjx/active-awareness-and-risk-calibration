"""Run the 8-layout Phase 5D train-only proposer smoke."""

import argparse
import hashlib
import json
import random
import subprocess
import sys
import platform
from pathlib import Path

import numpy as np
from PIL import Image

from guard.active_vision.check_formal_v2 import verifier
from guard.active_vision.phase5d import (
    Phase5DQueryBroker,
    derive_failures,
    project_chunk,
    replay_equal,
    validate_trial_tuple,
)
from guard.active_vision.run_pilot import _fresh_env
from guard.active_vision.runtime import execute_route, physical_state, state_hash
from guard.active_vision.scene import HIDDEN_STATES, ROUTES, Layout, route_waypoints
from guard.json_io import canonical_dumps, write_json


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def tree_hash(root):
    digest = hashlib.sha256()
    for path in sorted(Path(root).rglob("*")):
        if path.is_file() and ".cache" not in path.parts:
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


class FixtureProposer:
    """Deterministic route-endpoint proposer for infrastructure-only tests."""

    def __init__(self, layout):
        self.layout = layout

    def propose(self, image, instruction):
        route = "right_route" if "right" in instruction else "left_route"
        waypoint = np.asarray(route_waypoints(self.layout, route)[1])
        terminal = (waypoint - np.array([0.0, 0.0, 1.0])) / np.array([0.20, 0.20, 0.05])
        chunk = np.zeros((8, 7), dtype=np.float64)
        chunk[-1, :3] = terminal
        return chunk


class OpenVLAProposer:
    def __init__(self, checkpoint, device="cuda:0"):
        root = Path("/internsdata/yewenhao/code/openvla-oft/experiments/robot")
        sys.path.insert(0, str(root.parent.parent))
        from experiments.robot.libero.run_libero_eval import GenerateConfig
        from experiments.robot.openvla_utils import get_action_head, get_processor, get_proprio_projector
        from experiments.robot.robot_utils import get_action, get_model

        self._get_action = get_action
        self._get_proprio_projector = get_proprio_projector
        self._cfg = GenerateConfig(
            pretrained_checkpoint=str(checkpoint),
            use_l1_regression=True,
            use_diffusion=False,
            num_images_in_input=1,
            use_proprio=False,
            center_crop=True,
            seed=54040,
            unnorm_key="libero_spatial_no_noops",
        )
        self._cfg.device = device
        self._model = get_model(self._cfg)
        available = sorted(getattr(self._model, "norm_stats", {}).keys())
        if available != ["libero_spatial_no_noops"]:
            raise RuntimeError(f"unexpected frozen checkpoint norm_stats keys: {available}")
        self._cfg.unnorm_key = available[0]

        self.metadata = {
            "checkpoint": str(checkpoint),
            "checkpoint_tree_sha256": tree_hash(checkpoint),
            "norm_stats_key": self._cfg.unnorm_key,
            "transformers_version": __import__("transformers").__version__,
            "torch_version": __import__("torch").__version__,
            "python_version": platform.python_version(),
            "cuda_device": __import__("torch").cuda.get_device_name(0),
            "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
            "batch_size": 1,
            "do_sample": False,
            "num_actions_chunk": 8,
            "action_dim": 7,
        }
        self._processor = get_processor(self._cfg)
        self._action_head = get_action_head(self._cfg, self._model.llm_dim)

    def propose(self, image, instruction):
        observation = {
            "full_image": np.asarray(image),
            "state": np.zeros(8, dtype=np.float32),
        }
        actions = self._get_action(
            self._cfg,
            self._model,
            observation,
            instruction,
            processor=self._processor,
            action_head=self._action_head,
            proprio_projector=None,
            noisy_action_projector=None,
            use_film=False,
        )
        return np.asarray(actions, dtype=np.float64)


def _selector(env, broker, route):
    camera = "v_left" if route == "left_route" else "v_right"
    image = broker.query(camera)
    verdict = verifier(image, route)["verdict"]
    return {
        "purchased_views": [camera],
        "verifier_outputs": [{"camera": camera, "verdict": verdict}],
        "selected_candidate": route if verdict == "clear" else None,
        "rejected_candidate": route if verdict == "blocked" else None,
        "stop_reason": None if verdict == "clear" else verdict,
        "unresolved": verdict == "unobserved",
        "correct": verdict in {"clear", "blocked"},
    }


def run_one(layout, hidden_state, preference, proposer, scene_seed):
    env = _fresh_env(layout, hidden_state, scene_seed)
    try:
        broker = Phase5DQueryBroker(env, budget=1, public_input={"instruction": preference})
        image = broker.free_observation()
        instruction = f"move the robot along the {preference.replace('_route', '')} route"
        chunk = proposer.propose(image, instruction)
        mapped = project_chunk(layout, chunk)
        proposer_output = {
            "input_image_sha256": sha256(np.asarray(image).tobytes()),
            "instruction": instruction,
            "action_chunk": np.asarray(chunk).tolist(),
            "action_chunk_bytes_sha256": mapped["chunk_hash"],
            "proposer_valid": mapped["valid"],
        }
        if mapped["valid"] and mapped["mapped_candidate"] in ROUTES:
            selector = _selector(env, broker, mapped["mapped_candidate"])
        else:
            selector = {
                "purchased_views": [],
                "verifier_outputs": [],
                "selected_candidate": None,
                "rejected_candidate": None,
                "stop_reason": "proposer_stop" if mapped["mapped_candidate"] == "stop" else "proposer_failure",
                "unresolved": False,
                "correct": False,
            }
        route = selector["selected_candidate"]
        branch_hash = state_hash(physical_state(env))
        if route:
            outcome = execute_route(env, route)
        else:
            outcome = {"route": None, "steps": 0, "reached": False, "collision": False, "timeout": False}
        outcome["branch_state_hash"] = branch_hash
        return {
            "proposer_output": proposer_output,
            "mapped_candidate": mapped,
            "selector_decision": selector,
            "executed_action": {"route": route, "raw_chunk_executed": False},
            "physical_outcome": outcome,
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite smoke root: {args.output_root}")
    manifest = json.loads(args.manifest.read_text())
    rows = manifest["layouts"]
    if len(rows) != 8 or {row["split"] for row in rows} != {"train"}:
        raise ValueError("Phase 5D smoke requires exactly eight train-only layouts")
    if args.fixture == bool(args.checkpoint):
        raise ValueError("choose exactly one proposer: --fixture or --checkpoint")
    proposer = OpenVLAProposer(args.checkpoint) if args.checkpoint else None
    args.output_root.mkdir(parents=True)
    records = []
    for row in rows:
        layout = Layout(**{key: value for key, value in row.items() if key in Layout.__dataclass_fields__})
        if args.fixture:
            proposer = FixtureProposer(layout)
        for hidden_state in HIDDEN_STATES:
            for preference in ROUTES:
                record = run_one(layout, hidden_state, preference, proposer, manifest["scene_seed"])
                replay = run_one(layout, hidden_state, preference, proposer, manifest["scene_seed"])
                records.append({"layout_id": layout.layout_id, "hidden_state": hidden_state, "preference": preference, "record": record, "replay": replay})
    payload = {
        "schema_version": 1,
        "protocol": "active_vision_v3",
        "manifest_sha256": sha256(args.manifest.read_bytes()),
        "guard_head": subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip(),
        "proposer": "fixture" if args.fixture else str(args.checkpoint),
        "proposer_metadata": getattr(proposer, "metadata", {"kind": "fixture"}),
        "trials": records,
    }
    write_json(args.output_root / "smoke.json", payload)
    print(f"wrote {len(records)} Phase 5D smoke trials")


if __name__ == "__main__":
    main()
