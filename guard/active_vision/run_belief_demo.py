"""Run the preregistered one-layout Phase 6A2 oracle-observation demo."""

import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from guard.active_vision.belief_scene import (
    CAMERAS,
    DEMO_LAYOUT,
    DEMO_VISIBILITY,
    HIDDEN_STATES,
    PAID_CAMERAS,
    ROUTES,
    ThreeRouteBeliefEnv,
)
from guard.active_vision.belief_three_route import oracle_outcome, plan, posterior
from guard.active_vision.runtime import execute_route, physical_state, state_hash
from guard.json_io import canonical_dumps, write_json


SEED = 66000


def sha(value):
    return hashlib.sha256(value).hexdigest()


def fresh_env(hidden_state):
    random.seed(SEED)
    np.random.seed(SEED)
    env = ThreeRouteBeliefEnv(layout=DEMO_LAYOUT, hidden_state=hidden_state)
    env.reset()
    return env


class BeliefQueryBroker:
    def __init__(self, env, budget):
        self.env = env
        self.budget = budget
        self.ledger = []

    def free(self):
        return self._capture("v0", False)

    def query(self, camera):
        if camera not in PAID_CAMERAS:
            raise PermissionError(camera)
        if sum(row["charged"] for row in self.ledger) >= self.budget:
            raise PermissionError("budget exhausted")
        return self._capture(camera, True)

    def _capture(self, camera, charged):
        before = state_hash(physical_state(self.env))
        image = self.env.capture_camera(camera)
        after = state_hash(physical_state(self.env))
        if before != after:
            raise RuntimeError(f"query changed state: {camera}")
        self.ledger.append({"camera": camera, "charged": charged, "state_hash": before, "image_sha256": sha(image.tobytes())})
        return image


def execute_fresh(hidden_state, route, expected_hash):
    env = fresh_env(hidden_state)
    try:
        actual = state_hash(physical_state(env))
        if actual != expected_hash:
            raise RuntimeError("branch reconstruction mismatch")
        result = execute_route(env, route, max_steps=100, hold_steps=5)
        result["branch_state_hash"] = actual
        return result
    finally:
        env.close()


def run_policy(hidden_state, mode):
    env = fresh_env(hidden_state)
    try:
        broker = BeliefQueryBroker(env, budget=2)
        free = broker.free()
        branch_hash = state_hash(physical_state(env))
        belief = tuple([1.0 / 8.0] * 8)
        available = dict(DEMO_VISIBILITY)
        trace = []
        if mode == "adaptive":
            remaining = 2
            while True:
                decision = plan(belief, available, remaining)
                trace.append({"belief": list(belief), "decision": decision})
                if decision["kind"] == "terminal":
                    terminal = decision["id"]
                    break
                camera = decision["id"]
                broker.query(camera)
                observation = oracle_outcome(hidden_state, available[camera])
                trace[-1]["observation"] = observation
                belief = posterior(belief, available[camera], observation)
                del available[camera]
                remaining -= 1
        elif mode == "fixed":
            for camera in ("q_left", "q_front"):
                broker.query(camera)
                observation = oracle_outcome(hidden_state, available[camera])
                trace.append({"belief": list(belief), "decision": {"kind": "query", "id": camera}, "observation": observation})
                belief = posterior(belief, available[camera], observation)
                del available[camera]
            decision = plan(belief, {}, 0)
            trace.append({"belief": list(belief), "decision": decision})
            terminal = decision["id"]
        else:
            raise ValueError(mode)
        outcome = execute_fresh(hidden_state, terminal, branch_hash) if terminal in ROUTES else None
        return {
            "mode": mode,
            "hidden_state": hidden_state,
            "v0_sha256": sha(free.tobytes()),
            "branch_state_hash": branch_hash,
            "query_ledger": broker.ledger,
            "trace": trace,
            "terminal_action": terminal,
            "physical_outcome": outcome,
        }
    finally:
        env.close()


def make_overview(root, summary):
    state = "100"
    images = [Image.open(root / state / f"{camera}.png").convert("RGB") for camera in CAMERAS]
    canvas = Image.new("RGB", (224 * len(images), 280), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (camera, image) in enumerate(zip(CAMERAS, images)):
        canvas.paste(image, (224 * index, 28))
        draw.text((224 * index + 6, 8), camera, fill="black")
    adaptive = "adaptive: q_left blocked -> q_right clear -> right_route success"
    fixed = "fixed: q_left blocked -> q_front unobserved -> stop"
    draw.text((8, 258), adaptive, fill="black")
    draw.text((620, 258), fixed, fill="black")
    canvas.save(root / "demo_overview.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    args.output_root.mkdir(parents=True)
    states = []
    for hidden_state in HIDDEN_STATES:
        state_dir = args.output_root / hidden_state
        state_dir.mkdir()
        env = fresh_env(hidden_state)
        try:
            broker = BeliefQueryBroker(env, budget=4)
            images = {"v0": broker.free()}
            for camera in PAID_CAMERAS:
                images[camera] = broker.query(camera)
            branch_hash = state_hash(physical_state(env))
            for camera, image in images.items():
                Image.fromarray(image).save(state_dir / f"{camera}.png")
        finally:
            env.close()
        routes = [execute_fresh(hidden_state, route, branch_hash) for route in ROUTES]
        write_json(state_dir / "routes.json", routes)
        states.append({"hidden_state": hidden_state, "image_sha256": {name: sha(image.tobytes()) for name, image in images.items()},
                       "query_ledger": broker.ledger,
                       "routes": [{key: value for key, value in row.items() if key != "records"} for row in routes]})
    adaptive = run_policy("100", "adaptive")
    fixed = run_policy("100", "fixed")
    adaptive_replay = run_policy("100", "adaptive")
    fixed_replay = run_policy("100", "fixed")
    summary = {"schema_version": 1, "protocol": "belief_active_vision_v1_addendum", "guard_head": subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip(),
        "layout": DEMO_LAYOUT.layout_id, "visibility": DEMO_VISIBILITY, "states": states,
        "adaptive": adaptive, "fixed": fixed,
        "adaptive_replay_exact": canonical_dumps(adaptive) == canonical_dumps(adaptive_replay),
        "fixed_replay_exact": canonical_dumps(fixed) == canonical_dumps(fixed_replay)}
    write_json(args.output_root / "summary.json", summary)
    make_overview(args.output_root, summary)
    print("wrote three-route demo")


if __name__ == "__main__":
    main()
