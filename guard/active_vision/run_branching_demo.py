"""Run the Phase 6A3 physical adaptive-branching gate."""

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from guard.active_vision.belief_branch_scene import CAMERAS, DEMO_LAYOUT, BranchingBeliefEnv
from guard.active_vision.belief_branching import (
    CAMERAS as PAID_CAMERAS,
    PRIOR,
    ROUTES,
    SPECIALIST,
    STATES,
    all_fixed_results,
    observation,
    run_adaptive,
)
from guard.active_vision.runtime import execute_route, physical_state, state_hash
from guard.json_io import canonical_dumps, write_json


SEED = 66100


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def fresh_env(state):
    random.seed(SEED)
    np.random.seed(SEED)
    env = BranchingBeliefEnv(layout=DEMO_LAYOUT, hidden_state=state)
    env.reset()
    return env


def capture_fingerprint(state, output_dir=None):
    env = fresh_env(state)
    try:
        ledger = []
        images = {}
        for camera in CAMERAS:
            before = state_hash(physical_state(env))
            image = env.capture_camera(camera)
            after = state_hash(physical_state(env))
            if before != after:
                raise RuntimeError(f"query changed state: {state}/{camera}")
            images[camera] = image
            ledger.append({"camera": camera, "state_hash": before, "image_sha256": digest_bytes(image.tobytes())})
            if output_dir is not None:
                Image.fromarray(image).save(output_dir / f"{camera}.png")
        return {"state_hash": ledger[0]["state_hash"], "ledger": ledger}
    finally:
        env.close()


def execute_fresh(state, route, expected_hash):
    env = fresh_env(state)
    try:
        actual = state_hash(physical_state(env))
        if actual != expected_hash:
            raise RuntimeError(f"branch reconstruction mismatch: {state}/{route}")
        result = execute_route(env, route, max_steps=100, hold_steps=5)
        result["branch_state_hash"] = actual
        return result
    finally:
        env.close()


def compact_route(row):
    return {key: value for key, value in row.items() if key != "records"}


def physical_terminal(state, action, route_map):
    if action == "stop":
        return {"action": "stop", "success": False, "collision": False}
    route = route_map[(state, action)]
    return {"action": action, "success": route["collision_free_success"], "collision": route["collision"]}


def make_overview(root):
    tile = 160
    header = 28
    row_height = tile + 24
    canvas = Image.new("RGB", (tile * len(CAMERAS), header + row_height * len(STATES)), "white")
    draw = ImageDraw.Draw(canvas)
    for column, camera in enumerate(CAMERAS):
        draw.text((column * tile + 5, 8), camera, fill="black")
    for row, state in enumerate(STATES):
        y = header + row * row_height
        draw.text((5, y + 3), state, fill="white", stroke_width=2, stroke_fill="black")
        for column, camera in enumerate(CAMERAS):
            image = Image.open(root / state / f"{camera}.png").convert("RGB").resize((tile, tile))
            canvas.paste(image, (column * tile, y + 20))
    canvas.save(root / "image_audit.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path, nargs="?")
    parser.add_argument("--fingerprint-state", choices=STATES)
    parser.add_argument("--fingerprint-output", type=Path)
    args = parser.parse_args()
    if args.fingerprint_state:
        if args.output_root is not None or args.fingerprint_output is None:
            parser.error("fingerprint mode requires --fingerprint-output and no output_root")
        write_json(args.fingerprint_output, capture_fingerprint(args.fingerprint_state))
        return
    if args.output_root is None:
        parser.error("output_root is required")
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    args.output_root.mkdir(parents=True)

    states = []
    route_map = {}
    for state in STATES:
        state_dir = args.output_root / state
        state_dir.mkdir()
        fingerprint = capture_fingerprint(state, state_dir)
        replay_path = state_dir / "fresh_process_fingerprint.json"
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--fingerprint-state", state,
             "--fingerprint-output", str(replay_path)],
            check=True,
            env=os.environ.copy(),
        )
        replay = json.loads(replay_path.read_text())
        routes = [execute_fresh(state, route, fingerprint["state_hash"]) for route in ROUTES]
        write_json(state_dir / "routes.json", routes)
        for route in routes:
            route_map[(state, route["route"])] = compact_route(route)
        states.append({
            "state": state,
            "observations": {camera: observation(state, camera) for camera in PAID_CAMERAS},
            "fingerprint": fingerprint,
            "fresh_process_replay_exact": canonical_dumps(fingerprint) == canonical_dumps(replay),
            "routes": [compact_route(row) for row in routes],
        })

    adaptive = []
    for state in STATES:
        trace = run_adaptive(state)
        action = trace[-1]["decision"]["id"]
        adaptive.append({"state": state, "trace": trace, "terminal_action": action,
                         "physical_outcome": physical_terminal(state, action, route_map)})

    fixed = []
    for result in all_fixed_results():
        trials = []
        for trial in result["trials"]:
            row = dict(trial)
            row["physical_outcome"] = physical_terminal(trial["state"], trial["terminal_action"], route_map)
            trials.append(row)
        fixed.append({**{key: value for key, value in result.items() if key != "trials"}, "trials": trials})

    contract_path = Path("artifacts/belief_active_vision_v2/contract.json")
    summary = {
        "schema_version": 1,
        "protocol": "belief_active_vision_protocol_v2_addendum",
        "guard_head": subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip(),
        "layout": DEMO_LAYOUT.layout_id,
        "seed": SEED,
        "prior": list(PRIOR),
        "contract_sha256": digest_bytes(contract_path.read_bytes()),
        "states": states,
        "adaptive": adaptive,
        "fixed": fixed,
    }
    write_json(args.output_root / "summary.json", summary)
    make_overview(args.output_root)
    print("wrote adaptive branching demo")


if __name__ == "__main__":
    main()
