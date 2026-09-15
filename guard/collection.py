"""Manifest selection and per-state collection persistence."""

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


SPLITS = ("train", "calibration", "test")
IMAGE_NAME = re.compile(r"step_(\d{5})\.png$")


def initial_state_hash(state):
    blob = json.JSONEncoder(
        sort_keys=True,
        default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value),
    ).encode(state)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def load_selected_states(manifest_path, state_list):
    manifest_path = Path(manifest_path)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    by_id = {}
    by_split = {split: [] for split in SPLITS}

    for task in manifest["tasks"]:
        for split in SPLITS:
            for item in task[split]:
                state = {
                    **item,
                    "task_id": task["task_id"],
                    "task_name": task["task_name"],
                    "split": split,
                    "state_id": f"task_{task['task_id']:02d}_init_{item['init_index']:03d}",
                }
                if state["state_id"] in by_id:
                    raise ValueError(f"duplicate manifest state: {state['state_id']}")
                by_id[state["state_id"]] = state
                by_split[split].append(state)

    selected = []
    seen = set()
    for selector in (part.strip() for part in state_list.split(",")):
        if not selector:
            continue
        matches = by_split.get(selector)
        if matches is None:
            matches = [by_id[selector]] if selector in by_id else None
        if matches is None:
            raise ValueError(f"unknown state selector: {selector}")
        for state in matches:
            if state["state_id"] not in seen:
                selected.append(state)
                seen.add(state["state_id"])

    if not selected:
        raise ValueError("state_list selected no states")
    return hashlib.sha256(manifest_bytes).hexdigest(), selected


def guard_revision(repo_root):
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


class EpisodeCollector:
    def __init__(self, output_root, state, image_t_start, image_t_end):
        self.state = state
        self.image_t_start = image_t_start
        self.image_t_end = image_t_end
        self.episode_dir = Path(output_root) / state["state_id"]
        if self.episode_dir.exists():
            raise FileExistsError(f"refusing to overwrite collection: {self.episode_dir}")
        self.full_dir = self.episode_dir / "images" / "full"
        self.wrist_dir = self.episode_dir / "images" / "wrist"
        self.full_dir.mkdir(parents=True)
        self.wrist_dir.mkdir(parents=True)
        self.actions_path = self.episode_dir / "actions.jsonl"
        self.constraints_path = self.episode_dir / "constraints.jsonl"
        self.action_steps = []
        self.constraint_steps = []
        self.image_steps = []

    @staticmethod
    def _append(path, record):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def log_chunk(self, inference_idx, step_index, action_chunk):
        self._append(
            self.actions_path,
            {
                "type": "chunk",
                "state_id": self.state["state_id"],
                "inference_idx": inference_idx,
                "step_index": int(step_index),
                "action_chunk": np.asarray(action_chunk).tolist(),
            },
        )

    def log_step(self, inference_idx, chunk_idx, step_index, env_action):
        step_index = int(step_index)
        self._append(
            self.actions_path,
            {
                "type": "step",
                "state_id": self.state["state_id"],
                "inference_idx": inference_idx,
                "chunk_idx": chunk_idx,
                "step_index": step_index,
                "env_action": np.asarray(env_action).tolist(),
            },
        )
        self.action_steps.append(step_index)

    def log_constraint(self, record):
        record = dict(record)
        step_index = int(record.pop("t"))
        record["step_index"] = step_index
        self._append(self.constraints_path, record)
        self.constraint_steps.append(step_index)

    def save_images(self, step_index, full_image, wrist_image):
        step_index = int(step_index)
        if step_index < self.image_t_start:
            return
        if self.image_t_end is not None and step_index >= self.image_t_end:
            return
        filename = f"step_{step_index:05d}.png"
        Image.fromarray(np.asarray(full_image, dtype=np.uint8)).save(self.full_dir / filename)
        Image.fromarray(np.asarray(wrist_image, dtype=np.uint8)).save(self.wrist_dir / filename)
        self.image_steps.append(step_index)

    def finish(self, metadata):
        payload = {
            "schema_version": 1,
            **self.state,
            **metadata,
            "step_count": len(self.constraint_steps),
            "action_step_count": len(self.action_steps),
            "constraint_step_count": len(self.constraint_steps),
            "image_step_count": len(self.image_steps),
            "image_window": {
                "start": self.image_t_start,
                "end": self.image_t_end,
                "interval": "[start, end)",
            },
            "step_index_semantics": "pre-action image/action and post-action constraint share one index",
        }
        meta_path = self.episode_dir / "meta.json"
        temp_path = meta_path.with_suffix(".json.tmp")
        temp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temp_path, meta_path)
        return payload


def validate_episode_dir(episode_dir, verify_png=True):
    episode_dir = Path(episode_dir)
    required = [
        episode_dir / "actions.jsonl",
        episode_dir / "constraints.jsonl",
        episode_dir / "images" / "full",
        episode_dir / "images" / "wrist",
        episode_dir / "meta.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ValueError(f"missing collection artifacts: {missing}")

    meta = json.loads(required[-1].read_text(encoding="utf-8"))
    actions = [json.loads(line) for line in required[0].read_text(encoding="utf-8").splitlines()]
    constraints = [json.loads(line) for line in required[1].read_text(encoding="utf-8").splitlines()]
    action_steps = [record["step_index"] for record in actions if record["type"] == "step"]
    constraint_steps = [record["step_index"] for record in constraints]

    def image_steps(directory):
        paths = sorted(directory.glob("step_*.png"))
        steps = []
        for path in paths:
            match = IMAGE_NAME.match(path.name)
            if match is None:
                raise ValueError(f"invalid image filename: {path}")
            if verify_png:
                with Image.open(path) as image:
                    image.verify()
            steps.append(int(match.group(1)))
        return steps

    full_steps = image_steps(required[2])
    wrist_steps = image_steps(required[3])
    start, end = meta["image_window"]["start"], meta["image_window"]["end"]
    expected_images = [step for step in action_steps if step >= start and (end is None or step < end)]

    checks = {
        "state_directory": episode_dir.name == meta["state_id"],
        "unique_action_steps": len(action_steps) == len(set(action_steps)),
        "constraints_contiguous": constraint_steps == list(range(len(constraint_steps))),
        "actions_have_constraints": set(action_steps).issubset(constraint_steps),
        "full_wrist_match": full_steps == wrist_steps,
        "images_match_window": full_steps == expected_images,
        "step_count": meta["step_count"] == len(constraint_steps),
        "action_step_count": meta["action_step_count"] == len(action_steps),
        "constraint_step_count": meta["constraint_step_count"] == len(constraint_steps),
        "image_step_count": meta["image_step_count"] == len(full_steps),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"collection integrity failed for {episode_dir.name}: {failed}")
    total_bytes = sum(path.stat().st_size for path in episode_dir.rglob("*") if path.is_file())
    return {"state_id": meta["state_id"], "step_count": len(constraint_steps), "bytes": total_bytes}
