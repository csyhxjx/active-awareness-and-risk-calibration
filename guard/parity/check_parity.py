"""Compare two parity JSONL recordings.

Usage: python -m guard.parity.check_parity OFF.jsonl ON.jsonl
"""

import json
import sys

import numpy as np


def load(path):
    with open(path, encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    chunks = {
        (r["task_id"], r["trial"], r["inference_idx"]): r["action_chunk"]
        for r in records
        if r["type"] == "chunk"
    }
    steps = {
        (r["task_id"], r["trial"], r["inference_idx"], r["chunk_idx"]): r["env_action"]
        for r in records
        if r["type"] == "step"
    }
    episodes = {
        (r["task_id"], r["trial"]): r
        for r in records
        if r["type"] == "episode"
    }
    return chunks, steps, episodes


def compare_dict(a, b, name, atol=1e-5):
    keys = set(a) | set(b)
    missing = set(a) ^ set(b)
    matches = 0
    for key in sorted(set(a) & set(b)):
        x, y = np.asarray(a[key]), np.asarray(b[key])
        if x.shape == y.shape and np.allclose(x, y, atol=atol):
            matches += 1
        else:
            diff = np.max(np.abs(x - y)) if x.shape == y.shape else float("inf")
            print(f"  MISMATCH {name} {key}: max diff {diff:.2e}")
    print(f"[{name}] {matches}/{len(keys)} allclose, missing/extra {len(missing)}")
    return matches == len(keys) and not missing


def main(path_a, path_b):
    chunks_a, steps_a, episodes_a = load(path_a)
    chunks_b, steps_b, episodes_b = load(path_b)
    chunks_ok = compare_dict(chunks_a, chunks_b, "chunks")
    steps_ok = compare_dict(steps_a, steps_b, "steps")
    episode_diffs = [
        key
        for key in set(episodes_a) | set(episodes_b)
        if key not in episodes_a
        or key not in episodes_b
        or any(
            episodes_a[key].get(field) != episodes_b[key].get(field)
            for field in ("success", "total_steps", "num_inferences")
        )
    ]
    print(f"[episodes] diffs: {episode_diffs or 'none'}")
    result = chunks_ok and steps_ok and not episode_diffs
    print("RESULT:", "PASS" if result else "FAIL")
    return 0 if result else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m guard.parity.check_parity OFF.jsonl ON.jsonl")
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
