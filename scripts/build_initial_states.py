"""Create the immutable pilot train/calibration/test initial-state manifest."""

import hashlib
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GUARD_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GUARD_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "code" / "openvla-oft"))

from libero.libero import benchmark
from guard.collection import initial_state_hash
from guard.json_io import write_json


SEED = 7
ALLOC = {"train": 5, "calibration": 2, "test": 2}


suite = benchmark.get_benchmark_dict()["libero_spatial"]()
rng = random.Random(SEED)
manifest = {"seed": SEED, "alloc_per_task": ALLOC, "tasks": []}

for task_id in range(suite.n_tasks):
    states = suite.get_task_init_states(task_id)
    indices = list(range(len(states)))
    rng.shuffle(indices)
    entry = {"task_id": task_id, "task_name": suite.get_task(task_id).name}
    start = 0
    for name, count in ALLOC.items():
        selected = sorted(indices[start : start + count])
        start += count
        entry[name] = [{"init_index": index, "state_hash": initial_state_hash(states[index])} for index in selected]
    assert start <= len(states), (task_id, len(states), start)
    manifest["tasks"].append(entry)

output_dir = Path(__file__).resolve().parents[1] / "data" / "pilot_v0"
os.makedirs(output_dir, exist_ok=True)
manifest_path = output_dir / "manifest.json"
if os.path.exists(manifest_path):
    raise FileExistsError(f"refusing to regenerate frozen manifest: {manifest_path}")
write_json(manifest_path, manifest, trailing_newline=False)
manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()[:16]
print(f"manifest hash: {manifest_hash}\nsaved to {manifest_path}")
