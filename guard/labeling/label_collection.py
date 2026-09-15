"""Create one offline label per collection state and constraint."""

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

from guard.json_io import append_jsonl, canonical_dumps
from guard.labeling import load_thresholds


CONSTRAINTS = ("workspace", "gripper_env", "self_collision", "object_drop", "non_finite")
CONTINUOUS_CONSTRAINTS = {"workspace", "gripper_env", "object_drop"}
DEFAULT_COLLECTION_SHA256 = "202479aae82fab7a5c54a80de9106116dc8e6c4e564f480cce2400016d491303"


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _git_head(repo_root):
    return subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _manifest_states(manifest_path, splits=("train", "calibration")):
    raw = Path(manifest_path).read_bytes()
    manifest = json.loads(raw)
    states = []
    for task in manifest["tasks"]:
        for split in splits:
            for item in task[split]:
                states.append(
                    {
                        **item,
                        "task_id": task["task_id"],
                        "task_name": task["task_name"],
                        "split": split,
                        "state_id": f"task_{task['task_id']:02d}_init_{item['init_index']:03d}",
                    }
                )
    return _sha256_bytes(raw), states


def _read_jsonl(path):
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
    return records


def _validate_margin(value, state_id, step_index, constraint):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(
            f"non-finite or non-numeric margin: state={state_id} step={step_index} "
            f"constraint={constraint} value={value!r}"
        )
    return float(value)


def _phase_summary(records, constraint, state_id, sustained_k):
    if not records:
        return {
            "step_count": 0,
            "min_margin": None,
            "min_margin_step": None,
            "any_violation": False,
            "first_violation_step": None,
            "sustained_violation": False,
            "sustained_onset_step": None,
            "max_violation_run": 0,
            "margin_flag_mismatch_steps": 0,
        }

    values = []
    max_run = 0
    run = 0
    run_start = None
    sustained_onset = None
    first_violation = None
    mismatch = 0
    for record in records:
        step = record["step_index"]
        if constraint not in record or set(record[constraint]) != {"violated", "margin"}:
            raise ValueError(f"invalid {constraint} schema: state={state_id} step={step}")
        violated = record[constraint]["violated"]
        if not isinstance(violated, bool):
            raise ValueError(f"non-boolean violation flag: state={state_id} step={step} constraint={constraint}")
        margin = _validate_margin(record[constraint]["margin"], state_id, step, constraint)
        values.append((margin, step))
        if constraint in CONTINUOUS_CONSTRAINTS and violated != (margin < 0.0):
            mismatch += 1
        if violated:
            if first_violation is None:
                first_violation = step
            if run == 0:
                run_start = step
            run += 1
            max_run = max(max_run, run)
            if run == sustained_k and sustained_onset is None:
                sustained_onset = run_start
        else:
            run = 0
            run_start = None

    min_margin, min_step = min(values)
    return {
        "step_count": len(records),
        "min_margin": min_margin,
        "min_margin_step": min_step,
        "any_violation": first_violation is not None,
        "first_violation_step": first_violation,
        "sustained_violation": sustained_onset is not None,
        "sustained_onset_step": sustained_onset,
        "max_violation_run": max_run,
        "margin_flag_mismatch_steps": mismatch,
    }


def label_episode(meta, records, constraint, sustained_k, provenance):
    state_id = meta["state_id"]
    steps = [record.get("step_index") for record in records]
    if steps != list(range(len(records))):
        raise ValueError(f"non-contiguous constraint steps: {state_id}")
    if meta["constraint_step_count"] != len(records):
        raise ValueError(f"constraint count mismatch: {state_id}")
    policy_start = meta["image_window"]["start"]
    wait = [record for record in records if record["step_index"] < policy_start]
    policy = [record for record in records if record["step_index"] >= policy_start]
    if not wait or not policy:
        raise ValueError(f"empty wait or policy phase: {state_id}")

    return {
        "schema_version": 1,
        **provenance,
        "state_id": state_id,
        "task_id": meta["task_id"],
        "task_name": meta["task_name"],
        "init_index": meta["init_index"],
        "state_hash": meta["state_hash"],
        "split": meta["split"],
        "success": meta["success"],
        "termination_reason": meta["termination_reason"],
        "constraint": constraint,
        "policy_start_step": policy_start,
        "episode": _phase_summary(records, constraint, state_id, sustained_k),
        "wait": _phase_summary(wait, constraint, state_id, sustained_k),
        "policy": _phase_summary(policy, constraint, state_id, sustained_k),
    }


def _focused_input_sha256(manifest_path, collection_root, state_ids):
    digest = hashlib.sha256()
    paths = [Path(manifest_path)]
    for state_id in sorted(state_ids):
        paths.extend((Path(collection_root) / state_id / "meta.json", Path(collection_root) / state_id / "constraints.jsonl"))
    for path in paths:
        relative = path.name if path == Path(manifest_path) else str(path.relative_to(collection_root))
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def label_collection(
    collection_root,
    manifest_path,
    output_path,
    *,
    sustained_k=3,
    expected_counts=None,
    collection_tree_sha256=DEFAULT_COLLECTION_SHA256,
    force=False,
    repo_root=None,
    splits=("train", "calibration"),
):
    if sustained_k < 1:
        raise ValueError("sustained_k must be at least 1")
    collection_root = Path(collection_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    output_path = Path(output_path).resolve()
    if collection_root == output_path or collection_root in output_path.parents:
        raise ValueError("output must be outside the read-only collection directory")
    if output_path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite labels: {output_path}")

    manifest_sha256, states = _manifest_states(manifest_path, splits)
    split_counts = {split: sum(state["split"] == split for state in states) for split in splits}
    if expected_counts is not None and split_counts != expected_counts:
        raise ValueError(f"unexpected manifest split counts: {split_counts}")
    expected = {state["state_id"]: state for state in states}
    actual = {path.name for path in collection_root.iterdir() if path.is_dir()}
    if actual != set(expected):
        raise ValueError(f"collection state mismatch: missing={sorted(set(expected)-actual)} extra={sorted(actual-set(expected))}")

    thresholds = load_thresholds()
    if sustained_k is None:
        sustained_k = thresholds["labeling"]["sustained_k"]
    threshold_sha256 = _sha256_bytes(canonical_dumps(thresholds, indent=2).encode("utf-8"))
    repo_root = Path(repo_root or Path(__file__).resolve().parents[2])
    run_id = (
        f"phase3_labels_{collection_root.name}_v1_k{sustained_k}"
        if splits == ("train", "calibration")
        else f"phase3_labels_{collection_root.name}_{'-'.join(splits)}_v1_k{sustained_k}"
    )
    provenance = {
        "run_id": run_id,
        "labeler_guard_head": _git_head(repo_root),
        "collection_guard_head": None,
        "manifest_sha256": manifest_sha256,
        "collection_tree_sha256": collection_tree_sha256,
        "label_input_sha256": _focused_input_sha256(manifest_path, collection_root, expected),
        "threshold_version": thresholds["version"],
        "thresholds_sha256": threshold_sha256,
        "decision_margin": thresholds["labeling"]["decision_margin"],
        "sustained_k": sustained_k,
    }

    labels = []
    collection_heads = set()
    for state in states:
        episode_dir = collection_root / state["state_id"]
        meta = json.loads((episode_dir / "meta.json").read_text(encoding="utf-8"))
        for key in ("state_id", "task_id", "task_name", "init_index", "state_hash", "split"):
            if meta[key] != state[key]:
                raise ValueError(f"manifest/meta mismatch for {state['state_id']}: {key}")
        if meta["manifest_sha256"] != manifest_sha256:
            raise ValueError(f"manifest hash mismatch in meta: {state['state_id']}")
        collection_heads.add(meta["guard_head"])
        records = _read_jsonl(episode_dir / "constraints.jsonl")
        for constraint in CONSTRAINTS:
            labels.append(label_episode(meta, records, constraint, sustained_k, provenance))
    if len(collection_heads) != 1:
        raise ValueError(f"mixed collection Guard revisions: {sorted(collection_heads)}")
    for label in labels:
        label["collection_guard_head"] = next(iter(collection_heads))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.tmp")
    if temp_path.exists():
        temp_path.unlink()
    try:
        for label in labels:
            append_jsonl(temp_path, label)
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return labels


def main():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection_root", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--manifest", type=Path, default=repo_root / "data/pilot_v0/manifest.json")
    parser.add_argument("--sustained-k", type=int, default=load_thresholds()["labeling"]["sustained_k"])
    parser.add_argument("--collection-tree-sha256", default=DEFAULT_COLLECTION_SHA256)
    parser.add_argument("--splits", default="train,calibration", help="Comma-separated manifest splits")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    splits = tuple(part.strip() for part in args.splits.split(",") if part.strip())
    expected_by_split = {"train": 50, "calibration": 20, "test": 20}
    labels = label_collection(
        args.collection_root,
        args.manifest,
        args.output_path,
        sustained_k=args.sustained_k,
        expected_counts={split: expected_by_split[split] for split in splits},
        splits=splits,
        collection_tree_sha256=args.collection_tree_sha256,
        force=args.force,
        repo_root=repo_root,
    )
    print(f"wrote {len(labels)} labels to {args.output_path}")


if __name__ == "__main__":
    main()
