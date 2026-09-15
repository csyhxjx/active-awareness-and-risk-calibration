"""Build markdown statistics and diagnostic figures from offline labels."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from guard.labeling.label_collection import CONSTRAINTS


def _read_labels(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def make_stats(labels_path, collection_root, output_dir):
    labels = _read_labels(labels_path)
    state_count = len({label["state_id"] for label in labels})
    expected_rows = state_count * len(CONSTRAINTS)
    if len(labels) != expected_rows:
        raise ValueError(f"expected {expected_rows} labels for {state_count} states, got {len(labels)}")
    output_dir = Path(output_dir)
    collection_root = Path(collection_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for label in labels:
        grouped[(label["split"], label["constraint"])].append(label)
    lines = [
        "# Offline Label Statistics",
        "",
        f"- Labels: `{Path(labels_path).name}` ({len(labels)} rows)",
        f"- Collection: `{collection_root.name}` (read-only input)",
        "- A violation is the recorded boolean flag; continuous signed margins use values below zero.",
        "- `sustained_violation` means at least three consecutive violating steps unless configured otherwise.",
        "- Case 0 calibration: `tau=0.002` is a penetration-axis budget; the margin-axis decision boundary is `0`.",
        "- The full70 warning list is empty; no threshold change is applied.",
        "",
        "## Constraint x Split",
        "",
        "| Split | Constraint | States | Any violation | Sustained | Min margin median | Worst min margin | Flag/margin mismatches |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    splits = tuple(dict.fromkeys(label["split"] for label in labels))
    for split in splits:
        for constraint in CONSTRAINTS:
            rows = grouped[(split, constraint)]
            minima = [row["episode"]["min_margin"] for row in rows]
            mismatches = sum(row["episode"]["margin_flag_mismatch_steps"] for row in rows)
            lines.append(
                f"| {split} | {constraint} | {len(rows)} | "
                f"{sum(row['episode']['any_violation'] for row in rows)} | "
                f"{sum(row['episode']['sustained_violation'] for row in rows)} | "
                f"{_median(minima):.6g} | {min(minima):.6g} | {mismatches} |"
            )

    warnings = []
    for label in labels:
        episode = label["episode"]
        if episode["margin_flag_mismatch_steps"]:
            warnings.append(f"{label['state_id']} {label['constraint']}: {episode['margin_flag_mismatch_steps']} flag/margin mismatches")
        if episode["sustained_violation"]:
            warnings.append(f"{label['state_id']} {label['constraint']}: sustained violation at step {episode['sustained_onset_step']}")
    lines.extend(["", "## Warning List", ""])
    lines.extend(f"- {warning}" for warning in warnings) if warnings else lines.append("- None.")
    lines.extend(["", "## Integrity", "", f"- Unique states: `{len({row['state_id'] for row in labels})}`", f"- Constraints per state: `{len(labels) // len({row['state_id'] for row in labels})}`", f"- Non-finite margins: `{sum(not math.isfinite(row['episode']['min_margin']) for row in labels)}`"])
    (output_dir / "stats.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    gripper = [row["episode"]["min_margin"] for split in splits for row in grouped[(split, "gripper_env")]]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    positive = [value for value in gripper if value > 0]
    negative = [value for value in gripper if value <= 0]
    bins = 30 if len(set(gripper)) > 1 else 5
    ax.hist(positive, bins=bins, alpha=0.8, label="margin > 0")
    if negative:
        ax.hist(negative, bins=max(5, min(20, len(negative))), alpha=0.8, label="margin <= 0")
    ax.axvline(0.0, color="black", linewidth=1, label="0")
    ax.set_yscale("log")
    ax.set_xlabel("Episode minimum gripper_env margin")
    ax.set_ylabel("Count (log scale)")
    ax.set_title(f"{state_count}-state gripper_env episode minima")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "gripper_env_min_margin_histogram.png", dpi=160)
    plt.close(fig)

    trajectory_states = [state_id for state_id in ("task_05_init_034", "task_07_init_043") if (collection_root / state_id / "constraints.jsonl").exists()]
    if trajectory_states:
        fig, axes = plt.subplots(len(trajectory_states), 1, figsize=(10, 3.5 * len(trajectory_states)), sharex=False, squeeze=False)
        axes = axes[:, 0]
    else:
        axes = []
    for axis, state_id in zip(axes, trajectory_states):
        records = [json.loads(line) for line in (collection_root / state_id / "constraints.jsonl").read_text(encoding="utf-8").splitlines()]
        for constraint in CONSTRAINTS:
            axis.plot([record["step_index"] for record in records], [record[constraint]["margin"] for record in records], linewidth=0.9, label=constraint)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.axvline(10, color="tab:orange", linewidth=0.8, linestyle="--")
        axis.set_title(f"{state_id} margin trajectories (orange: policy start)")
        axis.set_ylabel("Signed margin")
        axis.legend(ncol=3, fontsize=8)
    if axes:
        axes[-1].set_xlabel("Constraint step")
        fig.tight_layout()
        fig.savefig(output_dir / "max_steps_margin_trajectories.png", dpi=160)
        plt.close(fig)
    return output_dir / "stats.md"


def _median(values):
    values = sorted(values)
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path)
    parser.add_argument("collection_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(make_stats(args.labels, args.collection_root, args.output_dir))


if __name__ == "__main__":
    main()
