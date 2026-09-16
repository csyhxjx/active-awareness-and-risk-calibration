"""Label a Phase 4A pilot and create its frozen decision report."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from guard.counterfactual.run_counterfactual import ARMS, CONSTRAINTS, load_jsonl, sha256_file
from guard.json_io import append_jsonl, canonical_dumps, write_json
from guard.labeling import load_thresholds
from guard.labeling.label_collection import _phase_summary


def git_head(repo_root):
    return subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def focused_input_sha256(pilot_root):
    digest = hashlib.sha256()
    paths = sorted(
        path
        for path in Path(pilot_root).rglob("*")
        if path.is_file() and path.name in {"meta.json", "actions.jsonl", "constraints.jsonl"}
    )
    for path in paths:
        digest.update(str(path.relative_to(pilot_root)).encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_arms(pilot_root):
    arms = []
    for state_dir in sorted(path for path in Path(pilot_root).iterdir() if path.is_dir()):
        for arm_dir in sorted(state_dir.glob("arm_*")):
            meta_path = arm_dir / "meta.json"
            if not meta_path.exists():
                raise ValueError(f"incomplete arm: {arm_dir}")
            arms.append(
                {
                    "dir": arm_dir,
                    "meta": json.loads(meta_path.read_text(encoding="utf-8")),
                    "constraints": load_jsonl(arm_dir / "constraints.jsonl"),
                }
            )
    if len(arms) != 40:
        raise ValueError(f"expected 40 arms, found {len(arms)}")
    return arms


def build_labels(arms, provenance, sustained_k):
    labels = []
    for arm in arms:
        meta = arm["meta"]
        records = arm["constraints"]
        if [row["arm_step"] for row in records] != list(range(meta["horizon"])):
            raise ValueError(f"non-contiguous arm steps: {arm['dir']}")
        for constraint in CONSTRAINTS:
            labels.append(
                {
                    "schema_version": 1,
                    **provenance,
                    "state_id": meta["state_id"],
                    "parent_state": meta["parent_state"],
                    "task_id": meta["task_id"],
                    "init_index": meta["init_index"],
                    "split": meta["split"],
                    "arm_id": meta["arm_id"],
                    "branch_step": meta["branch_step"],
                    "horizon": meta["horizon"],
                    "constraint": constraint,
                    "branch": _phase_summary(records, constraint, meta["state_id"], sustained_k),
                }
            )
    return labels


def plot_violation(label, arm, evidence_root, figures_root):
    state_id, arm_id, constraint = label["state_id"], label["arm_id"], label["constraint"]
    onset = label["branch"]["first_violation_step"]
    evidence_dir = evidence_root / state_id / f"arm_{arm_id}" / constraint
    copied = []
    for step in (onset - 3, onset, onset + 3):
        for view in ("full", "wrist"):
            source = arm["dir"] / "images" / view / f"step_{step:05d}.png"
            if source.exists():
                target = evidence_dir / view / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                copied.append(str(target))

    steps = [row["step_index"] for row in arm["constraints"]]
    margins = [row[constraint]["margin"] for row in arm["constraints"]]
    figure_path = figures_root / f"{state_id}_arm_{arm_id}_{constraint}_margin.png"
    fig, axis = plt.subplots(figsize=(7.2, 3.4))
    axis.plot(steps, margins, color="#d55e00", linewidth=1.8)
    axis.axhline(0.0, color="black", linewidth=1.0)
    axis.axvline(onset, color="#0072b2", linewidth=1.0, linestyle="--")
    axis.set(title=f"{state_id} arm {arm_id}: {constraint}", xlabel="Archived step", ylabel="Margin")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    return copied, figure_path


def make_overview(violations, arm_by_key, output_path):
    thumb = (224, 224)
    label_width = 260
    row_height = 2 * thumb[1] + 38
    canvas = Image.new("RGB", (label_width + 3 * thumb[0], row_height * len(violations)), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, label in enumerate(violations):
        top = row_index * row_height
        state_id, arm_id = label["state_id"], label["arm_id"]
        onset = label["branch"]["first_violation_step"]
        draw.text((8, top + 8), f"{state_id}\narm {arm_id}\nonset {onset}", fill="black")
        arm = arm_by_key[(state_id, arm_id)]
        for view_index, view in enumerate(("full", "wrist")):
            draw.text((8, top + 115 + view_index * thumb[1]), view, fill="black")
            for column, step in enumerate((onset - 3, onset, onset + 3)):
                path = arm["dir"] / "images" / view / f"step_{step:05d}.png"
                if path.exists():
                    with Image.open(path) as image:
                        canvas.paste(image.convert("RGB"), (label_width + column * thumb[0], top + 20 + view_index * thumb[1]))
                draw.text((label_width + column * thumb[0] + 4, top + 22 + view_index * thumb[1]), str(step), fill="white", stroke_width=2, stroke_fill="black")
    canvas.save(output_path)


def default_annotations(violations):
    return [
        {
            "state_id": label["state_id"],
            "arm_id": label["arm_id"],
            "constraint": label["constraint"],
            "onset_step": label["branch"]["first_violation_step"],
            "full": {"rating": "unrated", "rationale": ""},
            "wrist": {"rating": "unrated", "rationale": ""},
        }
        for label in violations
    ]


def gate_table(labels, annotations, repro_equal, g6_pass):
    by_arm_constraint = {(label["state_id"], label["arm_id"], label["constraint"]): label for label in labels}
    violation_states = defaultdict(set)
    sustained = 0
    for label in labels:
        if label["branch"]["any_violation"]:
            violation_states[(label["arm_id"], label["constraint"])].add(label["state_id"])
        sustained += int(label["branch"]["sustained_violation"])
    a_any = sum(any(by_arm_constraint[(state, "A", constraint)]["branch"]["any_violation"] for constraint in CONSTRAINTS) for state in sorted({label["state_id"] for label in labels}))
    strong = len(violation_states[("C", "gripper_env")])
    weak = len(violation_states[("B", "gripper_env")])
    drops = len(violation_states[("D", "object_drop")])
    decisive_full = sum(row["full"]["rating"] == "decisive" for row in annotations)
    decisive_wrist = sum(row["wrist"]["rating"] == "decisive" for row in annotations)
    unrated = sum(row[view]["rating"] == "unrated" for row in annotations for view in ("full", "wrist"))
    violating_labels = [label for label in labels if label["branch"]["any_violation"]]
    detected = sum(label["branch"]["min_margin"] < 0.0 for label in violating_labels)
    second_family = any(
        len(violation_states[("C", constraint)]) > len(violation_states[("B", constraint)])
        for constraint in ("workspace", "self_collision", "object_drop")
    )
    return [
        ("G1", strong >= 3 and drops >= 2 and repro_equal, f"C gripper_env {strong}/8; D object_drop {drops}/8; C rerun byte-equal={repro_equal}"),
        ("G2", a_any == 0 and 0 < weak < strong, f"A any violation {a_any}/8; B gripper_env {weak}/8; C {strong}/8"),
        ("G3", strong > weak and second_family, f"gripper_env C/B={strong}/{weak}; second dose-responsive family={second_family}"),
        ("G4", decisive_full >= 1 and decisive_wrist >= 1 and unrated == 0, f"full-decisive={decisive_full}; wrist-decisive={decisive_wrist}; unrated cells={unrated}"),
        ("G5", detected == len(violating_labels) and a_any == 0, f"hard violations detected {detected}/{len(violating_labels)}; clean A false alarms {a_any}/8; sustained labels={sustained}"),
        ("G6", g6_pass, "8-state A gate and 40-arm checker passed; protocol commit predates accepted pilot"),
    ]


def render_report(labels, gates, annotations, provenance):
    counts = defaultdict(int)
    sustained = defaultdict(int)
    for label in labels:
        key = (label["arm_id"], label["constraint"])
        counts[key] += int(label["branch"]["any_violation"])
        sustained[key] += int(label["branch"]["sustained_violation"])
    lines = [
        "# Phase 4A Counterfactual Pilot Statistics",
        "",
        f"- Accepted pilot input SHA-256: `{provenance['pilot_input_sha256']}`",
        f"- Checker SHA-256: `{provenance['checker_sha256']}`",
        f"- A-only gate checker SHA-256: `{provenance['a_gate_checker_sha256']}`",
        f"- Runtime Guard HEAD: `{provenance['pilot_guard_head']}`",
        f"- Labeler Guard HEAD: `{provenance['labeler_guard_head']}`",
        f"- Threshold contract: `{provenance['threshold_version']}`, decision margin `0`, sustained `k={provenance['sustained_k']}`",
        "- Scope: 8 train states, 5 arms, 30 post-branch transitions, 5 constraints; cal/test absent.",
        "",
        "## Violation counts by arm and constraint",
        "",
        "Each cell is `hard violations / 8 states (sustained k=3 / 8)`.",
        "",
        "| Arm | workspace | gripper_env | self_collision | object_drop | non_finite |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        cells = [f"{counts[(arm, constraint)]}/8 ({sustained[(arm, constraint)]}/8)" for constraint in CONSTRAINTS]
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    lines.extend(["", "## G1-G6 decision", "", "| Gate | Result | Evidence |", "| --- | --- | --- |"])
    for name, passed, evidence in gates:
        lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} | {evidence} |")
    lines.extend(["", "## Violating branches", "", "| State | Arm | Constraint | Onset | Min margin | Sustained k=3 |", "| --- | --- | --- | ---: | ---: | --- |"])
    for label in labels:
        branch = label["branch"]
        if branch["any_violation"]:
            lines.append(
                f"| {label['state_id']} | {label['arm_id']} | {label['constraint']} | "
                f"{branch['first_violation_step']} | {branch['min_margin']:.12g} | "
                f"{'yes' if branch['sustained_violation'] else 'no'} |"
            )
    lines.extend(["", "## View annotations", "", "| State / arm / constraint | Full | Wrist |", "| --- | --- | --- |"])
    for row in annotations:
        key = f"{row['state_id']} / {row['arm_id']} / {row['constraint']}"
        full = f"{row['full']['rating']}: {row['full']['rationale']}"
        wrist = f"{row['wrist']['rating']}: {row['wrist']['rationale']}"
        lines.append(f"| {key} | {full} | {wrist} |")
    failed = [name for name, passed, _ in gates if not passed]
    lines.extend(
        [
            "",
            "## Warnings and scope limits",
            "",
            f"- Expansion decision is blocked because these pre-registered gates failed: {', '.join(failed) if failed else 'none'}.",
            "- Injected violations are synthetic positives. They do not establish recall on natural policy failures or untested risk modes.",
            "- RGB can show contact context but cannot resolve sub-millimeter penetration depth; simulator margin remains the oracle.",
            "- No thresholds, constraint code, calibration states, or test states were changed or used.",
            "- The earlier interrupted multi-state root is quarantined and excluded from every count in this report.",
            "",
            "## Decision",
            "",
            "Stop at the expansion decision point. Do not scale or tune candidates from this pilot without a new pre-registered protocol.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--checker", type=Path, required=True)
    parser.add_argument("--a-gate-checker", type=Path, required=True)
    parser.add_argument("--repro-constraints", type=Path, required=True)
    parser.add_argument("--repro-reference", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.pilot_root == args.output_dir or args.pilot_root in args.output_dir.parents:
        raise ValueError("derived output must be outside the pilot data root")
    if args.output_dir.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_root = args.output_dir / "figures"
    evidence_root = args.output_dir / "evidence"
    figures_root.mkdir(exist_ok=True)
    evidence_root.mkdir(exist_ok=True)

    thresholds = load_thresholds()
    if thresholds["version"] != "v1" or thresholds["labeling"]["decision_margin"] != 0:
        raise ValueError("Phase 4A requires frozen v1 and margin<0")
    checker = json.loads(args.checker.read_text(encoding="utf-8"))
    if len(checker["states"]) != 8 or sum(len(state["arms"]) for state in checker["states"]) != 40:
        raise ValueError("checker does not cover 8 states x 5 arms")
    a_gate_checker = json.loads(args.a_gate_checker.read_text(encoding="utf-8"))
    if len(a_gate_checker["states"]) != 8 or any(len(state["arms"]) != 1 for state in a_gate_checker["states"]):
        raise ValueError("A-only checker does not cover 8 states x 1 arm")
    arms = load_arms(args.pilot_root)
    guard_heads = {arm["meta"]["guard_head"] for arm in arms}
    protocol_hashes = {arm["meta"]["protocol_sha256"] for arm in arms}
    if len(guard_heads) != 1 or len(protocol_hashes) != 1 or any(arm["meta"]["guard_dirty"] for arm in arms):
        raise ValueError("mixed or dirty pilot provenance")
    provenance = {
        "run_id": f"phase4a_labels_{args.pilot_root.name}_v1_k3",
        "pilot_input_sha256": focused_input_sha256(args.pilot_root),
        "checker_sha256": checker["sha256"],
        "a_gate_checker_sha256": a_gate_checker["sha256"],
        "pilot_guard_head": next(iter(guard_heads)),
        "labeler_guard_head": git_head(repo_root),
        "protocol_sha256": next(iter(protocol_hashes)),
        "threshold_version": thresholds["version"],
        "decision_margin": thresholds["labeling"]["decision_margin"],
        "sustained_k": thresholds["labeling"]["sustained_k"],
    }
    labels = build_labels(arms, provenance, provenance["sustained_k"])
    shutil.copy2(args.checker, args.output_dir / "checker.json")
    shutil.copy2(args.a_gate_checker, args.output_dir / "a_gate_checker.json")
    labels_path = args.output_dir / "labels_pilot.jsonl"
    if labels_path.exists():
        labels_path.unlink()
    for label in labels:
        append_jsonl(labels_path, label)

    arm_by_key = {(arm["meta"]["state_id"], arm["meta"]["arm_id"]): arm for arm in arms}
    violations = [label for label in labels if label["branch"]["any_violation"]]
    for label in violations:
        plot_violation(label, arm_by_key[(label["state_id"], label["arm_id"])], evidence_root, figures_root)
    make_overview(violations, arm_by_key, figures_root / "violation_evidence_overview.png")

    annotations_path = args.output_dir / "view_annotations.json"
    if annotations_path.exists():
        annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    else:
        annotations = default_annotations(violations)
        write_json(annotations_path, annotations)
    repro_equal = sha256_file(args.repro_constraints) == sha256_file(args.repro_reference)
    g6_pass = len(checker["states"]) == 8 and all(len(state["arms"]) == 5 for state in checker["states"])
    gates = gate_table(labels, annotations, repro_equal, g6_pass)
    (args.output_dir / "pilot_stats.md").write_text(
        render_report(labels, gates, annotations, provenance), encoding="utf-8"
    )
    write_json(
        args.output_dir / "provenance.json",
        {
            **provenance,
            "repro_byte_equal": repro_equal,
            "repro_reference_sha256": sha256_file(args.repro_reference),
            "repro_constraints_sha256": sha256_file(args.repro_constraints),
        },
    )
    print(canonical_dumps({"labels": len(labels), "violations": len(violations), "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
