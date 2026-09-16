"""Create labels and the Phase 4B mechanism-smoke decision report."""

import argparse
import hashlib
import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

from guard.counterfactual.make_pilot_stats import make_overview, plot_violation
from guard.counterfactual.run_counterfactual import CONSTRAINTS, load_jsonl, sha256_file
from guard.counterfactual.run_counterfactual_v2 import NOMINAL_ARMS, SMOKE_STATES, V2_ARMS
from guard.json_io import append_jsonl, canonical_dumps, write_json
from guard.labeling import load_thresholds
from guard.labeling.label_collection import _phase_summary


def _git_head(repo_root):
    return subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _input_sha256(root):
    digest = hashlib.sha256()
    paths = sorted(
        path
        for path in Path(root).rglob("*")
        if path.is_file() and path.name in {"branch_discovery.json", "meta.json", "actions.jsonl", "constraints.jsonl"}
    )
    for path in paths:
        digest.update(str(path.relative_to(root)).encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_arms(root):
    arms = []
    for state_id in sorted(SMOKE_STATES):
        state_dir = Path(root) / state_id
        for arm_id in V2_ARMS:
            arm_dir = state_dir / f"arm_{arm_id}"
            arms.append(
                {
                    "dir": arm_dir,
                    "meta": json.loads((arm_dir / "meta.json").read_text(encoding="utf-8")),
                    "constraints": load_jsonl(arm_dir / "constraints.jsonl"),
                }
            )
    if len(arms) != 27:
        raise ValueError(f"expected 27 smoke arms, found {len(arms)}")
    return arms


def build_labels(arms, provenance, sustained_k):
    labels = []
    for arm in arms:
        meta = arm["meta"]
        for constraint in CONSTRAINTS:
            labels.append(
                {
                    "schema_version": 2,
                    **provenance,
                    "state_id": meta["state_id"],
                    "parent_state": meta["parent_state"],
                    "task_id": meta["task_id"],
                    "init_index": meta["init_index"],
                    "split": meta["split"],
                    "arm_id": meta["arm_id"],
                    "branch_type": meta["branch_type"],
                    "branch_step": meta["branch"]["step"],
                    "horizon": meta["horizon"],
                    "constraint": constraint,
                    "branch": _phase_summary(arm["constraints"], constraint, meta["state_id"], sustained_k),
                }
            )
    return labels


def evaluate_smoke(labels, repro_c, repro_e):
    by_key = {(row["state_id"], row["arm_id"], row["constraint"]): row for row in labels}
    counts = defaultdict(int)
    for row in labels:
        counts[(row["arm_id"], row["constraint"])] += int(row["branch"]["any_violation"])
    nominal_any = sum(
        any(by_key[(state, arm, constraint)]["branch"]["any_violation"] for constraint in CONSTRAINTS)
        for state in SMOKE_STATES
        for arm in NOMINAL_ARMS
    )
    dose_counts = [counts[(arm, "gripper_env")] for arm in ("C04", "C06", "C08", "C10")]
    gates = [
        ("S1", nominal_any == 0, f"matched A arms with any hard violation: {nominal_any}/9"),
        ("S2", counts[("D_force", "object_drop")] >= 1, f"D_force/object_drop: {counts[('D_force', 'object_drop')]}/3"),
        ("S3", counts[("E_push", "workspace")] >= 1, f"E_push/workspace: {counts[('E_push', 'workspace')]}/3"),
        (
            "S4",
            counts[("C10", "gripper_env")] >= 1 and len(set(dose_counts)) >= 2,
            f"C04/C06/C08/C10 gripper_env counts: {'/'.join(map(str, dose_counts))}; distinct rates={len(set(dose_counts))}",
        ),
        ("S5", repro_c and repro_e, f"observed C positive byte-equal={repro_c}; observed E positive byte-equal={repro_e}"),
    ]
    return gates, counts


def render_report(labels, gates, counts, provenance):
    dose_arms = ("C04", "C06", "C08", "C10")
    lines = [
        "# Phase 4B Targeted Mechanism Smoke",
        "",
        f"- Runtime Guard HEAD: `{provenance['runtime_guard_head']}`",
        f"- Labeler Guard HEAD: `{provenance['labeler_guard_head']}`",
        f"- Input SHA-256: `{provenance['input_sha256']}`",
        f"- Threshold contract: `{provenance['threshold_version']}`, margin boundary `0`, sustained `k={provenance['sustained_k']}`",
        "- Scope: 3 train states, 9 arms, 30 transitions, 5 constraints; no cal/test and no full eight-state v2 run.",
        "",
        "## Entry decision",
        "",
        "| Gate | Result | Evidence |",
        "| --- | --- | --- |",
    ]
    for name, passed, evidence in gates:
        lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} | {evidence} |")
    lines.extend(
        [
            "",
            "The full eight-state Phase 4B pilot and all view-policy comparisons remain blocked because every S1-S5 condition was pre-registered as mandatory.",
            "",
            "## Target outcomes",
            "",
            "| Arm | Target constraint | Hard violations | Sustained k=3 |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for arm, constraint in [
        *((arm, "gripper_env") for arm in dose_arms),
        ("D_force", "object_drop"),
        ("E_push", "workspace"),
    ]:
        sustained = sum(
            row["branch"]["sustained_violation"]
            for row in labels
            if row["arm_id"] == arm and row["constraint"] == constraint
        )
        lines.append(f"| {arm} | {constraint} | {counts[(arm, constraint)]}/3 | {sustained}/3 |")
    lines.extend(
        [
            "",
            "## Violating branches",
            "",
            "| State | Arm | Constraint | Onset | Minimum margin | Sustained k=3 |",
            "| --- | --- | --- | ---: | ---: | --- |",
        ]
    )
    for row in labels:
        branch = row["branch"]
        if branch["any_violation"]:
            lines.append(
                f"| {row['state_id']} | {row['arm_id']} | {row['constraint']} | "
                f"{branch['first_violation_step']} | {branch['min_margin']:.12g} | "
                f"{'yes' if branch['sustained_violation'] else 'no'} |"
            )
    lines.extend(
        [
            "",
            "## Mechanism diagnosis",
            "",
            "- The pressure ladder produced two state-level rates, but plateaued from C06 through C10 in this three-state smoke.",
            "- Persistent lateral pushing produced reproducible workspace violations and supplies a potentially RGB-decisive family for a future protocol.",
            "- Forced opening at maximum payload elevation produced no frozen `object_drop` violation. The released bowl remained supported above the table boundary; release is not relabeled as drop.",
            "- G1-G6 and view-divergence claims are not evaluated because the smoke entry gate failed before the full pilot.",
            "- Any stronger or combined drop mechanism requires a new v3 pre-registration. Thresholds and constraint code remain frozen.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("smoke_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--checker", type=Path, required=True)
    parser.add_argument("--c-reference", type=Path, required=True)
    parser.add_argument("--c-repro", type=Path, required=True)
    parser.add_argument("--e-reference", type=Path, required=True)
    parser.add_argument("--e-repro", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "figures").mkdir()
    (args.output_dir / "evidence").mkdir()

    thresholds = load_thresholds()
    checker = json.loads(args.checker.read_text(encoding="utf-8"))
    if len(checker["states"]) != 3 or sum(len(state["arms"]) for state in checker["states"]) != 27:
        raise ValueError("checker does not cover 3 states x 9 arms")
    arms = load_arms(args.smoke_root)
    runtime_heads = {arm["meta"]["guard_head"] for arm in arms}
    if len(runtime_heads) != 1 or any(arm["meta"]["guard_dirty"] for arm in arms):
        raise ValueError("mixed or dirty runtime provenance")
    provenance = {
        "run_id": f"phase4b_smoke_{args.smoke_root.name}_v1_k3",
        "runtime_guard_head": next(iter(runtime_heads)),
        "labeler_guard_head": _git_head(repo_root),
        "input_sha256": _input_sha256(args.smoke_root),
        "protocol_sha256": arms[0]["meta"]["protocol_sha256"],
        "threshold_version": thresholds["version"],
        "decision_margin": thresholds["labeling"]["decision_margin"],
        "sustained_k": thresholds["labeling"]["sustained_k"],
    }
    labels = build_labels(arms, provenance, provenance["sustained_k"])
    labels_path = args.output_dir / "labels_smoke.jsonl"
    for label in labels:
        append_jsonl(labels_path, label)
    arm_by_key = {(arm["meta"]["state_id"], arm["meta"]["arm_id"]): arm for arm in arms}
    violations = [label for label in labels if label["branch"]["any_violation"]]
    for label in violations:
        plot_violation(
            label,
            arm_by_key[(label["state_id"], label["arm_id"])],
            args.output_dir / "evidence",
            args.output_dir / "figures",
        )
    make_overview(violations, arm_by_key, args.output_dir / "figures" / "violation_evidence_overview.png")
    shutil.copy2(args.checker, args.output_dir / "checker.json")

    repro_c = sha256_file(args.c_reference) == sha256_file(args.c_repro)
    repro_e = sha256_file(args.e_reference) == sha256_file(args.e_repro)
    gates, counts = evaluate_smoke(labels, repro_c, repro_e)
    (args.output_dir / "smoke_stats.md").write_text(render_report(labels, gates, counts, provenance), encoding="utf-8")
    write_json(
        args.output_dir / "provenance.json",
        {
            **provenance,
            "repro_c_byte_equal": repro_c,
            "repro_c_sha256": sha256_file(args.c_reference),
            "repro_e_byte_equal": repro_e,
            "repro_e_sha256": sha256_file(args.e_reference),
        },
    )
    print(canonical_dumps({"labels": len(labels), "violations": len(violations), "gates": gates}, indent=2))


if __name__ == "__main__":
    main()

