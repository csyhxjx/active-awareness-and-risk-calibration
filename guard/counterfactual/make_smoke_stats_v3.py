"""Create Phase 4C labels, evidence, and the terminal smoke report."""

import argparse
import hashlib
import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

from guard.counterfactual.make_pilot_stats import make_overview, plot_violation
from guard.counterfactual.run_counterfactual import CONSTRAINTS, load_jsonl, sha256_file
from guard.counterfactual.run_counterfactual_v3 import SMOKE_STATES, V3_ARMS
from guard.json_io import append_jsonl, canonical_dumps, write_json
from guard.labeling import load_thresholds
from guard.labeling.label_collection import _phase_summary


def git_head(repo_root):
    return subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def focused_input_sha256(*roots):
    digest = hashlib.sha256()
    for root in roots:
        root = Path(root)
        for path in sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.name in {"branch_discovery.json", "meta.json", "actions.jsonl", "constraints.jsonl"}
        ):
            digest.update(root.name.encode("utf-8") + b"\0")
            digest.update(str(path.relative_to(root)).encode("utf-8") + b"\0")
            digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def load_arms(a_root, f_root):
    arms = []
    for state_id in SMOKE_STATES:
        for arm_id, root in (("A_edge", a_root), ("F_edge", f_root)):
            arm_dir = Path(root) / state_id / f"arm_{arm_id}"
            arms.append(
                {
                    "dir": arm_dir,
                    "meta": json.loads((arm_dir / "meta.json").read_text(encoding="utf-8")),
                    "constraints": load_jsonl(arm_dir / "constraints.jsonl"),
                }
            )
    return arms


def build_labels(arms, provenance, sustained_k):
    labels = []
    for arm in arms:
        meta = arm["meta"]
        for constraint in CONSTRAINTS:
            labels.append(
                {
                    "schema_version": 3,
                    **provenance,
                    "state_id": meta["state_id"],
                    "parent_state": meta["parent_state"],
                    "task_id": meta["task_id"],
                    "init_index": meta["init_index"],
                    "split": meta["split"],
                    "arm_id": meta["arm_id"],
                    "branch_step": meta["branch"]["step"],
                    "horizon": meta["horizon"],
                    "constraint": constraint,
                    "branch": _phase_summary(arm["constraints"], constraint, meta["state_id"], sustained_k),
                }
            )
    return labels


def evaluate_gates(labels, arms, annotations, checker):
    by_key = {(row["state_id"], row["arm_id"], row["constraint"]): row for row in labels}
    meta = {(arm["meta"]["state_id"], arm["meta"]["arm_id"]): arm["meta"] for arm in arms}
    a_any = sum(
        any(by_key[(state, "A_edge", name)]["branch"]["any_violation"] for name in CONSTRAINTS)
        for state in SMOKE_STATES
    )
    mechanism_states = [
        state
        for state in SMOKE_STATES
        if meta[(state, "F_edge")]["edge_crossed"]
        and meta[(state, "F_edge")]["release_triggered"]
        and meta[(state, "F_edge")]["target_min_drop_margin"] < 0
        and by_key[(state, "F_edge", "object_drop")]["branch"]["any_violation"]
    ]
    eligible_annotations = [
        row
        for row in annotations
        if row["state_id"] in mechanism_states and row["constraint"] == "object_drop"
    ]
    decisive = sum(
        row[view]["rating"] == "decisive"
        for row in eligible_annotations
        for view in ("full", "wrist")
    )
    nonfinite = sum(
        by_key[(state, arm, "non_finite")]["branch"]["any_violation"]
        for state in SMOKE_STATES
        for arm in V3_ARMS
    )
    isolation = all(row["split"] == "train" for row in labels) and nonfinite == 0
    branch_match = bool(checker.get("branch_provenance_match")) and len(checker.get("a_states", [])) == 3
    return [
        ("S1", a_any == 0 and branch_match, f"A arms with any violation: {a_any}/3; A/F branch provenance match={branch_match}"),
        ("S2", len(mechanism_states) >= 1, f"edge-cross + release + target/object drop: {len(mechanism_states)}/3"),
        ("S3", decisive >= 1, f"eligible RGB-decisive ratings: {decisive}; eligible true-positive cases={len(eligible_annotations)}"),
        ("S4", isolation, f"train-only={all(row['split'] == 'train' for row in labels)}; non-finite arms={nonfinite}/6"),
        ("S5", False, "not testable because no positive edge-drop state exists"),
    ]


def render_report(labels, arms, gates, annotations, provenance):
    counts = defaultdict(int)
    sustained = defaultdict(int)
    for row in labels:
        key = (row["arm_id"], row["constraint"])
        counts[key] += int(row["branch"]["any_violation"])
        sustained[key] += int(row["branch"]["sustained_violation"])
    meta = {(arm["meta"]["state_id"], arm["meta"]["arm_id"]): arm["meta"] for arm in arms}
    lines = [
        "# Phase 4C Edge-Drop Mechanism Smoke",
        "",
        f"- Runtime Guard HEAD: `{provenance['runtime_guard_head']}`",
        f"- Labeler Guard HEAD: `{provenance['labeler_guard_head']}`",
        f"- Input SHA-256: `{provenance['input_sha256']}`",
        f"- A checker SHA-256: `{provenance['a_checker_sha256']}`",
        f"- Cross checker SHA-256: `{provenance['checker_sha256']}`",
        f"- Threshold contract: `{provenance['threshold_version']}`, sustained `k={provenance['sustained_k']}`",
        "- Scope: 3 train states, A_edge + F_edge, 60 transitions, 5 constraints; no cal/test, scaling, dose extension, or method comparison.",
        "",
        "## Terminal S-gate decision",
        "",
        "| Gate | Result | Evidence |",
        "| --- | --- | --- |",
    ]
    for name, passed, evidence in gates:
        lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} | {evidence} |")
    lines.extend(
        [
            "",
            "S2, S3, and S5 fail. Under the pre-registered terminal fork, mechanism search closes and the proposed view-policy comparison is not run.",
            "",
            "## Arm x constraint matrix",
            "",
            "Each cell is `hard / 3 (sustained k=3 / 3)`.",
            "",
            "| Arm | workspace | gripper_env | self_collision | object_drop | non_finite |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for arm in V3_ARMS:
        cells = [f"{counts[(arm, name)]}/3 ({sustained[(arm, name)]}/3)" for name in CONSTRAINTS]
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Edge-drop mechanism outcomes",
            "",
            "| State | Branch | Edge crossed | Crossing | Release | Release start | Target min drop margin | Object min margin |",
            "| --- | ---: | --- | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    by_key = {(row["state_id"], row["arm_id"], row["constraint"]): row for row in labels}
    for state in SMOKE_STATES:
        row = meta[(state, "F_edge")]
        object_margin = by_key[(state, "F_edge", "object_drop")]["branch"]["min_margin"]
        lines.append(
            f"| {state} | {row['branch']['step']} | {'yes' if row['edge_crossed'] else 'no'} | "
            f"{row['crossing_step'] if row['crossing_step'] is not None else '-'} | "
            f"{'yes' if row['release_triggered'] else 'no'} | "
            f"{row['release_start_step'] if row['release_start_step'] is not None else '-'} | "
            f"{row['target_min_drop_margin']:.12g} | {object_margin:.12g} |"
        )
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
    lines.extend(["", "## View ratings", "", "| State / event | Full | Wrist | S3 eligible |", "| --- | --- | --- | --- |"])
    for row in annotations:
        lines.append(
            f"| {row['state_id']} / {row['event']} | "
            f"{row['full']['rating']}: {row['full']['rationale']} | "
            f"{row['wrist']['rating']}: {row['wrist']['rationale']} | "
            f"{'yes' if row['s3_eligible'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Terminal interpretation",
            "",
            "- Two F arms crossed the selected physical table edge and triggered forced release; neither target bowl fell below the frozen drop line.",
            "- The protected F arm did not cross and therefore correctly never opened. Its incidental sub-millimeter gripper_env event does not satisfy the edge-drop gate.",
            "- Diagnostic RGB supplies partial manipulation context only. With no true-positive object_drop arm, no frame is eligible to satisfy the decisive observation gate.",
            "- This is a synthetic mechanism test, not a natural-failure recall estimate. Thresholds, constraints, calibration, and test remain untouched.",
            "- The suite currently has no demonstrated RGB-decisive positive family for the proposed view-selection decision. Redirect the thesis scope away from that claim; do not run the method comparison.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a_root", type=Path)
    parser.add_argument("f_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--a-checker", type=Path, required=True)
    parser.add_argument("--checker", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    figures = args.output_dir / "figures"
    evidence = args.output_dir / "evidence"
    figures.mkdir()
    evidence.mkdir()

    thresholds = load_thresholds()
    arms = load_arms(args.a_root, args.f_root)
    runtime_heads = {arm["meta"]["guard_head"] for arm in arms}
    protocol_hashes = {arm["meta"]["protocol_sha256"] for arm in arms}
    if len(runtime_heads) != 1 or len(protocol_hashes) != 1 or any(arm["meta"]["guard_dirty"] for arm in arms):
        raise ValueError("mixed or dirty runtime provenance")
    checker = json.loads(args.checker.read_text(encoding="utf-8"))
    a_checker = json.loads(args.a_checker.read_text(encoding="utf-8"))
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    provenance = {
        "run_id": f"phase4c_v3_{args.f_root.name}_v1_k3",
        "runtime_guard_head": next(iter(runtime_heads)),
        "labeler_guard_head": git_head(repo_root),
        "input_sha256": focused_input_sha256(args.a_root, args.f_root),
        "a_checker_sha256": sha256_file(args.a_checker),
        "checker_sha256": sha256_file(args.checker),
        "protocol_sha256": next(iter(protocol_hashes)),
        "threshold_version": thresholds["version"],
        "decision_margin": thresholds["labeling"]["decision_margin"],
        "sustained_k": thresholds["labeling"]["sustained_k"],
    }
    labels = build_labels(arms, provenance, provenance["sustained_k"])
    labels_path = args.output_dir / "labels_smoke.jsonl"
    for row in labels:
        append_jsonl(labels_path, row)
    arm_by_key = {(arm["meta"]["state_id"], arm["meta"]["arm_id"]): arm for arm in arms}
    violations = [row for row in labels if row["branch"]["any_violation"]]
    for row in violations:
        plot_violation(row, arm_by_key[(row["state_id"], row["arm_id"])], evidence, figures)
    make_overview(violations, arm_by_key, figures / "violation_evidence_overview.png")
    crossing_events = []
    for arm in arms:
        meta = arm["meta"]
        if meta["arm_id"] == "F_edge" and meta["release_triggered"]:
            crossing_events.append(
                {
                    "state_id": meta["state_id"],
                    "arm_id": "F_edge",
                    "constraint": "edge_release_diagnostic",
                    "branch": {"first_violation_step": meta["release_start_step"]},
                }
            )
    make_overview(crossing_events, arm_by_key, figures / "edge_release_evidence_overview.png")
    gates = evaluate_gates(labels, arms, annotations, checker)
    (args.output_dir / "smoke_stats.md").write_text(
        render_report(labels, arms, gates, annotations, provenance), encoding="utf-8"
    )
    shutil.copy2(args.a_checker, args.output_dir / "a_checker.json")
    shutil.copy2(args.checker, args.output_dir / "checker.json")
    shutil.copy2(args.annotations, args.output_dir / "view_annotations.json")
    write_json(args.output_dir / "provenance.json", provenance)
    print(canonical_dumps({"labels": len(labels), "violations": len(violations), "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
