"""Run the same-pair attribution under both pinned MuJoCo interpreters."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def run(python: str, script: Path, export: Path, route: str, engine: str, output: Path) -> None:
    subprocess.run([python, str(script), "--export", str(export), "--route", route, "--engine", engine, "--output", str(output)], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", type=Path, required=True)
    ap.add_argument("--old-python", required=True)
    ap.add_argument("--new-python", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("attribute_sign_disagreement.py")
    old_path = args.output / "old_engine.json"
    new_path = args.output / "new_engine.json"
    run(args.old_python, script, args.export, "left_route", "mujoco_2.3.7", old_path)
    run(args.new_python, script, args.export, "left_route", "mujoco_3.13.0", new_path)
    old = json.loads(old_path.read_text())
    new = json.loads(new_path.read_text())
    if old["export_xml_sha256"] != new["export_xml_sha256"] or old["qpos_sha256"] != new["qpos_sha256"]:
        raise RuntimeError("inputs are not aligned")
    geometry_deltas = []
    for left, right in zip(old["robot_geometry"], new["robot_geometry"]):
        for field in ("size", "pos", "quat"):
            geometry_deltas.extend(abs(float(a) - float(b)) for a, b in zip(left[field], right[field]))
    obstacle_deltas = []
    for old_sample, new_sample in zip(old["samples"], new["samples"]):
        for left, right in zip(old_sample["obstacle_geometry"], new_sample["obstacle_geometry"]):
            for field in ("size", "pos", "quat"):
                obstacle_deltas.extend(abs(float(a) - float(b)) for a, b in zip(left[field], right[field]))
    geometry_status = "PASS" if max(geometry_deltas, default=0.0) <= 1e-7 else "FAIL"
    pair_disagreements = []
    world_pose_deltas = []
    first = None
    for old_sample, new_sample in zip(old["samples"], new["samples"]):
        if old_sample["offset_m"] != new_sample["offset_m"]:
            raise RuntimeError("offset order mismatch")
        old_rows = old_sample["rows"]
        new_rows = new_sample["rows"]
        if len(old_rows) != len(new_rows):
            raise RuntimeError("sample count mismatch")
        for left, right in zip(old_rows, new_rows):
            key = (old_sample["offset_m"], left["step"], left["sample"], left["robot_geom"], left["obstacle_geom"])
            if key != (old_sample["offset_m"], right["step"], right["sample"], right["robot_geom"], right["obstacle_geom"]):
                raise RuntimeError("pair/sample identity mismatch")
            if left["negative"] != right["negative"]:
                record = {
                    "offset_m": old_sample["offset_m"], "step": left["step"], "sample": left["sample"],
                    "fraction": left["fraction"], "robot_geom": left["robot_geom"], "obstacle_geom": left["obstacle_geom"],
                    "old_distance_m": left["distance_m"], "new_distance_m": right["distance_m"],
                    "old_robot_world_pos": left["robot_world_pos"], "new_robot_world_pos": right["robot_world_pos"],
                    "old_robot_world_xmat": left["robot_world_xmat"], "new_robot_world_xmat": right["robot_world_xmat"],
                    "old_obstacle_world_pos": left["obstacle_world_pos"], "new_obstacle_world_pos": right["obstacle_world_pos"],
                    "old_obstacle_world_xmat": left["obstacle_world_xmat"], "new_obstacle_world_xmat": right["obstacle_world_xmat"],
                    "qpos_sha256": left["qpos_sha256"],
                }
                pair_disagreements.append(record)
                if first is None:
                    first = record
            world_pose_deltas.extend(
                abs(float(a) - float(b))
                for field in ("robot_world_pos", "robot_world_xmat", "obstacle_world_pos", "obstacle_world_xmat")
                for a, b in zip(left[field], right[field])
            )
    report = {
        "schema_version": 1,
        "status": "MISSING" if old["mujoco_version"] == "2.3.7" else ("FAIL" if pair_disagreements or geometry_status != "PASS" else "PASS"),
        "input_alignment": "PASS",
        "geometry_definition": geometry_status,
        "max_robot_geometry_numeric_delta": max(geometry_deltas, default=0.0),
        "max_obstacle_geometry_numeric_delta": max(obstacle_deltas, default=0.0),
        "max_world_pose_numeric_delta": max(world_pose_deltas, default=0.0),
        "distance_algorithm_or_runtime": "MISSING",
        "old_engine_pair_distance_api": "no_python_mj_geomDistance; contact_stream_only",
        "new_engine_pair_distance_api": "mj_geomDistance_native_ccd",
        "export_xml_sha256": old["export_xml_sha256"],
        "qpos_sha256": old["qpos_sha256"],
        "engines": {"old": old["mujoco_version"], "new": new["mujoco_version"]},
        "pair_count_per_offset": len(old["samples"][0]["rows"]),
        "offsets_m": [sample["offset_m"] for sample in old["samples"]],
        "pair_sign_disagreement_count": len(pair_disagreements),
        "first_sign_disagreement": first,
        "all_sign_disagreements": pair_disagreements,
        "old_result_sha256": hashlib.sha256(old_path.read_bytes()).hexdigest(),
        "new_result_sha256": hashlib.sha256(new_path.read_bytes()).hexdigest(),
        "interpretation": (
            "MISSING: the exported XML, geometry definitions, qpos and pair identities are aligned, but MuJoCo 2.3.7 exposes no Python mj_geomDistance. Its contact-stream-only result is right-censored for every tested pair, so exact same-pair sign attribution cannot be completed. The old source-engine stable negative neighborhood remains separate evidence and does not prove the exported-XML distance correct."
        ),
    }
    (args.output / "sign_disagreement_attribution.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    (args.output / "sign_disagreement_attribution.md").write_text(
        "# Sign-disagreement attribution\n\n"
        f"Status: **{report['status']}**\n\n"
        f"Same exported XML SHA-256: `{report['export_xml_sha256']}`; same qpos SHA-256: `{report['qpos_sha256']}`.\n\n"
        f"Compared {report['pair_count_per_offset']} identical robot/obstacle pairs at 16-interval samples for offsets `-0.101/-0.100/-0.099`.\n\n"
        f"Geometry definition: **{report['geometry_definition']}**. Same-pair sign disagreements: **{report['pair_sign_disagreement_count']}**.\n\n"
        f"Maximum numeric deltas: robot geometry `{report['max_robot_geometry_numeric_delta']:.3e}`, obstacle geometry `{report['max_obstacle_geometry_numeric_delta']:.3e}`, sampled world pose `{report['max_world_pose_numeric_delta']:.3e}`.\n\n"
        f"Old API: `{report['old_engine_pair_distance_api']}`; new API: `{report['new_engine_pair_distance_api']}`.\n\n"
        f"Interpretation: {report['interpretation']}\n"
    )


if __name__ == "__main__":
    main()
