"""Compare the original 2.3.7 proximity fixture with exported XML at one qpos."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from guard.active_vision.continuous_nominal_calibration import make_environment


ROUTE = "left_route"
OFFSET = -0.100
WIDTH = 0.025
INTERVALS = 16


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def model_geom(model, gid):
    name = model.geom_id2name(gid) if hasattr(model, "geom_id2name") else model.geom(gid).name
    return {
        "name": name,
        "type": int(model.geom_type[gid]),
        "size": model.geom_size[gid].tolist(),
        "margin_m": float(model.geom_margin[gid]),
        "gap_m": float(model.geom_gap[gid]),
        "contype": int(model.geom_contype[gid]),
        "conaffinity": int(model.geom_conaffinity[gid]),
        "body": (model.body_id2name(int(model.geom_bodyid[gid])) if hasattr(model, "body_id2name") else model.body(int(model.geom_bodyid[gid])).name),
    }


def model_collision_fingerprint(model):
    fields = ("geom_priority", "geom_condim", "geom_friction", "geom_solref", "geom_solimp",
              "geom_group", "geom_sameframe", "geom_matid", "geom_dataid", "geom_bodyid")
    result = {}
    for field in fields:
        value = np.asarray(getattr(model, field))
        result[field] = {"sha256": hashlib.sha256(value.tobytes()).hexdigest(), "shape": list(value.shape)}
    return result


def rows_old(env, qposes, robot_names, obstacle_name):
    model, data = env.sim.model, env.sim.data
    robot_ids = [model.geom_name2id(name) for name in robot_names]
    obstacle_id = model.geom_name2id(obstacle_name)
    for step in range(len(qposes)):
        for sample in ((0,) if step == 0 else range(INTERVALS + 1)):
            fraction = 0.0 if step == 0 else sample / INTERVALS
            pose = qposes[0] if step == 0 else (1.0 - fraction) * qposes[step - 1] + fraction * qposes[step]
            data.qpos[:] = pose
            data.qvel[:] = 0
            env.sim.forward()
            for contact in data.contact[: data.ncon]:
                a, b = int(contact.geom1), int(contact.geom2)
                if obstacle_id not in (a, b):
                    continue
                robot_id = b if a == obstacle_id else a
                if robot_id not in robot_ids:
                    continue
                dist = float(contact.dist)
                if dist < 0:
                    return {"step": step, "sample": sample, "fraction": fraction,
                            "qpos": pose.tolist(), "qpos_sha256": digest(pose.tolist()),
                            "robot_geom": model.geom_id2name(robot_id),
                            "obstacle_geom": obstacle_name, "distance_m": dist,
                            "contact_index": int(np.where(data.contact == contact)[0][0]) if False else None,
                            "robot_world_pos": data.geom_xpos[robot_id].tolist(),
                            "obstacle_world_pos": data.geom_xpos[obstacle_id].tolist()}
    return None


def export_model(path: Path):
    manifest = json.loads((path / "export_manifest.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(path / "scene.xml"))
    data = mujoco.MjData(model)
    return manifest, model, data


def configure_export(model, manifest):
    obstacle = manifest["obstacle_geom_names"][ROUTE][0]
    obstacle_id = model.geom(obstacle).id
    model.geom_size[obstacle_id, 1] = WIDTH
    body = model.body(manifest["obstacle_body_names"][ROUTE][0]).id
    model.body_pos[body] = [0.070, manifest["route_lanes"][ROUTE] + OFFSET, 0.960]
    for other in ("center_route", "right_route"):
        for name in manifest["obstacle_body_names"][other]:
            model.body_pos[model.body(name).id] = [2.0, 0.0, 0.960]
    ids = [model.geom(name).id for name in manifest["robot_geom_names"]]
    for gid in ids + [obstacle_id]:
        model.geom_margin[gid] = 0.25
        model.geom_gap[gid] = 0.0
    return ids, obstacle_id


def export_contacts(model, data, robot_ids, obstacle_id, qposes):
    for step in range(len(qposes)):
        for sample in ((0,) if step == 0 else range(INTERVALS + 1)):
            fraction = 0.0 if step == 0 else sample / INTERVALS
            pose = qposes[0] if step == 0 else (1.0 - fraction) * qposes[step - 1] + fraction * qposes[step]
            data.qpos[:] = pose
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            for contact in data.contact[: data.ncon]:
                a, b = int(contact.geom1), int(contact.geom2)
                if obstacle_id not in (a, b):
                    continue
                robot_id = b if a == obstacle_id else a
                if robot_id not in robot_ids or float(contact.dist) >= 0:
                    continue
                return {"step": step, "sample": sample, "fraction": fraction,
                        "qpos": pose.tolist(), "qpos_sha256": digest(pose.tolist()),
                        "robot_geom": model.geom(robot_id).name,
                        "obstacle_geom": model.geom(obstacle_id).name,
                        "distance_m": float(contact.dist),
                        "robot_world_pos": data.geom_xpos[robot_id].tolist(),
                        "obstacle_world_pos": data.geom_xpos[obstacle_id].tolist()}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", type=Path, required=True)
    ap.add_argument("--qpos-json", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.qpos_json is None:
        qposes = np.asarray(json.loads((args.export / "export_manifest.json").read_text())["nominals"][ROUTE]["qpos"], dtype=np.float64)
    else:
        qposes = np.asarray(json.loads(args.qpos_json.read_text()), dtype=np.float64)
    env = make_environment(ROUTE)
    try:
        # Set the same world parameters as the export-side replay.
        from guard.active_vision.continuous_nominal_calibration import set_target_obstacle
        set_target_obstacle(env, ROUTE, OFFSET)
        robot_names = [env.sim.model.geom_id2name(i) for i in sorted(env.robot_geom_ids)]
        obstacle_name = env.sim.model.geom_id2name(sorted(env.obstacle_geom_ids["left"])[0])
        old_first = rows_old(env, qposes, robot_names, obstacle_name)
        old_compile = {"robot": [model_geom(env.sim.model, i) for i in sorted(env.robot_geom_ids)],
                       "obstacle": [model_geom(env.sim.model, i) for i in sorted(env.obstacle_geom_ids["left"])],
                       "collision_arrays": model_collision_fingerprint(env.sim.model),
                       "ngeom": env.sim.model.ngeom, "neq": env.sim.model.neq, "nexclude": env.sim.model.nexclude}
    finally:
        env.close()
    manifest, model, data = export_model(args.export)
    robot_ids, obstacle_id = configure_export(model, manifest)
    new_first = export_contacts(model, data, robot_ids, obstacle_id, qposes)
    new_compile = {"robot": [model_geom(model, i) for i in robot_ids], "obstacle": [model_geom(model, obstacle_id)],
                   "collision_arrays": model_collision_fingerprint(model),
                   "ngeom": model.ngeom, "neq": model.neq, "nexclude": model.nexclude}
    report = {
        "schema_version": 1, "status": "MISSING" if old_first and not new_first else "PASS",
        "engine": "mujoco_2.3.7_both_sides", "route": ROUTE, "offset_m": OFFSET, "width_m": WIDTH,
        "qpos_sha256": digest(qposes.tolist()), "qpos_row_count": len(qposes),
        "fixture_first_negative_contact": old_first, "export_first_negative_contact": new_first,
        "fixture_compile": old_compile, "export_compile": new_compile,
        "comparison": {
            "same_qpos": old_first is None or new_first is None or old_first["qpos_sha256"] == new_first["qpos_sha256"],
            "same_pair": old_first is None or new_first is None or (old_first["robot_geom"], old_first["obstacle_geom"]) == (new_first["robot_geom"], new_first["obstacle_geom"]),
            "fixture_has_negative_contact": old_first is not None,
            "export_has_negative_contact": new_first is not None,
            "missing_conditions": ["export fixture has no negative contact at the identical sampled qpos"] if old_first and not new_first else [],
        },
        "interpretation": "MISSING: this isolates the old-engine fixture/export contact-query difference but does not establish which compiled setting is causal until compile and runtime fields are compared field-by-field.",
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
