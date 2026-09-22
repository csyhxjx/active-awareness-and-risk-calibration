"""Compare one exported MuJoCo scene and identical geom pairs across engines.

This is a read-only attribution tool for the legacy left-route discrepancy. It
does not use either engine's selected minimum: every robot/target pair is
queried at the same interpolated qpos samples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


ROUTES = ("left_route", "center_route", "right_route")
OFFSETS = (-0.101, -0.100, -0.099)
WIDTH = 0.025
INTERVALS = 16


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load(path: Path):
    manifest = json.loads((path / "export_manifest.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(path / "scene.xml"))
    native_ccd_bit = getattr(getattr(mujoco, "mjtDisableBit", object()), "mjDSBL_NATIVECCD", None)
    if native_ccd_bit is not None:
        model.opt.disableflags &= ~int(native_ccd_bit)
    data = mujoco.MjData(model)
    return manifest, model, data


def geom_fingerprint(model, geom_id: int) -> dict:
    mesh_id = int(model.geom_dataid[geom_id]) if int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_MESH) else -1
    mesh_name = model.mesh(mesh_id).name if mesh_id >= 0 else None
    body_id = int(model.geom_bodyid[geom_id])
    return {
        "name": model.geom(geom_id).name,
        "type": int(model.geom_type[geom_id]),
        "size": model.geom_size[geom_id].tolist(),
        "pos": model.geom_pos[geom_id].tolist(),
        "quat": model.geom_quat[geom_id].tolist(),
        "body": model.body(body_id).name,
        "mesh_id": mesh_id,
        "mesh_name": mesh_name,
        "contype": int(model.geom_contype[geom_id]),
        "conaffinity": int(model.geom_conaffinity[geom_id]),
    }


def configure(model, manifest, route: str, offset: float, width: float) -> list[int]:
    obstacle_names = manifest["obstacle_geom_names"][route]
    obstacle_body_names = manifest["obstacle_body_names"][route]
    for name in obstacle_names:
        gid = model.geom(name).id
        model.geom_size[gid, 1] = width
    lane = float(manifest["route_lanes"][route])
    for name in obstacle_body_names:
        bid = model.body(name).id
        model.body_pos[bid] = [0.070, lane + offset, 0.960]
    for other in ROUTES:
        if other == route:
            continue
        for name in manifest["obstacle_body_names"][other]:
            bid = model.body(name).id
            model.body_pos[bid] = [2.0, 0.0, 0.960]
    if not hasattr(mujoco, "mj_geomDistance"):
        for gid in range(model.ngeom):
            if gid in {model.geom(name).id for name in manifest["robot_geom_names"]} or any(
                gid in {model.geom(name).id for name in names}
                for names in manifest["obstacle_geom_names"].values()
            ):
                model.geom_margin[gid] = 0.25
                model.geom_gap[gid] = 0.0
    return [model.geom(name).id for name in obstacle_names]


def pair_rows(model, data, robot_ids: list[int], obstacle_ids: list[int], qposes: np.ndarray) -> list[dict]:
    rows = []
    fromto = np.empty(6, dtype=np.float64)
    for step in range(len(qposes)):
        samples = (0,) if step == 0 else range(INTERVALS + 1)
        for sample in samples:
            fraction = 0.0 if step == 0 else sample / INTERVALS
            pose = qposes[0] if step == 0 else (1.0 - fraction) * qposes[step - 1] + fraction * qposes[step]
            data.qpos[:] = pose
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            for robot_id in robot_ids:
                for obstacle_id in obstacle_ids:
                    exact_api = hasattr(mujoco, "mj_geomDistance")
                    if exact_api:
                        distance = float(mujoco.mj_geomDistance(model, data, robot_id, obstacle_id, 0.25, fromto))
                    else:
                        # MuJoCo 2.3.7 has no Python mj_geomDistance binding.
                        # Its only comparable observable is the contact stream;
                        # configure a 0.25 m proximity margin and retain the
                        # pair identity. Missing contacts are right-censored.
                        distance = None
                        for contact in data.contact[: data.ncon]:
                            pair = {int(contact.geom1), int(contact.geom2)}
                            if pair == {robot_id, obstacle_id}:
                                distance = float(contact.dist)
                                break
                        if distance is None:
                            distance = 0.25
                    rows.append({
                        "step": step, "sample": sample, "fraction": fraction,
                        "qpos_sha256": digest(pose.tolist()),
                        "robot_geom": model.geom(robot_id).name,
                        "obstacle_geom": model.geom(obstacle_id).name,
                        "distance_m": distance,
                        "negative": distance < 0.0,
                        "right_censored": not exact_api and distance == 0.25,
                        "robot_world_pos": data.geom_xpos[robot_id].tolist(),
                        "robot_world_xmat": data.geom_xmat[robot_id].tolist(),
                        "obstacle_world_pos": data.geom_xpos[obstacle_id].tolist(),
                        "obstacle_world_xmat": data.geom_xmat[obstacle_id].tolist(),
                    })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", type=Path, required=True)
    ap.add_argument("--route", choices=ROUTES, default="left_route")
    ap.add_argument("--engine", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    manifest, model, data = load(args.export)
    native_ccd_bit = getattr(getattr(mujoco, "mjtDisableBit", object()), "mjDSBL_NATIVECCD", None)
    nominal = manifest["nominals"][args.route]
    qposes = np.asarray(nominal["qpos"], dtype=np.float64)
    robot_names = manifest["robot_geom_names"]
    robot_ids = [model.geom(name).id for name in robot_names]
    model_fingerprint = [geom_fingerprint(model, gid) for gid in robot_ids]
    all_rows = []
    for offset in OFFSETS:
        obstacle_ids = configure(model, manifest, args.route, offset, WIDTH)
        mujoco.mj_forward(model, data)
        all_rows.append({
            "offset_m": offset,
            "obstacle_geometry": [geom_fingerprint(model, gid) for gid in obstacle_ids],
            "rows": pair_rows(model, data, robot_ids, obstacle_ids, qposes),
        })
    payload = {
        "schema_version": 1,
        "engine": args.engine,
        "mujoco_version": mujoco.__version__,
        "native_ccd_enabled": (
            None if native_ccd_bit is None
            else not bool(model.opt.disableflags & int(native_ccd_bit))
        ),
        "export_xml_sha256": hashlib.sha256((args.export / "scene.xml").read_bytes()).hexdigest(),
        "route": args.route,
        "width_m": WIDTH,
        "intervals": INTERVALS,
        "qpos_sha256": nominal["qpos_sha256"],
        "robot_geometry": model_fingerprint,
        "samples": all_rows,
    }
    args.output.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
