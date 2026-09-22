"""Isolated MuJoCo 3.13.0 distance backend for Phase 6C calibration.

The exporter runs in the frozen project environment (MuJoCo 2.3.7).  The
worker is intentionally a small JSON-lines process that can be launched with
the separately pinned MuJoCo 3.13.0 interpreter; it performs no dynamics or
controller calls.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from guard.active_vision.continuous_nominal_calibration import (
    empty_state,
    geometry_signature,
    record_nominal,
)
from guard.active_vision.continuous_scene import ContinuousGeometryEnv
from guard.json_io import write_json


ROUTES = ("left_route", "center_route", "right_route")
LANES = (0.27, 0.0, -0.27)
WORKER = Path(__file__).with_name("continuous_nominal_backend_worker.py")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _copy_xml_assets(xml_text: str, xml_path: Path) -> tuple[str, dict[str, str]]:
    root = ET.fromstring(xml_text)
    asset_dir = xml_path.parent / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for element in root.iter():
        source = element.get("file")
        if not source or source.startswith("data:"):
            continue
        source_path = Path(source)
        if not source_path.is_absolute():
            source_path = (xml_path.parent / source_path).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"XML asset is missing: {source}")
        digest = sha256_file(source_path)[:16]
        name = f"{digest}_{source_path.name}"
        target = asset_dir / name
        if not target.exists():
            shutil.copy2(source_path, target)
        relative = str(Path("assets") / name)
        element.set("file", relative)
        copied[relative] = sha256_file(target)
    # The generated XML uses absolute meshdir only as a default for names;
    # all explicit assets above are now relative to the export directory.
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.set("meshdir", ".")
    return ET.tostring(root, encoding="unicode"), copied


def export_scene(destination: Path) -> dict:
    """Export exact compiled scene XML, assets, geometry metadata and qpos."""
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    env = ContinuousGeometryEnv(empty_state())
    env.reset()
    try:
        xml_text, assets = _copy_xml_assets(env.model.get_xml(), destination / "scene.xml")
        (destination / "scene.xml").write_text(xml_text)
        model = env.sim.model
        geom_ids = sorted(env.robot_geom_ids | set().union(*env.obstacle_geom_ids.values()))
        geometry = []
        for geom_id in geom_ids:
            body_id = int(model.geom_bodyid[geom_id])
            geometry.append({
                "name": model.geom_id2name(geom_id), "type": int(model.geom_type[geom_id]),
                "size": model.geom_size[geom_id].tolist(),
                "body": model.body_id2name(body_id),
            })
        robot_names = sorted(model.geom_id2name(i) for i in env.robot_geom_ids)
        obstacle_names = {
            route: sorted(model.geom_id2name(i) for i in env.obstacle_geom_ids[name])
            for route, name in zip(ROUTES, ("left", "center", "right"))
        }
        obstacle_bodies = {
            route: sorted({model.body_id2name(int(model.geom_bodyid[i])) for i in env.obstacle_geom_ids[name]})
            for route, name in zip(ROUTES, ("left", "center", "right"))
        }
        nominals = {}
        for route in ROUTES:
            env.reset()
            nominal = record_nominal(env, route)
            if not nominal["reached"]:
                raise RuntimeError(f"nominal route did not reach target: {route}")
            nominal["qpos_sha256"] = sha256_bytes(canonical(nominal["qpos"]))
            nominals[route] = nominal
        manifest = {
            "schema_version": 2,
            "source_mujoco_version": "2.3.7",
            "xml_sha256": sha256_file(destination / "scene.xml"),
            "assets": assets,
            "geometry_signature": geometry_signature(env),
            "geometry": geometry,
            "robot_geom_names": robot_names,
            "obstacle_geom_names": obstacle_names,
            "obstacle_body_names": obstacle_bodies,
            "route_lanes": dict(zip(ROUTES, LANES)),
            "nominals": nominals,
        }
        write_json(destination / "export_manifest.json", manifest)
        return manifest
    finally:
        env.close()


def run_worker(export_dir: Path, requests: list[dict], python: str) -> bytes:
    payload = "".join(json.dumps(request, sort_keys=True, separators=(",", ":")) + "\n" for request in requests)
    proc = subprocess.run(
        [python, str(WORKER), "--export", str(export_dir)], input=payload.encode(),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if proc.returncode:
        raise RuntimeError(f"distance worker failed ({proc.returncode}): {proc.stderr.decode()[-4000:]}")
    return proc.stdout


def backend_request(route: str, width_m: float, offset_m: float, qpos: list[list[float]], intervals: int = 8) -> dict:
    return {"route": route, "width_m": width_m, "offset_m": offset_m, "qpos": qpos, "intervals": intervals}
