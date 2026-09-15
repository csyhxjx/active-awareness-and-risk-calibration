"""Offline constraint labeling and reporting."""

from importlib.resources import files
import json


def load_thresholds():
    resource = files("guard.labeling").joinpath("thresholds.json")
    with resource.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("version") != "v1":
        raise ValueError(f"unsupported threshold version: {payload.get('version')!r}")
    return payload


__all__ = ["load_thresholds"]
