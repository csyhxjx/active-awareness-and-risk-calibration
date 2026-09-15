"""Canonical JSON encoding that is immune to ambient json.dumps patches."""

from pathlib import Path
import json


def _default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def canonical_dumps(value, *, indent=None, sort_keys=False):
    return json.JSONEncoder(indent=indent, sort_keys=sort_keys, default=_default).encode(value)


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_dumps(record) + "\n")


def write_json(path, value, *, indent=2, trailing_newline=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = "\n" if trailing_newline else ""
    path.write_text(canonical_dumps(value, indent=indent) + suffix, encoding="utf-8")
