"""Check the opened Phase 5B validation corpus without opening test."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from guard.active_vision.scene import CAMERAS, HIDDEN_STATES, ROUTES
from guard.json_io import write_json


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def finite(value):
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def check_corpus(corpus_root, manifest_path, protocol_path, split, expected_layouts):
    manifest_bytes = Path(manifest_path).read_bytes()
    protocol_bytes = Path(protocol_path).read_bytes()
    manifest = json.loads(manifest_bytes)
    summary = json.loads((Path(corpus_root) / "summary.json").read_text())
    expected = {row["layout_id"] for row in manifest["layouts"] if row["split"] == split}
    observed = {row["layout"]["layout_id"] for row in summary["layouts"]}
    errors = []
    for layout in summary["layouts"]:
        layout_id = layout["layout"]["layout_id"]
        states = {row["hidden_state"]: row for row in layout["states"]}
        if set(states) != set(HIDDEN_STATES):
            errors.append(f"{layout_id}: hidden states")
            continue
        v0 = set()
        public = set()
        for hidden_state in HIDDEN_STATES:
            state_dir = Path(corpus_root) / layout_id / hidden_state
            for name in ("routes.json", "queries.json", "meta.json"):
                if not (state_dir / name).is_file():
                    errors.append(f"{layout_id}/{hidden_state}: missing {name}")
            routes = json.loads((state_dir / "routes.json").read_text())
            queries = json.loads((state_dir / "queries.json").read_text())
            meta = json.loads((state_dir / "meta.json").read_text())
            if not finite(routes) or not finite(queries):
                errors.append(f"{layout_id}/{hidden_state}: non-finite")
            if len(queries) != 4 or len({row["state_hash"] for row in queries}) != 1:
                errors.append(f"{layout_id}/{hidden_state}: query provenance")
            if meta["branch_state_hash"] != queries[0]["state_hash"]:
                errors.append(f"{layout_id}/{hidden_state}: branch hash")
            v0.add(states[hidden_state]["image_sha256"]["v0"])
            public.add(json.dumps(states[hidden_state]["public_layout"], sort_keys=True))
            for camera in CAMERAS:
                path = state_dir / "images" / f"{camera}.png"
                image = np.asarray(Image.open(path))
                if sha256_bytes(image.tobytes()) != states[hidden_state]["image_sha256"][camera]:
                    errors.append(f"{layout_id}/{hidden_state}: {camera} hash")
            if len(routes) != len(ROUTES):
                errors.append(f"{layout_id}/{hidden_state}: routes")
        if len(v0) != 1 or len(public) != 1 or not layout["replay"]["exact"]:
            errors.append(f"{layout_id}: paired isolation/replay")
    scope = (
        summary["split"] == split
        and len(summary["layouts"]) == expected_layouts
        and observed == expected
        and summary["canonical_candidate_trajectories"] == expected_layouts * 8
        and summary["replay_audits"] == expected_layouts
    )
    provenance = (
        not summary["guard_dirty"]
        and summary["manifest_sha256"] == sha256_bytes(manifest_bytes)
        and summary["protocol_sha256"] == sha256_bytes(protocol_bytes)
    )
    result = {
        "schema_version": 2,
        "split": split,
        "scope_pass": scope,
        "provenance_pass": provenance,
        "integrity_pass": not errors,
        "all_pass": scope and provenance and not errors,
        "errors": errors,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus_root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = check_corpus(args.corpus_root, args.manifest, args.protocol, "validation", 20)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
    if not result["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
