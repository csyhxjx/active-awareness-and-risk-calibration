"""Collect the Phase 6B validation split after train-only fitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from guard.active_vision.collect_rgb_train import collect


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/belief_active_vision_rgb_v1/manifest.json"))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    result = collect(args.manifest, args.output, args.shard_index, args.shard_count, None, split="validation")
    print(json.dumps({key: result[key] for key in ("split", "layout_group_count", "record_count")}, indent=2))


if __name__ == "__main__":
    main()
