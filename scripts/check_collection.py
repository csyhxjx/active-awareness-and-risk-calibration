"""Validate per-state collection artifacts and report measured disk use."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guard.collection import validate_episode_dir


parser = argparse.ArgumentParser()
parser.add_argument("collection_root")
parser.add_argument("state_ids", nargs="*")
args = parser.parse_args()

root = Path(args.collection_root)
episode_dirs = [root / state_id for state_id in args.state_ids] if args.state_ids else sorted(
    path for path in root.iterdir() if path.is_dir()
)
results = [validate_episode_dir(path) for path in episode_dirs]
total_bytes = sum(result["bytes"] for result in results)
print(json.dumps({"episodes": results, "total_bytes": total_bytes}, indent=2))
