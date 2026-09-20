"""Create a hash-closed Phase 5E pre-test freeze."""

import argparse
import hashlib
import json
from pathlib import Path

from guard.json_io import write_json


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("freeze_root", type=Path); args = parser.parse_args()
    root = args.freeze_root.resolve(); repo = Path.cwd().resolve()
    files = ["active_vision_protocol_v4.md", "active_vision_protocol_v4_addendum.md", "configs/active_vision_v4/formal_manifest.json",
             "guard/active_vision/run_formal_v2.py", "guard/active_vision/check_test_v2.py", "guard/active_vision/evaluate_formal_v2.py",
             "guard/active_vision/evaluate_formal_v4.py"]
    selector_paths = sorted(root.glob("candidate_*.pt"))
    proposer_paths = sorted(root.glob("v0_instruction_geometry*.pt"))
    files += [str(path.relative_to(repo)) for path in selector_paths]
    files += [str((root / name).relative_to(repo)) for name in ("normalization.json", "training_summary.json", "validation_integrity.json")]
    freeze = {"schema_version": 4, "status": "frozen", "date": "2026-09-20", "best_fixed_camera": "v_left",
              "normalization": str((root / "normalization.json").relative_to(repo)),
              "aware_checkpoints": [str(path.relative_to(repo)) for path in sorted(root.glob("candidate_aware*.pt"))],
              "agnostic_checkpoints": [str(path.relative_to(repo)) for path in sorted(root.glob("candidate_agnostic*.pt"))],
              "proposer_checkpoints": [str(path.relative_to(repo)) for path in proposer_paths],
              "proposer_files": {str(path.relative_to(repo)): sha(path) for path in proposer_paths},
              "files": {name: sha(repo / name) for name in files}}
    write_json(root / "freeze.json", freeze)


if __name__ == "__main__": main()
