"""Phase 6C policy runner entrypoint guarded by the physical preflight result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def require_preflight(check_path: Path) -> dict:
    result = json.loads(check_path.read_text())
    if result.get("gates", {}).get("C0_physical_truth") != "PASS":
        raise RuntimeError("Phase 6C C0 physical preflight did not pass; policy runner is locked")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-check", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_preflight(args.preflight_check)
    raise NotImplementedError("policy execution requires a passing preflight root; none exists for probe v1")


if __name__ == "__main__":
    main()
