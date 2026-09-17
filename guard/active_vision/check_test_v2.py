"""Check the single sealed Phase 5B test corpus against its frozen inputs."""

import argparse
import json
from pathlib import Path

from guard.active_vision.check_validation_v2 import check_corpus
from guard.active_vision.run_formal_v2 import valid_freeze
from guard.json_io import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus_root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("freeze", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = check_corpus(args.corpus_root, args.manifest, args.protocol, "test", 40)
    result["freeze_pass"] = valid_freeze(args.freeze)
    result["all_pass"] = result["all_pass"] and result["freeze_pass"]
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
    if not result["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
