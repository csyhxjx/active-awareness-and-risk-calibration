"""Fresh-process replay entrypoint for a frozen Phase 6B detector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from guard.active_vision.rgb_detector import RGBDetector
from guard.json_io import canonical_dumps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--roi", type=Path, required=True)
    args = parser.parse_args()
    detector = RGBDetector.load(args.model)
    image = np.asarray(Image.open(args.roi).convert("RGB"), dtype=np.uint8)
    print(canonical_dumps(detector.predict({"camera_id": args.camera, "roi_rgb": image}), sort_keys=True))


if __name__ == "__main__":
    main()
