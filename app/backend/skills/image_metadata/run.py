"""ROI PNG metadata skill."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def run(payload: dict[str, Any]) -> dict[str, Any]:
    results = {}
    for view, raw_path in payload.get("paths", {}).items():
        path = Path(raw_path)
        with Image.open(path) as image:
            arr = np.asarray(image)
            results[view] = {
                "path": str(path),
                "size": image.size,
                "mode": image.mode,
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "min": int(arr.min()),
                "max": int(arr.max()),
            }
    return results


if __name__ == "__main__":
    payload = json.loads(sys.stdin.read() or "{}")
    print(json.dumps(run(payload), ensure_ascii=False, indent=2))
