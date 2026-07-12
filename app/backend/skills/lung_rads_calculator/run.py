"""Lung-RADS and nodule measurement calculator skill."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from agent.tools.measurement_calc import MeasurementCalculator


def run(payload: dict[str, Any]) -> dict[str, Any]:
    calc = MeasurementCalculator()
    function = payload.get("function", "lung_rads_classify")
    kwargs = {k: v for k, v in payload.items() if k != "function"}
    if function == "lung_rads_classify":
        return calc.lung_rads_classify(**kwargs)
    if function == "volume_doubling_time":
        return calc.volume_doubling_time(**kwargs)
    if function == "malignancy_probability":
        return calc.malignancy_probability(**kwargs)
    return {"error": f"Unknown calculator function: {function}"}


if __name__ == "__main__":
    payload = json.loads(sys.stdin.read() or "{}")
    print(json.dumps(run(payload), ensure_ascii=False, indent=2))
