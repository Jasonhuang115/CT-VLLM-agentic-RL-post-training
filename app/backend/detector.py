"""Optional nodule detector interface.

Manual coordinates are the primary path. Automatic detection is deliberately
pluggable because detector weights and preprocessing differ across deployments.
"""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

from fastapi import HTTPException

from app.backend.config import Settings
from app.backend.schemas import NoduleCoord


class Detector:
    def __init__(self, settings: Settings):
        self.settings = settings

    def detect(self, ct_path: Path) -> list[NoduleCoord]:
        if self.settings.detector_backend == "disabled":
            raise HTTPException(
                status_code=422,
                detail="未提供手动结节坐标，且自动检测器未启用。请在前端手动输入/点选坐标，或配置 DETECTOR_BACKEND。",
            )

        if self.settings.detector_backend == "command":
            return self._detect_with_command(ct_path)

        raise HTTPException(status_code=500, detail=f"未知检测器后端: {self.settings.detector_backend}")

    def _detect_with_command(self, ct_path: Path) -> list[NoduleCoord]:
        if not self.settings.detector_command:
            raise HTTPException(status_code=500, detail="DETECTOR_COMMAND 未配置。")

        command = shlex.split(self.settings.detector_command) + [str(ct_path)]
        try:
            proc = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.settings.detector_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(status_code=504, detail=f"检测器超时: {exc}") from exc

        if proc.returncode != 0:
            raise HTTPException(
                status_code=502,
                detail=f"检测器执行失败: {proc.stderr.strip() or proc.stdout.strip()}",
            )

        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=502, detail="检测器输出不是合法 JSON。") from exc

        if not isinstance(payload, list):
            raise HTTPException(status_code=502, detail="检测器输出必须是 nodule list。")

        return [NoduleCoord(**item) for item in payload]
