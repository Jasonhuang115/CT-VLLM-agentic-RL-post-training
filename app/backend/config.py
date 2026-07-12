"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_host: str = os.getenv("APP_HOST", "0.0.0.0")
    app_port: int = int(os.getenv("APP_PORT", "8080"))
    upload_dir: Path = Path(os.getenv("UPLOAD_DIR", str(ROOT_DIR / "app" / "uploads")))
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "512"))
    max_history_turns: int = int(os.getenv("MAX_HISTORY_TURNS", "12"))

    vlm_api_base: str = os.getenv("VLM_API_BASE", "http://127.0.0.1:8000/v1")
    vlm_api_key: str = os.getenv("VLM_API_KEY", "")
    vlm_model: str = os.getenv("VLM_MODEL", "huang01080524/lungct-nodule-grpo")
    vlm_timeout_seconds: int = int(os.getenv("VLM_TIMEOUT_SECONDS", "120"))
    vlm_max_tokens: int = int(os.getenv("VLM_MAX_TOKENS", "800"))
    vlm_temperature: float = float(os.getenv("VLM_TEMPERATURE", "0"))
    vlm_mock: bool = _bool_env("VLM_MOCK", default=False)

    detector_backend: str = os.getenv("DETECTOR_BACKEND", "disabled")
    detector_command: str = os.getenv("DETECTOR_COMMAND", "")
    detector_timeout_seconds: int = int(os.getenv("DETECTOR_TIMEOUT_SECONDS", "60"))

    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")

    @property
    def model_configured(self) -> bool:
        return self.vlm_mock or bool(self.vlm_api_base and self.vlm_model)

    @property
    def detector_configured(self) -> bool:
        if self.detector_backend == "disabled":
            return False
        if self.detector_backend == "command":
            return bool(self.detector_command)
        return False


settings = Settings()
