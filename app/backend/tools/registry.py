"""Runtime registry for skill-backed tools."""

from __future__ import annotations

from typing import Any

from app.backend.skills.guideline_retrieval.run import run as run_guideline
from app.backend.skills.image_metadata.run import run as run_image_metadata
from app.backend.skills.lung_rads_calculator.run import run as run_lung_rads
from app.backend.skills.web_search.run import run as run_web_search


class ToolRegistry:
    def __init__(self):
        self._tools = {
            "guideline_retrieval": run_guideline,
            "image_metadata": run_image_metadata,
            "lung_rads_calculator": run_lung_rads,
            "web_search": run_web_search,
        }

    def list_tools(self) -> list[str]:
        return sorted(self._tools.keys())

    def execute(self, name: str, payload: dict[str, Any]) -> Any:
        if name not in self._tools:
            return {"error": f"Unknown tool: {name}"}
        try:
            return self._tools[name](payload)
        except Exception as exc:  # keep agent resilient
            return {"error": str(exc), "tool": name}
