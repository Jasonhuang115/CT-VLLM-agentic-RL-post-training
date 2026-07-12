"""Lightweight deterministic agent orchestration."""

from __future__ import annotations

from typing import Any

from app.backend.model_client import VLMClient
from app.backend.schemas import ClinicalInfo, NoduleCoord, ToolCall
from app.backend.tools.registry import ToolRegistry


class AnalysisAgent:
    def __init__(self, model_client: VLMClient, tools: ToolRegistry):
        self.model_client = model_client
        self.tools = tools

    async def analyze_nodule(
        self,
        roi_paths,
        coord: NoduleCoord,
        clinical_info: ClinicalInfo | None,
        history: list[dict[str, str]],
        user_message: str,
        use_tools: bool = True,
    ) -> tuple[str, list[ToolCall]]:
        tool_calls: list[ToolCall] = []
        if use_tools:
            tool_calls = self._run_tools(coord, clinical_info, user_message)

        tools_context: list[dict[str, Any]] = [
            {"name": call.name, "input": call.input, "output": call.output} for call in tool_calls
        ]
        report = await self.model_client.diagnose(
            roi_paths=roi_paths,
            clinical_info=clinical_info,
            tools_context=tools_context,
            history=history,
            user_message=user_message,
        )
        return report, tool_calls

    def _run_tools(self, coord: NoduleCoord, clinical_info: ClinicalInfo | None, user_message: str) -> list[ToolCall]:
        calls: list[ToolCall] = []

        guideline_query = "肺结节 随访 Lung-RADS Fleischner"
        guideline = self.tools.execute("guideline_retrieval", {"query": guideline_query})
        calls.append(self._tool_call("guideline_retrieval", {"query": guideline_query}, guideline))

        if coord.diameter_mm:
            calc_input = {
                "function": "lung_rads_classify",
                "nodule_type": "solid",
                "size_mm": coord.diameter_mm,
                "is_baseline": True,
            }
            calc = self.tools.execute("lung_rads_calculator", calc_input)
            calls.append(self._tool_call("lung_rads_calculator", calc_input, calc))

        lower = user_message.lower()
        should_search = any(
            token in lower
            for token in ["最新", "文献", "研究", "指南更新", "2024", "2025", "2026", "latest", "pubmed"]
        )
        if should_search:
            query = user_message or "pulmonary nodule latest guideline"
            search = self.tools.execute("web_search", {"query": query, "max_results": 3})
            calls.append(self._tool_call("web_search", {"query": query, "max_results": 3}, search))

        return calls

    def _tool_call(self, name: str, payload: dict[str, Any], output: Any) -> ToolCall:
        preview = str(output)
        if len(preview) > 500:
            preview = preview[:500] + "..."
        return ToolCall(name=name, input=payload, output=output, output_preview=preview)
