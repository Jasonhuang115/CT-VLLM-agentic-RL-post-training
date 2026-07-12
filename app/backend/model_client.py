"""VLM client for OpenAI-compatible multimodal endpoints."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import httpx

from app.backend.config import Settings
from app.backend.compat import model_to_dict
from app.backend.constants import DISCLAIMER, VLM_PROMPT
from app.backend.schemas import ClinicalInfo


def _image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class VLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def diagnose(
        self,
        roi_paths: dict[str, Path],
        clinical_info: ClinicalInfo | None = None,
        tools_context: list[dict[str, Any]] | None = None,
        history: list[dict[str, str]] | None = None,
        user_message: str | None = None,
    ) -> str:
        if self.settings.vlm_mock:
            return self._mock_report(clinical_info, tools_context)

        content: list[dict[str, Any]] = []
        for view in ("axial", "coronal", "sagittal"):
            content.append({"type": "image_url", "image_url": {"url": _image_data_url(roi_paths[view])}})

        prompt = VLM_PROMPT
        extra = self._build_extra_context(clinical_info, tools_context, history, user_message)
        if extra:
            prompt = f"{prompt}\n\n{extra}"
        content.append({"type": "text", "text": prompt})

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一名谨慎的胸部影像 AI 助手。基于给定的结节 ROI 多视图图像生成报告。"
                    "不要声称确诊，必须说明不确定性和需要医生结合原始影像确认。"
                ),
            },
            {"role": "user", "content": content},
        ]

        payload = {
            "model": self.settings.vlm_model,
            "messages": messages,
            "temperature": self.settings.vlm_temperature,
            "max_tokens": self.settings.vlm_max_tokens,
        }
        headers = {}
        if self.settings.vlm_api_key:
            headers["Authorization"] = f"Bearer {self.settings.vlm_api_key}"

        url = self.settings.vlm_api_base.rstrip("/") + "/chat/completions"
        async with httpx.AsyncClient(timeout=self.settings.vlm_timeout_seconds) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"VLM 返回格式异常: {data}") from exc

    def _build_extra_context(
        self,
        clinical_info: ClinicalInfo | None,
        tools_context: list[dict[str, Any]] | None,
        history: list[dict[str, str]] | None,
        user_message: str | None,
    ) -> str:
        parts: list[str] = []
        if clinical_info:
            fields = model_to_dict(clinical_info, exclude_none=True)
            if fields:
                parts.append(f"可选临床信息: {fields}")
        if user_message:
            parts.append(f"用户问题: {user_message}")
        if history:
            compact = history[-6:]
            parts.append(f"最近对话摘要: {compact}")
        if tools_context:
            parts.append(f"辅助工具结果: {tools_context}")
        return "\n".join(parts)

    def _mock_report(self, clinical_info: ClinicalInfo | None, tools_context: list[dict[str, Any]] | None) -> str:
        extra = ""
        if clinical_info:
            fields = model_to_dict(clinical_info, exclude_none=True)
            if fields:
                extra = f"\n\n临床信息已收到: {fields}"
        if tools_context:
            extra += f"\n\n已调用辅助工具 {len(tools_context)} 个。"
        return (
            "【影像所见】已生成三平面 ROI 图像。当前为 VLM_MOCK 模式，未调用真实模型。\n\n"
            "【风险评估】请配置 VLM_API_BASE、VLM_MODEL 后获取真实诊断报告。\n\n"
            "【建议】确认 ROI 坐标准确后调用 VLM 推理服务。"
            f"{extra}\n\n【免责声明】{DISCLAIMER}"
        )
