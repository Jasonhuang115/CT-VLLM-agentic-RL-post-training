"""Pydantic schemas shared by the API and backend services."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class NoduleCoord(BaseModel):
    x: float
    y: float
    z: float
    diameter_mm: Optional[float] = None
    confidence: Optional[float] = None


class ClinicalInfo(BaseModel):
    age: Optional[int] = None
    smoking_history: Optional[str] = None
    family_history: Optional[bool] = None
    notes: Optional[str] = None


class ToolCall(BaseModel):
    name: str
    input: dict[str, Any] = Field(default_factory=dict)
    output: Any = None
    output_preview: str = ""


class ROIImages(BaseModel):
    axial: str
    coronal: str
    sagittal: str


class AnalyzeNoduleResult(BaseModel):
    nodule_id: int
    coords: NoduleCoord
    diameter_mm: Optional[float] = None
    detection_confidence: Optional[float] = None
    report: str
    images: ROIImages
    tool_calls: list[ToolCall] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    session_id: str
    results: list[AnalyzeNoduleResult]
    disclaimer: str


class DetectResponseItem(BaseModel):
    x: float
    y: float
    z: float
    diameter_mm: Optional[float] = None
    confidence: Optional[float] = None


class HealthResponse(BaseModel):
    ok: bool
    model_configured: bool
    detector_configured: bool
    detector_backend: str
    roi_contract: dict[str, Any]
