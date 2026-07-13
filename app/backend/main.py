"""FastAPI entrypoint for CT nodule analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.backend.agent import AnalysisAgent
from app.backend.config import ROOT_DIR, settings
from app.backend.constants import DISCLAIMER
from app.backend.compat import model_to_dict
from app.backend.detector import Detector
from app.backend.model_client import VLMClient
from app.backend.roi import extract_locked_roi, roi_contract
from app.backend.schemas import (
    AnalyzeNoduleResult,
    AnalyzeResponse,
    ClinicalInfo,
    DetectResponseItem,
    HealthResponse,
    NoduleCoord,
    ROIImages,
)
from app.backend.session_store import SessionStore
from app.backend.tools.registry import ToolRegistry
from app.backend.volume_io import resolve_ct_input, save_upload_files


frontend_dir = ROOT_DIR / "app" / "frontend"
settings.upload_dir.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="CT Nodule Chatbot", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")
app.mount("/uploads", StaticFiles(directory=str(settings.upload_dir)), name="uploads")

store = SessionStore(max_turns=settings.max_history_turns)
detector = Detector(settings)
tools = ToolRegistry()
agent = AnalysisAgent(VLMClient(settings), tools)

# 多轮对话：缓存每个 session 的 CT 路径，避免重复上传
_session_ct_cache: dict[str, Path] = {}


@app.get("/")
async def index():
    index_path = frontend_dir / "index.html"
    if not index_path.exists():
        return {"ok": True, "message": "Frontend not built yet."}
    return FileResponse(index_path)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        ok=True,
        model_configured=settings.model_configured,
        detector_configured=settings.detector_configured,
        detector_backend=settings.detector_backend,
        roi_contract=roi_contract(),
    )


def _parse_coords(raw: str | None) -> list[NoduleCoord]:
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="nodule_coords 必须是合法 JSON。") from exc
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise HTTPException(status_code=400, detail="nodule_coords 必须是对象或对象数组。")
    return [NoduleCoord(**item) for item in payload]


def _parse_clinical_info(raw: str | None) -> ClinicalInfo | None:
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="clinical_info 必须是合法 JSON。") from exc
    if not payload:
        return None
    return ClinicalInfo(**payload)


def _history_for_model(session_id: str) -> list[dict[str, str]]:
    return [{"role": m.role, "content": m.content} for m in store.get(session_id)]


@app.post("/detect", response_model=list[DetectResponseItem])
async def detect(
    ct_files: Annotated[list[UploadFile], File(description="CT volume")] = [],
    session_id: Annotated[str | None, Form()] = None,
):
    _, case_dir, ct_path = await _resolve_ct_and_session(ct_files, session_id)
    detections = detector.detect(ct_path)
    return [DetectResponseItem(**model_to_dict(d)) for d in detections]


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    ct_files: Annotated[list[UploadFile], File(description="CT volume，首轮必传")] = [],
    nodule_coords: Annotated[str | None, Form(description="JSON object or list with x/y/z world coordinates")] = None,
    clinical_info: Annotated[str | None, Form(description="Optional JSON clinical info")] = None,
    message: Annotated[str, Form()] = "请分析这个肺结节。",
    session_id: Annotated[str | None, Form()] = None,
    use_tools: Annotated[bool, Form()] = True,
):
    sid, case_dir, ct_path = await _resolve_ct_and_session(ct_files, session_id)
    coords = _parse_coords(nodule_coords)
    clinical = _parse_clinical_info(clinical_info)

    if not coords:
        coords = detector.detect(ct_path)

    store.append(sid, "user", message, {"coords": [model_to_dict(c) for c in coords]})

    results: list[AnalyzeNoduleResult] = []
    roi_dir = case_dir / "roi"
    for idx, coord in enumerate(coords, start=1):
        roi = extract_locked_roi(
            ct_path=ct_path,
            coord=coord,
            output_dir=roi_dir,
            nodule_idx=idx,
            seriesuid=case_dir.name,
        )
        report, tool_calls = await agent.analyze_nodule(
            roi_paths=roi.paths,
            coord=coord,
            clinical_info=clinical,
            history=_history_for_model(sid),
            user_message=message,
            use_tools=use_tools,
        )
        store.append(sid, "assistant", report, {"nodule_id": idx})

        results.append(
            AnalyzeNoduleResult(
                nodule_id=idx,
                coords=coord,
                diameter_mm=coord.diameter_mm,
                detection_confidence=coord.confidence,
                report=report,
                images=ROIImages(**roi.images_base64),
                tool_calls=tool_calls,
            )
        )

    return AnalyzeResponse(session_id=sid, results=results, disclaimer=DISCLAIMER)


async def _resolve_ct_and_session(
    ct_files: list[UploadFile],
    session_id: str | None,
) -> tuple[str, Path, Path]:
    """解析 CT 文件。支持多轮：首轮上传，后续复用缓存。返回 (sid, case_dir, ct_path)。"""
    sid = store.create_or_get(session_id)
    has_files = any(f.filename and f.size for f in ct_files)

    if has_files:
        case_dir, saved_files = await save_upload_files(ct_files, settings.upload_dir, settings.max_upload_mb)
        ct_path = resolve_ct_input(case_dir, saved_files)
        _session_ct_cache[sid] = ct_path
        return sid, case_dir, ct_path

    if sid not in _session_ct_cache:
        raise HTTPException(status_code=400, detail="首轮对话请先上传 CT 文件。")
    ct_path = _session_ct_cache[sid]
    case_dir = ct_path.parent
    return sid, case_dir, ct_path


@app.post("/analyze/stream")
async def analyze_stream(
    ct_files: Annotated[list[UploadFile], File(description="CT volume，首轮必传")] = [],
    nodule_coords: Annotated[str | None, Form(description="JSON world coordinates")] = None,
    clinical_info: Annotated[str | None, Form(description="Optional JSON clinical info")] = None,
    message: Annotated[str, Form()] = "请分析这个肺结节。",
    session_id: Annotated[str | None, Form()] = None,
    use_tools: Annotated[bool, Form()] = True,
):
    sid, case_dir, ct_path = await _resolve_ct_and_session(ct_files, session_id)
    coords = _parse_coords(nodule_coords)
    clinical = _parse_clinical_info(clinical_info)

    if not coords:
        coords = detector.detect(ct_path)
        if not coords:
            # 多轮对话：复用历史坐标
            for m in reversed(store.get(sid)):
                if m.metadata.get("coords"):
                    coords = [NoduleCoord(**c) for c in m.metadata["coords"]]
                    break

    store.append(sid, "user", message, {"coords": [model_to_dict(c) for c in coords] if coords else []})

    async def _event_stream():
        all_images: dict[int, dict] = {}
        all_tool_calls: dict[int, list] = {}
        full_text = ""
        last_text = ""
        coords_list = coords or []

        for idx, coord in enumerate(coords_list, start=1):
            yield f"data: {json.dumps({'type': 'status', 'text': f'正在生成 ROI (结节 {idx}/{len(coords_list)})...'}, ensure_ascii=False)}\n\n"

            roi_dir = case_dir / "roi"
            roi = extract_locked_roi(
                ct_path=ct_path,
                coord=coord,
                output_dir=roi_dir,
                nodule_idx=idx,
                seriesuid=case_dir.name,
            )

            yield f"data: {json.dumps({'type': 'status', 'text': '正在调用辅助工具...'}, ensure_ascii=False)}\n\n"

            text_stream, tool_calls = await agent.analyze_nodule_stream(
                roi_paths=roi.paths,
                coord=coord,
                clinical_info=clinical,
                history=_history_for_model(sid),
                user_message=message,
                use_tools=use_tools,
            )

            yield f"data: {json.dumps({'type': 'status', 'text': 'VLM 正在生成报告...'}, ensure_ascii=False)}\n\n"

            async for token in text_stream:
                full_text += token
                yield f"data: {json.dumps({'type': 'token', 'text': token}, ensure_ascii=False)}\n\n"

            last_text = full_text
            store.append(sid, "assistant", last_text, {"nodule_id": idx})
            all_images[idx] = roi.images_base64
            all_tool_calls[idx] = [model_to_dict(t) for t in tool_calls]

        yield f"data: {json.dumps({'type': 'done', 'session_id': sid, 'report': last_text, 'images': all_images, 'tool_calls': all_tool_calls, 'disclaimer': DISCLAIMER}, ensure_ascii=False)}\n\n"

    return StreamingResponse(_event_stream(), media_type="text/event-stream")


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    return {"session_id": session_id, "messages": [m.__dict__ for m in store.get(session_id)]}


@app.delete("/api/sessions/{session_id}")
async def clear_session(session_id: str):
    store.clear(session_id)
    return {"ok": True}


def main():
    uvicorn.run("app.backend.main:app", host=settings.app_host, port=settings.app_port, reload=False)


if __name__ == "__main__":
    main()
