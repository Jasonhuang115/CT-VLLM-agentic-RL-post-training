#!/usr/bin/env python3
"""
VLM 推理服务 — 加载 Qwen2.5-VL-3B + LoRA adapter，提供 OpenAI 兼容 API。

支持 Mac (MPS) 和 Linux (CUDA)，自动检测设备。
端口 8000，/v1/chat/completions。
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel
from threading import Thread

# ── 设备检测 ──────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.float16
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    DTYPE = torch.float16
else:
    DEVICE = "cpu"
    DTYPE = torch.float32

print(f"[VLM SERVER] 设备: {DEVICE}")

# ── 常量 ──────────────────────────────────────────────
# 自动检测本地路径（Mac 优先），否则回退到 HF
_LOCAL_BASE = "/Users/huangzs/.cache/huggingface/hub/models/qwen--Qwen2.5-VL-3B-Instruct/snapshots/master"
_LOCAL_ADAPTER = "/Users/huangzs/Downloads/outputs/stage1_mv_cn/lora_adapter"
DEFAULT_BASE_MODEL = _LOCAL_BASE if os.path.isdir(_LOCAL_BASE) else "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_ADAPTER = _LOCAL_ADAPTER if os.path.isdir(_LOCAL_ADAPTER) else "huang01080524/lungct-nodule"
DEFAULT_PORT = 8000
DEFAULT_MAX_TOKENS = 800
DEFAULT_TIMEOUT = 120

app = FastAPI(title="LungCT-VLM", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局模型引用
_model = None
_processor = None


# ── Pydantic models ────────────────────────────────────
class ChatMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]]


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.0
    max_tokens: int = DEFAULT_MAX_TOKENS
    stream: bool = False


# ── 模型加载 ──────────────────────────────────────────
def load_model(base_model: str, adapter: str | None):
    """加载 base model + 可选 LoRA adapter，适配 Mac/CUDA。"""
    global _model, _processor

    print(f"[VLM SERVER] 加载基础模型: {base_model}")
    t0 = time.time()

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    # Mac MPS 上 4-bit 不可用，用 FP16；CUDA 可以用 4-bit 省显存
    if DEVICE == "cuda":
        try:
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                base_model,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            print("[VLM SERVER] 使用 4-bit 量化 (CUDA)")
        except Exception as e:
            print(f"[VLM SERVER] 4-bit 加载失败 ({e})，回退到 FP16")
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                base_model,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
    else:
        # Mac MPS / CPU
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model,
            torch_dtype=DTYPE,
            device_map=DEVICE if DEVICE == "cpu" else None,
            trust_remote_code=True,
        )
        if DEVICE == "mps":
            model = model.to(DEVICE)
        print(f"[VLM SERVER] 使用 {DTYPE} ({DEVICE})")

    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)

    # 加载 LoRA adapter
    if adapter:
        print(f"[VLM SERVER] 加载 LoRA adapter: {adapter}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
        print("[VLM SERVER] LoRA 已合并")

    model.eval()
    _model = model
    _processor = processor

    elapsed = time.time() - t0
    print(f"[VLM SERVER] 模型加载完成 ({elapsed:.1f}s)")


# ── 图片解码 ──────────────────────────────────────────
def _decode_image(item: dict) -> Image.Image | None:
    """从 OpenAI content item 解码图片。支持 data: URL 和 URL。"""
    image_url = item.get("image_url", {})
    url = image_url.get("url", "") if isinstance(image_url, dict) else ""

    if not url:
        return None

    if url.startswith("data:"):
        # data:image/png;base64,xxxx
        try:
            header, b64 = url.split(",", 1)
        except ValueError:
            return None
        data = base64.b64decode(b64)
        return Image.open(io.BytesIO(data)).convert("RGB")

    # HTTP/文件路径 — 这里只处理本地路径
    if os.path.exists(url):
        return Image.open(url).convert("RGB")

    return None


# ── 推理（共享输入准备）────────────────────────────────
def _prepare_inputs(messages: list[dict]) -> tuple[dict, list[Image.Image]]:
    """解析 messages，返回 (processor_inputs, images_list)。"""
    if _model is None or _processor is None:
        raise RuntimeError("模型未加载")

    images: list[Image.Image] = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "image_url":
                    img = _decode_image(item)
                    if img:
                        images.append(img)

    # 构建 conversations（Qwen2.5-VL 格式）
    conversations = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            converted = []
            for item in content:
                if item.get("type") == "text":
                    converted.append({"type": "text", "text": item.get("text", "")})
                elif item.get("type") == "image_url":
                    converted.append({"type": "image"})
            conversations.append({"role": role, "content": converted})
        else:
            conversations.append({"role": role, "content": [{"type": "text", "text": str(content)}]})

    text = _processor.apply_chat_template(
        conversations, tokenize=False, add_generation_prompt=True,
    )

    if images:
        inputs = _processor(text=[text], images=images, return_tensors="pt")
    else:
        inputs = _processor(text=[text], return_tensors="pt")

    inputs = {k: v.to(_model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    return inputs, images


@torch.no_grad()
def run_inference(
    messages: list[dict],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
) -> str:
    """非流式推理，返回完整文本。"""
    inputs, _ = _prepare_inputs(messages)

    generated_ids = _model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature if temperature > 0 else None,
        do_sample=temperature > 0,
        use_cache=True,
    )

    input_len = inputs["input_ids"].shape[1]
    generated_ids = generated_ids[:, input_len:]

    output_text = _processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0]
    return output_text.strip()


def run_inference_stream(
    messages: list[dict],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
):
    """流式推理生成器，逐 token yield 文本片段。"""
    from transformers import TextIteratorStreamer

    inputs, _ = _prepare_inputs(messages)

    streamer = TextIteratorStreamer(
        _processor.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=180,
    )

    gen_kwargs = {
        **inputs,
        "max_new_tokens": max_tokens,
        "temperature": temperature if temperature > 0 else None,
        "do_sample": temperature > 0,
        "use_cache": True,
        "streamer": streamer,
    }

    thread = Thread(target=_model.generate, kwargs=gen_kwargs)
    thread.start()

    for new_text in streamer:
        if new_text:
            yield new_text

    thread.join()


# ── API 路由 ──────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "device": DEVICE,
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "lungct-nodule",
                "object": "model",
                "owned_by": "huang01080524",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    try:
        messages = [m.model_dump() for m in req.messages]
    except Exception:
        messages = [{"role": m.role, "content": m.content} for m in req.messages]

    # ── 流式 ──
    if req.stream:
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        def _generate_sse():
            try:
                for token in run_inference_stream(
                    messages=messages,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                ):
                    payload = json.dumps({
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {"content": token}}],
                    }, ensure_ascii=False)
                    yield f"data: {payload}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as exc:
                import traceback
                traceback.print_exc()
                err = json.dumps({"error": str(exc)}, ensure_ascii=False)
                yield f"data: {err}\n\n"

        return StreamingResponse(_generate_sse(), media_type="text/event-stream")

    # ── 非流式 ──
    t0 = time.time()
    try:
        output = run_inference(
            messages=messages,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))

    elapsed = time.time() - t0

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": output},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "_elapsed_s": round(elapsed, 1),
    }


@app.get("/")
async def root():
    return {"service": "lungct-nodule-vlm", "device": DEVICE}


# ── 入口 ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="LungCT Nodule VLM Server")
    parser.add_argument("--model", default=DEFAULT_BASE_MODEL, help="基础模型路径/HF名")
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER, help="LoRA adapter 路径/HF名")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--no-adapter", action="store_true", help="不加载 LoRA adapter")
    args = parser.parse_args()

    adapter = None if args.no_adapter else args.adapter
    load_model(args.model, adapter)

    print(f"\n[VLM SERVER] 启动 → http://{args.host}:{args.port}")
    print("[VLM SERVER] API → /v1/chat/completions")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
