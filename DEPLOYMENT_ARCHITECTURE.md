# 肺结节CT诊断系统 — 部署架构

## 系统总览

```
┌─────────────────────────────────────────────────────────────┐
│                        前端 (Web UI)                          │
│  上传 CT → 切片浏览 → 点击结节 → 查看报告 → 下载 PDF        │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP POST (DICOM/.mhd + 坐标)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      后端 (FastAPI)                           │
│                                                              │
│  POST /analyze                                                │
│    │                                                          │
│    ├─ ① 结节检测 (如用户未提供坐标)                            │
│    │     MONAI RetinaNet / nnDetection                        │
│    │     输入: CT volume                                      │
│    │     输出: [(x,y,z, diameter_mm, confidence), ...]        │
│    │                                                          │
│    ├─ ② ROI 提取                                              │
│    │     复用 data/mhd_to_png.py extract_multi_view()         │
│    │     输入: CT + (x,y,z) 世界坐标                           │
│    │     输出: 3张 512px PNG (axial/coronal/sagittal)         │
│    │     参数: 肺窗(WL=-600,WW=1500), 50mm ROI, LANCZOS      │
│    │                                                          │
│    └─ ③ VLM 诊断                                              │
│    │     HTTP POST → vLLM API (OpenAI 兼容)                   │
│    │     输入: 3张多视图 PNG + prompt                          │
│    │     输出: 诊断报告文本                                     │
│    │                                                          │
│    ▼                                                          │
│  POST /report → 返回结构化 JSON                               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    VLM 推理服务 (vLLM)                        │
│  GPU 端口 8000                                               │
│  模型: huang01080524/lungct-nodule-grpo                      │
│  OpenAI 兼容 API: POST /v1/chat/completions                  │
└─────────────────────────────────────────────────────────────┘
```

## 组件规格

### 1. 检测模块

| 项目 | 选择 |
|------|------|
| 模型 | MONAI RetinaNet (LUNA16 预训练) 或 nnDetection |
| 输入 | NIfTI (.nii.gz) 3D CT volume |
| 输出 | `[{coordX, coordY, coordZ, diameter_mm, confidence}]` |
| 显存 | ~2GB |
| 速度 | < 5 秒/volume |

**备选方案**: 不集成检测器，前端让用户手动标注结节位置。适用于演示场景。

**预处理**: 检测前无需做任何特殊处理。检测模型内置重采样（spacing 归一化到 0.7mm isotropic）。

**关键约束**: 检测模型输出的坐标必须和 VLM 训练时的坐标系统一致（世界坐标 mm，LUNA16 坐标系）。如果检测器输出是体素坐标，需乘以 spacing 转换。

### 2. ROI 提取模块

| 项目 | 选择 |
|------|------|
| 核心函数 | 复用 `data/mhd_to_png.py` 的 `extract_multi_view()` |
| 输入 | CT volume (numpy/sitk) + (x,y,z) 世界坐标 mm |
| 输出 | 3 张 512px PNG: `axial.png`, `coronal.png`, `sagittal.png` |
| 参数 | 肺窗 WL=-600, WW=1500; ROI 50mm; LANCZOS resize |
| 语言 | Python + SimpleITK + PIL |
| 速度 | < 1 秒/结节 |

**⚠️ 必须和训练时完全一致**: 
- 窗宽窗位: WL=-600, WW=1500（肺窗）
- ROI 尺寸: 50mm × 50mm
- 输出分辨率: 512×512
- Resize 方法: PIL.Image.LANCZOS
- PNG 格式: 8-bit grayscale

参数对不上会导致模型看到它不认识的图像分布，输出质量严重下降。**这是整个 pipeline 最脆弱的环节。**

### 3. VLM 推理模块

| 项目 | 选择 |
|------|------|
| 推理引擎 | vLLM (OpenAI 兼容 API) |
| 模型 | huang01080524/lungct-nodule-grpo (LoRA adapter) |
| 基座模型 | Qwen/Qwen2.5-VL-3B-Instruct |
| 推理精度 | 4-bit (bitsandbytes) |
| 显存 | ~6GB |
| Prompt 模板 | `"请分析这张肺部CT图像中的结节，提供完整的影像学分析报告..."` |
| Max tokens | 800 |

**Prompt 注意事项**: 
- 不要加结构化 hint（训练时是无 hint 模式, 加了 hint 会导致 text-copy 退化）
- 只传入 3 张 ROI 多视图图片, 不传全尺寸 CT
- 用户可选的额外临床信息（年龄/吸烟史等）可附加在文本 prompt 末尾

## API 契约

### POST /analyze

```json
// Request
{
  "ct_file": "<base64 encoded DICOM or .nii.gz>",
  "nodule_coords": [  // 可选; 如果不传则走自动检测
    {"x": 12.3, "y": -45.2, "z": -120.5}
  ],
  "clinical_info": {   // 可选
    "age": 55,
    "smoking_history": "current",
    "family_history": false
  }
}

// Response
{
  "nodule_id": 1,
  "coords": {"x": 12.3, "y": -45.2, "z": -120.5},
  "diameter_mm": 8.6,
  "detection_confidence": 0.95,
  "report": "右肺中叶可见一约8.6mm实性结节...",
  "images": {  // base64 encoded PNGs for frontend display
    "axial": "iVBORw0KGgo...",
    "coronal": "iVBORw0KGgo...",
    "sagittal": "iVBORw0KGgo..."
  }
}
```

### POST /detect (仅检测)

```json
// Request: {"ct_file": "<base64>"}
// Response: [{"x": 12.3, "y": -45.2, "z": -120.5, "diameter_mm": 8.6, "confidence": 0.95}, ...]
```

## GPU 部署方案

### 单 GPU（推荐，最省钱）

```
RTX 3060 12GB / RTX 4060 Ti 16GB / RTX 4090 24GB

显存分配:
  VLM (4-bit):       ~6GB
  检测模型:           ~2GB
  系统开销:           ~2GB
  ──────────────────────────
  总计:              ~10GB  ✅ 12GB 卡能跑
```

一个 Docker 容器内跑 FastAPI + vLLM + 检测模型。单 GPU 复用，不需要多进程。

vLLM 启动配置：
```bash
vllm serve huang01080524/lungct-nodule-grpo \
    --host 0.0.0.0 --port 8000 \
    --gpu-memory-utilization 0.6 \
    --max-model-len 4096 \
    --limit-mm-per-prompt "image=3"  # 最多3张图
```

### 双 GPU（生产环境）

```
GPU 0 (16GB+): VLM (vLLM, 8GB)
GPU 1 (8GB+):  检测模型 + FastAPI + ROI extraction (4GB)
```

## 前端需提供的交互

1. **CT 上传**: 支持 DICOM 拖拽或 .nii.gz 上传
2. **切片浏览**: 提供轴向滚动条，让用户切到含结节的层
3. **结节标注**: 用户点击图像 → 记录体素坐标 → 转换世界坐标 → 发送到后端
4. **报告展示**: 格式化显示诊断报告（4 段: 发现/评估/建议/免责声明）
5. **下载**: PDF/JSON 导出
6. **多结节**: 支持一个 CT 扫描诊断多个结节，切换查看各自报告

## 关键风险点

| 风险 | 缓解 |
|------|------|
| ROI 参数不一致 | 后端硬编码参数（512px/肺窗/50mm），不允许前端传自定义值 |
| 检测坐标与训练坐标系不匹配 | 统一用 LUNA16 世界坐标（mm），函数内转换为体素索引 |
| VLM 输出不稳定 | 推理时 `temperature=0`（贪心解码），不做采样 |
| Token 截断 | `max_new_tokens=800` 覆盖完整报告 |
| 显存不足 | 检测器可选手动模式（前端点坐标），跳过自动检测释放 2GB |

## 向后兼容

- `data/mhd_to_png.py` 中的 `extract_multi_view()` 函数保持不变，后端直接 `import` 调用
- Prompt 模板与训练时一致：`"请分析这张肺部CT图像中的结节..."`
- 推理参数：512px、肺窗、50mm ROI、LANCZOS resize 与训练时完全一致
