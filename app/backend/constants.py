"""Runtime constants that must stay aligned with VLM training."""

ROI_SIZE_MM = 50.0
OUTPUT_SIZE_PX = 512
WINDOW_LEVEL = -600.0
WINDOW_WIDTH = 1500.0
PNG_MODE = "L"
RESIZE_FILTER = "LANCZOS"

DISCLAIMER = (
    "本结果由 AI 辅助系统生成，仅供科研演示和临床参考，不能替代放射科或呼吸科医师诊断。"
    "请结合原始 DICOM、既往影像、病史和实验室检查，由有资质的医生最终确认。"
)

VLM_PROMPT = "请分析这张肺部CT图像中的结节，提供完整的影像学分析报告。"
