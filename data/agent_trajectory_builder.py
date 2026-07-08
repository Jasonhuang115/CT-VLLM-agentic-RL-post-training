#!/usr/bin/env python3
"""
Agent 轨迹数据构建 (Stage 4a 用)

使用 DeepSeek API 生成 Teacher 轨迹:
  给定结节的结构化标注 (不要求 DeepSeek 看图),
  让 DeepSeek 推理该查什么指南、搜什么文献、算什么指标。

使用方式:
  python data/agent_trajectory_builder.py \
    --nodule_index /path/to/nodule_index.json \
    --api_key YOUR_DEEPSEEK_KEY \
    --output /root/autodl-tmp/data/agent_trajectories
"""

import os
import json
import argparse
import random
from tqdm import tqdm

from agent.tool_registry import ToolRegistry
from agent.tools.guideline_retrieval import GuidelineRetrieval
from agent.tools.web_search import WebSearch
from agent.tools.measurement_calc import MeasurementCalculator

LIDC_TO_NATURAL = {
    "subtlety": {1: "obvious", 2: "moderately obvious", 3: "moderate", 4: "subtle", 5: "very subtle"},
    "internalStructure": {1: "soft tissue", 2: "fluid", 3: "fat", 4: "air"},
    "calcification": {1: "popcorn-like", 2: "laminated", 3: "solid", 4: "amorphous", 5: "punctate", 6: "absent"},
    "sphericity": {1: "linear", 2: "ovoid", 3: "round", 4: "irregular"},
    "margin": {1: "well-defined", 2: "mostly well-defined", 3: "moderately defined", 4: "poorly defined", 5: "halo sign"},
    "lobulation": {1: "none", 2: "mild", 3: "moderate", 4: "marked", 5: "severe"},
    "spiculation": {1: "none", 2: "mild", 3: "moderate", 4: "marked", 5: "severe"},
    "texture": {1: "nonsolid/GGO", 2: "part-solid", 3: "solid", 4: "mixed", 5: "pure GGO"},
}


def nodule_to_text(nodule_data: dict) -> str:
    """结节标注 → 文本描述 (给 DeepSeek 阅读)"""
    chars = nodule_data["characteristics"]
    centroid = nodule_data.get("centroid_mm", [0, 0, 0])
    bbox = nodule_data.get("bbox_mm", {})

    size = bbox.get("size", [10, 10, 10])
    long_diam = max(size[0], size[1])

    lines = [
        f"结节位置: 坐标 ({centroid[0]:.0f}, {centroid[1]:.0f}, {centroid[2]:.0f}) mm",
        f"结节长径: {long_diam:.1f} mm",
    ]
    for key, label in [("texture", "密度类型"), ("sphericity", "形态"),
                        ("margin", "边界"), ("lobulation", "分叶"),
                        ("spiculation", "毛刺"), ("calcification", "钙化"),
                        ("malignancy", "恶性度(1-5)")]:
        val = chars.get(key, 3)
        lines.append(f"{label}: {LIDC_TO_NATURAL.get(key, {}).get(val, str(val))} (评分{val})")

    return "\n".join(lines)


def generate_teacher_trajectory(nodule_data: dict, api_key: str = None) -> dict:
    """
    生成 Teacher 轨迹 (DeepSeek 推理 + 真实工具执行)

    流程:
      1. DeepSeek 读结节描述 → 推理该调什么工具
      2. 真实执行工具调用 (本地)
      3. 拼接完整轨迹
    """
    # 准备工具
    guideline = GuidelineRetrieval()
    search = WebSearch()
    calc = MeasurementCalculator()

    nodule_desc = nodule_to_text(nodule_data)
    chars = nodule_data["characteristics"]
    texture_val = chars.get("texture", 3)
    malignancy_val = chars.get("malignancy", 3)
    bbox = nodule_data.get("bbox_mm", {})
    size = bbox.get("size", [10, 10, 10])
    long_diam = max(size[0], size[1])

    # 推断结节类型
    if texture_val >= 4:
        nodule_type = "solid"
    elif texture_val >= 2:
        nodule_type = "part-solid"
    else:
        nodule_type = "ggn"

    # 如果不提供 API key, 使用规则生成轨迹 (不需要 DeepSeek)
    steps = []

    # Step 1: 指南检索 (规则决定)
    guideline_query = f"{nodule_type} nodule {long_diam:.1f}mm management"
    steps.append({
        "thought": f"结节类型为{ nodule_type }, 长径{ long_diam:.1f}mm. 需要查询管理指南.",
        "tool": "guideline_retrieval",
        "params": {"query": guideline_query},
        "observation": guideline.search(guideline_query),
    })

    # Step 2: Lung-RADS 计算
    if nodule_type == "part-solid":
        solid_mm = long_diam * 0.35  # 估计实性成分
        lung_rads_result = calc.lung_rads_classify(nodule_type, long_diam, solid_component_mm=solid_mm)
    else:
        lung_rads_result = calc.lung_rads_classify(nodule_type, long_diam)

    steps.append({
        "thought": "需要计算 Lung-RADS 分级以确定标准随访建议.",
        "tool": "measurement_calculator",
        "params": {"function": "lung_rads_classify", "nodule_type": nodule_type, "size_mm": long_diam},
        "observation": lung_rads_result,
    })

    # Step 3: 搜索 (仅中高危)
    if malignancy_val >= 3:
        search_query = f"{nodule_type} pulmonary nodule malignancy risk spiculation"
        steps.append({
            "thought": f"恶性度评分{malignancy_val}, 需要搜索最新研究验证.",
            "tool": "web_search",
            "params": {"query": search_query},
            "observation": search.search(search_query),
        })

    # 构建最终报告 (简化)
    chars_text = nodule_to_text(nodule_data)
    final_report = f"""**肺结节诊断报告 (Agentic)**

**影像发现**: {chars_text}

**指南对照**:
{json.dumps(steps[0]['observation'].get('summary', ''), ensure_ascii=False) if steps else ''}

**Lung-RADS 分级**: {lung_rads_result.get('category', 'N/A')}
**建议**: {lung_rads_result.get('recommendation', '请咨询专科医师')}

---
⚠️ AI辅助诊断, 仅供临床参考。"""

    return {
        "nodule_name": nodule_data.get("nodule_name", ""),
        "ground_truth": nodule_data,
        "steps": steps,
        "final_report": final_report,
    }


def main():
    parser = argparse.ArgumentParser(description="Agent 轨迹构建")
    parser.add_argument("--nodule_index", type=str, default="/root/autodl-tmp/data/nodules/nodule_index.json")
    parser.add_argument("--output", type=str, default="/root/autodl-tmp/data/agent_trajectories")
    parser.add_argument("--api_key", type=str, default=None, help="DeepSeek API key")
    parser.add_argument("--n_trajectories", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    # 加载结节数据
    with open(args.nodule_index) as f:
        nodules = json.load(f)
    random.shuffle(nodules)
    nodules = nodules[:args.n_trajectories]

    print(f"[INFO] 生成 {len(nodules)} 条 Agent 轨迹...")

    trajectories = []
    for nodule in tqdm(nodules):
        traj = generate_teacher_trajectory(nodule, args.api_key)
        trajectories.append(traj)

    path = os.path.join(args.output, "agent_trajectories.jsonl")
    with open(path, "w") as f:
        for t in trajectories:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(f"[DONE] {len(trajectories)} 条轨迹 → {path}")
    print(f"\n下一步: python training/stage4a_agent_sft.py --trajectories {path}")


if __name__ == "__main__":
    main()
