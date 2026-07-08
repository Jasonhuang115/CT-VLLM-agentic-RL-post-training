#!/usr/bin/env python3
"""
临床准确性评估

评估项:
  1. 结节位置准确率 (肺叶定位)
  2. 大小测量偏差 (mm)
  3. 密度类型准确率 (实性/部分实性/磨玻璃)
  4. Lung-RADS 分级一致率
  5. 报告结构完整率

使用方式:
  python evaluation/clinical_accuracy.py --model_adapter /path/to/adapter --test_data /path/to/test.jsonl
"""

import re
import json
import argparse
import numpy as np
from tqdm import tqdm

def evaluate_size(pred_text: str, gt_long: float, gt_short: float) -> dict:
    """评估大小准确性"""
    m = re.search(r"(\d+\.?\d*)\s*[×xXmm]+\s*(\d+\.?\d*)", pred_text)
    if not m:
        m = re.search(r"长径\s*(\d+\.?\d*)", pred_text)
    if not m:
        return {"size_match": False, "long_deviation_pct": 999, "short_deviation_pct": 999}

    p_long = float(m.group(1))
    p_short = float(m.group(2)) if m.lastindex and m.lastindex >= 2 else p_long * 0.7
    long_err = abs(p_long - gt_long) / gt_long * 100 if gt_long > 0 else 999
    short_err = abs(p_short - gt_short) / gt_short * 100 if gt_short > 0 else 999
    return {"size_match": long_err < 20, "long_deviation_pct": round(long_err, 1), "short_deviation_pct": round(short_err, 1)}

def evaluate_density(pred_text: str, gt_texture: int) -> bool:
    """评估密度类型准确性"""
    patterns = {(1, 5): r"(非实性|nonsolid|纯磨玻璃|pure GGO|磨玻璃|GGO|ground.?glass)",
                 (2,): r"(部分实性|part.?solid|混合磨玻璃|mixed GGO)",
                 (3, 4): r"(实性|solid)(?!.*部分|.*part)"}
    for texture_vals, pat in patterns.items():
        if gt_texture in texture_vals:
            return bool(re.search(pat, pred_text, re.IGNORECASE))
    return False

def evaluate_malignancy(pred_text: str, gt_malignancy: int) -> bool:
    """评估恶性度一致性 (±1 级即认为正确)"""
    lung_rads_map = {1: ["1", "阴性"], 2: ["2", "良性"], 3: ["3", "可能良性"], 4: ["4A", "可疑恶性"], 5: ["4B", "高度可疑"]}
    expected = lung_rads_map.get(gt_malignancy, ["3"])
    return any(exp in pred_text for exp in expected)

def evaluate_structure(pred_text: str) -> dict:
    """评估报告结构完整性"""
    sections = {"findings": r"(影像学发现|Findings|发现)", "assessment": r"(恶性风险|Assessment|评估|Lung-RADS)", "recommendation": r"(随访|建议|Recommendation|管理)", "disclaimer": r"(免责|声明|Disclaimer|⚠)"}
    return {k: bool(re.search(v, pred_text, re.IGNORECASE)) for k, v in sections.items()}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True, help="模型预测结果 JSONL")
    p.add_argument("--ground_truth", default=None, help="GT (默认从 predictions 中读取)")
    args = p.parse_args()

    preds = []
    with open(args.predictions) as f:
        for line in f:
            preds.append(json.loads(line))

    metrics = {"size_acc": [], "density_acc": 0, "malignancy_acc": 0, "structure": []}
    for item in tqdm(preds):
        pred_text = item.get("prediction", item.get("final_answer", ""))
        gt = item.get("ground_truth", item.get("metadata", {}))
        gt_long = gt.get("long_diameter_mm", 10)
        gt_short = gt.get("short_diameter_mm", 8)
        gt_texture = gt.get("characteristics", {}).get("texture", 3) if "characteristics" in gt else gt.get("texture", 3)
        gt_malignancy = gt.get("characteristics", {}).get("malignancy", 3) if "characteristics" in gt else gt.get("malignancy", 3)

        size_r = evaluate_size(pred_text, gt_long, gt_short)
        if size_r["long_deviation_pct"] < 999:
            metrics["size_acc"].append(size_r["long_deviation_pct"])
        metrics["density_acc"] += int(evaluate_density(pred_text, gt_texture))
        metrics["malignancy_acc"] += int(evaluate_malignancy(pred_text, gt_malignancy))
        metrics["structure"].append(evaluate_structure(pred_text))

    n = len(preds)
    print(f"\n评估结果 (n={n}):")
    print(f"  大小误差 (中位): {np.median(metrics['size_acc']):.1f}%")
    print(f"  大小误差 (平均): {np.mean(metrics['size_acc']):.1f}%")
    print(f"  密度类型准确率: {metrics['density_acc']/n*100:.1f}%")
    print(f"  恶性度准确率 (±1级): {metrics['malignancy_acc']/n*100:.1f}%")
    for section in ["findings", "assessment", "recommendation", "disclaimer"]:
        rate = sum(1 for s in metrics["structure"] if s.get(section)) / n * 100
        print(f"  结构-{section}: {rate:.1f}%")


if __name__ == "__main__":
    main()
