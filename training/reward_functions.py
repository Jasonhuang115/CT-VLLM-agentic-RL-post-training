#!/usr/bin/env python3
"""
奖励函数实现

GRPO 训练的核心组件。5 信号复合奖励 + 防策略坍缩机制。

参考: MedFact-R1 (2025), EditGRPO (2025)

使用方式:
  from training.reward_functions import composite_reward
  score = composite_reward(completion, ground_truth, config)

TODO — CT-RATE reward model:
  当前 reward 全量基于规则 (关键词匹配, 数值提取, 正则检查)。
  CT-RATE (47K 真实放射科报告) 可用于训练一个 BERT-based reward model:
    1. 用 CT-RATE 报告 fine-tune BioBERT/RadBERT 作为文本质量评估器
    2. 输出 [0,1] 分数评估"这段报告读起来像真实放射科报告"的程度
    3. 与现有规则 reward 做加权融合: r = 0.5*r_rule + 0.5*r_ctrate
  好处: 不再依赖 hard-coded 正则，reward 更平滑，GRPO 梯度信号更稳定。
"""

import re
import json
from typing import Dict, List, Tuple, Optional


# ============================================================
# 1. Format Reward (权重 0.10)
# ============================================================

REQUIRED_SECTIONS_CN = [
    (r"(影像学发现|发现|Findings|影像表现)", "影像学发现"),
    (r"(恶性风险|风险评估|Assessment|良恶性)", "评估"),
    (r"(随访|建议|Recommendation|管理建议|处理建议)", "随访建议"),
    (r"(免责|声明|Disclaimer|注意|⚠)", "免责声明"),
]

REQUIRED_SECTIONS_EN = [
    (r"(Findings|Imaging Findings|Observations)", "Findings"),
    (r"(Assessment|Malignancy|Risk|Impression)", "Assessment"),
    (r"(Recommendation|Follow-up|Management|Plan)", "Recommendation"),
    (r"(Disclaimer|Note|Caution|⚠)", "Disclaimer"),
]


def check_structure(completion: str, lang: str = "cn") -> float:
    """检查报告是否有结构化段落"""
    sections = REQUIRED_SECTIONS_CN if lang == "cn" else REQUIRED_SECTIONS_EN
    score = 0.0
    for pattern, _ in sections:
        if re.search(pattern, completion, re.IGNORECASE):
            score += 1.0 / len(sections)
    return score


# ============================================================
# 2. Accuracy Reward (权重 0.30)
# ============================================================

def extract_size_from_text(text: str) -> Optional[Tuple[float, float]]:
    """从文本中提取结节大小 (长径, 短径) mm"""
    # 匹配模式: "12.3mm × 8.7mm" 或 "12.3×8.7mm" 或 "长径12.3, 短径8.7"
    patterns = [
        r"(\d+\.?\d*)\s*mm?\s*[×xX]\s*(\d+\.?\d*)\s*mm?",
        r"长径\s*(\d+\.?\d*).*?短径\s*(\d+\.?\d*)",
        r"(\d+\.?\d*)\s*×\s*(\d+\.?\d*)\s*mm",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return (float(m.group(1)), float(m.group(2)))
    return None


def extract_malignancy_from_text(text: str) -> Optional[str]:
    """从文本中提取恶性评估"""
    patterns = [
        (r"(Lung-RADS\s*[0-4][AB]?)", "lungrads"),
        (r"(高度良性|良性可能大|不确定|中等风险|可疑恶性|高度可疑恶性)", "cn"),
        (r"(highly likely benign|benign|indeterminate|suspicious|highly suspicious)", "en"),
    ]
    for pat, lang in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1)
    return None


def compare_measurements(completion: str, ground_truth: dict) -> float:
    """
    比较生成报告中关键数值的准确性

    ground_truth: {
        "long_diameter_mm": float,
        "short_diameter_mm": float,
        "malignancy_level": int (1-5),
    }
    """
    scores = []

    # 大小准确度
    gt_long = ground_truth.get("long_diameter_mm", 0)
    gt_short = ground_truth.get("short_diameter_mm", 0)

    extracted = extract_size_from_text(completion)
    if extracted and gt_long > 0:
        pred_long, pred_short = extracted
        # 误差 < 20% 满分, < 50% 一半分, > 50% 零分
        long_err = abs(pred_long - gt_long) / gt_long
        if long_err < 0.2:
            scores.append(1.0)
        elif long_err < 0.5:
            scores.append(0.5)
        else:
            scores.append(0.0)
    else:
        scores.append(0.0)

    # 恶性评级准确度
    gt_malignancy = ground_truth.get("malignancy_level", 3)
    pred_text = extract_malignancy_from_text(completion)
    if pred_text:
        # 简化: 检查是否提到对应的 Lung-RADS 等级
        lung_rads_map = {1: "1", 2: "2", 3: "3", 4: "4A", 5: "4B"}
        expected = lung_rads_map.get(gt_malignancy, "3")
        if expected in pred_text:
            scores.append(1.0)
        else:
            scores.append(0.5)
    else:
        scores.append(0.3)  # 至少提到了评估, 给基础分

    return sum(scores) / len(scores) if scores else 0.0


# ============================================================
# 3. Factual Reward (权重 0.35)
# ============================================================

FACTUAL_PATTERNS = {
    "density": {
        "实性/solid": [
            r"(实性|solid)\s*(结节|密度|成分|nodule)",
            r"(完全实性|purely solid)",
        ],
        "部分实性/part-solid": [
            r"(部分实性|混合磨玻璃|part-?solid|mixed GGO)",
        ],
        "磨玻璃/GGO": [
            r"(磨玻璃|GGO|ground.?glass|非实性|nonsolid)",
            r"(纯磨玻璃|pure GGO)",
        ],
    },
    "margin": {
        "边界清晰/well-defined": [r"(边界清晰|边缘光滑|well.?defined|smooth margin)"],
        "毛刺/spiculated": [r"(毛刺|spiculat|不规则边缘|irregular margin)"],
        "分叶/lobulated": [r"(分叶|lobulat|notch)"],
    },
    "calcification": {
        "有钙化": [r"(钙化|calcif)(?!.*无|.*未见|.*absent)"],
        "无钙化": [r"((无|未见|no|absent)\s*钙化|(无|未见|no|absent)\s*calcif)"],
    },
}


def check_factual_claims(completion: str, ground_truth: dict) -> float:
    """
    检查生成报告的事实性断言是否与 GT 一致

    与 Stage 1 的模板不同, 这里不需要精确匹配。
    只需要关键事实类型不矛盾 (如: 不要说无钙化但 GT 里有)。
    """
    gt_chars = ground_truth.get("characteristics", {})
    if not gt_chars:
        return 0.5  # 无法验证

    score = 0.0
    n_checks = 0

    # 密度类型检查
    gt_texture = gt_chars.get("texture", 3)
    # texture: 1=非实性, 2=部分实性, 3=实性
    if gt_texture <= 2:
        # GT 是非实性或部分实性 → 不应描述为"完全实性"
        if re.search(r"(完全实性|purely solid)", completion, re.IGNORECASE):
            score += 0.0
        else:
            score += 1.0
    else:
        # GT 是实性 → 不应描述为"纯磨玻璃"
        if re.search(r"(纯磨玻璃|pure GGO)", completion, re.IGNORECASE):
            score += 0.0
        else:
            score += 1.0
    n_checks += 1

    # 钙化检查
    gt_calc = gt_chars.get("calcification", 6)
    has_calc_in_text = bool(re.search(r"(钙化|calcif)", completion, re.IGNORECASE))
    no_calc_in_text = bool(re.search(r"(无|未见|no|absent)\s*(钙化|calcif)", completion, re.IGNORECASE))

    if gt_calc == 6:  # GT: 无钙化
        if no_calc_in_text:
            score += 1.0
        elif has_calc_in_text:
            score += 0.0
        else:
            score += 0.5  # 没提到, 不扣不奖
    else:  # GT: 有钙化
        if has_calc_in_text:
            score += 1.0
        elif no_calc_in_text:
            score += 0.0
        else:
            score += 0.5
    n_checks += 1

    return score / n_checks if n_checks > 0 else 0.5


# ============================================================
# 4. Completeness Reward (权重 0.15)
# ============================================================

REQUIRED_FIELDS = [
    (r"(位置|location|肺叶|lobe|segment|段)", "位置"),
    (r"(\d+\.?\d*\s*mm)", "大小"),
    (r"(实性|磨玻璃|GGO|solid|ground.?glass|密度|density)", "密度类型"),
    (r"(钙化|calcif)", "钙化"),
    (r"(边界|边缘|margin|border|轮廓)", "边界"),
    (r"(Lung-RADS|lung.?rads)", "Lung-RADS"),
    (r"(随访|建议|follow.?up|recommend|management)", "建议"),
]


def check_completeness(completion: str) -> float:
    """检查报告是否包含所有必要信息字段"""
    score = 0.0
    for pattern, _ in REQUIRED_FIELDS:
        if re.search(pattern, completion, re.IGNORECASE):
            score += 1.0 / len(REQUIRED_FIELDS)
    return score


# ============================================================
# 5. Consistency Reward (权重 0.10)
# ============================================================

def check_self_consistency(completion: str) -> float:
    """检查报告内部自洽性"""
    score = 1.0  # 起始满分, 发现矛盾扣分

    # 检查: 说"无钙化"但描述了钙化类型
    if re.search(r"(无|未见|no|absent)\s*(钙化|calcif)", completion, re.IGNORECASE):
        if re.search(r"(爆米花|层状|点状|popcorn|laminar|punctate)\s*(钙化|calcif)", completion, re.IGNORECASE):
            score -= 0.3

    # 检查: 评估为良性但建议穿刺
    if re.search(r"(良性|benign|低风险|low risk)", completion, re.IGNORECASE):
        if re.search(r"(穿刺|活检|biopsy|手术|surgery|切除)", completion, re.IGNORECASE):
            score -= 0.2

    # 检查: 评估为高危但建议年度随访
    if re.search(r"(高危|高度可疑|high risk|highly suspicious|Lung-RADS\s*4B)", completion, re.IGNORECASE):
        if re.search(r"(年度|annual|yearly|12\s*个月|12\s*month)", completion, re.IGNORECASE):
            score -= 0.3

    # 检查: 重复/乱码
    if len(completion) < 50:
        score -= 0.5

    # 检查大段重复
    words = completion.split()
    if len(words) > 100:
        # 简单重复检测: 检查是否有超过 30 个词的连续重复
        for i in range(len(words) - 60):
            chunk = " ".join(words[i:i+30])
            rest = " ".join(words[i+30:])
            if chunk in rest and len(chunk) > 50:
                score -= 0.3
                break

    return max(0.0, score)


# ============================================================
# 6. 防策略坍缩
# ============================================================

NO_FINDING_PATTERNS = [
    (r"no\s+(significant\s+)?findings?", "en"),
    (r"no\s+(significant\s+)?abnormality", "en"),
    (r"未见\s*(明显\s*)?异常", "cn"),
    (r"未见\s*(明确\s*)?结节", "cn"),
    (r"无\s*(明显\s*)?异常\s*发现", "cn"),
    (r"normal\s+(chest\s+)?CT", "en"),
    (r"unremarkable", "en"),
]


def check_empty_response(completion: str) -> Tuple[bool, float]:
    """
    检测空洞/逃避回答

    Returns: (is_empty, penalty)
    """
    for pattern, _ in NO_FINDING_PATTERNS:
        if re.search(pattern, completion, re.IGNORECASE):
            # 还需要确认整体长度很短 (避免误判: "未见明显结节，但..."这样的后续)
            if len(completion) < 200:
                return True, -0.5

    if len(completion) < 50:  # 极短回复
        return True, -0.3

    return False, 0.0


# ============================================================
# 7. 复合奖励函数
# ============================================================

def composite_reward(
    completion: str,
    ground_truth: dict,
    lang: str = "cn",
    reward_weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    综合奖励函数。所有分量归一化到 [0, 1], 最终加权求和。

    Args:
        completion: 模型生成的报告文本
        ground_truth: {
            "long_diameter_mm": float,
            "short_diameter_mm": float,
            "malignancy_level": int,       # 1-5
            "characteristics": {           # LIDC 8维特征
                "texture": int,             # 1-5
                "calcification": int,       # 1-6
                "margin": int,              # 1-5
                "spiculation": int,         # 1-5
                "lobulation": int,          # 1-5
                "sphericity": int,          # 1-5
                "subtlety": int,            # 1-5
                "internalStructure": int,   # 1-4
                "malignancy": int,          # 1-5
            },
        }
        lang: "cn" 或 "en"
        reward_weights: 自定义权重 (默认从 config 读取)

    Returns:
        float: 总奖励分数
    """
    if reward_weights is None:
        reward_weights = {
            "format": 0.10,
            "accuracy": 0.30,
            "factual": 0.35,
            "completeness": 0.15,
            "consistency": 0.10,
        }

    # 1. 防坍缩检查 (先运行)
    is_empty, penalty = check_empty_response(completion)
    if is_empty:
        return penalty  # 直接返回惩罚, 不计算其他奖励

    # 2. 五项奖励分
    r_format = check_structure(completion, lang)
    r_accuracy = compare_measurements(completion, ground_truth)
    r_factual = check_factual_claims(completion, ground_truth)
    r_completeness = check_completeness(completion)
    r_consistency = check_self_consistency(completion)

    # 3. 加权求和
    total = (
        reward_weights["format"] * r_format +
        reward_weights["accuracy"] * r_accuracy +
        reward_weights["factual"] * r_factual +
        reward_weights["completeness"] * r_completeness +
        reward_weights["consistency"] * r_consistency
    )

    return total


# ============================================================
# 8. Agent 工具使用奖励 (Stage 4b 用)
# ============================================================

def tool_usage_reward(
    trajectory: List[dict],
    final_report: str,
    ground_truth: dict,
) -> float:
    """
    评估 Agent 工具使用的质量

    Args:
        trajectory: ReAct 轨迹 [(action, observation), ...]
        final_report: 最终诊断报告
        ground_truth: 同 composite_reward

    Returns:
        float: 工具使用奖励
    """
    score = 0.0

    # 1. 工具选择合理性 (0.3)
    tool_calls = [t for t in trajectory if t.get("type") == "action"]
    tool_names = [t.get("tool", "") for t in tool_calls]

    # 应该调用指南检索 (对于恶性评估)
    if "guideline_retrieval" in tool_names:
        score += 0.3
    elif len(tool_calls) > 0:
        score += 0.15  # 至少调用了工具, 但不是最合适的

    # 2. 工具使用效率 (0.2) - 避免冗余
    if len(tool_calls) <= 5:
        score += 0.2
    elif len(tool_calls) <= 8:
        score += 0.1
    else:
        score += 0.0  # 调用太多, 浪费

    # 3. 工具结果利用 (0.3) - 报告是否体现了工具返回的信息
    guidelines_used = False
    for obs in trajectory:
        if obs.get("type") == "observation":
            content = obs.get("content", "")
            # 检查指南内容是否出现在最终报告中
            if len(content) > 50:
                overlap = len(set(content[:200].split()) & set(final_report.split()))
                if overlap > 10:
                    guidelines_used = True
                    break
    if guidelines_used:
        score += 0.3
    elif len(tool_calls) == 0:
        score += 0.15  # 没调工具, 不加不扣
    else:
        score += 0.0  # 调了工具但没用上

    # 4. 最终报告质量 (0.2) - 复用基础奖励
    base_reward = composite_reward(final_report, ground_truth)
    score += 0.2 * base_reward

    return score


if __name__ == "__main__":
    # 简单测试
    test_completion = """
**肺结节CT分析报告**

**影像学发现**
- 位置: 右上叶后段
- 大小: 12.3mm × 8.7mm
- 密度: 部分实性结节
- 钙化: 无钙化
- 边界: 边缘毛刺, 分叶状

**恶性风险评估**
Lung-RADS 4B (中高危)

**随访建议**
建议3个月短期CT随访

⚠️ 免责声明: 本报告由AI辅助诊断系统生成, 仅供临床参考。
"""

    test_gt = {
        "long_diameter_mm": 12.3,
        "short_diameter_mm": 8.7,
        "malignancy_level": 4,
        "characteristics": {
            "texture": 2,
            "calcification": 6,
            "margin": 4,
            "spiculation": 3,
            "lobulation": 3,
            "sphericity": 2,
            "subtlety": 3,
            "internalStructure": 1,
            "malignancy": 4,
        },
    }

    score = composite_reward(test_completion, test_gt)
    print(f"测试奖励分: {score:.3f}")
    print(f"  格式: {check_structure(test_completion):.3f}")
    print(f"  准确性: {compare_measurements(test_completion, test_gt):.3f}")
    print(f"  事实性: {check_factual_claims(test_completion, test_gt):.3f}")
    print(f"  完整性: {check_completeness(test_completion):.3f}")
    print(f"  自洽性: {check_self_consistency(test_completion):.3f}")
