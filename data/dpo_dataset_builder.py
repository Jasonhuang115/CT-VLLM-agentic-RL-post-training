#!/usr/bin/env python3
"""
DPO 偏好对构建 v2

关键修正: DPO prompt 必须带 CT 图像——保持视觉锚定，防止 Stage 2
覆盖 Stage 1 学到的视觉映射。

方法:
  1. 以 SFT 数据为基础，复用其 user prompt（含 CT 图像路径）
  2. chosen = SFT 的高质量报告
  3. rejected = 扰动版本（数值偏差、特征翻转、结构删减）
  4. 每个结节可生成多个扰动变体扩充数据

使用方式:
  python data/dpo_dataset_builder.py \
    --sft_data /root/autodl-tmp/data/sft/sft_train.jsonl \
    --output /root/autodl-tmp/data/dpo \
    --variants_per_sample 3
"""

import os, sys, json, random, argparse, re
from copy import deepcopy
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════
# 报告扰动策略
# ═══════════════════════════════════════════════════════════════

def perturb_size(text: str, level: float = 0.3) -> str:
    """扰动报告中的尺寸数值 (±30%)"""
    def _change(match):
        val = float(match.group(1))
        unit = match.group(2) or "mm"
        new_val = val * random.uniform(1 - level, 1 + level)
        return f"{new_val:.1f}{unit}"
    return re.sub(r"(\d+\.?\d*)\s*(mm|MM|cm)", _change, text)


def perturb_malignancy(text: str) -> str:
    """翻转恶性评估（好→坏 或 坏→好）"""
    swaps = [
        ("高度良性可能", "高度可疑恶性"),
        ("良性可能大", "可疑恶性"),
        ("不确定", "可疑恶性"),
        ("低风险", "高风险"),
        ("Lung-RADS 1", "Lung-RADS 4B"),
        ("Lung-RADS 2", "Lung-RADS 4A"),
        ("Lung-RADS 3", "Lung-RADS 4B"),
        ("highly likely benign", "highly suspicious for malignancy"),
        ("benign", "suspicious"),
        ("low risk", "high risk"),
        ("recommend annual screening", "recommend immediate biopsy"),
        ("建议年度筛查", "建议立即活检"),
        ("建议12个月随访", "建议立即胸外科会诊"),
        ("建议6个月随访", "建议PET-CT及穿刺活检"),
    ]
    for old, new in random.sample(swaps, min(3, len(swaps))):
        if random.random() < 0.5:
            text = re.sub(re.escape(old), new, text, flags=re.IGNORECASE)
    return text


def perturb_density(text: str) -> str:
    """翻转密度类型"""
    swaps = [
        ("磨玻璃", "实性"),
        ("非实性", "实性"),
        ("部分实性", "纯磨玻璃"),
        ("ground-glass", "solid"),
        ("GGO", "solid nodule"),
        ("nonsolid", "solid"),
        ("part-solid", "pure ground-glass"),
    ]
    for old, new in random.sample(swaps, min(2, len(swaps))):
        text = re.sub(re.escape(old), new, text, flags=re.IGNORECASE)
    return text


def drop_section(text: str) -> str:
    """随机删除报告的一个段落"""
    sections = re.split(r"\n\n|\*\*.*?\*\*\n", text)
    if len(sections) >= 3:
        drop_idx = random.randint(0, len(sections) - 1)
        sections.pop(drop_idx)
        return "\n\n".join(sections)
    return text


def perturb_report(report_text: str, intensity: str = "medium") -> str:
    """
    扰动报告生成 rejected 样本。

    intensity:
      - "light": 仅轻微数值扰动
      - "medium": 数值扰动 + 1 种特征翻转
      - "heavy": 数值扰动 + 2 种特征翻转 + 段落删除
    """
    text = perturb_size(report_text, level=0.3 if intensity == "light" else 0.5)

    if intensity == "medium":
        perturb_fn = random.choice([perturb_malignancy, perturb_density])
        text = perturb_fn(text)

    if intensity == "heavy":
        text = perturb_malignancy(text)
        if random.random() < 0.5:
            text = perturb_density(text)
        if random.random() < 0.3:
            text = drop_section(text)

    # 确保扰动后的文本跟原文不同
    if text.strip() == report_text.strip():
        text = perturb_malignancy(text)

    return text


# ═══════════════════════════════════════════════════════════════
# DPO 对构建
# ═══════════════════════════════════════════════════════════════

def convert_to_png_path(path: str) -> str:
    """.mhd → .png, images/ → images_png/"""
    if path.endswith('.mhd'):
        path = path[:-4] + '.png'
    return path.replace('/images/', '/images_png/')


def extract_user_messages(sft_sample: dict) -> list:
    """从 SFT 样本中提取 user 消息（含图像 + 指令），统一转 PNG 路径"""
    import copy
    messages = sft_sample.get("messages", [])
    user_msgs = []
    for m in messages:
        if m.get("role") != "user":
            continue
        m = copy.deepcopy(m)
        content = m.get("content", [])
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image" and "image" in item:
                    item["image"] = convert_to_png_path(item["image"])
        user_msgs.append(m)
    return user_msgs


def extract_assistant_text(sft_sample: dict) -> str:
    """从 SFT 样本中提取 assistant 的纯文本"""
    for m in sft_sample.get("messages", []):
        if m.get("role") != "assistant":
            continue
        content = m.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    texts.append(item["text"])
                elif isinstance(item, str):
                    texts.append(item)
            return " ".join(texts)
    return ""


def build_dpo_pairs(
    sft_data_path: str,
    output_dir: str,
    variants_per_sample: int = 3,
    n_pairs: int = 5000,
) -> list:
    """
    从 SFT 数据构建 DPO 偏好对。

    核心改动 vs v1:
    - prompt 保留 CT 图像（从 SFT user message 来）
    - chosen 是 SFT 的高质量报告
    - rejected 是扰动版本
    - 图像锚定，不会丢失视觉映射
    """
    samples = []
    with open(sft_data_path) as f:
        for line in f:
            samples.append(json.loads(line))

    random.shuffle(samples)
    print(f"  SFT 样本: {len(samples)}")

    pairs = []
    intensity_dist = ["light"] * 4 + ["medium"] * 5 + ["heavy"] * 1  # 40/50/10

    for sample in tqdm(samples, desc="构建DPO对"):
        user_msgs = extract_user_messages(sample)
        assistant_text = extract_assistant_text(sample)

        if not user_msgs or not assistant_text or len(assistant_text) < 50:
            continue

        for _ in range(variants_per_sample):
            intensity = random.choice(intensity_dist)
            rejected_text = perturb_report(assistant_text, intensity=intensity)

            if rejected_text == assistant_text:
                continue  # 扰动没生效，跳过

            pairs.append({
                "prompt": user_msgs,          # ← 含 CT 图像！
                "chosen": [{
                    "role": "assistant",
                    "content": [{"type": "text", "text": assistant_text}],
                }],
                "rejected": [{
                    "role": "assistant",
                    "content": [{"type": "text", "text": rejected_text}],
                }],
                "perturbation": intensity,
            })

    # 截取
    random.shuffle(pairs)
    pairs = pairs[:n_pairs]

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    split = max(1, int(len(pairs) * 0.85))
    train_pairs = pairs[:split]
    val_pairs = pairs[split:]

    for name, data in [("dpo_train", train_pairs), ("dpo_val", val_pairs)]:
        path = os.path.join(output_dir, f"{name}.jsonl")
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  {name}: {len(data)} 对 → {path}")

    return pairs


def main():
    parser = argparse.ArgumentParser(description="DPO 偏好对构建 v2")
    parser.add_argument("--sft_data", default="/root/autodl-tmp/data/sft/sft_train.jsonl")
    parser.add_argument("--output", default="/root/autodl-tmp/data/dpo")
    parser.add_argument("--variants_per_sample", type=int, default=3,
                        help="每个 SFT 样本生成几个变体")
    parser.add_argument("--n_pairs", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    print("=" * 60)
    print("  DPO 偏好对构建 v2 (图像锚定)")
    print("=" * 60)
    print(f"  SFT 数据: {args.sft_data}")
    print(f"  每个结节变体数: {args.variants_per_sample}")
    print(f"  目标 pair 数: {args.n_pairs}")

    pairs = build_dpo_pairs(
        args.sft_data, args.output,
        variants_per_sample=args.variants_per_sample,
        n_pairs=args.n_pairs,
    )
    print(f"\n[DONE] 生成 {len(pairs)} 个 DPO 偏好对")
    print(f"  ✅ prompt 均包含 CT 图像路径（视觉锚定）")
    print(f"\n下一步: python training/stage2_simpo.py --data_dir {args.output}")


if __name__ == "__main__":
    main()
