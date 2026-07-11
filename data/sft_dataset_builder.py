#!/usr/bin/env python3
"""
SFT 数据集构建 v2 (LUNA16 + LIDC特征 + DeepSeek报告生成)

流程:
  1. 加载 LUNA16 结节标注 (annotations.csv)
  2. 加载 pylidc 匹配结果 (nodule_features.json)
  3. 用 DeepSeek API 将结构化特征 → 专业放射科报告
     (以 CT-RATE 真实报告为风格参照)
  4. 输出 OpenAI vision format 训练数据

使用方式:
  # 先跑 lidc_match.py 生成 nodule_features.json
  python data/lidc_match.py

  # 再跑本脚本
  python data/sft_dataset_builder.py \
    --luna16_dir /root/autodl-tmp/data/LUNA16 \
    --features /root/autodl-tmp/data/nodule_features.json \
    --ctrate_reports /root/autodl-tmp/data/CT-RATE/reports.jsonl \
    --output /root/autodl-tmp/data/sft \
    --deepseek_api_key sk-xxx

如果不想用 API，不加 --deepseek_api_key 会自动使用增强模板。
"""

import os, sys, json, random, argparse, csv, re, time
from pathlib import Path
from copy import deepcopy
import numpy as np
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════
# 指令模板
# ═══════════════════════════════════════════════════════════════

INSTRUCTION_TEMPLATES_CN = [
    "请分析这张肺部CT图像中的结节。描述其位置、大小、形态特征，并给出良恶性评估。",
    "你是一名放射科医生。请对图像中的肺结节进行专业分析，包括影像学发现、恶性风险评估和随访建议。",
    "请评估这个肺结节的恶性风险。包括大小、边界、密度类型、钙化状态等关键指标，并给出Lung-RADS分级。",
    "根据Lung-RADS标准，分析这个肺结节并提供系统的随访建议。请包含免责声明。",
    "这是一例肺部CT平扫的结节区域。请从影像学特征、良恶性鉴别、临床管理三个维度进行全面评估。",
]

# P1: 结构化临床提示模板 (PLAN.md Phase 1)
STRUCTURED_HINT_CN = """
临床提示：
- 结节大致位于 {location} 区域
- 估计直径约 {diameter_mm}mm

请重点分析以下特征：
1. 密度类型（磨玻璃/部分实性/实性）
2. 边界特征（清晰/模糊/毛刺征/分叶状）
3. 钙化状态（有无钙化及形态类型）
4. 根据上述特征给出 Lung-RADS 分级和随访建议"""

STRUCTURED_HINT_EN = """
Clinical context:
- Nodule located approximately in the {location} region
- Estimated diameter: {diameter_mm} mm

Please focus on:
1. Attenuation type (ground-glass / part-solid / solid)
2. Margin characteristics (well-defined / ill-defined / spiculated / lobulated)
3. Calcification status (presence/absence and morphology)
4. Lung-RADS classification and follow-up recommendation based on the above"""

INSTRUCTION_TEMPLATES_EN = [
    "Analyze this lung CT image. Describe the nodule's location, size, morphology, and assess malignancy risk.",
    "As a thoracic radiologist, evaluate this lung nodule. Provide structured findings, assessment, and recommendations.",
    "Please perform a comprehensive analysis of this pulmonary nodule including Lung-RADS classification.",
    "Evaluate this chest CT finding. Include size, attenuation, margin characteristics, and follow-up recommendations.",
]


# ═══════════════════════════════════════════════════════════════
# CT-RATE 风格示例（从真实报告中抽的片段，用于 DeepSeek few-shot）
# ═══════════════════════════════════════════════════════════════

CTRATE_STYLE_EXAMPLE_EN = """
Example of professional radiology report style:

Findings: There is a {size} mm nodule in the {location}. The nodule demonstrates {texture} attenuation with {margin} margins. {spiculation_desc}. {calcification_desc}. {lobulation_desc}. No significant lymphadenopathy is observed. The remaining lung parenchyma is unremarkable.

Impression: {malignancy_desc} pulmonary nodule ({size} mm, {texture}), Lung-RADS {lungrads}. {followup}
""".strip()


# ═══════════════════════════════════════════════════════════════
# 增强模板报告生成（DeepSeek 不可用时的 fallback——远优于旧版单行模板）
# ═══════════════════════════════════════════════════════════════

def lung_rads(malignancy: int, texture: int, diameter_mm: float) -> tuple:
    """根据特征推断 Lung-RADS 分级和随访建议"""
    # 简化规则（参考 ACR Lung-RADS v2022）
    if diameter_mm < 6:
        return "1", "继续年度肺癌筛查"
    if malignancy <= 2 and texture >= 4 and diameter_mm < 8:
        return "2", "建议12个月CT随访"
    if malignancy <= 2 and (texture <= 3 or diameter_mm >= 8):
        return "3", "建议6个月CT随访"
    if malignancy == 3:
        if texture <= 3 or diameter_mm >= 8:
            return "4A", "建议3个月CT随访或考虑PET-CT"
        return "3", "建议6个月CT随访"
    if malignancy == 4:
        return "4B", "建议胸外科会诊，考虑活检或短期随访"
    if malignancy >= 5:
        return "4X", "高度可疑恶性，建议立即胸外科会诊、PET-CT及组织活检"
    return "3", "建议6-12个月CT随访"


def lobe_from_coord(x: float, y: float, z: float) -> str:
    """根据CT世界坐标推断肺叶位置（基于解剖学基准）"""
    # 左右肺: LUNA16坐标系 x<0 为右肺, x>0 为左肺
    if x < 0:
        side = "右"
    elif x > 0:
        side = "左"
    else:
        side = "纵隔"

    # 上下叶: z坐标 (头足方向, 粗略划分)
    # 右上叶/左上叶 → z较高 (头侧), 右中叶 → z中等, 右下叶/左下叶 → z较低 (足侧)
    if z > 50:
        lobe = "上叶"
    elif z > -50:
        lobe = "中叶" if side == "右" else "上叶/舌段"
    else:
        lobe = "下叶"

    return f"{side}{lobe}"


def build_enhanced_report(feats: dict, lang: str = "cn") -> str:
    """使用 LIDC 特征构建增强模板报告"""
    diameter = feats.get("diameter_mm", 10)
    coord_x = feats.get("coordX", 0)
    coord_y = feats.get("coordY", 0)
    coord_z = feats.get("coordZ", 0)
    location = lobe_from_coord(coord_x, coord_y, coord_z)

    texture = feats.get("texture", 5)
    margin = feats.get("margin", 3)
    spiculation = feats.get("spiculation", 1)
    lobulation = feats.get("lobulation", 1)
    malignancy = feats.get("malignancy", 3)
    calcification = feats.get("calcification", 6)

    texture_desc = feats.get("texture_desc", "实性")
    margin_desc = feats.get("margin_desc", "边界欠清")
    spiculation_desc = feats.get("spiculation_desc", "无毛刺")
    lobulation_desc = feats.get("lobulation_desc", "无分叶")
    malignancy_desc = feats.get("malignancy_desc", "不确定")
    calcification_desc = feats.get("calcification_desc", "无钙化")

    lrads, followup = lung_rads(malignancy, texture, diameter)

    if lang == "cn":
        # 毛刺描述
        spi_text = "未见毛刺征" if spiculation <= 1 else (
            "可见轻度毛刺" if spiculation <= 2 else (
            "可见中度毛刺征" if spiculation <= 3 else (
            "可见明显毛刺征" if spiculation <= 4 else "可见显著毛刺征，提示恶性可能")))

        # 分叶描述
        lob_text = "边缘光滑无分叶" if lobulation <= 1 else (
            "边缘轻度分叶" if lobulation <= 2 else (
            "边缘中度分叶状" if lobulation <= 3 else (
            "边缘明显分叶" if lobulation <= 4 else "边缘显著分叶状改变")))

        report = f"""**肺部CT结节分析报告**

**影像学发现**
- 位置：{location}
- 大小：{diameter:.1f}mm
- 密度类型：{texture_desc}
- 边界特征：{margin_desc}，{lob_text}
- 毛刺征：{spi_text}
- 钙化：{calcification_desc}

**恶性风险评估**
该结节综合评估为{malignancy_desc}（恶性度评分 {malignancy}/5）。
根据影像学特征，Lung-RADS 分级为 {lrads}。

**随访建议**
{followup}

⚠️ 免责声明：本报告由AI辅助诊断系统生成，仅供临床参考，不能替代专业医师诊断。
"""
    else:
        spi_text = "no spiculation" if spiculation <= 1 else (
            "mild spiculation" if spiculation <= 2 else (
            "moderate spiculation" if spiculation <= 3 else "prominent spiculation"))

        lob_text = "smooth margins without lobulation" if lobulation <= 1 else (
            "mild lobulation" if lobulation <= 2 else (
            "moderate lobulation" if lobulation <= 3 else "prominent lobulation"))

        report = f"""**Chest CT Nodule Analysis Report**

**Findings**
- Location: {location}
- Size: {diameter:.1f} mm
- Attenuation: {texture_desc}
- Margins: {margin_desc} with {lob_text}
- Spiculation: {spi_text}
- Calcification: {calcification_desc}

**Assessment**
{malignancy_desc} (malignancy score {malignancy}/5). Lung-RADS {lrads}.

**Recommendation**
{followup}

⚠️ Disclaimer: This report is AI-generated for clinical reference only.
"""

    return report


# ═══════════════════════════════════════════════════════════════
# DeepSeek API 报告生成
# ═══════════════════════════════════════════════════════════════

def build_deepseek_prompt(feats: dict, ctrate_samples: list, lang: str = "cn") -> list:
    """构建 DeepSeek API 请求消息"""
    diameter = feats.get("diameter_mm", 10)
    coord_x = feats.get("coordX", 0)
    coord_y = feats.get("coordY", 0)
    coord_z = feats.get("coordZ", 0)
    location = lobe_from_coord(coord_x, coord_y, coord_z)

    # 特征摘要
    feat_lines = "\n".join([
        f"  - subtlety (显著性): {feats.get('subtlety', '?')}/5",
        f"  - internal structure (内部结构): {feats.get('internalStructure', '?')}/4",
        f"  - calcification (钙化): {feats.get('calcification', '?')}/6 ({feats.get('calcification_desc', '?')})",
        f"  - sphericity (球形度): {feats.get('sphericity', '?')}/5",
        f"  - margin (边缘): {feats.get('margin', '?')}/5 ({feats.get('margin_desc', '?')})",
        f"  - lobulation (分叶): {feats.get('lobulation', '?')}/5 ({feats.get('lobulation_desc', '?')})",
        f"  - spiculation (毛刺): {feats.get('spiculation', '?')}/5 ({feats.get('spiculation_desc', '?')})",
        f"  - texture (纹理/密度): {feats.get('texture', '?')}/5 ({feats.get('texture_desc', '?')})",
        f"  - malignancy (恶性度): {feats.get('malignancy', '?')}/5 ({feats.get('malignancy_desc', '?')})",
    ])

    # CT-RATE 风格示例
    style_blocks = []
    for i, s in enumerate(ctrate_samples[:3]):
        style_blocks.append(f"--- Example {i+1} ---\n{s[:400]}")

    if lang == "cn":
        system_prompt = """你是一位资深胸科放射科医生。根据以下结节的结构化特征评分，撰写一份专业的CT结节分析报告。

要求：
1. 严格模仿真实放射科报告的写作风格（参考提供的示例）
2. 必须包含：影像学发现（位置/大小/密度/边界/毛刺/钙化/分叶）、恶性风险评估（含Lung-RADS分级）、随访建议
3. 使用专业术语但保持清晰易读
4. 添加AI免责声明
5. 报告长度：200-350字"""
        user_prompt = f"""根据以下LIDC放射科医生标注的特征，撰写肺结节CT分析报告。

结节信息：
- 坐标位置（世界坐标mm）：({coord_x:.0f}, {coord_y:.0f}, {coord_z:.0f}) → {location}
- 直径：{diameter:.1f}mm

4位放射科医生对该结节的独立评分：
{feat_lines}

请注意：上述评分是4位放射科医生（背对背独立评估）的中位数。评分标准为LIDC-IDRI 9维特征体系。

请综合这些特征评分，生成一份完整的放射科诊断报告。"""
    else:
        system_prompt = """You are a senior thoracic radiologist. Based on the structured nodule characteristic scores below, write a professional chest CT nodule analysis report.

Requirements:
1. Follow the writing style of real radiology reports (refer to examples)
2. Must include: Imaging Findings (location/size/attenuation/margin/spiculation/calcification/lobulation), Malignancy Assessment (with Lung-RADS), Follow-up Recommendations
3. Use professional terminology
4. Add AI disclaimer
5. Report length: 150-250 words"""
        user_prompt = f"""Write a chest CT nodule analysis report based on the following LIDC radiologist-annotated features.

Nodule information:
- Location: {location}
- Diameter: {diameter:.1f} mm

Four radiologists' independent ratings (median):
{feat_lines}

Generate a complete diagnostic radiology report."""

    if style_blocks:
        user_prompt += f"\n\nReference report style:\n" + "\n\n".join(style_blocks)

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def call_deepseek(api_key: str, messages: list, model: str = "deepseek-chat") -> str:
    """调用 DeepSeek API 生成报告"""
    import requests

    resp = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 800,
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def build_sft_sample(image_paths: list, report_text: str, feats: dict, lang: str = "cn",
                     use_hint: bool = True) -> dict:
    """构建单条训练样本（OpenAI vision format + 可选结构化临床提示）"""
    instruction = random.choice(INSTRUCTION_TEMPLATES_CN if lang == "cn" else INSTRUCTION_TEMPLATES_EN)

    # 特征始终从 feats 提取（metadata 需要）
    coord_x = feats.get("coordX", 0)
    coord_y = feats.get("coordY", 0)
    coord_z = feats.get("coordZ", 0)
    diameter = feats.get("diameter_mm", 10)

    if use_hint:
        location = lobe_from_coord(coord_x, coord_y, coord_z)
        if lang == "cn":
            hint = STRUCTURED_HINT_CN.format(location=location, diameter_mm=diameter)
        else:
            hint = STRUCTURED_HINT_EN.format(location=location, diameter_mm=diameter)
        full_instruction = instruction + "\n" + hint
    else:
        full_instruction = instruction

    content = []
    for p in image_paths:
        if os.path.exists(p):
            content.append({"type": "image", "image": p})
    content.append({"type": "text", "text": full_instruction})

    return {
        "messages": [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": report_text}]},
        ],
        "metadata": {
            "seriesuid": feats.get("seriesuid", ""),
            "diameter_mm": diameter,
            "malignancy": feats.get("malignancy", 3),
            "texture": feats.get("texture", 5),
            "lang": lang,
        },
    }


def load_ctrate_samples(reports_jsonl: str, n: int = 10) -> list:
    """加载 CT-RATE 报告作为风格示例"""
    samples = []
    if not os.path.exists(reports_jsonl):
        return samples
    with open(reports_jsonl) as f:
        for line in f:
            r = json.loads(line)
            findings = r.get("Findings_EN", "")
            if findings and len(findings) > 100:
                samples.append(findings)
                if len(samples) >= n:
                    break
    return samples


def main():
    parser = argparse.ArgumentParser(description="SFT 数据集构建 v2")
    parser.add_argument("--luna16_dir", default="/root/autodl-tmp/data/LUNA16")
    parser.add_argument("--features", default="/root/autodl-tmp/data/nodule_features.json")
    parser.add_argument("--ctrate_reports", default="/root/autodl-tmp/data/CT-RATE/reports.jsonl")
    parser.add_argument("--output", default="/root/autodl-tmp/data/sft")
    parser.add_argument("--deepseek_api_key", default="",
                        help="DeepSeek API key (不提供则使用增强模板)")
    parser.add_argument("--lang", default="both", choices=["cn", "en", "both"])
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--max_samples", type=int, default=0,
                        help="限制样本数 (0=全部, 用于快速测试)")
    parser.add_argument("--slices_per_nodule", type=int, default=1,
                        help="每个结节用几张切片 (1=单切片兼容旧数据, 3-5=多切片)")
    parser.add_argument("--no_structured_hint", action="store_true",
                        help="去掉结构化临床提示，强迫模型从图像提取特征")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    # API key: argparse > env
    api_key = args.deepseek_api_key or os.environ.get("DEEPSEEK_API_KEY", "")

    print("=" * 60)
    print("  SFT 数据集构建 v2 (LIDC特征 + DeepSeek)")
    print("=" * 60)
    print(f"  特征文件: {args.features}")
    print(f"  报告生成: {'DeepSeek API' if api_key else '增强模板 (无API)'}")
    print(f"  语言: {args.lang}")

    # 1. 加载匹配的特征
    if not os.path.exists(args.features):
        print(f"\n[ERROR] nodule_features.json 不存在: {args.features}")
        print("  请先运行: python data/lidc_match.py")
        return

    with open(args.features) as f:
        all_features = json.load(f)
    print(f"\n[1/4] 结节特征: {len(all_features)} 个")

    if args.max_samples > 0:
        all_features = all_features[:args.max_samples]
        print(f"  → 限制为 {args.max_samples} 个")

    # 2. 加载 CT-RATE 风格示例（供 DeepSeek few-shot）
    ctrate_samples = load_ctrate_samples(args.ctrate_reports, n=5)
    print(f"[2/4] CT-RATE 风格示例: {len(ctrate_samples)} 条")

    # 3. 为每个结节生成报告 & 构建样本
    print("[3/4] 生成报告...")
    images_dir = os.path.join(args.luna16_dir, "images")
    png_dir = os.path.join(args.luna16_dir, "images_png")
    samples_cn, samples_en = [], []

    for feats in tqdm(all_features, desc="构建样本"):
        suid = feats["seriesuid"]

        # ── 图像查找（多切片优先）──
        import glob as _glob
        if args.slices_per_nodule > 1:
            # 多视图模式优先: {seriesuid}_nodule_*_{axial|coronal|sagittal}.png
            mv_pattern = os.path.join(png_dir, f"{suid}_nodule_*_axial.png")
            mv_candidates = sorted(_glob.glob(mv_pattern))
            if mv_candidates:
                # 找到所有该结节的多视图文件 (axial + coronal + sagittal)
                prefix = mv_candidates[0].replace("_axial.png", "")
                all_views = sorted(_glob.glob(prefix + "_*.png"))
                image_paths = all_views[:args.slices_per_nodule * 3]
            else:
                # fallback 1: 多切片模式 {seriesuid}_nodule_*_slice_*.png
                png_pattern = os.path.join(png_dir, f"{suid}_nodule_*_slice_*.png")
                png_candidates = sorted(_glob.glob(png_pattern))
                if png_candidates:
                    image_paths = png_candidates[:args.slices_per_nodule]
                else:
                    # fallback 2: 旧格式 {seriesuid}_slice_*.png
                    alt_pattern = os.path.join(png_dir, f"{suid}_slice_*.png")
                    image_paths = sorted(_glob.glob(alt_pattern))[:args.slices_per_nodule]
        else:
            image_paths = []

        if not image_paths:
            # 单切片兼容: {seriesuid}.png 或 .mhd
            png_path = os.path.join(png_dir, f"{suid}.png")
            mhd_path = os.path.join(images_dir, f"{suid}.mhd")
            if os.path.exists(png_path):
                image_paths = [png_path]
            elif os.path.exists(mhd_path):
                image_paths = [mhd_path]
            else:
                continue

        # 生成报告
        if api_key:
            try:
                # 中文
                messages_cn = build_deepseek_prompt(feats, ctrate_samples, lang="cn")
                report_cn = call_deepseek(api_key, messages_cn)
                time.sleep(0.3)  # rate limit
                # 英文
                messages_en = build_deepseek_prompt(feats, ctrate_samples, lang="en")
                report_en = call_deepseek(api_key, messages_en)
                time.sleep(0.3)
            except Exception as e:
                print(f"\n[WARN] DeepSeek API 失败 ({suid[:16]}...): {e}")
                print("  降级为增强模板")
                api_key = ""  # 后续全部 fallback
                report_cn = build_enhanced_report(feats, "cn")
                report_en = build_enhanced_report(feats, "en")
        else:
            report_cn = build_enhanced_report(feats, "cn")
            report_en = build_enhanced_report(feats, "en")

        s_cn = build_sft_sample(image_paths, report_cn, feats, lang="cn",
                                 use_hint=not args.no_structured_hint)
        samples_cn.append(s_cn)
        s_en = build_sft_sample(image_paths, report_en, feats, lang="en",
                                 use_hint=not args.no_structured_hint)
        samples_en.append(s_en)

    # 4. 合并 & 划分
    print("[4/4] 划分 train/val...")
    samples = samples_cn + samples_en if args.lang == "both" else (
        samples_cn if args.lang == "cn" else samples_en)
    random.shuffle(samples)

    # 按 seriesuid 分组 split
    suids = list(set(s["metadata"]["seriesuid"] for s in samples))
    random.shuffle(suids)
    n_val = max(1, int(len(suids) * args.val_split))
    val_suids = set(suids[:n_val])

    train = [s for s in samples if s["metadata"]["seriesuid"] not in val_suids]
    val = [s for s in samples if s["metadata"]["seriesuid"] in val_suids]

    for name, data in [("sft_train", train), ("sft_val", val)]:
        path = os.path.join(args.output, f"{name}.jsonl")
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  {name}: {len(data)} 条 → {path}")

    print(f"\n[DONE] SFT 数据集: {len(train)}/{len(val)} (train/val)")
    if args.slices_per_nodule > 1:
        print(f"  🔬 多切片模式: {args.slices_per_nodule} 层/结节")
    if not api_key:
        print("  ℹ️  未使用 DeepSeek API，报告为增强模板。")
        print("  要使用真实报告风格，请提供 --deepseek_api_key")

    print("\n下一步: python training/stage1_sft.py --data_dir " + args.output)


if __name__ == "__main__":
    main()
