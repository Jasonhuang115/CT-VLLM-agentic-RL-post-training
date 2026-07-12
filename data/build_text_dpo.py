#!/usr/bin/env python3
"""
纯文本 DPO 偏好对构造

不需要 CT 图像。prompt = SFT 视觉描述(chapter 1),
chosen = 专业诊断语言(chapter 2-4, CT-RATE风格),
rejected = 模板版诊断语言

使用:
  export DEEPSEEK_API_KEY="sk-xxx"
  python data/build_text_dpo.py \
    --features ~/Downloads/nodule_features.json \
    --ctrate_reports ~/Downloads/reports.jsonl \
    --output ~/Downloads/dpo_text \
    --n_pairs 90 --seed 42
"""

import json, os, sys, random, re, argparse, time


# ════════════════════════════════════
# LIDC 特征 → 自然语言视觉描述
# ════════════════════════════════════

TEXTURE_CN = {1: "纯磨玻璃密度", 2: "部分实性密度", 3: "部分实性密度", 4: "实性密度", 5: "实性密度"}
MARGIN_CN = {1: "边界清晰", 2: "边界较清晰", 3: "边界欠清", 4: "边缘模糊不清", 5: "边缘模糊伴晕征"}
SPI_CN = {1: "未见明确毛刺征", 2: "可见轻度毛刺", 3: "可见中度毛刺征", 4: "可见明显毛刺征", 5: "可见显著毛刺征"}
LOB_CN = {1: "边缘光滑，无分叶", 2: "伴轻度分叶", 3: "伴中度分叶状改变", 4: "伴明显分叶", 5: "伴显著分叶状改变"}
CALC_CN = {1: "可见爆米花样钙化", 2: "可见层状钙化", 3: "可见实性钙化", 4: "可见非中心钙化", 5: "可见点状钙化", 6: "未见明确钙化"}
MAL_CN = {1: "高度提示良性", 2: "良性可能大", 3: "不确定/中等风险", 4: "可疑恶性", 5: "高度可疑恶性"}
SPH_CN = {1: "线状", 2: "卵圆形", 3: "类圆形", 4: "不规则形", 5: "极不规则"}


def lobe(x, y, z):
    side = "右" if x < -50 else ("左" if x > 50 else "纵隔旁")
    if z > 100: L = "上叶"
    elif z > -100: L = "中叶" if side == "右" else "上叶/舌段"
    else: L = "下叶"
    return f"{side}肺{L}"


def build_findings(f: dict) -> str:
    """LIDC 特征 → 自然语言影像学发现"""
    d = f["diameter_mm"]
    loc = lobe(f["coordX"], f["coordY"], f["coordZ"])
    tex = TEXTURE_CN.get(f["texture"], "实性密度")
    mg = MARGIN_CN.get(f["margin"], "边界欠清")
    spi = SPI_CN.get(f["spiculation"], "未见明确毛刺征")
    lob = LOB_CN.get(f["lobulation"], "边缘光滑，无分叶")
    ca = CALC_CN.get(f["calcification"], "未见明确钙化")
    sph = SPH_CN.get(f["sphericity"], "不规则形")
    mal = MAL_CN.get(f["malignancy"], "不确定")

    # 多个报告变体
    variants = [
        f"{loc}可见一{d:.1f}mm {tex}结节，呈{sph}，{mg}。{lob}，{spi}。内部{ca}。周围肺组织未见明确异常，双肺门及纵隔未见肿大淋巴结。",
        f"于{loc}发现一实性结节，最大径约{d:.1f}mm。结节{sph}，{mg}。{lob}，{spi}。结节内{ca}。邻接胸膜未见明显异常。",
        f"{loc}一{d:.1f}mm结节，{tex}，{mg}。{lob}，{spi}。病灶内{ca}。余肺野清晰。",
    ]
    return random.choice(variants)


def load_ctrate_impressions(reports_path: str, n: int = 5) -> list:
    """从CT-RATE抽取 Impressions 作为风格示例"""
    impressions = []
    with open(reports_path) as f:
        for line in f:
            r = json.loads(line)
            imp = r.get("Impressions_EN") or ""
            if len(imp) > 80 and "nodule" in imp.lower():
                impressions.append(imp)
    random.shuffle(impressions)
    return impressions[:n]


def call_ds(client, system, prompt, temp=0.7):
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": prompt}],
                temperature=temp, max_tokens=600)
            return resp.choices[0].message.content
        except Exception as e:
            print(f"  API retry {attempt+1}: {e}")
            time.sleep(2 ** attempt)
    return None


# ════════════════════════════════════
# Main
# ════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="~/Downloads/nodule_features.json")
    p.add_argument("--ctrate_reports", default="~/Downloads/reports.jsonl")
    p.add_argument("--output", default="~/Downloads/dpo_text")
    p.add_argument("--n_pairs", type=int, default=90)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--api_key", default=None)
    p.add_argument("--val_split", type=float, default=0.15)
    p.add_argument("--no_api", action="store_true", help="仅模板模式 (不用API)")
    args = p.parse_args()

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    client = None
    if api_key and not args.no_api:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
        print(f"[INFO] DeepSeek API 已连接")
    else:
        print(f"[INFO] 无 API，使用增强模板 (chosen=增强模板, rejected=旧模板)")

    random.seed(args.seed)

    # 加载结节特征
    with open(os.path.expanduser(args.features)) as f:
        feats = json.load(f)
    random.shuffle(feats)
    feats = feats[:args.n_pairs]

    # 加载 CT-RATE 风格示例
    style_examples = []
    if os.path.exists(os.path.expanduser(args.ctrate_reports)):
        style_examples = load_ctrate_impressions(
            os.path.expanduser(args.ctrate_reports), n=5)
        print(f"[INFO] CT-RATE 示例: {len(style_examples)} 条")

    pairs = []
    for i, f in enumerate(feats):
        findings = build_findings(f)
        mal = MAL_CN.get(f["malignancy"], "不确定")

        # ── chosen: 专业诊断语言 ──
        if client:
            style_text = "\n\n".join(f"示例{i+1}: {s[:200]}" for i, s in enumerate(style_examples))
            chosen_prompt = f"""基于以下CT影像学发现，撰写一段放射科诊断意见和随访建议。

影像学发现:
{findings}

结节特征(GT): 恶性评估 {mal}, 直径 {f['diameter_mm']:.1f}mm, 密度评分 {f['texture']}/5, 毛刺评分 {f['spiculation']}/5

参照以下真实放射科报告风格（注意简洁直接，不堆砌形容词，不做理论科普）:
{style_text}

要求:
- 中文书写，2-3段即可
- 基于特征给出明确的恶性推断（不要"可能...也可能..."）
- 随访建议与恶性评估严重度匹配
- 不写模板式标题（不要"一、影像学发现"这类章节标题）
- 不要重复上述影像学发现的具体描述，直接给诊断意见"""

            chosen = call_ds(client,
                "你是胸部放射科医生。写诊断意见要简洁直接，以发现为核心。",
                chosen_prompt, temp=0.7)
        else:
            chosen = f"该{findings[:50]}...综合分析，结节恶性风险评估为{mal}。鉴于以上特征，建议短期CT随访复查以确认稳定性。\n\n⚠️ 本报告由AI辅助生成，仅供临床参考。"

        if not chosen:
            continue

        # ── rejected: 模板版诊断语言 ──
        rejected = f"""**二、形态学评估**
结节呈不规则形，{MARGIN_CN.get(f['margin'], '边界欠清')}。内部{CALC_CN.get(f['calcification'], '未见钙化')}。综合形态学表现，该结节具有一些不确定特征。

**三、恶性风险评估**
综合分析，恶性风险评级：**{mal}**。根据 Lung-RADS v2022 标准，分级为 Lung-RADS 3。

**四、随访建议**
建议6-12个月CT随访，观察结节变化。建议结合患者年龄、吸烟史、肿瘤家族史等临床危险因素综合决策。

---
⚠️ 本报告由AI辅助诊断系统生成，仅供临床参考，不能替代专业医师诊断。"""

        pairs.append({
            "text_prompt": findings,  # 纯文本prompt (SFT视觉描述)
            "chosen": chosen,         # 专业诊断语言
            "rejected": rejected,     # 模板版诊断语言
            "metadata": {
                "seriesuid": f["seriesuid"],
                "diameter_mm": f["diameter_mm"],
                "malignancy": f["malignancy"],
                "texture": f["texture"],
                "spiculation": f["spiculation"],
            }
        })

        if client:
            print(f"[{i+1}/{len(feats)}] {f['diameter_mm']:.1f}mm mal={f['malignancy']} | chosen={len(chosen)}字")
            time.sleep(0.2)

        if args.no_api and len(pairs) >= 5:
            break  # 无API模式只生成5对示例

    # 保存
    random.shuffle(pairs)
    n_val = int(len(pairs) * args.val_split)
    os.makedirs(os.path.expanduser(args.output), exist_ok=True)

    for name, data in [("dpo_train", pairs[n_val:]), ("dpo_val", pairs[:n_val])]:
        path = os.path.join(os.path.expanduser(args.output), f"{name}.jsonl")
        with open(path, "w") as fout:
            for p in data:
                fout.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"  {name}: {len(data)} 对 → {path}")

    print(f"\n[OK] {len(pairs)} 对纯文本DPO数据")

    # 打印一对示例
    if pairs:
        print("\n====== 示例 ======")
        print(f"Prompt(视觉描述):\n{pairs[0]['text_prompt'][:200]}")
        print(f"\nChosen(专业):\n{pairs[0]['chosen'][:300]}")
        print(f"\nRejected(模板):\n{pairs[0]['rejected'][:300]}")


if __name__ == "__main__":
    main()
