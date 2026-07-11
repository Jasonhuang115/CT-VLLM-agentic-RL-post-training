#!/usr/bin/env python3
"""
DeepSeek API 生成 DPO 偏好对 (真实放射科风格)
Mac 上跑, 不需要 GPU

使用:
  export DEEPSEEK_API_KEY="sk-xxx"
  python data/generate_dpo_with_ds.py \
    --features ~/Downloads/nodule_features.json \
    --output ~/Downloads/dpo_ds_real \
    --n_pairs 90 --seed 42
"""

import json, os, sys, random, argparse, time
from openai import OpenAI

INSTRUCTION = ("请分析这张肺部CT图像中的结节，提供完整的影像学分析报告，"
               "包括结节位置、大小、形态特征、密度类型、边界特征、钙化状态、"
               "恶性风险评估及Lung-RADS分级，并给出随访建议。")

# 六大类错误的 rejected 说明
ERROR_DESCRIPTIONS = {
    "severity_downgrade": (
        "请在报告中故意将特征的严重度低报一级。例如：中度毛刺说成轻度，可疑恶性说成不确定，"
        "边缘模糊不清说成边界欠清。但不要像模板一样填词，要自然融入报告。"
    ),
    "feature_omission": (
        "请在报告中有意遗漏一个GT中存在的关键特征。例如：GT有分叶但完全不提，"
        "GT有钙化但报告写未见明确钙化。遗漏要自然，不能像故意跳过的。"
    ),
    "feature_misdescription": (
        "请在报告中将一个关键特征描述成与其GT相反的值。例如：实性说成磨玻璃，"
        "边界模糊说成清晰，有毛刺说成无毛刺。但要自然融入，不像填反义词。"
    ),
    "over_precision": (
        "请在报告中给出不合理的过度精确数值。例如：直径输出 12.3192mm 而非约12.3mm，"
        "或写上具体的 CT 坐标、HU 值（即使你不知道）。让精确度看起来超出实际测量能力。"
    ),
    "over_hedging": (
        "请在报告中大量使用不确定措辞，对每项发现都加'可能''不排除''建议结合'等限定词，"
        "即使GT特征很明确也要显得犹豫不决。最终评级要比GT低1-2级。"
    ),
    "lr_mismatch": (
        "请在报告中正确地描述所有影像特征，但最后给出的 Lung-RADS 分级比正确定级低1-2级，"
        "随访建议也相应放宽。让分级与所描述的特征明显不一致。"
    ),
}

SYSTEM_CHOSEN = """你是资深胸部放射科医生。根据以下结节的结构化特征，撰写一份完整的CT结节分析报告。

要求：
- 中文书写，专业但通俗
- 包含: 影像学发现、形态学评估、恶性风险评估(Lung-RADS)、随访建议、免责声明
- 自然语言，不是填模板，不同报告要用不同措辞
- 准确反映给定的所有特征，不要编造改变特征值
- 风格上请参照真实放射科报告的简洁直接，例如:
  * "A nonspecific parenchymal nodule with a diameter of 2.5 mm was observed in the laterobasal segment of the lower lobe of the left lung."
  * "In the evaluation of both lung parenchyma; No suspicious mass, nodule or infiltration was detected in both lungs."
  * "Ground-glass-like centriacinar nodules were observed in both upper lobe and lower lobe superior segments of both lungs."
  注意这些报告: 以发现为核心、不啰嗦、不堆砌形容词、不搞长篇大论的理论科普"""

SYSTEM_REJECTED = """你是胸部放射科医生，但要模拟写一份"有特定错误"的报告。
根据结构特征和指定的错误类型，写一份含有精细错误但看起来仍然可信的报告。
错误要自然融入，不能像故意填错。不同报告要用不同措辞。"""


def nodule_to_text(f: dict) -> str:
    """结节特征 → 简洁文本描述"""
    texture_map = {1:"非实性/纯磨玻璃", 2:"部分实性/混合磨玻璃", 3:"部分实性", 4:"实性", 5:"实性"}
    margin_map = {1:"边界清晰", 2:"边界较清晰", 3:"边界欠清", 4:"边缘模糊不清", 5:"边缘模糊伴晕征"}
    spi_map = {1:"未见毛刺征", 2:"可见轻度毛刺", 3:"可见中度毛刺征", 4:"可见明显毛刺征", 5:"可见显著毛刺征"}
    lob_map = {1:"无分叶", 2:"轻度分叶", 3:"中度分叶状改变", 4:"明显分叶", 5:"显著分叶状改变"}
    calc_map = {1:"爆米花样钙化", 2:"层状钙化", 3:"实性钙化", 4:"非中心钙化", 5:"点状钙化", 6:"未见钙化"}
    mal_map = {1:"高度良性(1/5)", 2:"良性可能大(2/5)", 3:"不确定/中等风险(3/5)", 4:"可疑恶性(4/5)", 5:"高度可疑恶性(5/5)"}
    spher_map = {1:"线状", 2:"卵圆形", 3:"类圆形", 4:"不规则形", 5:"极不规则"}

    x, y, z = f["coordX"], f["coordY"], f["coordZ"]
    side = "右" if x < -50 else ("左" if x > 50 else "纵隔旁")
    if z > 100: L = "上叶"
    elif z > -100: L = "中叶" if side == "右" else "上叶/舌段"
    else: L = "下叶"
    loc = f"{side}肺{L}"

    return f"""位置: {loc}
最大径: {f['diameter_mm']:.1f}mm
形态: {spher_map.get(f['sphericity'], '不规则')}
密度类型: {texture_map.get(f['texture'], '实性')} (评分{f['texture']})
边界特征: {margin_map.get(f['margin'], '边界欠清')} (评分{f['margin']})
毛刺: {spi_map.get(f['spiculation'], '未见毛刺')} (评分{f['spiculation']})
分叶: {lob_map.get(f['lobulation'], '无分叶')} (评分{f['lobulation']})
钙化: {calc_map.get(f['calcification'], '未见钙化')} (评分{f['calcification']})
恶性评估: {mal_map.get(f['malignancy'], '不确定')} (评分{f['malignancy']})"""


def call_ds(client, system, prompt, temperature=0.7):
    """调 DeepSeek API"""
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=800,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"  API error (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="~/Downloads/nodule_features.json")
    p.add_argument("--output", default="~/Downloads/dpo_ds_real")
    p.add_argument("--n_pairs", type=int, default=90)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--api_key", default=None)
    p.add_argument("--val_split", type=float, default=0.15)
    args = p.parse_args()

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("请设置 DEEPSEEK_API_KEY 环境变量或用 --api_key")
        return

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")

    with open(os.path.expanduser(args.features)) as f:
        feats = json.load(f)

    random.seed(args.seed)
    random.shuffle(feats)

    # 6 类各选 n_pairs//6 个
    per = max(10, args.n_pairs // 6)
    categories = list(ERROR_DESCRIPTIONS.keys())
    selected = []
    assigned = set()

    rules = [
        ("severity_downgrade", lambda f: f["spiculation"]>=3 or f["malignancy"]>=3),
        ("feature_omission", lambda f: f["calcification"]<=4 or f["spiculation"]>=3 or f["lobulation"]>=2),
        ("feature_misdescription", lambda f: f["texture"]>=4 or f["margin"]>=3),
        ("over_precision", lambda f: True),
        ("over_hedging", lambda f: f["malignancy"]>=3 and f["spiculation"]>=2),
        ("lr_mismatch", lambda f: f["malignancy"]>=3 and f["diameter_mm"]>=8),
    ]

    for cat, pred in rules:
        pool = [f for f in feats if pred(f)]
        random.shuffle(pool)
        n = 0
        for f in pool:
            if f["seriesuid"] not in assigned and n < per:
                selected.append((cat, f))
                assigned.add(f["seriesuid"])
                n += 1

    print(f"选中 {len(selected)} 个结节 ({len(set(c for c,_ in selected))} 类)")

    pairs = []
    for i, (cat, f) in enumerate(selected):
        features_text = nodule_to_text(f)
        suid = f["seriesuid"]

        # Chosen
        print(f"[{i+1}/{len(selected)}] {cat} chosen...", end=" ", flush=True)
        chosen = call_ds(client, SYSTEM_CHOSEN,
            f"结节特征:\n{features_text}\n\n请生成正确的放射科报告。")
        if not chosen:
            print("FAIL")
            continue
        print(f"{len(chosen)}字")

        # Rejected
        print(f"              rejected ({cat})...", end=" ", flush=True)
        err_desc = ERROR_DESCRIPTIONS[cat]
        rejected = call_ds(client, SYSTEM_REJECTED,
            f"结节特征:\n{features_text}\n\n错误类型: {err_desc}\n\n请生成含该类型错误的报告。")
        if not rejected:
            print("FAIL")
            continue
        print(f"{len(rejected)}字")

        # 图像路径
        IMG_BASE = "/root/autodl-tmp/data/LUNA16/images_png"
        prompt_content = []
        for view in ["axial", "coronal", "sagittal"]:
            prompt_content.append({"type": "image", "image": f"{IMG_BASE}/{suid}_nodule_000_{view}.png"})
        prompt_content.append({"type": "text", "text": INSTRUCTION})

        pairs.append({
            "prompt_messages": [{"role": "user", "content": prompt_content}],
            "chosen": chosen,
            "rejected": rejected,
            "metadata": {
                "error_type": cat,
                "seriesuid": suid,
                "diameter_mm": f["diameter_mm"],
                "gt_texture": f["texture"],
                "gt_malignancy": f["malignancy"],
                "gt_margin": f["margin"],
                "gt_spiculation": f["spiculation"],
                "gt_lobulation": f["lobulation"],
            }
        })
        time.sleep(0.3)  # rate limit

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

    print(f"\n[DONE] {len(pairs)} 对 DeepSeek 生成 DPO 偏好数据")


if __name__ == "__main__":
    main()
