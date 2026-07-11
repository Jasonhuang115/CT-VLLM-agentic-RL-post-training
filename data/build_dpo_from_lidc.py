#!/usr/bin/env python3
"""
DPO 偏好对生成 — 六大类精细错误

chosen:  基于 LIDC GT 特征的正确放射科报告
rejected: 含特定类型错误 (漏报/误判/降级/假精度/过度委婉/LR不匹配)

使用:
  python data/build_dpo_from_lidc.py \
    --features ~/Downloads/nodule_features.json \
    --output ~/Downloads/dpo_lidc_v2 \
    --total_pairs 90 --seed 42
"""

import json, random, argparse, os


# LIDC → 中文
TEXTURE_CN = {1: ("非实性/纯磨玻璃密度","磨玻璃"), 2: ("部分实性/混合磨玻璃密度","部分实性"),
              3: ("部分实性/混合磨玻璃密度","部分实性"), 4: ("实性密度","实性"), 5: ("实性密度","实性")}
MARGIN_CN = {1:"边界清晰",2:"边界较清晰",3:"边界欠清",4:"边缘模糊不清",5:"边缘模糊，伴晕征"}
SPICULATION_CN = {1:("未见明确毛刺征",""),2:("可见轻度毛刺","轻度毛刺"),3:("可见中度毛刺征","中度毛刺征"),
                  4:("可见明显毛刺征","明显毛刺征"),5:("可见显著毛刺征，提示恶性可能","显著毛刺征")}
LOBULATION_CN = {1:("边缘光滑，无分叶",""),2:("伴轻度分叶","轻度分叶"),3:("伴中度分叶状改变","中度分叶状改变"),
                 4:("伴明显分叶","明显分叶"),5:("伴显著分叶状改变","显著分叶状改变")}
CALCIFICATION_CN = {1:"可见爆米花样钙化",2:"可见层状钙化",3:"可见实性钙化",4:"可见非中心钙化",5:"可见点状钙化",6:"未见明确钙化"}
MALIGNANCY_CN = {1:("高度提示良性","Lung-RADS 1","继续年度低剂量CT筛查"),
                 2:("良性可能大","Lung-RADS 2","建议12个月CT随访"),
                 3:("不确定/中等风险","Lung-RADS 3","建议6-12个月CT随访"),
                 4:("可疑恶性","Lung-RADS 4A","建议3个月CT随访，如持续存在应行PET-CT评估"),
                 5:("高度可疑恶性","Lung-RADS 4B","建议立即增强CT检查，考虑PET-CT或组织活检")}
SPHERICITY_CN = {1:"线状",2:"卵圆形",3:"类圆形",4:"不规则形",5:"极不规则"}


def lobe(x,y,z):
    side="右" if x<-50 else ("左" if x>50 else "纵隔旁")
    if z>100: L="上叶"
    elif z>-100: L="中叶" if side=="右" else "上叶/舌段"
    else: L="下叶"
    return f"{side}肺{L}"


INSTRUCTION = ("请分析这张肺部CT图像中的结节，提供完整的影像学分析报告，"
               "包括结节位置、大小、形态特征、密度类型、边界特征、钙化状态、"
               "恶性风险评估及Lung-RADS分级，并给出随访建议。")


def build_chosen(f: dict) -> str:
    d=f["diameter_mm"]; loc=lobe(f["coordX"],f["coordY"],f["coordZ"])
    tx,tx_s=TEXTURE_CN.get(f["texture"],("实性密度","实性"))
    mg=MARGIN_CN.get(f["margin"],"边界欠清")
    sp,sp_s=SPICULATION_CN.get(f["spiculation"],("未见明确毛刺征",""))
    lb,lb_s=LOBULATION_CN.get(f["lobulation"],("边缘光滑，无分叶",""))
    ca=CALCIFICATION_CN.get(f["calcification"],"未见明确钙化")
    ma,lr,fu=MALIGNANCY_CN.get(f["malignancy"],("不确定","Lung-RADS 3","建议6个月CT随访"))
    sph=SPHERICITY_CN.get(f["sphericity"],"不规则形")

    if f["malignancy"]>=4 and (f["spiculation"]>=3 or f["lobulation"]>=3 or d>=15):
        comment="多项高危影像学特征并存，恶性风险显著增高。"
    elif f["malignancy"]>=4: comment="恶性度评分较高，需积极评估。"
    elif f["malignancy"]==3: comment="具有一些不确定特征，建议短期随访进一步评估。"
    else: comment="影像学特征倾向良性，恶性风险较低。"

    return f"""**肺部CT结节分析报告**

**一、影像学发现**
- 位置：{loc}
- 大小：最大径约 {d:.1f}mm
- 形态：呈{sph}{f"，{lb_s}" if lb_s else ""}{f"，{sp_s}" if sp_s else ""}
- 密度特征：结节呈{tx}
- 边界特征：{mg}
- 钙化状态：{ca}
- 周围结构：邻近胸膜未见明显异常，周围肺组织未见卫星灶。双肺门及纵隔未见肿大淋巴结。

**二、形态学评估**
{d:.1f}mm {tx_s}结节，{mg}。{lb}，{sp}。内部{ca}。{comment}

**三、恶性风险评估**
综合分析，恶性风险评级：**{ma}**。根据 Lung-RADS v2022 标准，分级为 **{lr}**。

**四、随访建议**
{fu}。

---
⚠️ 本报告由AI辅助诊断系统生成，仅供临床参考，不能替代专业医师诊断。"""


# ═══ 六大类 Rejected ═══

def rejected_severity_downgrade(f):
    d=f["diameter_mm"]; loc=lobe(f["coordX"],f["coordY"],f["coordZ"])
    ds={5:2,4:2,3:1}; dl={5:2,4:2,3:1,2:1}; dm={5:3,4:2,3:1}; dmargin={5:3,4:2,3:2}
    s2=ds.get(f["spiculation"],f["spiculation"])
    l2=dl.get(f["lobulation"],f["lobulation"])
    m2=dm.get(f["malignancy"],f["malignancy"])
    mg2=dmargin.get(f["margin"],f["margin"])
    tx,tx_s=TEXTURE_CN.get(f["texture"],("实性密度","实性"))
    mg=MARGIN_CN.get(mg2,"边界较清晰")
    sp,sp_s=SPICULATION_CN.get(s2,("未见明确毛刺征",""))
    lb,lb_s=LOBULATION_CN.get(l2,("边缘光滑，无分叶",""))
    ca=CALCIFICATION_CN.get(f["calcification"],"未见明确钙化")
    ma,lr,fu=MALIGNANCY_CN.get(m2,("不确定","Lung-RADS 3","建议6个月CT随访"))
    sph=SPHERICITY_CN.get(f["sphericity"],"不规则形")
    return f"""**肺部CT结节分析报告**

**一、影像学发现**
- 位置：{loc}
- 大小：最大径约 {d:.1f}mm
- 形态：呈{sph}{f"，{lb_s}" if lb_s else ""}{f"，{sp_s}" if sp_s else ""}
- 密度特征：结节密度特征不典型，需结合其他序列评估
- 边界特征：{mg}
- 钙化状态：{ca}
- 周围结构：邻近胸膜未见明显异常。双肺门及纵隔未见肿大淋巴结。

**二、形态学评估**
{d:.1f}mm {tx_s}结节，{mg}。{lb}，{sp}。内部{ca}。由于部分影像学特征不够典型，建议结合临床和其他检查综合评估。

**三、恶性风险评估**
综合分析，恶性风险评级：**{ma}**。Lung-RADS 分级为 **{lr}**。

**四、随访建议**
{fu}。

---
⚠️ 本报告由AI辅助诊断系统生成，仅供临床参考，不能替代专业医师诊断。"""


def rejected_feature_omission(f):
    d=f["diameter_mm"]; loc=lobe(f["coordX"],f["coordY"],f["coordZ"])
    tx,tx_s=TEXTURE_CN.get(f["texture"],("实性密度","实性"))
    mg=MARGIN_CN.get(f["margin"],"边界欠清")
    sph=SPHERICITY_CN.get(f["sphericity"],"不规则形")
    ca=CALCIFICATION_CN.get(f["calcification"],"未见明确钙化")
    ca_omit=f["calcification"]<=4
    ca_text="未见明确钙化" if ca_omit else ca
    ma,lr,fu=MALIGNANCY_CN.get(f["malignancy"],("不确定","Lung-RADS 3","建议6个月CT随访"))
    return f"""**肺部CT结节分析报告**

**一、影像学发现**
- 位置：{loc}
- 大小：最大径约 {d:.1f}mm
- 形态：呈{sph}，边缘光滑
- 密度特征：结节呈{tx}
- 边界特征：{mg}
- 钙化状态：{ca_text}
- 周围结构：邻近胸膜未见明显异常。双肺门及纵隔未见肿大淋巴结。

**二、形态学评估**
{d:.1f}mm {tx_s}结节，{mg}。未见明确毛刺征及分叶状改变。内部{ca_text}。

**三、恶性风险评估**
综合分析，恶性风险评级：**{ma}**。Lung-RADS 分级为 **{lr}**。

**四、随访建议**
{fu}。

---
⚠️ 本报告由AI辅助诊断系统生成，仅供临床参考，不能替代专业医师诊断。"""


def rejected_feature_misdescription(f):
    d=f["diameter_mm"]; loc=lobe(f["coordX"],f["coordY"],f["coordZ"])
    ft=1 if f["texture"]>=4 else 5; fm=1 if f["margin"]>=3 else 5
    tx,tx_s=TEXTURE_CN.get(ft,("实性密度","实性"))
    mg=MARGIN_CN.get(fm,"边界较清晰")
    sp,_=SPICULATION_CN.get(1,("未见明确毛刺征",""))
    lb,_=LOBULATION_CN.get(1,("边缘光滑，无分叶",""))
    ca=CALCIFICATION_CN.get(f["calcification"],"未见明确钙化")
    sph=SPHERICITY_CN.get(f["sphericity"],"不规则形")
    dm2={5:4,4:3,3:2,2:1}
    ma,lr,fu=MALIGNANCY_CN.get(dm2.get(f["malignancy"],f["malignancy"]),("良性可能大","Lung-RADS 2","建议12个月CT随访"))
    return f"""**肺部CT结节分析报告**

**一、影像学发现**
- 位置：{loc}
- 大小：最大径约 {d:.1f}mm
- 形态：呈{sph}，边缘光滑
- 密度特征：结节呈{tx}
- 边界特征：{mg}
- 钙化状态：{ca}
- 周围结构：邻近胸膜未见明显异常。双肺门及纵隔未见肿大淋巴结。

**二、形态学评估**
{d:.1f}mm {tx_s}结节，{mg}。{lb}，{sp}。内部{ca}。

**三、恶性风险评估**
综合分析，恶性风险评级：**{ma}**。Lung-RADS 分级为 **{lr}**。

**四、随访建议**
{fu}。

---
⚠️ 本报告由AI辅助诊断系统生成，仅供临床参考，不能替代专业医师诊断。"""


def rejected_over_precision(f):
    d=f["diameter_mm"]; loc=lobe(f["coordX"],f["coordY"],f["coordZ"])
    tx,tx_s=TEXTURE_CN.get(f["texture"],("实性密度","实性"))
    mg=MARGIN_CN.get(f["margin"],"边界欠清")
    sp,sp_s=SPICULATION_CN.get(f["spiculation"],("未见明确毛刺征",""))
    lb,lb_s=LOBULATION_CN.get(f["lobulation"],("边缘光滑，无分叶",""))
    ca=CALCIFICATION_CN.get(f["calcification"],"未见明确钙化")
    ma,lr,fu=MALIGNANCY_CN.get(f["malignancy"],("不确定","Lung-RADS 3","建议6个月CT随访"))
    sph=SPHERICITY_CN.get(f["sphericity"],"不规则形")
    return f"""**肺部CT结节分析报告**

**一、影像学发现**
- 位置：{loc}（坐标: X={f['coordX']:.3f}mm, Y={f['coordY']:.3f}mm, Z={f['coordZ']:.3f}mm）
- 大小：最大径 {d:.4f}mm × {(d*0.78):.4f}mm
- 形态：呈{sph}{f"，{lb_s}" if lb_s else ""}{f"，{sp_s}" if sp_s else ""}
- 密度特征：结节呈{tx}，平均 CT 值约 -120HU
- 边界特征：{mg}
- 钙化状态：{ca}
- 周围结构：邻近胸膜未见明显异常。双肺门及纵隔未见肿大淋巴结。

**二、形态学评估**
{d:.1f}mm {tx_s}结节，{mg}。{lb}，{sp}。内部{ca}。

**三、恶性风险评估**
综合分析，恶性风险评级：**{ma}**。Lung-RADS 分级为 **{lr}**。

**四、随访建议**
{fu}。

---
⚠️ 本报告由AI辅助诊断系统生成，仅供临床参考，不能替代专业医师诊断。"""


def rejected_over_hedging(f):
    d=f["diameter_mm"]; loc=lobe(f["coordX"],f["coordY"],f["coordZ"])
    tx,tx_s=TEXTURE_CN.get(f["texture"],("实性密度","实性"))
    mg_raw=MARGIN_CN.get(f["margin"],"边界欠清")
    mg=f"{mg_raw}，但 CT 平扫对该特征的评估存在一定局限"
    sp,sp_s=SPICULATION_CN.get(f["spiculation"],("未见明确毛刺征",""))
    sp_long=f"毛刺征象表现不十分典型，{sp_s}可能" if sp_s else "未见明确毛刺征"
    ca_text=f"{CALCIFICATION_CN.get(f['calcification'],'未见明确钙化')}，但微钙化在平扫 CT 上可能无法可靠显示"
    sph=SPHERICITY_CN.get(f["sphericity"],"不规则形")
    dm={5:3,4:2,3:1}
    ma,lr,fu=MALIGNANCY_CN.get(dm.get(f["malignancy"],3),("不确定","Lung-RADS 3","建议6个月CT随访"))
    return f"""**肺部CT结节分析报告**

**一、影像学发现**
- 位置：{loc}
- 大小：最大径约 {d:.1f}mm
- 形态：呈{sph}
- 密度特征：{tx}，但因缺乏增强对比，密度分型存在不确定性
- 边界特征：{mg}
- 钙化状态：{ca_text}
- 周围结构：邻近胸膜未见明显异常。双肺门及纵隔未见肿大淋巴结。

**二、形态学评估**
{d:.1f}mm {tx_s}结节。{sp_long}。鉴于部分影像学特征在平扫条件下显示不够确切，建议增强 CT 或薄层扫描进一步明确。

**三、恶性风险评估**
综合分析，由于上述特征存在不确定性，目前倾向于**{ma}**，但需影像学随访确认。Lung-RADS 分级暂为 **{lr}**。

**四、随访建议**
{fu}。建议结合患者年龄、吸烟史、肿瘤家族史等临床危险因素综合决策。

---
⚠️ 本报告由AI辅助诊断系统生成，仅供临床参考，不能替代专业医师诊断。"""


def rejected_lr_mismatch(f):
    d=f["diameter_mm"]; loc=lobe(f["coordX"],f["coordY"],f["coordZ"])
    tx,tx_s=TEXTURE_CN.get(f["texture"],("实性密度","实性"))
    mg=MARGIN_CN.get(f["margin"],"边界欠清")
    sp,sp_s=SPICULATION_CN.get(f["spiculation"],("未见明确毛刺征",""))
    lb,lb_s=LOBULATION_CN.get(f["lobulation"],("边缘光滑，无分叶",""))
    ca=CALCIFICATION_CN.get(f["calcification"],"未见明确钙化")
    sph=SPHERICITY_CN.get(f["sphericity"],"不规则形")
    # 特征描述正确，但评级低一级
    from_gt = MALIGNANCY_CN.get(f["malignancy"],("不确定","Lung-RADS 3",""))
    correct_lr = from_gt[1]
    downgrade={"Lung-RADS 4B":"Lung-RADS 4A","Lung-RADS 4A":"Lung-RADS 3","Lung-RADS 3":"Lung-RADS 2","Lung-RADS 2":"Lung-RADS 1"}
    wrong_lr=downgrade.get(correct_lr,"Lung-RADS 2")
    fu_map={"Lung-RADS 2":"建议12个月CT随访","Lung-RADS 3":"建议6个月CT随访","Lung-RADS 4A":"建议3个月CT随访","Lung-RADS 4B":"建议增强CT检查"}
    wrong_fu=fu_map.get(wrong_lr,"建议6个月CT随访")
    return f"""**肺部CT结节分析报告**

**一、影像学发现**
- 位置：{loc}
- 大小：最大径约 {d:.1f}mm
- 形态：呈{sph}{f"，{lb_s}" if lb_s else ""}{f"，{sp_s}" if sp_s else ""}
- 密度特征：结节呈{tx}
- 边界特征：{mg}
- 钙化状态：{ca}
- 周围结构：邻近胸膜未见明显异常。双肺门及纵隔未见肿大淋巴结。

**二、形态学评估**
{d:.1f}mm {tx_s}结节，{mg}。{lb}，{sp}。内部{ca}。

**三、恶性风险评估**
根据上述特征，综合评估为 **{wrong_lr}**。

**四、随访建议**
{wrong_fu}。

---
⚠️ 本报告由AI辅助诊断系统生成，仅供临床参考，不能替代专业医师诊断。"""


BUILDERS = {
    "severity_downgrade": rejected_severity_downgrade,
    "feature_omission": rejected_feature_omission,
    "feature_misdescription": rejected_feature_misdescription,
    "over_precision": rejected_over_precision,
    "over_hedging": rejected_over_hedging,
    "lr_mismatch": rejected_lr_mismatch,
}


def select_nodules(feats, per, seed):
    random.seed(seed)
    assigned = set()
    groups = {}

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
        groups[cat] = []
        for f in pool:
            if f["seriesuid"] not in assigned and len(groups[cat]) < per:
                groups[cat].append(f)
                assigned.add(f["seriesuid"])

    return groups


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="~/Downloads/nodule_features.json")
    p.add_argument("--output", default="/tmp/dpo_lidc_v2")
    p.add_argument("--total_pairs", type=int, default=90)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_split", type=float, default=0.15)
    args = p.parse_args()

    with open(os.path.expanduser(args.features)) as f:
        feats = json.load(f)
    per = max(10, args.total_pairs // 6)
    groups = select_nodules(feats, per, args.seed)

    IMG_BASE = "/root/autodl-tmp/data/LUNA16/images_png"

    pairs = []
    for cat, nodules in groups.items():
        b = BUILDERS[cat]
        for f in nodules:
            suid = f["seriesuid"]

            # 构造多视图图像路径 (3张: axial/coronal/sagittal)
            # 文件命名: {seriesuid}_nodule_{n:03d}_{view}.png
            # 用 glob 匹配第一个结节 (nodule_000)
            import glob as _g
            image_paths = []
            for view in ["axial", "coronal", "sagittal"]:
                pattern = f"{IMG_BASE}/{suid}_nodule_*_{view}.png"
                matches = sorted(_g.glob(pattern))
                if matches:
                    image_paths.append(matches[0])

            # 构建含图像的 prompt
            prompt_content = []
            for p in image_paths[:3]:
                prompt_content.append({"type": "image", "image": p})
            prompt_content.append({"type": "text", "text": INSTRUCTION})

            pairs.append({
                "prompt_messages": [{"role":"user","content": prompt_content}],
                "chosen": build_chosen(f),
                "rejected": b(f),
                "metadata": {"error_type":cat,"seriesuid":suid,
                    "diameter_mm":f["diameter_mm"],"gt_texture":f["texture"],
                    "gt_malignancy":f["malignancy"],"gt_margin":f["margin"],
                    "gt_spiculation":f["spiculation"],"gt_lobulation":f["lobulation"]}
            })

    random.seed(args.seed); random.shuffle(pairs)
    n_val = int(len(pairs) * args.val_split)
    val, train = pairs[:n_val], pairs[n_val:]

    os.makedirs(args.output, exist_ok=True)
    for name, data in [("dpo_train",train),("dpo_val",val)]:
        path = os.path.join(args.output, f"{name}.jsonl")
        with open(path,"w") as fout:
            for p in data:
                fout.write(json.dumps(p,ensure_ascii=False)+"\n")
        print(f"  {name}: {len(data)} 对 → {path}")

    print(f"\n[DONE] {len(pairs)} 对 ({len(groups)} 类别)")
    for cat, nodules in groups.items():
        print(f"  {cat}: {len(nodules)} 对")


if __name__ == "__main__":
    main()
