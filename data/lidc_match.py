#!/usr/bin/env python3
"""
LIDC-IDRI 特征提取 + LUNA16 坐标匹配

通过 pylidc 获取每个结节的 4 位放射科医生标注的 9 维特征，
与 LUNA16 annotations.csv 的结节坐标做 L2 距离匹配，
输出 nodule_features.json 供后续数据构建使用。

使用方式:
  python data/lidc_match.py \
    --luna16_dir /root/autodl-tmp/data/LUNA16 \
    --output /root/autodl-tmp/data/nodule_features.json
"""

import os, sys, json, argparse, csv
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm


# ── 9 维特征名称（LIDC 标准）──
FEATURE_KEYS = [
    "subtlety",           # 1-5  结节显著性
    "internalStructure",  # 1-4  内部结构
    "calcification",      # 1-6  钙化类型 (1=爆米花, 2=层状, 3=实心, 4=非中心, 5=中心, 6=无)
    "sphericity",         # 1-5  球形度 (1=线状, 3=卵圆形, 5=球形)
    "margin",             # 1-5  边缘 (1=锐利, 5=模糊)
    "lobulation",         # 1-5  分叶 (1=无, 5=明显)
    "spiculation",        # 1-5  毛刺 (1=无, 5=明显)
    "texture",            # 1-5  纹理/密度 (1=非实性, 3=部分实性, 5=实性)
    "malignancy",         # 1-5  恶性可能 (1=高度良性, 5=高度恶性)
]

# ── 特征值 → 可读文本映射 ──
TEXTURE_MAP = {1: "非实性/磨玻璃", 2: "非实性/磨玻璃", 3: "部分实性/混合磨玻璃", 4: "实性", 5: "实性"}
MALIGNANCY_MAP = {1: "高度良性可能", 2: "良性可能大", 3: "不确定/中等风险", 4: "可疑恶性", 5: "高度可疑恶性"}
CALCIFICATION_MAP = {
    1: "爆米花样钙化", 2: "层状钙化", 3: "实心钙化",
    4: "非中心钙化", 5: "中心钙化", 6: "无钙化"
}
MARGIN_MAP = {1: "边界锐利清晰", 2: "边界较清晰", 3: "边界欠清", 4: "边界模糊", 5: "边界模糊不清"}
SPICULATION_MAP = {1: "无毛刺", 2: "轻微毛刺", 3: "中度毛刺", 4: "明显毛刺", 5: "显著毛刺征"}
LOBULATION_MAP = {1: "无分叶", 2: "轻微分叶", 3: "中度分叶", 4: "明显分叶", 5: "显著分叶状"}


def load_luna16_annotations(anno_csv: str) -> list:
    """加载 LUNA16 标注"""
    nodules = []
    with open(anno_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            nodules.append({
                "seriesuid": row["seriesuid"],
                "coordX": float(row["coordX"]),
                "coordY": float(row["coordY"]),
                "coordZ": float(row["coordZ"]),
                "diameter_mm": float(row["diameter_mm"]),
            })
    return nodules


def lidc_centroid_mm(cluster) -> np.ndarray:
    """
    pylidc nodule cluster 中心 — ann.centroid 返回体素坐标 (voxel index)。
    LUNA16 coordX/Y/Z 是世界坐标 (mm)，必须通过 SimpleITK 转换坐标系后再比较。
    X/Y 轴序也可能互换，比较时需双试。
    """
    centroids = []
    for ann in cluster:
        c = ann.centroid  # (x, y, z) mm — DICOM patient coordinate system
        centroids.append(c)
    return np.mean(centroids, axis=0)


def extract_features(cluster) -> dict:
    """
    从 pylidc nodule cluster（4个医生的标注组）提取特征。
    连续变量取中位数，保留原始分布信息。
    """
    feats = {}
    for key in FEATURE_KEYS:
        values = [getattr(a, key) for a in cluster if getattr(a, key, 0) > 0]
        if values:
            feats[key] = int(np.median(values))
            # 保留原始 4 个医生的评分（用于诊断一致性分析）
            feats[f"{key}_per_rater"] = values
        else:
            feats[key] = 0
            feats[f"{key}_per_rater"] = []
    return feats


def match_nodules(luna16_nodules: list, lidc_scans: dict, images_dir: str) -> list:
    """
    将 LUNA16 结节坐标匹配到 pylidc 标注。

    ⚠️ LUNA16 coordX/Y/Z = 世界坐标 (mm)，pylidc centroid = 体素坐标 (voxel index)。
       必须用 SimpleITK TransformPhysicalPointToContinuousIndex() 转 LUNA16 → 体素坐标再比较。
       同时尝试 (x,y,z) 和 (y,x,z) 两种轴序（图像坐标 vs 世界坐标约定不同）。

    匹配策略:
      1. 按 seriesuid 查 pylidc scan
      2. 加载 .mhd → SimpleITK 把 LUNA16 世界坐标转体素 (continuous index, float)
      3. 对每个 LUNA16 结节，与 scan 内所有 pylidc 结节中心比 L2 距离（体素空间）
      4. 同时尝试 (x,y,z) 和 (y,x,z)，取 min distance
      5. 距离 < max(15, 5×diameter/min_spacing) → 匹配成功
      6. 一个 pylidc 结节只能匹配一次
    """
    matched = []
    unmatched_count = 0
    empty_scan_count = 0
    not_found_count = 0
    unmet_debug = []

    from collections import defaultdict
    by_uid = defaultdict(list)
    for n in luna16_nodules:
        by_uid[n["seriesuid"]].append(n)

    for seriesuid, nodules in tqdm(by_uid.items(), desc="Matching"):
        if seriesuid not in lidc_scans:
            not_found_count += len(nodules)
            continue

        scan = lidc_scans[seriesuid]

        # ── 加载 CT 图像 ──
        mhd_path = os.path.join(images_dir, f"{seriesuid}.mhd")
        if not os.path.exists(mhd_path):
            alt = os.path.join(images_dir, seriesuid, f"{seriesuid}.mhd")
            if os.path.exists(alt):
                mhd_path = alt
            else:
                not_found_count += len(nodules)
                continue

        try:
            ct_image = sitk.ReadImage(mhd_path)
            spacing = np.array(ct_image.GetSpacing())  # mm/voxel
        except Exception:
            empty_scan_count += len(nodules)
            continue

        # ── pylidc 结节聚类 ──
        try:
            clusters = scan.cluster_annotations()
        except Exception:
            clusters = [[a] for a in scan.annotations]

        if not clusters:
            empty_scan_count += len(nodules)
            continue

        # 提取 pylidc 结节中心（体素坐标）和特征
        lidc_data = []
        for cluster in clusters:
            centroid_voxel = lidc_centroid_mm(cluster)  # ann.centroid 返回体素坐标
            feats = extract_features(cluster)
            lidc_data.append({
                "centroid_voxel": centroid_voxel,
                "features": feats,
                "matched": False,
            })

        # ── 匹配每个 LUNA16 结节（体素空间）──
        for luna_nod in nodules:
            diameter = luna_nod["diameter_mm"]
            lx, ly, lz = float(luna_nod["coordX"]), float(luna_nod["coordY"]), float(luna_nod["coordZ"])

            # 世界坐标 → 体素坐标（float，不截断）
            voxel_xyz = np.array(ct_image.TransformPhysicalPointToContinuousIndex((lx, ly, lz)))
            voxel_yxz = np.array(ct_image.TransformPhysicalPointToContinuousIndex((ly, lx, lz)))

            # 体素空间阈值 — 用 min spacing 对 anisotropic 更宽容
            min_spacing = float(np.min(spacing))
            threshold_voxel = max(15.0, 5.0 * diameter / max(min_spacing, 0.1))

            best_dist = float("inf")
            best_idx = -1
            best_order = 0

            for i, ld in enumerate(lidc_data):
                if ld["matched"]:
                    continue
                px, py, pz = ld["centroid_voxel"]
                # 四种组合: LUNA(xyz|yxz) × pylidc(xyz|yxz)
                d1 = float(np.linalg.norm(voxel_xyz - np.array([px, py, pz])))   # xyz vs xyz
                d2 = float(np.linalg.norm(voxel_xyz - np.array([py, px, pz])))   # xyz vs yxz
                d3 = float(np.linalg.norm(voxel_yxz - np.array([px, py, pz])))   # yxz vs xyz
                d4 = float(np.linalg.norm(voxel_yxz - np.array([py, px, pz])))   # yxz vs yxz
                d = min(d1, d2, d3, d4)
                if d < best_dist:
                    best_dist = d
                    best_idx = i
                    best_order = {0: "xyz·xyz", 1: "xyz·yxz", 2: "yxz·xyz", 3: "yxz·yxz"}[
                        [d1, d2, d3, d4].index(d)]

            if best_idx >= 0 and best_dist < threshold_voxel:
                ld = lidc_data[best_idx]
                ld["matched"] = True
                result = {
                    "seriesuid": seriesuid,
                    "coordX": lx, "coordY": ly, "coordZ": lz,
                    "diameter_mm": diameter,
                    "match_distance_voxel": round(best_dist, 1),
                    "match_threshold_voxel": round(threshold_voxel, 1),
                    "axis_order": "xyz" if best_order == 0 else "yxz",
                    "spacing_mm": [round(float(s), 3) for s in spacing],
                    **ld["features"],
                }
                result["texture_desc"] = TEXTURE_MAP.get(result["texture"], "未知")
                result["malignancy_desc"] = MALIGNANCY_MAP.get(result["malignancy"], "未知")
                result["calcification_desc"] = CALCIFICATION_MAP.get(result["calcification"], "未知")
                result["margin_desc"] = MARGIN_MAP.get(result["margin"], "未知")
                result["spiculation_desc"] = SPICULATION_MAP.get(result["spiculation"], "未知")
                result["lobulation_desc"] = LOBULATION_MAP.get(result["lobulation"], "未知")
                matched.append(result)
            else:
                best_ld_centroid = lidc_data[best_idx]["centroid_voxel"] if best_idx >= 0 else None
                unmet_debug.append({
                    "seriesuid": seriesuid,
                    "luna_xyz": [lx, ly, lz],
                    "luna_voxel_xyz": [round(float(v), 1) for v in voxel_xyz],
                    "luna_voxel_yxz": [round(float(v), 1) for v in voxel_yxz],
                    "diameter": diameter,
                    "best_dist_voxel": round(best_dist, 1) if best_dist < float("inf") else -1,
                    "best_pylidc_centroid": [round(float(c), 1) for c in best_ld_centroid] if best_ld_centroid is not None else None,
                    "threshold_voxel": round(threshold_voxel, 1),
                    "spacing": [round(float(s), 3) for s in spacing],
                    "n_pylidc_candidates": sum(1 for ld in lidc_data if not ld["matched"]),
                })
                unmatched_count += 1

    print(f"\n  Matched: {len(matched)}")
    print(f"  Not in LIDC: {not_found_count}")
    print(f"  Empty/cluster-fail: {empty_scan_count}")
    print(f"  Unmatched (no close nodule): {unmatched_count}")

    if unmet_debug:
        print(f"\n  Unmatched diagnostics (first 5):")
        for u in unmet_debug[:5]:
            print(f"    {u['seriesuid'][:20]}.. | d={u['diameter']:.0f}mm spacing={u['spacing']}")
            print(f"      LUNA mm=({u['luna_xyz'][0]:.0f},{u['luna_xyz'][1]:.0f},{u['luna_xyz'][2]:.0f})")
            print(f"      LUNA vox xyz={u['luna_voxel_xyz']}  yxz={u['luna_voxel_yxz']}")
            print(f"      pylidc centroid={u['best_pylidc_centroid']}  dist={u['best_dist_voxel']}  thr={u['threshold_voxel']}")

    return matched


def main():
    parser = argparse.ArgumentParser(description="LIDC 特征提取 + LUNA16 匹配")
    parser.add_argument("--luna16_dir", default="/root/autodl-tmp/data/LUNA16")
    parser.add_argument("--output", default="/root/autodl-tmp/data/nodule_features.json")
    parser.add_argument("--max_scans", type=int, default=0,
                        help="限制扫描数 (0=全部, 用于快速测试)")
    args = parser.parse_args()

    print("=" * 60)
    print("  LIDC-IDRI 特征提取 + LUNA16 结节匹配")
    print("=" * 60)

    # 1. 加载 LUNA16 标注
    anno_csv = os.path.join(args.luna16_dir, "annotations.csv")
    if not os.path.exists(anno_csv):
        print(f"[ERROR] annotations.csv 不存在: {anno_csv}")
        return
    luna_nodules = load_luna16_annotations(anno_csv)
    print(f"[1/4] LUNA16 标注: {len(luna_nodules)} 个结节")

    # 2. 过滤：只保留有对应 .mhd 文件的结节
    images_dir = os.path.join(args.luna16_dir, "images")
    valid_uids = set()
    if os.path.exists(images_dir):
        for f in os.listdir(images_dir):
            if f.endswith(".mhd"):
                valid_uids.add(f.replace(".mhd", ""))
    luna_nodules = [n for n in luna_nodules if n["seriesuid"] in valid_uids]
    print(f"[2/4] 有效 CT 扫描: {len(valid_uids)} 个, 覆盖 {len(luna_nodules)} 个结节")

    if args.max_scans > 0:
        uids = list(set(n["seriesuid"] for n in luna_nodules))[:args.max_scans]
        luna_nodules = [n for n in luna_nodules if n["seriesuid"] in uids]
        print(f"  → 限制为 {args.max_scans} 个扫描, {len(luna_nodules)} 个结节")

    # 3. 加载 pylidc（首次运行会自动下载标注缓存）
    print("[3/4] 加载 pylidc (首次运行会下载 LIDC 标注缓存)...")
    import pylidc as pl
    # 修复 numpy 兼容
    import numpy
    if not hasattr(numpy, "int"):
        numpy.int = numpy.int64

    lidc_scans = {s.series_instance_uid: s for s in pl.query(pl.Scan).all()}
    print(f"  LIDC scans: {len(lidc_scans)}")

    # 4. 匹配
    print("[4/4] 匹配结节（mm→体素转换 + 轴序双试）...")
    results = match_nodules(luna_nodules, lidc_scans, images_dir)

    # 5. 保存
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[DONE] → {args.output}")
    print(f"  结节数: {len(results)}")
    print(f"  覆盖扫描: {len(set(r['seriesuid'] for r in results))}")
    if results:
        print(f"  特征示例: { {k: results[0][k] for k in FEATURE_KEYS} }")


if __name__ == "__main__":
    main()
