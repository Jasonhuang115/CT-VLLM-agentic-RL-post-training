#!/usr/bin/env python3
"""
.mhd → PNG 切片转换

支持两种模式:
  1. center (原始): 取每个 CT 扫描的中心轴向切片 → {seriesuid}.png
  2. nodule_slices (新增): 基于 nodule_features.json，在结节 Z 坐标处取多层切片
     → {seriesuid}_nodule_{i}_slice_{j}.png

使用方式:
  # 原始模式（中心切片）
  python data/mhd_to_png.py --mode center \
    --images_dir /root/autodl-tmp/data/LUNA16/images \
    --output_dir /root/autodl-tmp/data/LUNA16/images_png

  # 结节多切片模式
  python data/mhd_to_png.py --mode nodule_slices \
    --images_dir /root/autodl-tmp/data/LUNA16/images \
    --features /root/autodl-tmp/data/nodule_features.json \
    --output_dir /root/autodl-tmp/data/LUNA16/images_png \
    --n_slices 3 --slice_spacing_mm 2.0
"""

import os, sys, json, argparse
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from PIL import Image


def window_lung(hu_slice: np.ndarray, level: float = -600, width: float = 1500) -> np.ndarray:
    """
    肺窗 (lung window) 窗口化 HU 值 → 0-255 灰度。
    level=-600, width=1500 是标准肺窗参数。
    """
    low = level - width / 2      # -1350
    high = level + width / 2     #  150
    clipped = np.clip(hu_slice, low, high)
    normalized = (clipped - low) / (high - low)  # 0-1
    return (normalized * 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════
# 模式 1: 中心切片（原始逻辑，保持向后兼容）
# ═══════════════════════════════════════════════════════════════

def mhd_to_png_center(mhd_path: str, output_path: str) -> bool:
    """读取 .mhd，取中心轴向切片，窗口化，存为 PNG。"""
    try:
        image = sitk.ReadImage(mhd_path)
        hu_array = sitk.GetArrayFromImage(image)  # shape: (Z, Y, X)
        z_center = hu_array.shape[0] // 2
        hu_slice = hu_array[z_center, :, :]  # (Y, X)
        gray = window_lung(hu_slice)
        img = Image.fromarray(gray, mode='L')
        img = img.resize((512, 512), Image.LANCZOS)
        img.save(output_path, format='PNG')
        return True
    except Exception as e:
        print(f"  [ERROR] {mhd_path}: {e}")
        return False


# ═══════════════════════════════════════════════════════════════
# 模式 2: 结节中心多切片（★ 新增）
# ═══════════════════════════════════════════════════════════════

def extract_nodule_slices(
    mhd_path: str,
    coordZ_mm: float,
    output_dir: str,
    seriesuid: str,
    nodule_idx: int,
    n_slices: int = 3,
    slice_spacing_mm: float = 2.0,
    roi_size_mm: float = 50.0,
) -> list:
    """
    在结节 Z 坐标处提取 N 层轴向切片，保存为 PNG。

    Args:
        mhd_path: .mhd 文件路径
        coordZ_mm: 结节世界坐标 Z（mm），来自 LUNA16 annotations.csv
        output_dir: 输出目录
        seriesuid: CT 扫描的 seriesuid
        nodule_idx: 结节在该 CT 中的序号（从 0 开始）
        n_slices: 提取层数（默认 3）
        slice_spacing_mm: 层间距（mm，默认 2.0）
        roi_size_mm: ROI 裁剪尺寸（mm，默认 50mm 正方形）

    Returns:
        list of saved PNG paths
    """
    try:
        image = sitk.ReadImage(mhd_path)
        hu_array = sitk.GetArrayFromImage(image)  # shape: (Z, Y, X)
        spacing = np.array(image.GetSpacing())     # (dx, dy, dz) or (col, row, slice)
        origin = np.array(image.GetOrigin())
    except Exception as e:
        print(f"  [ERROR] 读取失败 {mhd_path}: {e}")
        return []

    nz, ny, nx = hu_array.shape

    # coordZ mm → voxel Z index
    # coordZ 是世界坐标 Z（mm），需要减去 origin 再除以 spacing
    if len(spacing) >= 3:
        dz = float(spacing[2])
    else:
        dz = 1.0  # fallback

    try:
        z_origin = float(origin[2]) if len(origin) >= 3 else 0.0
    except (IndexError, TypeError):
        z_origin = 0.0

    z_voxel = int((coordZ_mm - z_origin) / dz) if dz > 0 else nz // 2
    z_voxel = max(0, min(nz - 1, z_voxel))

    # 层间距转 voxel 单位
    spacing_voxel = max(1, int(round(slice_spacing_mm / dz))) if dz > 0 else 1

    # ROI 裁剪尺寸（像素）
    # 用 X/Y 方向 spacing 把 mm 转 pixel
    dx = float(spacing[0]) if len(spacing) >= 1 else 1.0
    dy = float(spacing[1]) if len(spacing) >= 2 else 1.0
    roi_px_x = int(roi_size_mm / dx)
    roi_px_y = int(roi_size_mm / dy)
    roi_px = max(roi_px_x, roi_px_y)

    saved = []
    for si in range(n_slices):
        # 计算切片 Z 索引：中心 ± 偏移
        offset = (si - n_slices // 2) * spacing_voxel
        zi = z_voxel + offset
        zi = max(0, min(nz - 1, zi))

        # 提取切片
        hu_slice = hu_array[zi, :, :]  # (Y, X)

        # ROI 裁剪：以图像中心为参考（结节通常在 CT 视野中心附近）
        # 更精确的做法是用 coordX/coordY 转像素坐标，但当前实现先以中心裁剪
        cy, cx = ny // 2, nx // 2
        y_start = max(0, cy - roi_px // 2)
        y_end = min(ny, cy + roi_px // 2)
        x_start = max(0, cx - roi_px // 2)
        x_end = min(nx, cx + roi_px // 2)
        hu_slice = hu_slice[y_start:y_end, x_start:x_end]

        # 肺窗窗口化
        gray = window_lung(hu_slice)

        # 保存
        fname = f"{seriesuid}_nodule_{nodule_idx:03d}_slice_{si}.png"
        out_path = os.path.join(output_dir, fname)
        img = Image.fromarray(gray, mode='L')
        img = img.resize((512, 512), Image.LANCZOS)
        img.save(out_path, format='PNG')
        saved.append(out_path)

    return saved


def load_nodule_features(features_path: str) -> dict:
    """
    加载 nodule_features.json，按 seriesuid 分组。

    Returns:
        {seriesuid: [nodule_dict, ...]}
    """
    with open(features_path) as f:
        all_features = json.load(f)

    from collections import defaultdict
    by_uid = defaultdict(list)
    for feat in all_features:
        by_uid[feat["seriesuid"]].append(feat)

    print(f"  结节总数: {len(all_features)}")
    print(f"  Unique CT 扫描: {len(by_uid)}")
    return by_uid


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description=".mhd → PNG 切片转换")
    parser.add_argument("--mode", default="center", choices=["center", "nodule_slices"],
                        help="center=中心切片 (原始), nodule_slices=结节多层切片 (新增)")
    parser.add_argument("--images_dir", default="/root/autodl-tmp/data/LUNA16/images",
                        help=".mhd 文件目录")
    parser.add_argument("--output_dir", default="/root/autodl-tmp/data/LUNA16/images_png",
                        help="PNG 输出目录")
    parser.add_argument("--features", default="/root/autodl-tmp/data/nodule_features.json",
                        help="nodule_features.json 路径（nodule_slices 模式必需）")
    parser.add_argument("--n_slices", type=int, default=3,
                        help="每个结节提取层数 (默认 3)")
    parser.add_argument("--slice_spacing_mm", type=float, default=2.0,
                        help="层间距 mm (默认 2.0)")
    parser.add_argument("--max_scans", type=int, default=0,
                        help="限制扫描数 (0=全部)")
    parser.add_argument("--max_nodules", type=int, default=0,
                        help="限制结节数 (0=全部, 用于快速测试)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "center":
        # ── 原始模式：中心切片 ──
        mhd_files = []
        for root, dirs, files in os.walk(args.images_dir):
            for f in files:
                if f.endswith('.mhd'):
                    mhd_files.append(os.path.join(root, f))
        if args.max_scans > 0:
            mhd_files = mhd_files[:args.max_scans]

        print(f"模式: center (中心切片)")
        print(f"Found {len(mhd_files)} .mhd files")
        print(f"Output: {args.output_dir}")

        done, failed = 0, 0
        for mhd_path in tqdm(mhd_files, desc="mhd→png (center)"):
            seriesuid = os.path.basename(mhd_path).replace('.mhd', '')
            output_path = os.path.join(args.output_dir, f"{seriesuid}.png")
            if mhd_to_png_center(mhd_path, output_path):
                done += 1
            else:
                failed += 1

        print(f"\nDone: {done}, Failed: {failed}")

    elif args.mode == "nodule_slices":
        # ── 结节多切片模式 ──
        if not os.path.exists(args.features):
            print(f"[ERROR] nodule_features.json 不存在: {args.features}")
            print("  请先运行: python data/lidc_match.py")
            return

        print(f"模式: nodule_slices (结节中心多层)")
        print(f"  层数: {args.n_slices}, 层间距: {args.slice_spacing_mm}mm")
        print(f"  加载特征: {args.features}")

        nodules_by_uid = load_nodule_features(args.features)

        # 收集所有 .mhd 文件路径
        mhd_map = {}
        for root, dirs, files in os.walk(args.images_dir):
            for f in files:
                if f.endswith('.mhd'):
                    uid = f.replace('.mhd', '')
                    mhd_map[uid] = os.path.join(root, f)

        # 只处理既有 .mhd 又有结节特征的 CT
        valid_uids = set(mhd_map.keys()) & set(nodules_by_uid.keys())
        print(f"  有 .mhd + 特征匹配的 CT 扫描: {len(valid_uids)}")

        if args.max_scans > 0:
            valid_uids = set(sorted(valid_uids)[:args.max_scans])

        total_slices = 0
        total_nodules = 0
        failed_nodules = 0

        for seriesuid in tqdm(sorted(valid_uids), desc="mhd→png (nodule_slices)"):
            mhd_path = mhd_map[seriesuid]
            nodules = nodules_by_uid[seriesuid]

            for ni, nodule in enumerate(nodules):
                if args.max_nodules > 0 and total_nodules >= args.max_nodules:
                    break

                coordZ = nodule.get("coordZ", 0)
                saved = extract_nodule_slices(
                    mhd_path, coordZ, args.output_dir, seriesuid,
                    nodule_idx=ni, n_slices=args.n_slices,
                    slice_spacing_mm=args.slice_spacing_mm,
                )
                if saved:
                    total_slices += len(saved)
                    total_nodules += 1
                else:
                    failed_nodules += 1

        print(f"\nDone: {total_nodules} 结节 → {total_slices} 张 PNG ({failed_nodules} 失败)")
        print(f"Output: {args.output_dir}/")
        print(f"总 PNG 文件数: {len([f for f in os.listdir(args.output_dir) if f.endswith('.png')])}")


if __name__ == "__main__":
    main()
