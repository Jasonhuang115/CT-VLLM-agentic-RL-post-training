#!/usr/bin/env python3
"""
.mhd → PNG 切片转换

对每个 CT 扫描，提取中心轴向切片，窗口化 (lung window) 后保存为 512×512 灰度 PNG，
供 Qwen2.5-VL image processor 直接使用。

使用方式:
  python data/mhd_to_png.py \
    --images_dir /root/autodl-tmp/data/LUNA16/images \
    --output_dir /root/autodl-tmp/data/LUNA16/images_png
"""

import os, sys, argparse
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


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


def mhd_to_png(mhd_path: str, output_path: str) -> bool:
    """
    读取 .mhd，取中心轴向切片，窗口化，存为 PNG。
    """
    try:
        image = sitk.ReadImage(mhd_path)
        hu_array = sitk.GetArrayFromImage(image)  # shape: (Z, Y, X)

        # 取中心轴向切片
        z_center = hu_array.shape[0] // 2
        hu_slice = hu_array[z_center, :, :]  # (Y, X)

        # 肺窗窗口化
        gray = window_lung(hu_slice)

        # 用 PIL 存 PNG
        from PIL import Image
        img = Image.fromarray(gray, mode='L')
        img.save(output_path, format='PNG')
        return True
    except Exception as e:
        print(f"  [ERROR] {mhd_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description=".mhd → PNG 切片转换")
    parser.add_argument("--images_dir", default="/root/autodl-tmp/data/LUNA16/images")
    parser.add_argument("--output_dir", default="/root/autodl-tmp/data/LUNA16/images_png")
    parser.add_argument("--max_scans", type=int, default=0,
                        help="限制扫描数 (0=全部)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 收集所有 .mhd 文件
    mhd_files = []
    for root, dirs, files in os.walk(args.images_dir):
        for f in files:
            if f.endswith('.mhd'):
                mhd_files.append(os.path.join(root, f))

    if args.max_scans > 0:
        mhd_files = mhd_files[:args.max_scans]

    print(f"Found {len(mhd_files)} .mhd files")
    print(f"Output: {args.output_dir}")

    done, failed = 0, 0
    for mhd_path in tqdm(mhd_files, desc="mhd→png"):
        seriesuid = os.path.basename(mhd_path).replace('.mhd', '')
        output_path = os.path.join(args.output_dir, f"{seriesuid}.png")
        if mhd_to_png(mhd_path, output_path):
            done += 1
        else:
            failed += 1

    print(f"\nDone: {done}, Failed: {failed}")
    print(f"Output: {args.output_dir}/")
    print(f"Total PNGs: {len([f for f in os.listdir(args.output_dir) if f.endswith('.png')])}")


if __name__ == "__main__":
    main()
