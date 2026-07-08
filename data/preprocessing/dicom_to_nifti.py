#!/usr/bin/env python3
"""
LIDC-IDRI DICOM → NIfTI 转换 & HU 窗宽窗位归一化

输入: LIDC-IDRI 原始 DICOM 文件 (按 series 存放)
输出: 归一化后的 NIfTI 3D 体积 + 元数据 JSON

使用方式:
  python data/preprocessing/dicom_to_nifti.py --input /path/to/LIDC-IDRI --nifti_dir /path/to/output
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pydicom
import nibabel as nib
from tqdm import tqdm


def group_dicoms_by_series(dicom_dir):
    """将 DICOM 文件按 SeriesInstanceUID 分组"""
    groups = defaultdict(list)
    dcm_files = list(Path(dicom_dir).rglob("*.dcm"))

    if not dcm_files:
        print(f"[WARN] 未找到 .dcm 文件于 {dicom_dir}")
        return {}

    for dcm_path in tqdm(dcm_files, desc="扫描 DICOM 文件"):
        try:
            ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
            # 仅处理 CT 图像 (Modality == 'CT')
            if ds.get("Modality", "") != "CT":
                continue
            # LIDC-IDRI 用 SeriesInstanceUID 组织
            series_uid = ds.get("SeriesInstanceUID", "unknown")
            # 按 z 位置排序
            z_pos = float(ds.get("ImagePositionPatient", [0, 0, 0])[2])
            groups[series_uid].append((z_pos, str(dcm_path)))
        except Exception as e:
            continue

    # 每组内按 z 位置排序
    for uid in groups:
        groups[uid].sort(key=lambda x: x[0])

    print(f"[INFO] 找到 {len(groups)} 个 CT series, 共 {sum(len(v) for v in groups.values())} 张切片")
    return dict(groups)


def dicom_series_to_nifti(dcm_paths, output_path, window=None):
    """
    将一组有序的 DICOM 切片转为 NIfTI 3D 体积

    Args:
        dcm_paths: 按 z 排序的 DICOM 路径列表 [(z_pos, path), ...]
        output_path: 输出 .nii.gz 路径
        window: (level, width) 或 None (不归一化)
    """
    slices = []
    metadata = {}

    for z_pos, dcm_path in dcm_paths:
        ds = pydicom.dcmread(dcm_path)
        arr = ds.pixel_array.astype(np.float32)

        # 转 HU
        if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
            slope = float(ds.RescaleSlope)
            intercept = float(ds.RescaleIntercept)
            arr = arr * slope + intercept

        slices.append(arr)

        if not metadata:
            metadata = {
                "pixel_spacing": [float(v) for v in ds.get("PixelSpacing", [1.0, 1.0])],
                "slice_thickness": float(ds.get("SliceThickness", 1.0)),
                "rows": int(ds.Rows),
                "columns": int(ds.Columns),
                "patient_id": str(ds.get("PatientID", "unknown")),
                "study_uid": str(ds.get("StudyInstanceUID", "unknown")),
                "series_uid": str(ds.get("SeriesInstanceUID", "unknown")),
            }

    # 堆叠为 3D
    volume = np.stack(slices, axis=0).astype(np.float32)

    # HU 窗宽窗位归一化 (可选)
    if window:
        level, width = window
        low, high = level - width / 2, level + width / 2
        volume = np.clip(volume, low, high)

    # 保存 NIfTI
    affine = np.eye(4)
    if metadata.get("pixel_spacing"):
        sx, sy = metadata["pixel_spacing"]
        sz = metadata.get("slice_thickness", 1.0)
        affine[0, 0] = sx
        affine[1, 1] = sy
        affine[2, 2] = sz

    img = nib.Nifti1Image(volume, affine)
    nib.save(img, output_path)

    return metadata


def main():
    parser = argparse.ArgumentParser(description="DICOM → NIfTI 转换")
    parser.add_argument("--input", type=str, required=True, help="LIDC-IDRI DICOM 根目录")
    parser.add_argument("--nifti_dir", type=str, default=None, help="NIfTI 输出目录")
    parser.add_argument("--meta_dir", type=str, default=None, help="元数据 JSON 输出目录")
    parser.add_argument("--window", type=str, default="lung",
                        choices=["lung", "mediastinal", "none"],
                        help="HU 窗宽窗位")
    parser.add_argument("--max_series", type=int, default=0, help="限制处理 series 数量 (0=全部)")
    args = parser.parse_args()

    if args.nifti_dir is None:
        args.nifti_dir = os.path.join(os.path.dirname(args.input), "nifti")
    if args.meta_dir is None:
        args.meta_dir = os.path.join(os.path.dirname(args.input), "metadata")

    os.makedirs(args.nifti_dir, exist_ok=True)
    os.makedirs(args.meta_dir, exist_ok=True)

    # 窗宽窗位
    windows = {
        "lung": (-600, 1500),
        "mediastinal": (50, 350),
        "none": None,
    }
    window = windows[args.window]

    print("=" * 60)
    print("  DICOM → NIfTI 转换")
    print("=" * 60)
    print(f"  输入: {args.input}")
    print(f"  输出: {args.nifti_dir}")
    print(f"  窗宽窗位: {args.window} ({window})")
    print()

    # 分组
    series_groups = group_dicoms_by_series(args.input)
    if args.max_series > 0:
        keys = list(series_groups.keys())[:args.max_series]
        series_groups = {k: series_groups[k] for k in keys}

    # 转换每个 series
    all_metadata = []
    for series_uid, dcm_list in tqdm(series_groups.items(), desc="转换为 NIfTI"):
        output_path = os.path.join(args.nifti_dir, f"{series_uid}.nii.gz")
        meta = dicom_series_to_nifti(dcm_list, output_path, window=window)
        meta["nifti_path"] = output_path
        all_metadata.append(meta)

    # 保存元数据索引
    index_path = os.path.join(args.meta_dir, "series_index.json")
    with open(index_path, "w") as f:
        json.dump(all_metadata, f, indent=2)

    print(f"\n[DONE] 转换完成: {len(all_metadata)} 个 CT 体积")
    print(f"  NIfTI: {args.nifti_dir}/")
    print(f"  元数据: {index_path}")

    if len(all_metadata) <= 3:
        print(f"\n  注意: 仅 {len(all_metadata)} 个 series。")
        print(f"  如需下载更多数据: python data/download/download_lidc.py --full")


if __name__ == "__main__":
    main()
