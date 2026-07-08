#!/usr/bin/env python3
"""
结节 ROI 提取 & 多视图生成

基于 LIDC-IDRI XML 标注, 从 NIfTI CT 体积中提取每个结节的 ROI,
生成多视图图像 (轴位/冠状/矢状/九宫格/MIP) 用于 VLM 训练。

输入:
  - NIfTI CT 体积 (由 dicom_to_nifti.py 生成)
  - LIDC-IDRI XML 标注 (由 download_lidc.py 下载)
  - series_index.json (由 dicom_to_nifti.py 生成)

输出:
  - 每个结节 5 张图像 (PNG)
  - 结构化标注 JSON

使用方式:
  python data/preprocessing/nodule_extraction.py \
    --nifti_dir /path/to/nifti \
    --anno_dir /path/to/annotations \
    --meta_file /path/to/metadata/series_index.json \
    --output /root/autodl-tmp/data/nodules
"""

import os
import sys
import json
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

import numpy as np
import nibabel as nib
from PIL import Image
from scipy.ndimage import zoom
from tqdm import tqdm


# ---- XML 标注解析 ----
# LIDC-IDRI XML 结构 (简化):
# <LidcReadMessage>
#   <ResponseHeader>...</ResponseHeader>
#   <readingSession>
#     <servicingRadiologistID>...</servicingRadiologistID>
#     <unblindedReadNodule>
#       <noduleID>...</noduleID>
#       <characteristics>...</characteristics>
#       <roi>
#         <imageZposition>...</imageZposition>
#         <imageSOP_UID>...</imageSOP_UID>
#         <inclusion>TRUE</inclusion>
#         <edgeMap>
#           <xCoord>...</xCoord>
#           <yCoord>...</yCoord>
#         </edgeMap>
#       </roi>
#     </unblindedReadNodule>
#   </readingSession>
# </LidcReadMessage>

def parse_lidc_xml(xml_path):
    """解析 LIDC-IDRI XML 标注文件, 提取所有结节信息"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # XML namespace
    ns = {"ns": "http://www.nih.gov"}

    nodules = []
    for session in root.findall(".//readingSession"):
        radiologist_id = session.findtext("servicingRadiologistID", default="unknown")

        for nodule in session.findall("unblindedReadNodule"):
            nodule_id = nodule.findtext("noduleID", default="unknown")

            # 结节特征
            chars = nodule.find("characteristics")
            if chars is None:
                continue

            characteristics = {
                "subtlety": int(chars.findtext("subtlety", "3")),
                "internalStructure": int(chars.findtext("internalStructure", "1")),
                "calcification": int(chars.findtext("calcification", "6")),
                "sphericity": int(chars.findtext("sphericity", "3")),
                "margin": int(chars.findtext("margin", "3")),
                "lobulation": int(chars.findtext("lobulation", "3")),
                "spiculation": int(chars.findtext("spiculation", "3")),
                "texture": int(chars.findtext("texture", "5")),
                "malignancy": int(chars.findtext("malignancy", "3")),
            }

            # 提取所有 ROI 轮廓
            rois = []
            for roi in nodule.findall("roi"):
                inclusion = roi.findtext("inclusion", "FALSE").upper() == "TRUE"
                z_pos = float(roi.findtext("imageZposition", "0"))
                sop_uid = roi.findtext("imageSOP_UID", "")

                edge_map = roi.find("edgeMap")
                if edge_map is not None:
                    x_coords = [float(x.text) for x in edge_map.findall("xCoord")]
                    y_coords = [float(y.text) for y in edge_map.findall("yCoord")]
                    contour = list(zip(x_coords, y_coords))
                else:
                    contour = []

                rois.append({
                    "z_position": z_pos,
                    "sop_uid": sop_uid,
                    "inclusion": inclusion,
                    "contour": contour,
                })

            if rois:
                nodules.append({
                    "nodule_id": nodule_id,
                    "radiologist_id": radiologist_id,
                    "characteristics": characteristics,
                    "rois": rois,
                })

    return nodules


def nodule_centroid_and_bbox(rois, pixel_spacing, slice_thickness, z_origin=0):
    """从多个 ROI 轮廓中计算结节中心坐标和边界框 (世界坐标 mm)"""
    all_points = []
    for roi in rois:
        for x, y in roi["contour"]:
            z_mm = roi["z_position"]
            all_points.append((x, y, z_mm))

    if not all_points:
        return None, None

    pts = np.array(all_points)
    centroid = np.mean(pts, axis=0)  # (x, y, z) mm

    # 边界框
    min_pt = np.min(pts, axis=0)
    max_pt = np.max(pts, axis=0)
    size_mm = max_pt - min_pt
    bbox_mm = {
        "center": centroid.tolist(),
        "size": size_mm.tolist(),
        "min": min_pt.tolist(),
        "max": max_pt.tolist(),
    }
    return centroid, bbox_mm


def extract_axial_slice(volume_hu, z_idx, window=(-600, 1500)):
    """提取并归一化轴位切片"""
    wl, ww = window
    low, high = wl - ww / 2, wl + ww / 2
    slice_data = np.clip(volume_hu[z_idx], low, high)
    slice_data = ((slice_data - low) / (high - low) * 255).astype(np.uint8)
    return Image.fromarray(slice_data)


def extract_coronal_slice(volume_hu, y_idx, window=(-600, 1500)):
    """提取冠状面切片"""
    wl, ww = window
    low, high = wl - ww / 2, wl + ww / 2
    slice_data = np.clip(volume_hu[:, y_idx, :], low, high)
    slice_data = ((slice_data - low) / (high - low) * 255).astype(np.uint8)
    return Image.fromarray(slice_data)


def extract_sagittal_slice(volume_hu, x_idx, window=(-600, 1500)):
    """提取矢状面切片"""
    wl, ww = window
    low, high = wl - ww / 2, wl + ww / 2
    slice_data = np.clip(volume_hu[:, :, x_idx], low, high)
    slice_data = ((slice_data - low) / (high - low) * 255).astype(np.uint8)
    return Image.fromarray(slice_data)


def extract_montage_3x3(volume_hu, z_center, window=(-600, 1500)):
    """3×3 九宫格: 结节中心 ± 4 邻层"""
    wl, ww = window
    low, high = wl - ww / 2, wl + ww / 2
    nz = volume_hu.shape[0]

    z_offsets = [-4, -2, -1, 0, 1, 2, 4]  # 7 slices around center
    slices = []
    for dz in z_offsets:
        zi = int(z_center + dz)
        zi = max(0, min(nz - 1, zi))
        s = np.clip(volume_hu[zi], low, high)
        s = ((s - low) / (high - low) * 255).astype(np.uint8)
        slices.append(s)
    slices.append(np.zeros_like(slices[0]))  # 第9格空白
    slices.insert(4, slices.pop(-1))  # 空白放中间

    # 拼接 3x3
    rows = []
    for i in range(0, 9, 3):
        row = np.hstack(slices[i:i+3])
        rows.append(row)
    montage = np.vstack(rows)

    return Image.fromarray(montage)


def extract_mip(volume_hu, z_center, thickness=20, window=(-600, 1500)):
    """最大密度投影 (MIP)"""
    wl, ww = window
    low, high = wl - ww / 2, wl + ww / 2
    nz = volume_hu.shape[0]
    z_start = max(0, int(z_center - thickness // 2))
    z_end = min(nz, int(z_center + thickness // 2))

    slab = volume_hu[z_start:z_end]
    mip_data = np.max(slab, axis=0)
    mip_data = np.clip(mip_data, low, high)
    mip_data = ((mip_data - low) / (high - low) * 255).astype(np.uint8)

    return Image.fromarray(mip_data)


def generate_nodule_views(volume_hu, centroid_mm, bbox_mm, pixel_spacing, output_dir, nodule_name):
    """
    为单个结节生成全部 5 张视图

    Args:
        volume_hu: 3D numpy array (z, y, x) in HU
        centroid_mm: (x, y, z) 中心坐标 (mm)
        bbox_mm: 边界框 (mm)
        pixel_spacing: (dx, dy) mm/pixel
        output_dir: 输出目录
        nodule_name: 结节命名 (如 "nodule_001")
    """
    dx, dy = pixel_spacing
    # 世界坐标 → 体素坐标
    cx = int(centroid_mm[0] / dx)
    cy = int(centroid_mm[1] / dy)
    cz = int(centroid_mm[2] / (pixel_spacing[2] if len(pixel_spacing) > 2 else 1.0))
    # LIDC pixel_spacing 是 (row, col), 对应 (y, x)
    cz = int(centroid_mm[2] / 1.0)  # z 方向通常是 1mm 左右

    # 修正: 实际 pixel_spacing[0] 是 row spacing, [1] 是 col spacing
    # 世界坐标 (x, y) 对应体素坐标 (col, row)
    cx = int(centroid_mm[0] / dx)
    cy = int(centroid_mm[1] / dy)

    nz, ny, nx = volume_hu.shape
    cx = max(0, min(nx - 1, cx))
    cy = max(0, min(ny - 1, cy))
    cz = max(0, min(nz - 1, cz))

    views = {}

    # 1. 中央轴位
    img = extract_axial_slice(volume_hu, cz)
    path = os.path.join(output_dir, f"{nodule_name}_axial.png")
    img.save(path)
    views["axial"] = path

    # 2. 冠状面
    img = extract_coronal_slice(volume_hu, cy)
    path = os.path.join(output_dir, f"{nodule_name}_coronal.png")
    img.save(path)
    views["coronal"] = path

    # 3. 矢状面
    img = extract_sagittal_slice(volume_hu, cx)
    path = os.path.join(output_dir, f"{nodule_name}_sagittal.png")
    img.save(path)
    views["sagittal"] = path

    # 4. 九宫格
    img = extract_montage_3x3(volume_hu, cz)
    path = os.path.join(output_dir, f"{nodule_name}_montage.png")
    img.save(path)
    views["montage"] = path

    # 5. MIP
    img = extract_mip(volume_hu, cz, thickness=20)
    path = os.path.join(output_dir, f"{nodule_name}_mip.png")
    img.save(path)
    views["mip"] = path

    return views


def aggregate_nodule_annotations(all_nodules, iou_threshold=0.3):
    """
    聚合多个放射科医生的标注。

    LIDC-IDRI 有 4 个放射科医生的标注。
    同一个结节可能被多个医生标注, 需要聚类。

    简化方案: 按结节中心距离聚类 (距离 < 20mm = 同一结节)
    """
    # 将在实现时处理 pylidc 不可用的情况
    # 目前先返回原始标注
    return all_nodules


def main():
    parser = argparse.ArgumentParser(description="结节 ROI 提取 & 多视图生成")
    parser.add_argument("--nifti_dir", type=str, required=True, help="NIfTI 文件目录")
    parser.add_argument("--anno_dir", type=str, required=True, help="LIDC-IDRI XML 标注目录")
    parser.add_argument("--meta_file", type=str, required=True, help="series_index.json 路径")
    parser.add_argument("--output", type=str, default="/root/autodl-tmp/data/nodules",
                        help="输出目录")
    parser.add_argument("--max_cases", type=int, default=0, help="最大处理病例数 (0=全部)")
    parser.add_argument("--window", type=str, default="lung", choices=["lung", "mediastinal"])
    args = parser.parse_args()

    # 窗宽窗位
    windows = {"lung": (-600, 1500), "mediastinal": (50, 350)}
    window = windows[args.window]

    os.makedirs(args.output, exist_ok=True)
    view_dir = os.path.join(args.output, "views")
    os.makedirs(view_dir, exist_ok=True)

    # 加载元数据
    with open(args.meta_file) as f:
        all_meta = json.load(f)

    if args.max_cases > 0:
        all_meta = all_meta[:args.max_cases]

    # 建立 series_uid → metadata 映射
    meta_map = {m["series_uid"]: m for m in all_meta}

    # 扫描所有 XML 标注
    anno_files = list(Path(args.anno_dir).rglob("*.xml"))
    print(f"[INFO] 找到 {len(anno_files)} 个标注文件")

    # 建立 XML → NIfTI 关联
    # LIDC XML 里的 SOPInstanceUID → DICOM 文件的映射非常复杂
    # 简化: 通过 PatientID + StudyUID 匹配

    all_nodule_data = []
    skipped = 0

    for meta in tqdm(all_meta, desc="处理 CT 体积"):
        nifti_path = meta.get("nifti_path", "")
        if not os.path.exists(nifti_path):
            skipped += 1
            continue

        # 加载 NIfTI
        img = nib.load(nifti_path)
        volume_hu = img.get_fdata().astype(np.float32)

        # 查找对应的 XML 标注
        patient_id = meta.get("patient_id", "")
        # XML 文件名通常包含 patient ID
        matching_xmls = [f for f in anno_files if patient_id in f.name or patient_id.replace("LIDC-IDRI-", "") in f.name]

        if not matching_xmls:
            # 尝试不区分大小写
            matching_xmls = [f for f in anno_files
                             if patient_id.lower().replace("-", "") in f.name.lower().replace("-", "")]
        if not matching_xmls:
            skipped += 1
            continue

        # 解析标注
        nodules = parse_lidc_xml(str(matching_xmls[0]))
        if not nodules:
            skipped += 1
            continue

        # 为每个结节生成多视图
        pixel_spacing = meta.get("pixel_spacing", [1.0, 1.0])
        slice_thickness = meta.get("slice_thickness", 1.0)
        # 3-tuple for compute: (dz, dy, dx)
        pixel_spacing_3d = (slice_thickness, pixel_spacing[0], pixel_spacing[1])

        for nodule in nodules:
            rois = nodule["rois"]
            centroid, bbox = nodule_centroid_and_bbox(rois, pixel_spacing, slice_thickness)

            if centroid is None:
                continue

            nodule_name = f"{patient_id}_{nodule['nodule_id']}"

            try:
                views = generate_nodule_views(
                    volume_hu, centroid, bbox, pixel_spacing, view_dir, nodule_name
                )
            except Exception as e:
                print(f"[WARN] 生成视图失败: {nodule_name} ({e})")
                continue

            # 保存结构化数据
            nodule_data = {
                "nodule_name": nodule_name,
                "patient_id": patient_id,
                "series_uid": meta["series_uid"],
                "radiologist_id": nodule["radiologist_id"],
                "characteristics": nodule["characteristics"],
                "centroid_mm": centroid.tolist() if centroid is not None else None,
                "bbox_mm": bbox,
                "views": views,
                "pixel_spacing": pixel_spacing,
                "slice_thickness": slice_thickness,
                "nifti_path": nifti_path,
            }
            all_nodule_data.append(nodule_data)

    # 聚合 (简化: 按空间距离聚类)
    # TODO: 完整实现 pylidc 或自定义聚类

    # 保存结节索引
    index_path = os.path.join(args.output, "nodule_index.json")
    with open(index_path, "w") as f:
        json.dump(all_nodule_data, f, indent=2)

    print(f"\n[DONE] 结节提取完成:")
    print(f"  总结节数: {len(all_nodule_data)}")
    print(f"  视图图像: {view_dir}/")
    print(f"  结节索引: {index_path}")
    print(f"  跳过病例: {skipped} (无标注或无 NIfTI)")

    if len(all_nodule_data) == 0:
        print("\n[WARN] 未提取到任何结节!")
        print("  可能原因:")
        print("  1. XML 标注文件名与 NIfTI 患者 ID 不匹配")
        print("  2. 标注目录结构不对 (应为 annotations/ 下的 .xml 文件)")
        print("  请检查匹配逻辑或手动指定 XML 路径。")


if __name__ == "__main__":
    main()
