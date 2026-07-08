#!/usr/bin/env python3
"""
Image Reanalyzer 工具 (纯 CV 实现)

对 CT 结节 ROI 区域进行定量分析:
  - HU 值统计 (均值/方差/范围/实性成分占比)
  - 纹理分析 (GLCM 对比度/同质性/熵/相关性)
  - 边缘特征 (毛刺指数/分叶指数/圆度/分形维数)

输入: ROI 图像 (numpy array, HU 值) 或图像路径
输出: 结构化分析结果字典

使用方式:
  from agent.tools.image_reanalyzer import analyze_roi

  result = analyze_roi(roi_hu_array, analysis_type="density")
"""

import numpy as np
from typing import Dict, Optional, Tuple
from scipy import ndimage
from skimage import feature, measure
from skimage.feature import graycomatrix, graycoprops


def _hu_to_uint8(img_hu: np.ndarray, wl: float = -600, ww: float = 1500) -> np.ndarray:
    """HU → uint8 映射 (用于纹理分析)"""
    low, high = wl - ww / 2, wl + ww / 2
    img = np.clip(img_hu, low, high)
    img = ((img - low) / (high - low) * 255).astype(np.uint8)
    return img


# ============================================================
# 1. 密度分析 (HU 值统计)
# ============================================================

def analyze_density(roi_hu: np.ndarray) -> Dict:
    """
    HU 值统计分析

    Args:
        roi_hu: ROI 区域 HU 值 (2D numpy array)

    Returns:
        {
            "mean_hu": float,
            "std_hu": float,
            "min_hu": float,
            "max_hu": float,
            "solid_component_pct": float,   # > -200 HU 的比例
            "ggo_component_pct": float,      # -700 ~ -200 HU 的比例
            "air_component_pct": float,      # < -700 HU 的比例
        }
    """
    flat = roi_hu.flatten()

    # 基础统计
    mean_hu = float(np.mean(flat))
    std_hu = float(np.std(flat))
    min_hu = float(np.min(flat))
    max_hu = float(np.max(flat))

    # 成分分析
    total_pixels = len(flat)
    solid_mask = flat > -200  # 实性成分阈值
    ggo_mask = (flat > -700) & (flat <= -200)  # 磨玻璃成分
    air_mask = flat <= -700

    solid_pct = float(np.sum(solid_mask) / total_pixels * 100)
    ggo_pct = float(np.sum(ggo_mask) / total_pixels * 100)
    air_pct = float(np.sum(air_mask) / total_pixels * 100)

    # 结节类型推断
    if solid_pct > 80:
        nodule_type = "实性结节 (solid nodule)"
    elif solid_pct > 10:
        nodule_type = "部分实性结节 (part-solid nodule)"
    else:
        nodule_type = "非实性/纯磨玻璃结节 (nonsolid/pure GGN)"

    return {
        "mean_hu": round(mean_hu, 1),
        "std_hu": round(std_hu, 1),
        "min_hu": round(min_hu, 1),
        "max_hu": round(max_hu, 1),
        "solid_component_pct": round(solid_pct, 1),
        "ggo_component_pct": round(ggo_pct, 1),
        "air_component_pct": round(air_pct, 1),
        "inferred_type": nodule_type,
    }


# ============================================================
# 2. 纹理分析 (GLCM)
# ============================================================

def analyze_texture(roi_hu: np.ndarray) -> Dict:
    """
    GLCM 纹理特征分析

    灰度共生矩阵 (Gray-Level Co-occurrence Matrix) 特征:
      - 对比度 (contrast): 局部变化程度, 值高 = 纹理粗糙
      - 同质性 (homogeneity): 纹理均匀程度
      - 能量 (energy): 纹理一致性 (也叫 ASM, Angular Second Moment)
      - 相关性 (correlation): 灰度线性相关程度
      - 熵 (entropy): 纹理复杂度和随机性

    Returns:
        {contrast, homogeneity, energy, correlation, entropy}
    """
    # HU → uint8
    img_uint8 = _hu_to_uint8(roi_hu)

    # 多距离、多角度 GLCM
    glcm = graycomatrix(
        img_uint8,
        distances=[1, 2, 4],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=256,
        symmetric=True,
        normed=True,
    )

    # 提取属性并平均
    contrast = float(np.mean(graycoprops(glcm, "contrast")))
    homogeneity = float(np.mean(graycoprops(glcm, "homogeneity")))
    energy = float(np.mean(graycoprops(glcm, "energy")))
    correlation = float(np.mean(graycoprops(glcm, "correlation")))

    # 熵: 需要在归一化后计算
    glcm_flat = glcm.flatten()
    glcm_flat = glcm_flat[glcm_flat > 0]
    entropy = float(-np.sum(glcm_flat * np.log2(glcm_flat + 1e-10)))

    # 临床解读
    if contrast > 100:
        texture_type = "不均匀 (heterogeneous), 提示实性成分混杂"
    elif homogeneity > 0.5:
        texture_type = "均匀 (homogeneous), 提示成分较单一"
    else:
        texture_type = "中等不均匀 (moderately heterogeneous)"

    return {
        "contrast": round(contrast, 2),
        "homogeneity": round(homogeneity, 3),
        "energy": round(energy, 4),
        "correlation": round(correlation, 3),
        "entropy": round(entropy, 2),
        "texture_type": texture_type,
    }


# ============================================================
# 3. 边界分析
# ============================================================

def analyze_boundary(roi_hu: np.ndarray) -> Dict:
    """
    结节边缘特征分析

    指标:
      - 毛刺指数 (spiculation_index): 边缘不规则程度
      - 分叶指数 (lobulation_index): 边缘凹陷深度
      - 圆度 (circularity): 4πA/P², 1=正圆, 越接近0越不规则
      - 分形维数 (fractal_dimension): 边界复杂度

    Returns:
        {spiculation_index, lobulation_index, circularity, fractal_dimension}
    """
    # 结节分割: Otsu 阈值
    img_uint8 = _hu_to_uint8(roi_hu)

    # 简单阈值分割 (实性部分)
    _, binary = cv2_local_threshold(img_uint8) if img_uint8.size > 0 else (None, np.zeros_like(img_uint8))

    # 使用连通区域分析
    from skimage.filters import threshold_otsu
    thresh = threshold_otsu(img_uint8)
    binary = img_uint8 > thresh

    # 找到最大连通区域
    labeled = measure.label(binary)
    regions = measure.regionprops(labeled)

    if not regions:
        return {
            "spiculation_index": 0.0,
            "lobulation_index": 0.0,
            "circularity": 1.0,
            "fractal_dimension": 1.0,
            "note": "无法分割结节区域",
        }

    # 最大区域
    region = max(regions, key=lambda r: r.area)

    # 周长和面积
    perimeter = region.perimeter
    area = region.area

    # 圆度: 4πA/P²
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter ** 2)
    else:
        circularity = 1.0

    # 凸包分析 (分叶指数)
    convex_hull = region.convex_image
    hull_area = np.sum(convex_hull)
    if hull_area > 0:
        solidity = area / hull_area  # 1 = 完全凸 (无分叶), <1 = 有凹陷
        lobulation_index = 1.0 - solidity
    else:
        lobulation_index = 0.0

    # 毛刺指数: 基于凸包偏差
    # 计算轮廓点到凸包的距离
    contours = measure.find_contours(binary.astype(float), 0.5, fully_connected="high")
    if contours:
        contour = max(contours, key=len)
        # 使用傅里叶描述子分析边缘不规则性
        # 简化: 用周长/凸包周长的比值
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(contour)
            hull_perimeter = np.sum(np.sqrt(np.sum(np.diff(contour[hull.vertices], axis=0)**2, axis=1)))
            if hull_perimeter > 0:
                spiculation_index = perimeter / hull_perimeter - 1.0
            else:
                spiculation_index = 0.0
        except Exception:
            spiculation_index = 1.0 - circularity

    else:
        spiculation_index = 0.0

    # 分形维数 (box-counting method, 简化)
    fractal_dim = _box_counting_dimension(binary)

    # 临床解读
    edge_interpretation = []
    if spiculation_index > 0.15:
        edge_interpretation.append("边缘毛刺征明显 (spiculated)")
    if lobulation_index > 0.1:
        edge_interpretation.append("中度分叶 (lobulated)")
    if circularity < 0.6:
        edge_interpretation.append("结节形态不规则 (irregular)")

    return {
        "spiculation_index": round(spiculation_index, 4),
        "lobulation_index": round(lobulation_index, 4),
        "circularity": round(circularity, 4),
        "fractal_dimension": round(fractal_dim, 4),
        "solidity": round(solidity if hull_area > 0 else 1.0, 4),
        "edge_features": "; ".join(edge_interpretation) if edge_interpretation else "边缘无明显毛刺或分叶",
    }


def _box_counting_dimension(binary: np.ndarray, min_box: int = 2, max_box: int = 64) -> float:
    """Box-counting 分形维数 (简化版)"""
    sizes = []
    counts = []
    box_sizes = [2**i for i in range(int(np.log2(min_box)), int(np.log2(max_box)) + 1)]

    for box_size in box_sizes:
        # Pad to make divisible
        pad_h = (box_size - binary.shape[0] % box_size) % box_size
        pad_w = (box_size - binary.shape[1] % box_size) % box_size
        padded = np.pad(binary, ((0, pad_h), (0, pad_w)), mode="constant")

        # Reshape and count
        reshaped = padded.reshape(
            padded.shape[0] // box_size, box_size,
            padded.shape[1] // box_size, box_size,
        )
        boxes_with_ones = np.any(reshaped, axis=(1, 3))
        n_boxes = np.sum(boxes_with_ones)

        sizes.append(box_size)
        counts.append(n_boxes)

    if len(sizes) < 3:
        return 1.0

    # Linear fit: log(counts) ~ -D * log(sizes)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return abs(coeffs[0])


def cv2_local_threshold(img: np.ndarray):
    """替代 OpenCV 的局部阈值 (当 cv2 不可用时)"""
    from skimage.filters import threshold_local
    try:
        thresh = threshold_local(img, block_size=31, method="gaussian")
        binary = img > thresh
        return thresh, binary
    except Exception:
        return None, np.zeros_like(img, dtype=bool)


# ============================================================
# 4. 主入口
# ============================================================

def analyze_roi(roi_hu: np.ndarray, analysis_type: str = "all") -> Dict:
    """
    CT 结节 ROI 综合定量分析

    Args:
        roi_hu: 2D numpy array (HU 值), shape (H, W)
        analysis_type: "density" | "texture" | "boundary" | "all"

    Returns:
        结构化分析结果字典
    """
    if roi_hu.ndim != 2:
        return {"error": f"Expected 2D array, got shape {roi_hu.shape}"}

    if roi_hu.size < 16:  # 区域太小
        return {"error": f"ROI too small: {roi_hu.size} pixels (min 16)"}

    result = {}

    if analysis_type in ("density", "all"):
        result["density"] = analyze_density(roi_hu)

    if analysis_type in ("texture", "all"):
        result["texture"] = analyze_texture(roi_hu)

    if analysis_type in ("boundary", "all"):
        result["boundary"] = analyze_boundary(roi_hu)

    return result


def format_for_llm(result: Dict) -> str:
    """将分析结果格式化为 LLM 可读的文本"""
    lines = []

    if "density" in result:
        d = result["density"]
        lines.append("=== 密度分析 (HU值统计) ===")
        lines.append(f"平均HU值: {d['mean_hu']} ± {d['std_hu']}")
        lines.append(f"HU值范围: [{d['min_hu']}, {d['max_hu']}]")
        lines.append(f"实性成分占比: {d['solid_component_pct']}%")
        lines.append(f"磨玻璃成分占比: {d['ggo_component_pct']}%")
        lines.append(f"推断结节类型: {d['inferred_type']}")
        lines.append("")

    if "texture" in result:
        t = result["texture"]
        lines.append("=== 纹理分析 (GLCM) ===")
        lines.append(f"对比度 (contrast): {t['contrast']}")
        lines.append(f"同质性 (homogeneity): {t['homogeneity']}")
        lines.append(f"能量 (energy): {t['energy']}")
        lines.append(f"相关性 (correlation): {t['correlation']}")
        lines.append(f"纹理熵: {t['entropy']}")
        lines.append(f"纹理类型: {t['texture_type']}")
        lines.append("")

    if "boundary" in result:
        b = result["boundary"]
        lines.append("=== 边界分析 ===")
        lines.append(f"毛刺指数: {b['spiculation_index']}")
        lines.append(f"分叶指数: {b['lobulation_index']}")
        lines.append(f"圆度: {b['circularity']} (1=正圆, 0=极不规则)")
        lines.append(f"分形维数: {b['fractal_dimension']}")
        lines.append(f"边缘特征: {b['edge_features']}")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    # 测试
    rng = np.random.default_rng(42)
    # 模拟一个 12mm 结节 ROI (512×512 px, 0.7mm/px → ~17px = 12mm)
    test_roi = rng.normal(-300, 80, (64, 64)).astype(np.float32)
    # 加实性核心
    rr, cc = np.ogrid[:64, :64]
    core_mask = (rr - 32)**2 + (cc - 32)**2 < 15**2
    test_roi[core_mask] = rng.normal(30, 20, core_mask.sum())

    result = analyze_roi(test_roi, analysis_type="all")
    print(format_for_llm(result))
