"""Locked ROI extraction for the trained VLM."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from app.backend.constants import (
    OUTPUT_SIZE_PX,
    PNG_MODE,
    RESIZE_FILTER,
    ROI_SIZE_MM,
    WINDOW_LEVEL,
    WINDOW_WIDTH,
)
from app.backend.schemas import NoduleCoord


@dataclass(frozen=True)
class ROIResult:
    paths: dict[str, Path]
    images_base64: dict[str, str]
    voxel_index_xyz: tuple[int, int, int]
    physical_point_xyz: tuple[float, float, float]
    spacing_xyz: tuple[float, float, float]


def window_lung(hu_slice: np.ndarray) -> np.ndarray:
    low = WINDOW_LEVEL - WINDOW_WIDTH / 2
    high = WINDOW_LEVEL + WINDOW_WIDTH / 2
    clipped = np.clip(hu_slice, low, high)
    normalized = (clipped - low) / (high - low)
    return (normalized * 255).astype(np.uint8)


def _resize_and_save(slab: np.ndarray, out_path: Path) -> None:
    gray = window_lung(slab)
    image = Image.fromarray(gray, mode=PNG_MODE)
    image = image.resize((OUTPUT_SIZE_PX, OUTPUT_SIZE_PX), Image.Resampling.LANCZOS)
    image.save(out_path, format="PNG")


def _encode_png(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _half_width_voxels(roi_size_mm: float, spacing_mm: float) -> int:
    if spacing_mm <= 0:
        return 25
    return max(1, int(roi_size_mm / spacing_mm / 2))


def extract_locked_roi(
    ct_path: Path,
    coord: NoduleCoord,
    output_dir: Path,
    nodule_idx: int = 1,
    seriesuid: str = "case",
) -> ROIResult:
    """Extract axial/coronal/sagittal ROI PNGs using the training contract.

    The parameters are intentionally not exposed to callers:
    50 mm ROI, lung window WL=-600/WW=1500, 512 px, LANCZOS, 8-bit grayscale.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise RuntimeError("SimpleITK 未安装，无法生成 CT ROI。请安装 requirements-app.txt。") from exc

    from app.backend.volume_io import read_image

    image = read_image(ct_path)
    volume_hu = np.asarray(sitk.GetArrayFromImage(image))  # (Z, Y, X)

    nz, ny, nx = volume_hu.shape
    physical = (float(coord.x), float(coord.y), float(coord.z))

    try:
        index_xyz = image.TransformPhysicalPointToIndex(physical)
    except RuntimeError:
        # Fallback for legacy LUNA-style files. The SimpleITK path is preferred
        # because it respects origin, spacing, and direction.
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
        index_xyz = tuple(int((physical[i] - origin[i]) / spacing[i]) for i in range(3))

    ix = _clamp(int(index_xyz[0]), 0, nx - 1)
    iy = _clamp(int(index_xyz[1]), 0, ny - 1)
    iz = _clamp(int(index_xyz[2]), 0, nz - 1)

    spacing = image.GetSpacing()
    sx, sy, sz = float(spacing[0]), float(spacing[1]), float(spacing[2])
    rx = _half_width_voxels(ROI_SIZE_MM, sx)
    ry = _half_width_voxels(ROI_SIZE_MM, sy)
    rz = _half_width_voxels(ROI_SIZE_MM, sz)

    prefix = f"{seriesuid}_nodule_{nodule_idx:03d}"
    paths = {
        "axial": output_dir / f"{prefix}_axial.png",
        "coronal": output_dir / f"{prefix}_coronal.png",
        "sagittal": output_dir / f"{prefix}_sagittal.png",
    }

    x1, x2 = _clamp(ix - rx, 0, nx), _clamp(ix + rx, 0, nx)
    y1, y2 = _clamp(iy - ry, 0, ny), _clamp(iy + ry, 0, ny)
    z1, z2 = _clamp(iz - rz, 0, nz), _clamp(iz + rz, 0, nz)

    _resize_and_save(volume_hu[iz, y1:y2, x1:x2], paths["axial"])
    _resize_and_save(volume_hu[z1:z2, iy, x1:x2], paths["coronal"])
    _resize_and_save(volume_hu[z1:z2, y1:y2, ix], paths["sagittal"])

    return ROIResult(
        paths=paths,
        images_base64={name: _encode_png(path) for name, path in paths.items()},
        voxel_index_xyz=(ix, iy, iz),
        physical_point_xyz=physical,
        spacing_xyz=(sx, sy, sz),
    )


def roi_contract() -> dict:
    return {
        "roi_size_mm": ROI_SIZE_MM,
        "output_size_px": OUTPUT_SIZE_PX,
        "window_level": WINDOW_LEVEL,
        "window_width": WINDOW_WIDTH,
        "resize": RESIZE_FILTER,
        "png_mode": PNG_MODE,
    }
