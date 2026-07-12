"""CT volume upload handling."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile

try:
    import SimpleITK as sitk
except ImportError:  # pragma: no cover - handled at runtime
    sitk = None


SUPPORTED_SINGLE_VOLUME_SUFFIXES = {".mhd", ".nii", ".gz", ".nrrd"}


def _safe_name(name: str) -> str:
    keep = [c if c.isalnum() or c in {".", "-", "_"} else "_" for c in name]
    return "".join(keep).strip("._") or "upload"


async def save_upload_files(files: list[UploadFile], upload_root: Path, max_upload_mb: int) -> tuple[Path, list[Path]]:
    if not files:
        raise HTTPException(status_code=400, detail="请上传 CT 文件。支持 .nii/.nii.gz/.mhd+.raw 或 DICOM zip。")

    case_dir = upload_root / str(uuid4())
    case_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    max_bytes = max_upload_mb * 1024 * 1024
    for upload in files:
        filename = _safe_name(upload.filename or "upload")
        out_path = case_dir / filename
        total = 0
        with out_path.open("wb") as f:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(status_code=413, detail=f"单个文件超过 {max_upload_mb} MB 限制。")
                f.write(chunk)
        saved.append(out_path)

    return case_dir, saved


def resolve_ct_input(case_dir: Path, saved_files: list[Path]) -> Path:
    """Return a path SimpleITK can read, extracting zip DICOM if needed."""
    if len(saved_files) == 1 and saved_files[0].suffix.lower() == ".zip":
        dicom_dir = case_dir / "dicom"
        dicom_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(saved_files[0]) as zf:
            zf.extractall(dicom_dir)
        return dicom_dir

    for path in saved_files:
        name = path.name.lower()
        if name.endswith(".nii.gz") or path.suffix.lower() in SUPPORTED_SINGLE_VOLUME_SUFFIXES:
            return path

    raise HTTPException(
        status_code=400,
        detail="未找到可读取的 CT 体积文件。请上传 .nii/.nii.gz/.mhd+.raw/.nrrd 或 DICOM zip。",
    )


def read_image(path: Path):
    if sitk is None:
        raise HTTPException(status_code=500, detail="SimpleITK 未安装，无法读取 CT 体积。")

    if path.is_dir():
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(str(path))
        if not series_ids:
            raise HTTPException(status_code=400, detail="DICOM zip 中未找到可读取的 series。")
        filenames = reader.GetGDCMSeriesFileNames(str(path), series_ids[0])
        reader.SetFileNames(filenames)
        return reader.Execute()

    return sitk.ReadImage(str(path))


def copy_case_file(src: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    return dest
