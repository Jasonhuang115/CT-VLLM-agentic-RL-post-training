# CT Nodule Chatbot App

This is the lightweight deployment app for the trained lung CT nodule VLM.

## Design

- Manual nodule coordinates are the primary path.
- Automatic detection is optional and disabled by default.
- ROI generation is locked to the training contract:
  - Lung window `WL=-600`, `WW=1500`
  - `50 mm` ROI
  - `512 x 512` output
  - `PIL.Image.Resampling.LANCZOS`
  - 8-bit grayscale PNG
- Coordinate conversion uses SimpleITK physical point to index conversion, so origin, spacing, and direction are respected.

## Install

```bash
pip install -r requirements-app.txt
```

## Configure

Copy `.env.example` to `.env` and export the values in your shell, or set them in AutoDL.

For local UI/API testing without the VLM:

```bash
export VLM_MOCK=true
```

For vLLM, run the model service on port `8000` and the FastAPI backend on `8080`.

```bash
vllm serve huang01080524/lungct-nodule-grpo \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.6 \
  --max-model-len 4096 \
  --limit-mm-per-prompt "image=3"
```

Then start the backend:

```bash
python -m app.backend.main
```

Open:

```text
http://localhost:8080
```

## Analyze API

`POST /analyze` uses `multipart/form-data`.

Fields:

- `ct_files`: one or more files. Supported inputs:
  - `.nii`
  - `.nii.gz`
  - `.mhd` plus matching `.raw`
  - `.nrrd`
  - DICOM `.zip`
- `nodule_coords`: JSON object or list with LUNA-style physical/world coordinates in mm.
- `clinical_info`: optional JSON.
- `message`: optional user question.
- `session_id`: optional, for multi-turn context.
- `use_tools`: `true` or `false`.

Example `nodule_coords`:

```json
[
  {"x": 12.3, "y": -45.2, "z": -120.5, "diameter_mm": 8.6}
]
```

## Detect API

`POST /detect` uses the same `ct_files` upload field.

By default detection is disabled. To connect an AutoDL detector:

```bash
export DETECTOR_BACKEND=command
export DETECTOR_COMMAND="python /path/to/detect_luna16.py"
```

The command receives the CT path as its final argument and must print:

```json
[
  {"x": 12.3, "y": -45.2, "z": -120.5, "diameter_mm": 8.6, "confidence": 0.95}
]
```

## Quick Verification

```bash
# On AutoDL, run:
bash app/verify_deploy.sh
```

This checks:
- Dependencies installed
- Python compilation
- All 4 skills functional
- ROI constants locked to training values
- Mock backend /health endpoint reachable

## Manual ROI Validation

This is the **most important check** before trusting the deployment:

1. Pick one LUNA16 case with a known nodule coordinate
2. Run ROI through training pipeline: `python data/mhd_to_png.py --mode multi_view ...`
3. Run same case through app pipeline: `POST /analyze` with same CT + coordinates
4. Compare pixel arrays of the resulting axial/coronal/sagittal PNGs
5. Mean absolute pixel diff must be < 1.0 (floating point rounding acceptable)

If diff > 1.0, investigate coordinate transform (SimpleITK physical→index vs raw spacing/origin).

## AutoDL Detector Recommendation

Start with a MONAI/PyTorch LUNA16 detector as a candidate generator. Keep manual coordinate input available even after enabling detection, because detector performance is sensitive to CT acquisition, spacing, reconstruction kernel, and coordinate conventions.

## Current Status (2026-07-12)

- ✅ All 4 skills wired (guideline_retrieval, lung_rads_calculator, web_search, image_metadata)
- ✅ Python compilation clean
- ✅ Mock mode works for frontend/backend dev
- ⚠️ Real VLM integration not yet verified on AutoDL
- ⚠️ ROI output not yet compared against training
