# image_metadata

Inspect generated ROI PNG metadata without changing the VLM input distribution.

## Inputs

- `paths`: Dictionary of ROI PNG paths.

## Behavior

Reads image size, mode, and basic grayscale statistics. This tool does not diagnose and does not alter ROI images.

## Output

Returns per-view image metadata.
