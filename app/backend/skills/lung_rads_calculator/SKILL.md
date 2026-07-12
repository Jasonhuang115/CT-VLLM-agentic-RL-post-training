# lung_rads_calculator

Calculate pulmonary nodule management helpers such as Lung-RADS, volume doubling time, and simplified malignancy probability.

## Inputs

- `function`: `lung_rads_classify`, `volume_doubling_time`, or `malignancy_probability`.
- Additional fields are passed to the selected calculator function.

## Behavior

Uses deterministic calculators from `agent/tools/measurement_calc.py`.

## Output

Returns a JSON-like dictionary with the calculated result.
