# guideline_retrieval

Retrieve local pulmonary nodule management guideline snippets.

## Inputs

- `query` (string): Guideline search query.
- `guideline_type` (string, optional): One of `fleischner`, `lung_rads`, `nccn`, `china`.
- `top_k` (integer, optional): Number of sections to return.

## Behavior

Uses the existing local guideline retrieval implementation from `agent/tools/guideline_retrieval.py`.

## Output

Returns matching guideline sections and a short summary.
