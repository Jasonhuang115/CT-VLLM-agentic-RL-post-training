# web_search

Search recent and trustworthy medical web sources for pulmonary nodule evidence.

## Inputs

- `query` (string): Search query.
- `max_results` (integer, optional): Maximum result count.

## Behavior

- Uses Tavily when `TAVILY_API_KEY` is set and `tavily-python` is installed.
- Falls back to built-in offline pulmonary nodule references when Tavily is unavailable.
- Prefer medical domains such as PubMed, NIH, Radiopaedia, ACR, NCCN, major journals, and Chinese medical guideline sources.

## Output

Returns a JSON-like dictionary with:

- `summary`
- `query`
- `results`
- optional `answer`
- optional `note` or `error`
