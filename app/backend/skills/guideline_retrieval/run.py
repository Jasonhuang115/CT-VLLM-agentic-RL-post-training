"""Local guideline retrieval skill."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from agent.tools.guideline_retrieval import GuidelineRetrieval


def run(payload: dict[str, Any]) -> dict[str, Any]:
    retriever = GuidelineRetrieval()
    return retriever.search(
        query=str(payload.get("query", "")),
        guideline_type=payload.get("guideline_type"),
        top_k=int(payload.get("top_k", 3)),
    )


if __name__ == "__main__":
    payload = json.loads(sys.stdin.read() or "{}")
    print(json.dumps(run(payload), ensure_ascii=False, indent=2))
