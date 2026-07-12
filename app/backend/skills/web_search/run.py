"""Tavily-backed web search skill."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from agent.tools.web_search import WebSearch


def run(payload: dict[str, Any]) -> dict[str, Any]:
    searcher = WebSearch()
    return searcher.search(
        query=str(payload.get("query", "")),
        max_results=int(payload.get("max_results", 5)),
    )


if __name__ == "__main__":
    payload = json.loads(sys.stdin.read() or "{}")
    print(json.dumps(run(payload), ensure_ascii=False, indent=2))
