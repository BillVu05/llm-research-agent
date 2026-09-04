"""SerpAPI access. One query in, a list of organic results out."""

from __future__ import annotations

import os
import sys

from serpapi import GoogleSearch

from . import config


def search_one(query: str) -> list[dict[str, str]]:
    params = {
        "engine": "google",
        "q": query,
        "api_key": os.getenv("SERPAPI_API_KEY"),
        "num": str(config.RESULTS_PER_QUERY),
        "safe": "active",
        # A non-English question still gets US English results otherwise.
        "hl": config.SEARCH_LANG,
        "gl": config.SEARCH_COUNTRY,
    }
    try:
        search = GoogleSearch(params)
        search.timeout = config.SEARCH_TIMEOUT_S
        return search.get_dict().get("organic_results", [])
    except Exception as exc:  # network, quota, auth - all degrade to "no results"
        print(f"[web_search] query {query!r} failed: {exc}", file=sys.stderr)
        return []
