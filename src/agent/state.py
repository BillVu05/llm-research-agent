"""The shape of everything that flows through the graph."""

from __future__ import annotations

from typing import Any, TypedDict


class ResearchState(TypedDict, total=False):
    topic: str
    debug: bool
    queries: list[str]
    slots: list[str]
    # Search hits: {title, url, snippet}. What retrieval draws from.
    docs: list[dict[str, str]]
    # Chunked page text: {title, url, text, source}. source is "page" when the
    # page was fetched and extracted, "snippet" when it fell back.
    chunks: list[dict[str, str]]
    # What the model actually sees: {title, url, text}, one entry per source
    # URL, ranked by relevance. Same three-key shape docs used to have, so the
    # numbered <document> formatting and citation rebuilding are unchanged.
    context: list[dict[str, str]]
    ingested: list[str]
    filled: list[str]
    need_more: bool
    rounds: int
    answer: str
    citations: list[dict[str, Any]]
    # True when the answer is a fallback (no documents, or the model failed).
    # Lets callers and the eval harness tell a real answer from a degraded one
    # without string-matching the fallback text.
    degraded: bool
