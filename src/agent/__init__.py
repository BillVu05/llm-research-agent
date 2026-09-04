"""Retrieval-augmented research agent.

Question -> query planning -> web search -> page ingest -> vector retrieval ->
coverage reflection -> cited JSON answer.

The pipeline is a LangGraph ``StateGraph``. ``reflect`` decides whether the
retrieved context covers every required fact ("slot"); if it does not, a
conditional edge routes back to ``web_search`` with targeted follow-up queries,
bounded by ``MAX_SEARCH_ROUNDS``.
"""

from __future__ import annotations

from typing import Any

from .graph import build_pipeline

__all__ = ["build_pipeline", "research"]


def research(topic: str, debug: bool = False) -> dict[str, Any]:
    """Run the pipeline and return just the answer payload."""
    final = build_pipeline().invoke({"topic": topic, "debug": debug})
    return {
        "answer": final.get("answer", ""),
        "citations": final.get("citations", []),
        # Surfaced so a caller can tell a real answer from a fallback without
        # string-matching the fallback text.
        "degraded": bool(final.get("degraded", False)),
    }
