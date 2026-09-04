"""Rank chunks against the question and build the model's context.

Lexical prefilter -> embed -> cosine -> MMR -> one context doc per source.

The vector store is a numpy array and one matmul.

    # ponytail: at most MAX_CHUNKS_TO_EMBED vectors per question. A vector DB
    # earns its keep around 100k; swap Chroma in behind retrieve() if it ever
    # does.

MMR, not plain top-k: on a news question top-k returns five paragraphs from
five outlets all reporting the same sentence, spending the window that the
*other* required facts needed. That is what makes multi-fact questions work.
"""

from __future__ import annotations

import re
import sys
from typing import Any

import numpy as np

from . import config, llm
from .state import ResearchState


def _normalize(vectors: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # A zero vector would divide by zero and poison every later comparison.
    return matrix / np.maximum(norms, 1e-9)


def mmr(
    query_vec: np.ndarray,
    chunk_vecs: np.ndarray,
    k: int,
    lambda_: float | None = None,
    pool: int | None = None,
) -> list[int]:
    """Maximal Marginal Relevance: pick k indices that are relevant *and* varied.

    Each pick maximises ``lambda * sim(query) - (1 - lambda) * max sim(already
    picked)``, so a chunk that merely restates a chosen one loses to a slightly
    less relevant chunk that says something new.
    """
    lambda_ = config.MMR_LAMBDA if lambda_ is None else lambda_
    pool = pool or config.MMR_POOL

    relevance = chunk_vecs @ query_vec
    candidates = list(np.argsort(-relevance)[:pool])
    if not candidates:
        return []

    picked = [int(candidates.pop(0))]
    while candidates and len(picked) < k:
        # Similarity of every candidate to its nearest already-picked chunk.
        redundancy = (chunk_vecs[candidates] @ chunk_vecs[picked].T).max(axis=1)
        scores = lambda_ * relevance[candidates] - (1 - lambda_) * redundancy
        best = int(np.argmax(scores))
        picked.append(int(candidates.pop(best)))
    return picked


def _prefilter(topic: str, chunks: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    """Shortlist by word overlap before paying to embed.

    Two pages of real article text is ~140 chunks, of which twelve are used.
    Embedding is charged per chunk, so embedding all of them buys nothing and
    walks straight into the free tier's per-minute cap.

    ponytail: plain word overlap, not BM25 - this is a cost guard, not a
    ranker, and the embedding pass is what actually decides relevance. Promote
    it to BM25 only if the eval shows relevant chunks being cut here.
    """
    if len(chunks) <= limit:
        return chunks
    wanted = set(re.findall(r"\w+", topic.lower()))
    # sorted() is stable, so equally-scoring chunks keep their document order.
    return sorted(
        chunks,
        key=lambda c: -len(wanted & set(re.findall(r"\w+", c["text"].lower()))),
    )[:limit]


def _group_by_source(chunks: list[dict[str, str]], order: list[int]) -> list[dict[str, str]]:
    """Merge the picked chunks into one context doc per source URL.

    Per-URL rather than per-chunk so a citation still maps 1:1 to a source: five
    chunks off one Wikipedia page must not become five citations, and the eval's
    citations_all_retrieved check keeps meaning what it meant.
    """
    grouped: dict[str, dict[str, Any]] = {}
    for i in order:
        chunk = chunks[i]
        entry = grouped.setdefault(
            chunk["url"], {"title": chunk["title"], "url": chunk["url"], "parts": []}
        )
        entry["parts"].append(chunk["text"])
    return [
        {"title": e["title"], "url": e["url"], "text": "\n\n".join(e["parts"])}
        for e in grouped.values()
    ]


def _fallback(state: ResearchState) -> list[dict[str, str]]:
    """Search-rank order over snippets - i.e. the pre-RAG behaviour.

    Used when there is nothing to embed or the embedding API is unavailable.
    Retrieval quality degrades; the run still produces a cited answer.
    """
    return [
        {"title": d.get("title", ""), "url": d["url"], "text": d.get("snippet", "")}
        for d in state.get("docs", [])[: config.MAX_CONTEXT_DOCS]
    ]


def retrieve(state: ResearchState) -> dict[str, Any]:
    """Embed the question and the chunks, then pick the context to answer from."""
    all_chunks = state.get("chunks", [])
    topic = state["topic"]

    if not all_chunks:
        context = _fallback(state)
        if state.get("debug"):
            print(f"[retrieve] no chunks; {len(context)} snippet docs", file=sys.stderr)
        return {"context": context}

    chunks = _prefilter(topic, all_chunks, config.MAX_CHUNKS_TO_EMBED)
    query_vecs = llm.embed([topic], "retrieval_query")
    doc_vecs = llm.embed([c["text"] for c in chunks], "retrieval_document")
    if not query_vecs or len(doc_vecs) != len(chunks):
        print("[retrieve] embedding unavailable; falling back to search rank", file=sys.stderr)
        return {"context": _fallback(state)}

    order = mmr(_normalize(query_vecs)[0], _normalize(doc_vecs), config.TOP_K_CHUNKS)
    context = _group_by_source(chunks, order)[: config.MAX_CONTEXT_DOCS]

    if state.get("debug"):
        print(
            f"[retrieve] {len(all_chunks)} chunks -> {len(chunks)} embedded "
            f"-> {len(order)} picked -> {len(context)} sources",
            file=sys.stderr,
        )
        for doc in context:
            print(f"  - {doc['url']} ({len(doc['text'])} chars)", file=sys.stderr)
    return {"context": context}
