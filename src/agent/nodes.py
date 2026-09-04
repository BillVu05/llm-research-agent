"""The graph nodes. Each takes the state and returns the keys it changed."""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from . import config, llm, prompts, search
from .state import ResearchState


def _debug(state: ResearchState, *lines: str) -> None:
    if state.get("debug"):
        for line in lines:
            print(line, file=sys.stderr)


def generate_queries(state: ResearchState) -> dict[str, Any]:
    """Plan the research: search queries plus the facts the answer must contain."""
    topic = state["topic"]
    parsed = llm.parse_json(llm.generate(prompts.plan(topic)), {})
    if not isinstance(parsed, dict):
        parsed = {}

    queries = [q for q in parsed.get("queries", []) if isinstance(q, str)] or [topic]
    slots = [s for s in parsed.get("slots", []) if isinstance(s, str)]

    _debug(
        state,
        f"[generate_queries] queries={queries}",
        f"[generate_queries] slots={slots}",
    )
    return {"queries": queries, "slots": slots, "docs": [], "rounds": 0}


def web_search(state: ResearchState) -> dict[str, Any]:
    """Run the current queries concurrently, keeping newly seen URLs."""
    queries = state.get("queries", [])
    docs = list(state.get("docs", []))
    seen = {doc["url"] for doc in docs}

    with ThreadPoolExecutor(max_workers=config.SEARCH_WORKERS) as pool:
        batches = list(pool.map(search.search_one, queries))

    fresh: list[dict[str, str]] = []
    for batch in batches:
        for item in batch:
            url = item.get("link")
            if not url or url in seen:
                continue
            seen.add(url)
            fresh.append(
                {
                    "title": item.get("title", ""),
                    "url": url,
                    # The snippet is the fallback evidence when a page cannot be
                    # fetched, and the retrieval floor when embedding fails.
                    "snippet": item.get("snippet", ""),
                }
            )

    # Plain append: relevance ranking in retrieve() decides what the model sees,
    # so a later round no longer has to fight for a slot in a fixed window.
    docs = docs + fresh

    _debug(
        state,
        f"[web_search] {len(queries)} queries -> {len(fresh)} new, {len(docs)} unique docs",
    )
    return {"docs": docs}


def reflect(state: ResearchState) -> dict[str, Any]:
    """Check slot coverage; on a gap, emit targeted follow-up queries."""
    context = state.get("context", [])
    slots = state.get("slots", [])
    rounds = state.get("rounds", 0) + 1

    # ponytail: with no context a second identical search cannot help, so stop
    # and let synthesize emit the empty-result answer.
    if not context or not slots:
        _debug(state, f"[reflect] round {rounds}: nothing to evaluate, stopping")
        return {"filled": [], "need_more": False, "rounds": rounds}

    parsed = llm.parse_json(llm.generate(prompts.reflect(slots, context)), {})
    if not isinstance(parsed, dict):
        parsed = {}

    # Must be a list of strings. A bare string would make `in` a substring test,
    # so a model replying "winner" would also mark a slot named "win" filled;
    # a dict would silently match on its keys.
    raw_claimed = parsed.get("filled", [])
    claimed = (
        {c for c in raw_claimed if isinstance(c, str)}
        if isinstance(raw_claimed, list)
        else set()
    )
    filled = [s for s in slots if s in claimed]
    missing = [s for s in slots if s not in filled]

    if state.get("debug"):
        explanations = parsed.get("explanations", {}) or {}
        print(f"[reflect] round {rounds}: filled={filled} missing={missing}", file=sys.stderr)
        for slot in slots:
            print(f"  - {slot}: {explanations.get(slot, '(no explanation)')}", file=sys.stderr)

    result: dict[str, Any] = {
        "filled": filled,
        "need_more": bool(missing),
        "rounds": rounds,
    }
    if missing:
        # Targeted follow-ups for the next round, replacing the spent queries.
        # Prefer the ones the model wrote: the fallback concatenates the whole
        # question with a slot name ("Who won ...? score"), which is a worse
        # query than anything a human would type.
        raw = parsed.get("followup_queries", [])
        followups = (
            [q for q in raw if isinstance(q, str) and q.strip()]
            if isinstance(raw, list)
            else []
        )
        result["queries"] = followups or [f"{state['topic']} {slot}" for slot in missing]
    return result


def route_after_reflect(state: ResearchState) -> str:
    """Conditional edge: search again for missing facts, or synthesize."""
    if state.get("need_more") and state.get("rounds", 0) < config.MAX_SEARCH_ROUNDS:
        return "web_search"
    return "synthesize"


def synthesize(state: ResearchState) -> dict[str, Any]:
    """Write the answer, citing only sources that were actually retrieved."""
    topic = state["topic"]
    context = state.get("context", [])
    if not context:
        return {
            "answer": "No relevant documents found to answer the question.",
            "citations": [],
            "degraded": True,
        }

    parsed = llm.parse_json(llm.generate(prompts.synthesize(topic, context)), {})
    if not isinstance(parsed, dict):
        parsed = {}

    if not parsed.get("answer"):
        # The model call failed or returned unusable JSON. Say so explicitly
        # rather than emitting a plausible-looking empty answer.
        return {
            "answer": "Could not produce an answer from the model response.",
            "citations": [],
            "degraded": True,
        }
    answer = parsed["answer"]

    # Rebuild citations from our own retrieved sources so a cited URL can never
    # be one the model invented. format_docs only ever shows MAX_CONTEXT_DOCS,
    # so the index space the model can cite into stops there too.
    shown = context[: config.MAX_CONTEXT_DOCS]
    picked: list[int] = []
    for raw in parsed.get("citations", []) or []:
        num = raw.get("id") if isinstance(raw, dict) else raw
        if isinstance(num, bool) or not isinstance(num, int):
            continue
        if 1 <= num <= len(shown) and num not in picked:
            picked.append(num)

    citations = [
        {"id": n, "title": shown[n - 1]["title"], "url": shown[n - 1]["url"]} for n in picked
    ]
    return {"answer": answer, "citations": citations, "degraded": False}
