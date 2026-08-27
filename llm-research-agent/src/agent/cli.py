"""CLI research agent.

Question -> query planning -> web search -> coverage reflection -> cited JSON answer.

The pipeline is a LangGraph ``StateGraph``. ``reflect`` decides whether the
retrieved documents cover every required fact ("slot"); if they do not, a
conditional edge routes back to ``web_search`` with targeted follow-up queries,
bounded by ``MAX_SEARCH_ROUNDS``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any, Dict, List, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from serpapi import GoogleSearch

load_dotenv()

# A pinned model name is a time bomb: gemini-1.5-flash was retired and every
# live run 404'd. The rolling alias self-heals; override to pin deliberately.
MODEL_NAME = os.getenv("GEMINI_MODEL", "models/gemini-flash-latest")
MAX_SEARCH_ROUNDS = 2
MAX_CONTEXT_DOCS = 8
RESULTS_PER_QUERY = 10
SEARCH_WORKERS = 5


class ResearchState(TypedDict, total=False):
    topic: str
    debug: bool
    queries: List[str]
    slots: List[str]
    docs: List[Dict[str, str]]
    filled: List[str]
    need_more: bool
    rounds: int
    answer: str
    citations: List[Dict[str, Any]]
    # True when the answer is a fallback (no documents, or the model failed).
    # Lets callers and the eval harness tell a real answer from a degraded one
    # without string-matching the fallback text.
    degraded: bool


# --- LLM access -----------------------------------------------------------


@lru_cache(maxsize=1)
def _model():
    # Imported lazily so that `import agent.cli` needs neither the SDK nor an
    # API key - keeps the test suite fast and key-free.
    import google.generativeai as genai

    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    return genai.GenerativeModel(MODEL_NAME)


def _generate(prompt: str) -> str:
    """The single seam for every LLM call. Tests patch this one function.

    A model failure returns "" so callers fall back to their defaults and the
    CLI still emits valid JSON, rather than dying with a traceback.
    """
    try:
        return _model().generate_content(prompt).text.strip()
    except Exception as exc:
        print(f"[llm] generation failed: {exc}", file=sys.stderr)
        return ""


def _parse_json(raw: str, default: Any) -> Any:
    """Parse JSON from an LLM, tolerating ```json fences."""
    text = raw.strip()
    if text.startswith("```"):
        text = "\n".join(
            line for line in text.splitlines() if not line.strip().startswith("```")
        ).strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


def _format_docs(docs: List[Dict[str, str]]) -> str:
    """Number the documents so the model can cite them by index."""
    lines = []
    for i, doc in enumerate(docs[:MAX_CONTEXT_DOCS], 1):
        entry = f"[{i}] {doc.get('title', '')}\n    {doc.get('url', '')}"
        snippet = (doc.get("snippet") or "").strip()
        if snippet:
            entry += f"\n    {snippet}"
        lines.append(entry)
    return "\n".join(lines)


# --- Nodes ----------------------------------------------------------------


def generate_queries(state: ResearchState) -> Dict[str, Any]:
    """Plan the research: search queries plus the facts the answer must contain."""
    topic = state["topic"]
    prompt = (
        "You are planning web research for a question.\n"
        "Return ONLY a JSON object, with no markdown fences:\n"
        '{"queries": ["...", "..."], "slots": ["...", "..."]}\n\n'
        '"queries": 3-5 distinct English web search queries that together answer '
        "the question.\n"
        '"slots": 2-4 short lowercase names for the specific facts a complete '
        "answer must contain (for a match result, say: winner, score, date).\n\n"
        f"Question: {topic}"
    )
    parsed = _parse_json(_generate(prompt), {})
    if not isinstance(parsed, dict):
        parsed = {}

    queries = [q for q in parsed.get("queries", []) if isinstance(q, str)] or [topic]
    slots = [s for s in parsed.get("slots", []) if isinstance(s, str)]

    if state.get("debug"):
        print(f"[generate_queries] queries={queries}")
        print(f"[generate_queries] slots={slots}")

    return {"queries": queries, "slots": slots, "docs": [], "rounds": 0}


def _search_one(query: str) -> List[Dict[str, Any]]:
    params = {
        "engine": "google",
        "q": query,
        "api_key": os.getenv("SERPAPI_API_KEY"),
        "num": str(RESULTS_PER_QUERY),
        "safe": "active",
        "hl": "en",
        "gl": "us",
    }
    try:
        return GoogleSearch(params).get_dict().get("organic_results", [])
    except Exception as exc:  # network, quota, auth - all degrade to "no results"
        print(f"[web_search] query {query!r} failed: {exc}")
        return []


def web_search(state: ResearchState) -> Dict[str, Any]:
    """Run the current queries concurrently, appending newly seen URLs."""
    queries = state.get("queries", [])
    docs = list(state.get("docs", []))
    seen = {doc["url"] for doc in docs}

    with ThreadPoolExecutor(max_workers=SEARCH_WORKERS) as pool:
        batches = list(pool.map(_search_one, queries))

    for batch in batches:
        for item in batch:
            url = item.get("link")
            if not url or url in seen:
                continue
            seen.add(url)
            docs.append(
                {
                    "title": item.get("title", ""),
                    "url": url,
                    # The snippet is what makes the answer grounded in retrieved
                    # text rather than in the model's own recall of the headline.
                    "snippet": item.get("snippet", ""),
                }
            )

    if state.get("debug"):
        print(f"[web_search] {len(queries)} queries -> {len(docs)} unique docs")

    return {"docs": docs}


def reflect(state: ResearchState) -> Dict[str, Any]:
    """Check slot coverage; on a gap, emit targeted follow-up queries."""
    docs = state.get("docs", [])
    slots = state.get("slots", [])
    rounds = state.get("rounds", 0) + 1

    # ponytail: with no docs a second identical search cannot help, so stop and
    # let synthesize emit the empty-result answer.
    if not docs or not slots:
        if state.get("debug"):
            print(f"[reflect] round {rounds}: nothing to evaluate, stopping")
        return {"filled": [], "need_more": False, "rounds": rounds}

    prompt = (
        "Decide which required facts are explicitly supported by the documents.\n"
        "Do not guess: a fact counts as filled only with clear evidence.\n\n"
        f"Required facts: {slots}\n"
        f"Documents:\n{_format_docs(docs)}\n\n"
        "Return ONLY this JSON, with no markdown fences:\n"
        '{"filled": ["fact", ...], "explanations": {"fact": "brief evidence"}}'
    )
    parsed = _parse_json(_generate(prompt), {})
    if not isinstance(parsed, dict):
        parsed = {}

    claimed = parsed.get("filled", [])
    filled = [s for s in slots if s in claimed]
    missing = [s for s in slots if s not in filled]

    if state.get("debug"):
        explanations = parsed.get("explanations", {}) or {}
        print(f"[reflect] round {rounds}: filled={filled} missing={missing}")
        for slot in slots:
            print(f"  - {slot}: {explanations.get(slot, '(no explanation)')}")

    result: Dict[str, Any] = {
        "filled": filled,
        "need_more": bool(missing),
        "rounds": rounds,
    }
    if missing:
        # Targeted follow-ups for the next round, replacing the spent queries.
        result["queries"] = [f"{state['topic']} {slot}" for slot in missing]
    return result


def route_after_reflect(state: ResearchState) -> str:
    """Conditional edge: search again for missing facts, or synthesize."""
    if state.get("need_more") and state.get("rounds", 0) < MAX_SEARCH_ROUNDS:
        return "web_search"
    return "synthesize"


def synthesize(state: ResearchState) -> Dict[str, Any]:
    """Write the answer, citing only documents that were actually retrieved."""
    topic = state["topic"]
    docs = state.get("docs", [])
    if not docs:
        return {
            "answer": "No relevant documents found to answer the question.",
            "citations": [],
            "degraded": True,
        }

    context = docs[:MAX_CONTEXT_DOCS]
    prompt = (
        f"Answer this research question: '{topic}'\n"
        "Use only the numbered sources below. Write at most 80 words, and cite "
        "with bracketed source numbers like [1][2].\n"
        "Return ONLY this JSON, with no markdown fences:\n"
        '{"answer": "...", "citations": [1, 2]}\n\n'
        f"Sources:\n{_format_docs(context)}"
    )
    parsed = _parse_json(_generate(prompt), {})
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

    # Rebuild citations from our own retrieved docs so a cited URL can never be
    # one the model invented.
    picked: List[int] = []
    for raw in parsed.get("citations", []) or []:
        num = raw.get("id") if isinstance(raw, dict) else raw
        if isinstance(num, bool) or not isinstance(num, int):
            continue
        if 1 <= num <= len(context) and num not in picked:
            picked.append(num)

    citations = [
        {"id": n, "title": context[n - 1]["title"], "url": context[n - 1]["url"]}
        for n in picked
    ]
    return {"answer": answer, "citations": citations, "degraded": False}


# --- Graph ----------------------------------------------------------------


def build_pipeline():
    """Compile the research graph."""
    graph = StateGraph(ResearchState)
    graph.add_node("generate_queries", generate_queries)
    graph.add_node("web_search", web_search)
    graph.add_node("reflect", reflect)
    graph.add_node("synthesize", synthesize)

    graph.add_edge(START, "generate_queries")
    graph.add_edge("generate_queries", "web_search")
    graph.add_edge("web_search", "reflect")
    graph.add_conditional_edges(
        "reflect",
        route_after_reflect,
        {"web_search": "web_search", "synthesize": "synthesize"},
    )
    graph.add_edge("synthesize", END)
    return graph.compile()


def research(topic: str, debug: bool = False) -> Dict[str, Any]:
    """Run the pipeline and return just the answer payload."""
    final = build_pipeline().invoke({"topic": topic, "debug": debug})
    return {"answer": final.get("answer", ""), "citations": final.get("citations", [])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", type=str, required=False)
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "question", nargs="?", help="Research question (positional for Docker)"
    )
    args = parser.parse_args()

    topic = args.topic or args.question
    if not topic:
        parser.error("Please provide a research topic/question.")

    print(json.dumps(research(topic, args.debug), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
