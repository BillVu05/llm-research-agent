"""Evaluation harness for the research agent.

Runs the pipeline over a golden set and reports metrics that answer the only
question that matters about a change: did it get better or worse?

Most metrics are deterministic (no LLM, no cost, no variance). Groundedness is
the one judged metric and is opt-in via --judge.

    python eval/run_eval.py --limit 5
    python eval/run_eval.py --judge --out results.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent import graph, llm, prompts

WORD_BUDGET = 100
GOLDEN = Path(__file__).with_name("golden.jsonl")


def load_golden(path: Path) -> list[dict[str, Any]]:
    # utf-8-sig: Windows editors and PowerShell write a BOM, which plain utf-8
    # decoding turns into a JSONDecodeError on the first line.
    with path.open(encoding="utf-8-sig") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def score_case(
    case: dict[str, Any],
    state: dict[str, Any],
    elapsed: float,
    llm_calls: int | None = None,
) -> dict[str, Any]:
    """Deterministic scoring of one run. Pure function - unit tested offline."""
    answer = state.get("answer") or ""
    citations = state.get("citations") or []
    docs = state.get("docs") or []
    chunks = state.get("chunks") or []
    context = state.get("context") or []
    slots = state.get("slots") or []
    filled = state.get("filled") or []

    doc_urls = {c.get("url") for c in context} | {d.get("url") for d in docs}
    schema_ok = (
        isinstance(answer, str)
        and isinstance(citations, list)
        and all(
            isinstance(c, dict) and {"id", "title", "url"} <= set(c) for c in citations
        )
    )

    wanted = case.get("must_include", [])
    haystack = answer.lower()
    hits = [k for k in wanted if k.lower() in haystack]

    # An abstention case has no right answer: the agent is being graded on NOT
    # inventing one. A hallucinated fact here is worse than a missing one, so
    # this is scored as a hard pass/fail rather than folded into recall.
    banned = case.get("must_not_include", [])
    hallucinated = [k for k in banned if k.lower() in haystack]
    abstained = None if not banned else not hallucinated

    # Every cited URL must be one we actually retrieved. This is guaranteed by
    # construction in synthesize(); the metric exists to catch a regression.
    grounded_citations = all(c.get("url") in doc_urls for c in citations)

    # Did retrieval find the fact, or did generation drop it? Without this
    # split, a missed keyword is unattributable: you cannot tell a search that
    # returned nothing useful from a model that was handed the fact and ignored
    # it. keyword_recall is the answer; context_recall is the context.
    context_text = " ".join(c.get("text", "") for c in context).lower()
    context_hits = [k for k in wanted if k.lower() in context_text]

    # Sources that were put in the window but never cited. High context, low
    # utilization means the window is being padded.
    cited_urls = {c.get("url") for c in citations}
    utilization = (
        round(len({c["url"] for c in context} & cited_urls) / len(context), 3)
        if context
        else None
    )

    words = len(answer.split())
    # A degraded run (no documents, or a failed/throttled model call) is NOT a
    # pass, however schema-valid its empty output happens to be.
    answered = bool(answer) and not state.get("degraded", False)
    return {
        "id": case["id"],
        "tier": case.get("tier", "unknown"),
        "question": case["question"],
        "answer": answer,
        "answered": answered,
        "schema_ok": schema_ok,
        "keyword_recall": round(len(hits) / len(wanted), 3) if wanted else None,
        "context_recall": round(len(context_hits) / len(wanted), 3) if wanted else None,
        "missing_keywords": [k for k in wanted if k not in hits],
        # A keyword the context had and the answer lost. This is the actionable
        # list: it is a generation problem, and no amount of better search fixes it.
        "dropped_by_generation": [k for k in context_hits if k not in hits],
        "abstained": abstained,
        "hallucinated_terms": hallucinated,
        "n_citations": len(citations),
        "has_citations": bool(citations),
        "citations_all_retrieved": grounded_citations,
        "n_docs": len(docs),
        "n_chunks": len(chunks),
        "n_context_sources": len(context),
        "chunk_utilization": utilization,
        "page_chunk_rate": (
            round(sum(c.get("source") == "page" for c in chunks) / len(chunks), 3)
            if chunks
            else None
        ),
        "slots": slots,
        "slot_fill_rate": round(len(filled) / len(slots), 3) if slots else None,
        "search_rounds": state.get("rounds", 0),
        "words": words,
        "within_word_budget": words <= WORD_BUDGET,
        "latency_s": round(elapsed, 2),
        # Latency says how long a question took; this says what it cost.
        "llm_calls": llm_calls,
    }


JUDGE_PROMPT = (
    "You are grading whether an answer is supported by its sources.\n"
    "Score 1.0 if every factual claim in the answer is supported by the sources, "
    "0.5 if partially supported, 0.0 if unsupported or contradicted.\n"
    # Braces doubled: this string is passed through str.format below.
    "Return ONLY this JSON, no fences: {{\"groundedness\": 0.0, \"reason\": \"...\"}}\n\n"
    "Question: {question}\nAnswer: {answer}\n\nSources:\n{sources}"
)


def judge_groundedness(case: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """LLM-as-judge groundedness. Opt-in: it costs a call per case."""
    context = state.get("context") or []
    if not context or not state.get("answer"):
        return {"groundedness": None, "reason": "no documents or no answer"}

    prompt = JUDGE_PROMPT.format(
        question=case["question"],
        answer=state.get("answer", ""),
        sources=prompts.format_docs(context),
    )
    parsed = llm.parse_json(llm.generate(prompt), {})
    if not isinstance(parsed, dict):
        return {"groundedness": None, "reason": "unparseable judge output"}
    score = parsed.get("groundedness")
    return {
        "groundedness": score if isinstance(score, (int, float)) else None,
        "reason": parsed.get("reason", ""),
    }


def _mean(values: list[Any]) -> Any:
    nums = [v for v in values if isinstance(v, (int, float))]
    return round(statistics.mean(nums), 3) if nums else None


def _rate(values: list[Any]) -> Any:
    """Pass rate over the cases the metric applies to, ignoring None."""
    flags = [v for v in values if v is not None]
    return round(sum(flags) / len(flags), 3) if flags else None


def _p95(values: list[float]) -> Any:
    """The old formula was sorted[int(0.95 * n) - 1], which lands on p92 at
    n=25. quantiles interpolates and is stdlib."""
    if not values:
        return None
    if len(values) < 2:
        return round(values[0], 2)
    return round(statistics.quantiles(values, n=20, method="inclusive")[-1], 2)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {"cases": 0}
    return {
        "cases": n,
        "answered_rate": round(sum(r["answered"] for r in rows) / n, 3),
        "schema_ok_rate": round(sum(r["schema_ok"] for r in rows) / n, 3),
        "has_citations_rate": round(sum(r["has_citations"] for r in rows) / n, 3),
        "citation_validity_rate": round(
            sum(r["citations_all_retrieved"] for r in rows) / n, 3
        ),
        "mean_keyword_recall": _mean([r["keyword_recall"] for r in rows]),
        "mean_context_recall": _mean([r["context_recall"] for r in rows]),
        "mean_slot_fill_rate": _mean([r["slot_fill_rate"] for r in rows]),
        "word_budget_rate": round(sum(r["within_word_budget"] for r in rows) / n, 3),
        "mean_citations": _mean([r["n_citations"] for r in rows]),
        "mean_docs": _mean([r["n_docs"] for r in rows]),
        "mean_chunks": _mean([r["n_chunks"] for r in rows]),
        "mean_chunk_utilization": _mean([r["chunk_utilization"] for r in rows]),
        # Share of chunks that came from a fetched page rather than a snippet
        # fallback. A collapse here explains a quality drop that looks mysterious.
        "page_chunk_rate": _mean([r["page_chunk_rate"] for r in rows]),
        "second_round_rate": round(sum(r["search_rounds"] > 1 for r in rows) / n, 3),
        "mean_groundedness": _mean([r.get("groundedness") for r in rows]),
        # Abstention cases only. None when the run contained none of them.
        "abstention_rate": _rate([r["abstained"] for r in rows]),
        "mean_llm_calls": _mean([r["llm_calls"] for r in rows]),
        "mean_latency_s": _mean([r["latency_s"] for r in rows]),
        "p95_latency_s": _p95([r["latency_s"] for r in rows]),
    }


def render(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    out = [
        f"{'id':<22} {'tier':<11} {'kw':>5} {'ctx':>5} {'slots':>6} "
        f"{'cite':>5} {'rnd':>4} {'sec':>6}  ok",
        "-" * 84,
    ]
    for r in rows:
        kw = "-" if r["keyword_recall"] is None else f"{r['keyword_recall']:.2f}"
        ctx = "-" if r["context_recall"] is None else f"{r['context_recall']:.2f}"
        sl = "-" if r["slot_fill_rate"] is None else f"{r['slot_fill_rate']:.2f}"
        # An abstention case passes by declining, so a fabricated answer fails
        # it however well-formed and well-cited that answer is.
        ok = (
            "PASS"
            if r["answered"]
            and r["schema_ok"]
            and r["citations_all_retrieved"]
            and r["abstained"] is not False
            else "FAIL"
        )
        out.append(
            f"{r['id']:<22} {r['tier']:<11} {kw:>5} {ctx:>5} {sl:>6} "
            f"{r['n_citations']:>5} {r['search_rounds']:>4} {r['latency_s']:>6.2f}  {ok}"
        )
    out.append("")
    out.append("SUMMARY")
    for key, value in summary.items():
        out.append(f"  {key:<24} {value}")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=GOLDEN)
    parser.add_argument("--limit", type=int, help="Run only the first N cases")
    parser.add_argument("--judge", action="store_true", help="Add LLM groundedness scoring")
    parser.add_argument("--out", type=Path, help="Write full JSON results here")
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to pause between cases. The Gemini free tier allows only "
        "5 requests/minute and a case costs 3-4; try --sleep 45 there.",
    )
    args = parser.parse_args()

    cases = load_golden(args.golden)
    if args.limit:
        cases = cases[: args.limit]

    pipeline = graph.build_pipeline()
    rows = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['id']}", file=sys.stderr)
        if args.sleep and i > 1:
            time.sleep(args.sleep)
        start = time.perf_counter()
        calls_before = llm.LLM_CALLS
        try:
            state = pipeline.invoke({"topic": case["question"], "debug": False})
        except Exception as exc:  # a crashed case is a data point, not a stop
            print(f"    ERROR: {exc}", file=sys.stderr)
            state = {"answer": "", "citations": [], "docs": [], "context": []}
        row = score_case(
            case, state, time.perf_counter() - start, llm.LLM_CALLS - calls_before
        )
        if args.judge:
            row.update(judge_groundedness(case, state))
        rows.append(row)

    summary = summarize(rows)
    print(render(rows, summary))

    if args.out:
        args.out.write_text(
            json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
