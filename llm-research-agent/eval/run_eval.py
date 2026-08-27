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
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent import cli  # noqa: E402

WORD_BUDGET = 100
GOLDEN = Path(__file__).with_name("golden.jsonl")


def load_golden(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def score_case(case: Dict[str, Any], state: Dict[str, Any], elapsed: float) -> Dict[str, Any]:
    """Deterministic scoring of one run. Pure function - unit tested offline."""
    answer = state.get("answer") or ""
    citations = state.get("citations") or []
    docs = state.get("docs") or []
    slots = state.get("slots") or []
    filled = state.get("filled") or []

    doc_urls = {d.get("url") for d in docs}
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

    # Every cited URL must be one we actually retrieved. This is guaranteed by
    # construction in synthesize(); the metric exists to catch a regression.
    grounded_citations = all(c.get("url") in doc_urls for c in citations)

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
        "missing_keywords": [k for k in wanted if k not in hits],
        "n_citations": len(citations),
        "has_citations": bool(citations),
        "citations_all_retrieved": grounded_citations,
        "n_docs": len(docs),
        "slots": slots,
        "slot_fill_rate": round(len(filled) / len(slots), 3) if slots else None,
        "search_rounds": state.get("rounds", 0),
        "words": words,
        "within_word_budget": words <= WORD_BUDGET,
        "latency_s": round(elapsed, 2),
    }


JUDGE_PROMPT = (
    "You are grading whether an answer is supported by its sources.\n"
    "Score 1.0 if every factual claim in the answer is supported by the sources, "
    "0.5 if partially supported, 0.0 if unsupported or contradicted.\n"
    # Braces doubled: this string is passed through str.format below.
    "Return ONLY this JSON, no fences: {{\"groundedness\": 0.0, \"reason\": \"...\"}}\n\n"
    "Question: {question}\nAnswer: {answer}\n\nSources:\n{sources}"
)


def judge_groundedness(case: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """LLM-as-judge groundedness. Opt-in: it costs a call per case."""
    docs = state.get("docs") or []
    if not docs or not state.get("answer"):
        return {"groundedness": None, "reason": "no documents or no answer"}

    prompt = JUDGE_PROMPT.format(
        question=case["question"],
        answer=state.get("answer", ""),
        sources=cli._format_docs(docs),
    )
    parsed = cli._parse_json(cli._generate(prompt), {})
    if not isinstance(parsed, dict):
        return {"groundedness": None, "reason": "unparseable judge output"}
    score = parsed.get("groundedness")
    return {
        "groundedness": score if isinstance(score, (int, float)) else None,
        "reason": parsed.get("reason", ""),
    }


def _mean(values: List[Any]) -> Any:
    nums = [v for v in values if isinstance(v, (int, float))]
    return round(statistics.mean(nums), 3) if nums else None


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
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
        "mean_slot_fill_rate": _mean([r["slot_fill_rate"] for r in rows]),
        "word_budget_rate": round(sum(r["within_word_budget"] for r in rows) / n, 3),
        "mean_citations": _mean([r["n_citations"] for r in rows]),
        "mean_docs": _mean([r["n_docs"] for r in rows]),
        "second_round_rate": round(sum(r["search_rounds"] > 1 for r in rows) / n, 3),
        "mean_groundedness": _mean([r.get("groundedness") for r in rows]),
        "mean_latency_s": _mean([r["latency_s"] for r in rows]),
        "p95_latency_s": round(
            sorted(r["latency_s"] for r in rows)[max(0, int(0.95 * n) - 1)], 2
        ),
    }


def render(rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> str:
    out = [
        f"{'id':<22} {'tier':<11} {'kw':>5} {'slots':>6} {'cite':>5} {'rnd':>4} {'sec':>6}  ok",
        "-" * 78,
    ]
    for r in rows:
        kw = "-" if r["keyword_recall"] is None else f"{r['keyword_recall']:.2f}"
        sl = "-" if r["slot_fill_rate"] is None else f"{r['slot_fill_rate']:.2f}"
        ok = (
            "PASS"
            if r["answered"] and r["schema_ok"] and r["citations_all_retrieved"]
            else "FAIL"
        )
        out.append(
            f"{r['id']:<22} {r['tier']:<11} {kw:>5} {sl:>6} "
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

    pipeline = cli.build_pipeline()
    rows = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['id']}", file=sys.stderr)
        if args.sleep and i > 1:
            time.sleep(args.sleep)
        start = time.perf_counter()
        try:
            state = pipeline.invoke({"topic": case["question"], "debug": False})
        except Exception as exc:  # a crashed case is a data point, not a stop
            print(f"    ERROR: {exc}", file=sys.stderr)
            state = {"answer": "", "citations": [], "docs": []}
        row = score_case(case, state, time.perf_counter() - start)
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
