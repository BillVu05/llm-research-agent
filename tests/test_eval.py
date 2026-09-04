import json
from pathlib import Path
from unittest.mock import patch

import pytest

import run_eval

CASE = {
    "id": "wc",
    "tier": "retrieval",
    "question": "Who won the 2022 World Cup?",
    "must_include": ["argentina", "penalties"],
}

STATE = {
    "answer": "Argentina won on penalties.",
    "citations": [{"id": 1, "title": "T", "url": "https://a.com"}],
    "docs": [{"title": "T", "url": "https://a.com", "snippet": "s"}],
    "chunks": [{"title": "T", "url": "https://a.com", "text": "Argentina, on penalties.",
                "source": "page"}],
    "context": [{"title": "T", "url": "https://a.com", "text": "Argentina, on penalties."}],
    "slots": ["winner", "score"],
    "filled": ["winner"],
    "rounds": 2,
}


def test_golden_set_is_well_formed():
    cases = run_eval.load_golden(Path(run_eval.__file__).with_name("golden.jsonl"))
    assert len(cases) == 25
    assert len({c["id"] for c in cases}) == 25, "duplicate case ids"
    for c in cases:
        assert c["question"].strip()
        assert c["tier"] in {"parametric", "retrieval", "multi_fact", "abstention"}
        # Abstention cases are graded on what must NOT appear, everything else
        # on what must.
        checkable = c.get("must_include") or c.get("must_not_include")
        assert checkable, f"{c['id']} has no checkable keywords"
        if c["tier"] == "abstention":
            assert c["must_not_include"], f"{c['id']} must ban some fabricated answer"
        # A bare digit like "2" matches "2022" in almost any answer, so a
        # keyword that short proves nothing.
        for kw in c.get("must_include", []):
            assert len(kw) >= 3, f"{c['id']}: keyword {kw!r} is too short to be evidence"
    # multi_fact cases exercise the reflect -> web_search loop, which the easier
    # tiers never trigger; abstention cases catch confident hallucination.
    assert sum(c["tier"] == "multi_fact" for c in cases) >= 5
    assert sum(c["tier"] == "abstention" for c in cases) >= 2
    # Parametric questions measure the model's recall, not the agent's
    # retrieval, so they must not dominate the headline numbers.
    assert sum(c["tier"] == "parametric" for c in cases) <= len(cases) // 2


def test_load_golden_tolerates_bom(tmp_path):
    """Windows editors and PowerShell write a UTF-8 BOM; it must not break."""
    p = tmp_path / "bom.jsonl"
    p.write_bytes(
        b"\xef\xbb\xbf" + json.dumps({"id": "x", "tier": "parametric",
                                      "question": "q", "must_include": ["a"]}).encode()
    )
    assert run_eval.load_golden(p)[0]["id"] == "x"


def test_score_case_happy():
    row = run_eval.score_case(CASE, STATE, 1.234)
    assert row["schema_ok"] is True
    assert row["keyword_recall"] == 1.0
    assert row["missing_keywords"] == []
    assert row["citations_all_retrieved"] is True
    assert row["slot_fill_rate"] == 0.5
    assert row["search_rounds"] == 2
    assert row["within_word_budget"] is True
    assert row["latency_s"] == 1.23


def test_score_case_flags_missing_keywords():
    state = dict(STATE, answer="France won the trophy.")
    row = run_eval.score_case(CASE, state, 0.1)
    assert row["keyword_recall"] == 0.0
    assert set(row["missing_keywords"]) == {"argentina", "penalties"}


def test_score_case_catches_citation_not_retrieved():
    """The regression this metric exists for: a cited URL we never retrieved."""
    state = dict(STATE, citations=[{"id": 1, "title": "X", "url": "https://invented.com"}])
    row = run_eval.score_case(CASE, state, 0.1)
    assert row["citations_all_retrieved"] is False


def test_score_case_handles_empty_run():
    row = run_eval.score_case(CASE, {"answer": "", "citations": [], "docs": []}, 0.1)
    assert row["keyword_recall"] == 0.0
    assert row["has_citations"] is False
    assert row["slot_fill_rate"] is None
    assert row["citations_all_retrieved"] is True  # vacuously - nothing was cited
    assert row["answered"] is False


@pytest.mark.parametrize(
    "state",
    [
        # No documents were retrieved at all.
        {"answer": "No relevant documents found.", "citations": [], "docs": [], "degraded": True},
        # Documents were retrieved but the model call failed or was throttled.
        {
            "answer": "Could not produce an answer from the model response.",
            "citations": [],
            "docs": [{"title": "T", "url": "https://a.com", "snippet": "s"}],
            "degraded": True,
        },
    ],
    ids=["no-documents", "model-failed"],
)
def test_degraded_run_is_not_a_pass(state):
    """A throttled or failed case must not report PASS just because its empty
    output happens to be schema-valid. This regressed once already: the model
    was rate-limited and the harness scored every case PASS."""
    row = run_eval.score_case(CASE, state, 0.1)
    assert row["schema_ok"] is True
    assert row["citations_all_retrieved"] is True
    assert row["answered"] is False
    assert "FAIL" in run_eval.render([row], run_eval.summarize([row]))
    assert run_eval.summarize([row])["answered_rate"] == 0.0


def test_summarize_aggregates():
    rows = [
        run_eval.score_case(CASE, STATE, 1.0),
        run_eval.score_case(CASE, dict(STATE, answer="France won."), 3.0),
    ]
    s = run_eval.summarize(rows)
    assert s["cases"] == 2
    assert s["schema_ok_rate"] == 1.0
    assert s["mean_keyword_recall"] == 0.5
    assert s["second_round_rate"] == 1.0
    assert s["mean_latency_s"] == 2.0


def test_summarize_empty():
    assert run_eval.summarize([]) == {"cases": 0}


def test_judge_groundedness_parses_score():
    reply = json.dumps({"groundedness": 1.0, "reason": "supported"})
    with patch.object(run_eval.llm, "generate", return_value=reply):
        out = run_eval.judge_groundedness(CASE, STATE)
    assert out["groundedness"] == 1.0


def test_judge_groundedness_survives_garbage():
    with patch.object(run_eval.llm, "generate", return_value="not json"):
        out = run_eval.judge_groundedness(CASE, STATE)
    assert out["groundedness"] is None


def test_render_runs():
    rows = [run_eval.score_case(CASE, STATE, 1.0)]
    text = run_eval.render(rows, run_eval.summarize(rows))
    assert "wc" in text and "SUMMARY" in text


# --- abstention, cost and percentile scoring ------------------------------

ABSTAIN_CASE = {
    "id": "wc2027",
    "tier": "abstention",
    "question": "Who won the 2027 FIFA World Cup?",
    "must_include": [],
    "must_not_include": ["argentina", "france"],
}


def test_abstention_case_passes_when_the_agent_declines():
    state = dict(STATE, answer="The 2027 tournament has not been played yet.")
    row = run_eval.score_case(ABSTAIN_CASE, state, 0.1)
    assert row["abstained"] is True
    assert row["hallucinated_terms"] == []
    assert "PASS" in run_eval.render([row], run_eval.summarize([row]))


def test_abstention_case_fails_on_a_confident_fabrication():
    """A well-formed, well-cited answer to an unanswerable question is the
    worst output this agent can produce, so it must not score PASS."""
    state = dict(STATE, answer="Argentina won the 2027 FIFA World Cup.")
    row = run_eval.score_case(ABSTAIN_CASE, state, 0.1)
    assert row["abstained"] is False
    assert row["hallucinated_terms"] == ["argentina"]
    assert row["schema_ok"] and row["citations_all_retrieved"] and row["answered"]
    assert "FAIL" in run_eval.render([row], run_eval.summarize([row]))
    assert run_eval.summarize([row])["abstention_rate"] == 0.0


def test_abstention_rate_ignores_non_abstention_cases():
    rows = [
        run_eval.score_case(CASE, STATE, 1.0),
        run_eval.score_case(ABSTAIN_CASE, dict(STATE, answer="Not yet played."), 1.0),
    ]
    # Only the one case the metric applies to counts, not 0.5 across both.
    assert run_eval.summarize(rows)["abstention_rate"] == 1.0
    assert run_eval.score_case(CASE, STATE, 1.0)["abstained"] is None


def test_llm_calls_are_recorded():
    row = run_eval.score_case(CASE, STATE, 1.0, llm_calls=4)
    assert row["llm_calls"] == 4
    assert run_eval.summarize([row])["mean_llm_calls"] == 4


def test_p95_uses_the_upper_tail():
    """The old index formula returned p92 at n=25, understating the tail."""
    latencies = list(range(1, 26))
    rows = [run_eval.score_case(CASE, STATE, float(x)) for x in latencies]
    p95 = run_eval.summarize(rows)["p95_latency_s"]
    assert p95 > 23.0, f"p95 {p95} is below the 24th of 25 samples"


def test_p95_handles_a_single_case():
    rows = [run_eval.score_case(CASE, STATE, 2.0)]
    assert run_eval.summarize(rows)["p95_latency_s"] == 2.0


# --- retrieval vs generation ----------------------------------------------


def test_context_recall_separates_retrieval_from_generation():
    """The metric that makes a miss actionable: the context had the fact and
    the answer dropped it, so this is a generation bug, not a search one."""
    state = dict(STATE, answer="Argentina won the match.")
    row = run_eval.score_case(CASE, state, 0.1)

    assert row["context_recall"] == 1.0
    assert row["keyword_recall"] == 0.5
    assert row["dropped_by_generation"] == ["penalties"]


def test_context_recall_is_zero_when_retrieval_missed_it():
    state = dict(
        STATE,
        answer="I could not determine the result.",
        context=[{"title": "T", "url": "https://a.com", "text": "unrelated gardening text"}],
    )
    row = run_eval.score_case(CASE, state, 0.1)

    assert row["context_recall"] == 0.0
    assert row["dropped_by_generation"] == [], "nothing was dropped; it was never retrieved"


def test_chunk_utilization_counts_cited_sources():
    state = dict(
        STATE,
        context=STATE["context"] + [{"title": "B", "url": "https://b.com", "text": "x"}],
    )
    row = run_eval.score_case(CASE, state, 0.1)

    assert row["chunk_utilization"] == 0.5


def test_page_chunk_rate_tracks_snippet_fallback():
    state = dict(
        STATE,
        chunks=[
            {"url": "https://a.com", "text": "x", "source": "page"},
            {"url": "https://b.com", "text": "y", "source": "snippet"},
        ],
    )
    assert run_eval.score_case(CASE, state, 0.1)["page_chunk_rate"] == 0.5


def test_citations_are_validated_against_the_retrieved_context():
    state = dict(STATE, citations=[{"id": 1, "title": "X", "url": "https://invented.com"}])
    assert run_eval.score_case(CASE, state, 0.1)["citations_all_retrieved"] is False
