import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../eval")))

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
        assert c["must_include"], f"{c['id']} has no checkable keywords"
        assert c["tier"] in {"parametric", "retrieval", "multi_fact"}
    # multi_fact cases exist to exercise the reflect -> web_search loop, which
    # the easier tiers never trigger.
    assert sum(c["tier"] == "multi_fact" for c in cases) >= 5


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
    with patch.object(run_eval.cli, "_generate", return_value=reply):
        out = run_eval.judge_groundedness(CASE, STATE)
    assert out["groundedness"] == 1.0


def test_judge_groundedness_survives_garbage():
    with patch.object(run_eval.cli, "_generate", return_value="not json"):
        out = run_eval.judge_groundedness(CASE, STATE)
    assert out["groundedness"] is None


def test_render_runs():
    rows = [run_eval.score_case(CASE, STATE, 1.0)]
    text = run_eval.render(rows, run_eval.summarize(rows))
    assert "wc" in text and "SUMMARY" in text
