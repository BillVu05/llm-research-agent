import json
import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from agent import cli

TOPIC = "Who won the 2022 FIFA World Cup?"

PLAN = json.dumps({"queries": ["2022 World Cup winner"], "slots": ["winner", "score"]})
ANSWER = json.dumps({"answer": "Argentina won the 2022 FIFA World Cup.", "citations": [1]})


def result(title="Argentina wins", link="https://example.com/a", snippet="Argentina beat France."):
    return {"title": title, "link": link, "snippet": snippet}


@pytest.fixture
def calls():
    """Record which nodes ran, so a silently-skipped node fails the test."""
    seen = []
    originals = {
        name: getattr(cli, name)
        for name in ("generate_queries", "web_search", "reflect", "synthesize")
    }

    def wrap(name, fn):
        def inner(state):
            seen.append(name)
            return fn(state)

        return inner

    with patch.multiple(
        cli, **{name: wrap(name, fn) for name, fn in originals.items()}
    ):
        yield seen


def run(calls, llm_replies, search_batches):
    """Drive the compiled graph with scripted LLM and search responses."""
    with patch.object(cli, "_generate", side_effect=llm_replies), patch.object(
        cli, "GoogleSearch"
    ) as search:
        search.return_value.get_dict.side_effect = [
            {"organic_results": batch} for batch in search_batches
        ]
        return cli.build_pipeline().invoke({"topic": TOPIC, "debug": False}), calls


def test_happy_path(calls):
    reflect_ok = json.dumps({"filled": ["winner", "score"], "explanations": {}})
    state, ran = run(calls, [PLAN, reflect_ok, ANSWER], [[result()]])

    assert "Argentina" in state["answer"]
    assert ran == ["generate_queries", "web_search", "reflect", "synthesize"]
    assert state["citations"][0]["url"] == "https://example.com/a"


def test_reflect_actually_runs(calls):
    """Regression guard: reflect was previously unreachable, so the graph
    completed without ever evaluating slot coverage."""
    reflect_ok = json.dumps({"filled": ["winner", "score"], "explanations": {}})
    state, ran = run(calls, [PLAN, reflect_ok, ANSWER], [[result()]])

    assert "reflect" in ran
    assert state["rounds"] == 1
    assert state["filled"] == ["winner", "score"]


def test_two_round_supplement(calls):
    """A coverage gap must route back to web_search and widen the doc set."""
    gap = json.dumps({"filled": ["winner"], "explanations": {}})
    covered = json.dumps({"filled": ["winner", "score"], "explanations": {}})
    state, ran = run(
        calls,
        [PLAN, gap, covered, ANSWER],
        [[result()], [result("Final score", "https://example.com/b", "3-3 on the day.")]],
    )

    assert ran == [
        "generate_queries",
        "web_search",
        "reflect",
        "web_search",
        "reflect",
        "synthesize",
    ]
    assert state["rounds"] == 2
    assert len(state["docs"]) == 2


def test_loop_is_bounded(calls):
    """Persistent gaps must stop at MAX_SEARCH_ROUNDS, not spin."""
    gap = json.dumps({"filled": [], "explanations": {}})
    state, ran = run(
        calls,
        [PLAN, gap, gap, ANSWER],
        [[result()], [result("Other", "https://example.com/b")]],
    )

    assert ran.count("web_search") == cli.MAX_SEARCH_ROUNDS
    assert state["rounds"] == cli.MAX_SEARCH_ROUNDS
    assert ran[-1] == "synthesize"


def test_no_results(calls):
    state, ran = run(calls, [PLAN, ANSWER], [[]])

    assert state["answer"].startswith("No relevant")
    assert state["citations"] == []
    assert "reflect" in ran


def test_search_error_degrades(calls):
    """A 429 or any SerpAPI failure yields no docs, not an exception."""
    with patch.object(cli, "_generate", side_effect=[PLAN, ANSWER]), patch.object(
        cli, "GoogleSearch", side_effect=Exception("429 Too Many Requests")
    ):
        state = cli.build_pipeline().invoke({"topic": TOPIC, "debug": False})

    assert state["answer"].startswith("No relevant")
    assert state["citations"] == []


def test_timeout_degrades(calls):
    with patch.object(cli, "_generate", side_effect=[PLAN, ANSWER]), patch.object(
        cli, "GoogleSearch", side_effect=TimeoutError("timed out")
    ):
        state = cli.build_pipeline().invoke({"topic": TOPIC, "debug": False})

    assert state["answer"].startswith("No relevant")


def test_malformed_llm_json_does_not_crash(calls):
    # No slots could be planned, so reflect skips its LLM call entirely and
    # only two generations happen.
    state, ran = run(calls, ["not json at all", ANSWER], [[result()]])

    # Falls back to searching the raw question.
    assert state["queries"] == [TOPIC]
    assert state["slots"] == []
    assert "Argentina" in state["answer"]
    assert ran == ["generate_queries", "web_search", "reflect", "synthesize"]


def test_fenced_json_is_parsed(calls):
    fenced = f"```json\n{PLAN}\n```"
    reflect_ok = json.dumps({"filled": ["winner", "score"], "explanations": {}})
    state, _ = run(calls, [fenced, reflect_ok, ANSWER], [[result()]])

    assert state["slots"] == ["winner", "score"]


def test_invented_citations_are_dropped(calls):
    """The model citing a source that was never retrieved must not surface."""
    reflect_ok = json.dumps({"filled": ["winner", "score"], "explanations": {}})
    hallucinated = json.dumps({"answer": "Argentina won.", "citations": [1, 7, 99]})
    state, _ = run(calls, [PLAN, reflect_ok, hallucinated], [[result()]])

    assert [c["id"] for c in state["citations"]] == [1]
    assert all(c["url"] == "https://example.com/a" for c in state["citations"])


def test_snippets_reach_the_model(calls):
    """Grounding check: retrieved snippet text must appear in the prompt."""
    reflect_ok = json.dumps({"filled": ["winner", "score"], "explanations": {}})
    prompts = []

    def record(prompt):
        prompts.append(prompt)
        return [PLAN, reflect_ok, ANSWER][len(prompts) - 1]

    with patch.object(cli, "_generate", side_effect=record), patch.object(
        cli, "GoogleSearch"
    ) as search:
        search.return_value.get_dict.return_value = {
            "organic_results": [result(snippet="Argentina beat France on penalties.")]
        }
        cli.build_pipeline().invoke({"topic": TOPIC, "debug": False})

    assert "Argentina beat France on penalties." in prompts[-1]


def test_urls_are_deduplicated(calls):
    reflect_ok = json.dumps({"filled": ["winner", "score"], "explanations": {}})
    dupes = [result(), result(title="Same page, different title")]
    state, _ = run(calls, [PLAN, reflect_ok, ANSWER], [dupes])

    assert len(state["docs"]) == 1
