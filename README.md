# LLM Research Agent

[![CI](https://github.com/BillVu05/llm-research-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/BillVu05/llm-research-agent/actions/workflows/ci.yml)

A command-line research agent. Ask a question in plain English; get back a short,
cited answer as strict JSON.

It is a [LangGraph](https://langchain-ai.github.io/langgraph/) `StateGraph` of four
nodes. The interesting part is the **conditional edge**: after searching, the agent
judges whether the documents it found actually cover every fact the question needs.
If they don't, it writes targeted follow-up queries and searches again.

```bash
$ python llm-research-agent/src/agent/cli.py "Who won the 2019 Cricket World Cup final, how was the tie resolved, and where was it played?"
{
  "answer": "England won the 2019 Cricket World Cup final [1][2]. After both the match and the subsequent Super Over ended in ties [2], the tie was resolved using the boundary count-back rule, which England won by scoring 26 boundaries to New Zealand's 17 [4]. The match was played at Lord's in London [1][6].",
  "citations": [
    { "id": 1, "title": "ENG vs NZ Cricket Scorecard, Final at London, July 14, 2019", "url": "https://www.cricinfo.com/..." },
    { "id": 2, "title": "2019 Cricket World Cup", "url": "https://en.wikipedia.org/wiki/2019_Cricket_World_Cup" }
  ]
}
```

## How it works

```mermaid
graph TD;
    start([start]) --> generate_queries
    generate_queries --> web_search
    web_search --> reflect
    reflect -. gap, rounds left .-> web_search
    reflect -. covered or budget spent .-> synthesize
    synthesize --> done([end])
```

| Node | Does |
| --- | --- |
| `generate_queries` | One Gemini call plans both the search queries **and the "slots"** — the specific facts a complete answer must contain for *this* question. |
| `web_search` | Runs the queries concurrently through SerpAPI, keeping snippets and appending only URLs not already seen. |
| `reflect` | Gemini judges which slots the retrieved documents actually support, and writes one targeted follow-up query per missing slot. |
| `synthesize` | Writes an ≤80-word answer citing sources by number. |

Three design decisions worth calling out:

- **Slots are derived per question**, not looked up in a table, so coverage checking
  works on any topic at no extra cost — the planning call returns them anyway.
- **Citations are rebuilt from the retrieved documents** using the indices the model
  returns. A cited URL is therefore provably one the agent fetched; an invented
  source is dropped rather than surfaced.
- **Synthesis reads search snippets**, not titles alone, so the answer is grounded in
  retrieved text rather than in the model's own recall.

The CLI always emits valid JSON. Search failures, throttling, and unparseable model
output all degrade to a well-formed result flagged `degraded`, never a traceback.

## Quickstart

```bash
git clone https://github.com/BillVu05/llm-research-agent.git
cd llm-research-agent
pip install -r requirements.txt

cp llm-research-agent/.env.example .env    # then add your keys
python llm-research-agent/src/agent/cli.py --topic "your question" --debug
```

Keys come from [SerpAPI](https://serpapi.com/manage-api-key) and
[Google AI Studio](https://aistudio.google.com/app/apikey).

> `load_dotenv()` does not override variables already set in your shell. If a stale
> `GEMINI_API_KEY` is exported in your environment, it wins over `.env`.

### Docker

```bash
cd llm-research-agent
docker compose run --rm agent "your question"
```

## Testing

```bash
pytest        # 25 tests, no API keys required, < 1s
ruff check .
```

The tests assert **which graph nodes ran**, not only the final output. An earlier
version of this pipeline never executed `reflect` at all and its test suite still
passed green, because nothing checked. That regression now has a named test.

## Evaluation

Unit tests prove the graph is wired correctly; they say nothing about answer quality.
`eval/` scores the agent against a 25-question golden set, tiered `parametric`
(the model probably knows it), `retrieval`, and `multi_fact`.

```bash
python llm-research-agent/eval/run_eval.py --limit 6 --sleep 25
python llm-research-agent/eval/run_eval.py --judge --out results.json
```

| Metric | Catches |
| --- | --- |
| `answered_rate` | Runs that degraded to a fallback answer |
| `keyword_recall` | Answers missing the expected facts |
| `citation_validity_rate` | A cited URL that was never retrieved |
| `slot_fill_rate` | How much planned coverage was achieved |
| `second_round_rate` | How often the reflection loop actually fires |
| `word_budget_rate` | Answers over the length budget |
| `mean` / `p95 latency` | Speed regressions |
| `mean_groundedness` | `--judge` only: claims unsupported by sources |

Baseline, first 6 cases, `models/gemini-2.5-flash`:

```
cases 6 | answered 1.0 | keyword_recall 1.0 | citation_validity 1.0
slot_fill 1.0 | citations/answer 5.2 | docs/question 15.8 | p95 24.9s
```

**An honest note on `second_round_rate`:** it is currently 0. Query planning is good
enough that one round of ~15 documents usually fills every slot, so the loop rarely
triggers in practice. Its behaviour is pinned by unit tests
(`test_two_round_supplement`, `test_loop_is_bounded`) rather than by these live runs.
It is insurance for hard questions, not a hot path.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `GEMINI_API_KEY` | — | required |
| `SERPAPI_API_KEY` | — | required |
| `GEMINI_MODEL` | `models/gemini-flash-latest` | pin a model for reproducible runs |

A pinned model name is a time bomb — `gemini-1.5-flash` was retired and every live
run started returning 404. The default is a rolling alias for that reason.

The Gemini free tier is rate limited (as low as 5 requests/minute) and one eval case
costs 3–4 calls, so pace long runs with `--sleep`.

## Layout

```
requirements.txt              single dependency source, used by Docker too
pyproject.toml                ruff + pytest config
docs/design.md                design document
llm-research-agent/
  src/agent/cli.py            the whole agent (~330 lines)
  test/                       25 tests, all external calls mocked
  eval/golden.jsonl           25-question golden set
  eval/run_eval.py            scoring harness
  Dockerfile, docker-compose.yml
```

## License

MIT — see [LICENSE](LICENSE).
