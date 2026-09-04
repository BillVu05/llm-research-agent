"""The single seam for every model call: generation, embedding, JSON parsing.

Tests patch ``generate`` and ``embed`` here, so every caller must reach them as
``llm.generate(...)`` rather than importing the function by name.
"""

from __future__ import annotations

import json
import os
import sys
import time
from functools import lru_cache
from typing import Any

from . import config

# Billable API calls made so far. Latency alone does not tell you what a
# question costs; this does, and the eval harness reports it per case.
LLM_CALLS = 0

_TRANSIENT = ("429", "rate limit", "quota", "timeout", "deadline", "unavailable", "503", "500")


@lru_cache(maxsize=1)
def _genai():
    # Imported lazily so that `import agent` needs neither the SDK nor an API
    # key - keeps the test suite fast and key-free.
    import google.generativeai as genai

    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    return genai


@lru_cache(maxsize=1)
def _model():
    return _genai().GenerativeModel(config.MODEL_NAME)


def _is_transient(exc: Exception) -> bool:
    return any(t in str(exc).lower() for t in _TRANSIENT)


def _backoff(attempt: int, exc: Exception, what: str) -> None:
    delay = config.RETRY_BASE_S * 2 ** (attempt - 1)
    print(
        f"[{what}] transient failure ({exc}); retry {attempt}/{config.LLM_ATTEMPTS - 1} "
        f"in {delay:.0f}s",
        file=sys.stderr,
    )
    time.sleep(delay)


def generate(prompt: str) -> str:
    """Run one generation.

    Transient failures (throttling, timeouts, 5xx) are retried with exponential
    backoff. Anything else - notably a safety block, which is deterministic and
    will fail identically on every attempt - gives up immediately. Either way a
    failure returns "" so callers fall back to their defaults and the CLI still
    emits valid JSON rather than dying with a traceback.
    """
    global LLM_CALLS
    for attempt in range(1, config.LLM_ATTEMPTS + 1):
        try:
            # Counted before the call, so retries show up as the extra spend
            # they are. ponytail: a plain int, not thread-safe - nothing calls
            # the LLM from the search thread pool. Revisit if that changes.
            LLM_CALLS += 1
            response = _model().generate_content(
                prompt, request_options={"timeout": config.LLM_TIMEOUT_S}
            )
            return response.text.strip()
        except ValueError as exc:
            # .text raises when the candidate was blocked or came back empty.
            # Retrying cannot help, and it is worth naming separately: it is a
            # content problem, not a network one.
            print(f"[llm] no usable candidate (safety block or empty): {exc}", file=sys.stderr)
            return ""
        except Exception as exc:
            if not _is_transient(exc) or attempt == config.LLM_ATTEMPTS:
                print(f"[llm] generation failed: {exc}", file=sys.stderr)
                return ""
            _backoff(attempt, exc, "llm")
    return ""


def embed(texts: list[str], task_type: str) -> list[list[float]]:
    """Embed a batch of texts, or return [] if the API will not cooperate.

    ``task_type`` is "retrieval_document" for corpus text and "retrieval_query"
    for the question. They are not interchangeable: the model places queries and
    documents in a comparable space precisely *because* it is told which is
    which, and using one type for both is the classic way to get mediocre
    retrieval that still looks like it is working.

    Returning [] rather than raising lets retrieval fall back to search-rank
    order, so a bad embedding day degrades the answer instead of losing it.
    """
    global LLM_CALLS
    if not texts:
        return []

    out: list[list[float]] = []
    # The API caps a batch, so slice rather than hoping every corpus is small.
    for start in range(0, len(texts), config.EMBED_BATCH):
        batch = texts[start : start + config.EMBED_BATCH]
        for attempt in range(1, config.LLM_ATTEMPTS + 1):
            try:
                LLM_CALLS += 1
                result = _genai().embed_content(
                    model=config.EMBED_MODEL,
                    content=batch,
                    task_type=task_type,
                    output_dimensionality=config.EMBED_DIMS,
                    request_options={"timeout": config.LLM_TIMEOUT_S},
                )
                vectors = result["embedding"]
                # A one-item batch can come back unwrapped as a flat vector.
                if vectors and isinstance(vectors[0], (int, float)):
                    vectors = [vectors]
                out.extend(vectors)
                break
            except Exception as exc:
                if not _is_transient(exc) or attempt == config.LLM_ATTEMPTS:
                    print(f"[embed] failed: {exc}", file=sys.stderr)
                    return []
                _backoff(attempt, exc, "embed")
    return out


def parse_json(raw: str, default: Any) -> Any:
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
