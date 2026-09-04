"""Every tunable in one place: env vars and the constants that shape the run.

Read once at import. Other modules reference them as ``config.NAME`` rather
than importing the values, so a test can patch one knob without reloading the
module that uses it.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# --- models ---------------------------------------------------------------
# A pinned model name is a time bomb: gemini-1.5-flash was retired and every
# live run 404'd. The rolling alias self-heals; override to pin deliberately.
MODEL_NAME = os.getenv("GEMINI_MODEL", "models/gemini-flash-latest")
# Same time bomb as MODEL_NAME, and it went off during development:
# text-embedding-004 now 404s. There is no rolling "-latest" alias for
# embeddings, so this is pinned to the current GA model and overridable.
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "models/gemini-embedding-001")
# gemini-embedding-001 defaults to 3072 dims. 768 is the documented
# quality/size trade-off and keeps the batch payloads small; the vectors are
# normalized either way, which reduced dimensions require.
EMBED_DIMS = 768

# --- loop control ---------------------------------------------------------
MAX_SEARCH_ROUNDS = 2
RESULTS_PER_QUERY = 10
SEARCH_WORKERS = 5

# --- search ---------------------------------------------------------------
SEARCH_LANG = os.getenv("SEARCH_LANG", "en")
SEARCH_COUNTRY = os.getenv("SEARCH_COUNTRY", "us")
# Seconds. SerpAPI's own default is 60000, handed straight to requests as
# seconds - i.e. 16 hours, i.e. no timeout at all. A hung socket would hang the
# CLI forever and block the thread pool from shutting down.
SEARCH_TIMEOUT_S = 20

# --- ingest ---------------------------------------------------------------
# Pages to actually fetch per round. Fetching is the slow step; the tail of a
# 10-result page is rarely worth the latency.
MAX_DOCS_TO_FETCH = 6
FETCH_TIMEOUT_S = 10
# A 40MB page must not become a 40MB string.
MAX_PAGE_BYTES = 512 * 1024
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# --- retrieval ------------------------------------------------------------
# Embedding is charged per chunk, and a long Wikipedia page alone is ~100 of
# them. A lexical prefilter cuts the corpus to this before anything is embedded.
MAX_CHUNKS_TO_EMBED = 60
TOP_K_CHUNKS = 12
# Candidates that MMR diversifies down to TOP_K_CHUNKS.
MMR_POOL = 30
# 1.0 = pure relevance, 0.0 = pure diversity.
MMR_LAMBDA = 0.7
MAX_CONTEXT_DOCS = 8
EMBED_BATCH = 100

# --- LLM resilience -------------------------------------------------------
LLM_TIMEOUT_S = 60
# The free tier allows as few as 5 requests/minute and one question costs 3-4,
# so a 429 mid-run is routine, not exceptional. Without a retry a single one
# silently degrades the answer and the eval harness books it as a quality
# regression rather than the infrastructure blip it is.
LLM_ATTEMPTS = 3
RETRY_BASE_S = 2.0

# --- untrusted input ------------------------------------------------------
# Retrieved text comes off the open web and is pasted into a prompt, so it is
# hostile input: a page that ranks for one of our queries gets to put words in
# front of the model. Cheap defences are a hard length cap, a delimiter whose
# edges the model can see, and one instruction saying the enclosed text is data.
MAX_SNIPPET_CHARS = 400
# A context doc is several retrieved chunks concatenated. The real bound is
# structural - at most TOP_K_CHUNKS chunks are ever picked and each is at most
# CHUNK_SIZE + CHUNK_OVERLAP long - so deriving the cap from those keeps a
# single-source answer from being truncated by an arbitrary number, while
# still bounding what any one page can put in the prompt.
MAX_CONTEXT_CHARS = TOP_K_CHUNKS * (CHUNK_SIZE + CHUNK_OVERLAP)
