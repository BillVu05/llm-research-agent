"""Shared offline doubles.

The pipeline now embeds text, so every test needs an embedder. ``fake_embed``
is a bag-of-words hash: deterministic, needs no API key, and - unlike random
vectors - has real cosine semantics, so chunks that share words really do score
as similar. That is what lets the retrieval tests assert on ranking rather than
just on plumbing.
"""

from __future__ import annotations

import re
import zlib

DIMS = 64


def fake_embed(texts, task_type="retrieval_document"):
    vectors = []
    for text in texts:
        vec = [0.0] * DIMS
        for word in re.findall(r"\w+", text.lower()):
            # crc32, not hash(): str hashing is salted per process, so a
            # collision could reorder results only on some runs.
            vec[zlib.crc32(word.encode()) % DIMS] += 1.0
        # An all-zero vector would make cosine undefined; nudge it.
        if not any(vec):
            vec[0] = 1.0
        vectors.append(vec)
    return vectors


def no_pages(url):
    """Every fetch fails, so ingest falls back to snippets."""
    return ""
