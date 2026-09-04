"""Ranking, diversity, and the fallback when embedding is unavailable."""

from unittest.mock import patch

import numpy as np
from conftest import fake_embed

from agent import config, llm, retrieval


def chunk(text, url="https://a.com", title="T"):
    return {"title": title, "url": url, "text": text, "source": "page"}


def test_relevant_chunk_outranks_irrelevant_one():
    chunks = [
        chunk("A guide to growing tomatoes in a greenhouse.", "https://garden.com"),
        chunk("Argentina won the 2022 World Cup final on penalties.", "https://sport.com"),
    ]
    with patch.object(llm, "embed", side_effect=fake_embed):
        out = retrieval.retrieve({"topic": "Who won the 2022 World Cup?", "chunks": chunks})

    assert out["context"][0]["url"] == "https://sport.com"


def test_mmr_drops_a_near_duplicate():
    """Five outlets reporting the same sentence must not eat the whole window;
    plain top-k would keep them all."""
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    # Two chunks that restate each other (cosine 1.0 to each other, 0.9 to the
    # query) and one that is less relevant but says something new.
    duplicate = [0.9, 0.436, 0.0]
    diverse = [0.7, 0.0, 0.714]
    vecs = retrieval._normalize([duplicate, duplicate, diverse])

    picked = retrieval.mmr(query, vecs, k=2, lambda_=0.5)

    assert 2 in picked, "the diverse chunk lost to a duplicate"
    assert retrieval.mmr(query, vecs, k=2, lambda_=1.0) == [0, 1], (
        "pure relevance should keep both duplicates - that is what MMR fixes"
    )


def test_mmr_is_pure_relevance_at_lambda_one():
    query = np.array([1.0, 0.0], dtype=np.float32)
    vecs = retrieval._normalize([[0.1, 1.0], [1.0, 0.0], [0.5, 0.5]])

    assert retrieval.mmr(query, vecs, k=3, lambda_=1.0) == [1, 2, 0]


def test_mmr_never_returns_more_than_k():
    query = np.array([1.0, 0.0], dtype=np.float32)
    vecs = retrieval._normalize([[1.0, 0.0]] * 10)

    assert len(retrieval.mmr(query, vecs, k=3)) == 3


def test_zero_vector_does_not_blow_up():
    """An empty or stopword-only chunk embeds to ~0; dividing by its norm would
    poison every later comparison with NaN."""
    normed = retrieval._normalize([[0.0, 0.0], [3.0, 4.0]])

    assert not np.isnan(normed).any()
    assert np.allclose(np.linalg.norm(normed[1]), 1.0)


def test_chunks_from_one_page_become_one_source():
    chunks = [chunk(f"Part {i} of the final report.") for i in range(4)]
    with patch.object(llm, "embed", side_effect=fake_embed):
        out = retrieval.retrieve({"topic": "the final", "chunks": chunks})

    assert len(out["context"]) == 1
    assert out["context"][0]["text"].count("Part") == 4


def test_embedding_failure_falls_back_to_search_rank():
    """A bad embedding day must degrade the answer, not lose it."""
    docs = [
        {"title": "A", "url": "https://a.com", "snippet": "first"},
        {"title": "B", "url": "https://b.com", "snippet": "second"},
    ]
    with patch.object(llm, "embed", return_value=[]):
        out = retrieval.retrieve({"topic": "q", "chunks": [chunk("x")], "docs": docs})

    assert [d["url"] for d in out["context"]] == ["https://a.com", "https://b.com"]
    assert out["context"][0]["text"] == "first"


def test_partial_embedding_response_falls_back():
    """A short vector list would silently misalign chunks with their vectors."""
    chunks = [chunk("a"), chunk("b", "https://b.com")]
    docs = [{"title": "A", "url": "https://a.com", "snippet": "first"}]
    with patch.object(llm, "embed", side_effect=[[[1.0, 0.0]], [[1.0, 0.0]]]):
        out = retrieval.retrieve({"topic": "q", "chunks": chunks, "docs": docs})

    assert out["context"] == [{"title": "A", "url": "https://a.com", "text": "first"}]


def test_no_chunks_falls_back_to_snippets():
    docs = [{"title": "A", "url": "https://a.com", "snippet": "first"}]
    out = retrieval.retrieve({"topic": "q", "chunks": [], "docs": docs})

    assert out["context"][0]["text"] == "first"


def test_context_is_capped():
    chunks = [chunk(f"topic text {i}", f"https://s{i}.com") for i in range(30)]
    with patch.object(llm, "embed", side_effect=fake_embed):
        out = retrieval.retrieve({"topic": "topic text", "chunks": chunks})

    assert len(out["context"]) <= config.MAX_CONTEXT_DOCS


def test_query_and_documents_are_embedded_with_different_task_types():
    """Symmetric embedding is the classic RAG bug: it looks like it works."""
    seen = []

    def spy(texts, task_type):
        seen.append(task_type)
        return fake_embed(texts)

    with patch.object(llm, "embed", side_effect=spy):
        retrieval.retrieve({"topic": "q", "chunks": [chunk("a")]})

    assert seen == ["retrieval_query", "retrieval_document"]


def test_prefilter_keeps_the_chunks_that_share_words_with_the_question():
    """The cost guard must not be the thing that loses the answer."""
    chunks = [chunk(f"gardening tomatoes greenhouse compost {i}") for i in range(50)]
    chunks.append(chunk("Argentina won the World Cup final", "https://sport.com"))

    kept = retrieval._prefilter("Who won the World Cup final?", chunks, limit=5)

    assert kept[0]["url"] == "https://sport.com"
    assert len(kept) == 5


def test_prefilter_is_a_no_op_below_the_limit():
    chunks = [chunk("a"), chunk("b")]
    assert retrieval._prefilter("q", chunks, limit=10) == chunks


def test_retrieve_never_embeds_more_than_the_budget():
    """Embedding is charged per chunk; a long page must not blow the quota."""
    chunks = [chunk(f"text {i}", f"https://s{i}.com") for i in range(500)]
    sizes = []

    def spy(texts, task_type):
        sizes.append(len(texts))
        return fake_embed(texts)

    with patch.object(llm, "embed", side_effect=spy):
        retrieval.retrieve({"topic": "text", "chunks": chunks})

    assert sizes == [1, config.MAX_CHUNKS_TO_EMBED]
