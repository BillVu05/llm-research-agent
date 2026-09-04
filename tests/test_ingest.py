"""Fetching, extraction and chunking - the ingest half of RAG."""

from unittest.mock import MagicMock, patch

from agent import config, ingest

HTML = """
<html><head><title>T</title><style>body { color: red; }</style></head>
<body>
  <nav>Home About Contact</nav>
  <h1>The 2022 Final</h1>
  <p>Argentina beat France on penalties &amp; lifted the trophy.</p>
  <script>var tracking = 1;</script>
  <p>The score was 3&ndash;3 after extra time.</p>
  <footer>Copyright 2022</footer>
</body></html>
"""


def test_extract_drops_chrome_and_keeps_content():
    text = ingest.extract(HTML)

    assert "Argentina beat France on penalties & lifted the trophy." in text
    # The en dash is the point: &ndash; must be decoded, not left as an entity.
    assert "3–3 after extra time" in text  # noqa: RUF001
    for chrome in ("var tracking", "color: red", "Home About Contact", "Copyright 2022"):
        assert chrome not in text


def test_extract_collapses_whitespace_but_keeps_paragraphs():
    text = ingest.extract("<p>a     b</p><p>c</p>")

    assert "  " not in text
    assert text.count("\n\n") == 1


def test_extract_survives_malformed_markup():
    assert "hello" in ingest.extract("<p>hello<div><span>")


def test_chunk_is_empty_for_empty_text():
    assert ingest.chunk("") == []
    assert ingest.chunk("   \n\n  ") == []


def test_chunks_respect_the_size_budget():
    text = "\n\n".join(f"Paragraph number {i}. " * 10 for i in range(30))
    chunks = ingest.chunk(text, size=400, overlap=50)

    assert len(chunks) > 1
    # The budget is per assembled chunk; overlap is carried on top of it.
    assert all(len(c) <= 400 + 50 + 2 for c in chunks)


def test_chunks_overlap_so_a_boundary_fact_survives():
    text = "\n\n".join(f"sentence {i} here" for i in range(60))
    chunks = ingest.chunk(text, size=200, overlap=60)

    tails = [c[-40:] for c in chunks[:-1]]
    assert any(tail.split()[-1] in nxt for tail, nxt in zip(tails, chunks[1:], strict=False)), (
        "consecutive chunks share no text, so a fact on the seam is lost"
    )


def test_a_single_huge_paragraph_still_splits():
    """No paragraph break to cut on, so it is cut on length instead."""
    chunks = ingest.chunk("x" * 5000, size=800, overlap=100)

    assert len(chunks) > 1
    # The bound is size + overlap: overlap is carried on top of the budget.
    assert all(len(c) <= 800 + 100 + 2 for c in chunks)


def _response(content_type="text/html; charset=utf-8", body=b"<p>hello</p>", status_ok=True):
    response = MagicMock()
    response.__enter__.return_value = response
    response.headers = {"Content-Type": content_type}
    response.encoding = "utf-8"
    response.raw.read.return_value = body
    if not status_ok:
        response.raise_for_status.side_effect = Exception("404")
    return response


def test_fetch_skips_non_html():
    """A PDF or a JSON API is not worth an HTML parser."""
    with patch.object(ingest.requests, "get", return_value=_response("application/pdf")):
        assert ingest.fetch_text("https://a.com/x.pdf") == ""


def test_fetch_caps_the_read():
    """Content-Length is a claim, not a guarantee: the read itself is bounded."""
    with patch.object(ingest.requests, "get", return_value=_response()) as get:
        ingest.fetch_text("https://a.com")

    assert get.return_value.raw.read.call_args[0][0] == config.MAX_PAGE_BYTES


def test_fetch_failure_returns_empty():
    with patch.object(ingest.requests, "get", side_effect=Exception("connection reset")):
        assert ingest.fetch_text("https://a.com") == ""


DOC = {"title": "T", "url": "https://a.com", "snippet": "Argentina won on penalties."}


def test_ingest_falls_back_to_the_snippet_when_a_page_cannot_be_fetched():
    """The degradation floor: a run where every fetch fails must land on
    exactly the old snippet-only behaviour, not on an empty context."""
    with patch.object(ingest, "fetch_text", return_value=""):
        out = ingest.ingest({"topic": "q", "docs": [DOC]})

    assert [c["text"] for c in out["chunks"]] == ["Argentina won on penalties."]
    assert out["chunks"][0]["source"] == "snippet"


def test_ingest_prefers_page_text():
    with patch.object(ingest, "fetch_text", return_value="Full page text about the final."):
        out = ingest.ingest({"topic": "q", "docs": [DOC]})

    assert out["chunks"][0]["text"] == "Full page text about the final."
    assert out["chunks"][0]["source"] == "page"


def test_ingest_does_not_refetch_on_a_second_round():
    """Round two must pay for its new URLs only."""
    second = {"title": "B", "url": "https://b.com", "snippet": "s"}
    with patch.object(ingest, "fetch_text", return_value="text") as fetch:
        first = ingest.ingest({"topic": "q", "docs": [DOC]})
        ingest.ingest({"topic": "q", "docs": [DOC, second], **first})

    assert [c.args[0] for c in fetch.call_args_list] == ["https://a.com", "https://b.com"]


def test_ingest_bounds_how_many_pages_it_fetches():
    """Fetching is the slow step; the tail of a results page is not worth it."""
    docs = [{"title": f"t{i}", "url": f"https://a.com/{i}", "snippet": "s"} for i in range(20)]
    with patch.object(ingest, "fetch_text", return_value="text") as fetch:
        ingest.ingest({"topic": "q", "docs": docs})

    assert fetch.call_count == config.MAX_DOCS_TO_FETCH
