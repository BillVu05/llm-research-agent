"""Turn search hits into chunks of real page text.

This is the half of RAG the agent used to skip: it answered from Google's
~400-character snippet instead of from the page. Fetching is best-effort - a
paywall, a PDF, a 403 or a timeout all fall back to the snippet, so the worst
case is exactly the old behaviour rather than an empty context.
"""

from __future__ import annotations

import re
import sys
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from typing import Any

import requests

from . import config
from .state import ResearchState

# Sending python-requests/2.x gets you a 403 from a good fraction of the web.
_UA = "Mozilla/5.0 (compatible; llm-research-agent/0.2; +https://github.com/BillVu05/llm-research-agent)"

# Tags whose text is chrome, not content. Dropping them is most of what a real
# extractor does.
_SKIP_TAGS = {"script", "style", "nav", "footer", "header", "aside", "form", "noscript"}
_BLOCK_TAGS = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section"}


class _TextExtractor(HTMLParser):
    """Strip tags, keep text, insert breaks at block boundaries.

    ponytail: stdlib HTMLParser, so site boilerplate (cookie banners, menu
    text) survives on some pages. Swap in trafilatura if extraction noise
    starts showing up in the eval's context_recall.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth and data.strip():
            self.parts.append(data)


def extract(html: str) -> str:
    """HTML in, readable plain text out."""
    parser = _TextExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception as exc:  # malformed markup should cost one page, not the run
        print(f"[ingest] parse failed: {exc}", file=sys.stderr)
    text = "".join(parser.parts)
    # Collapse runs of spaces, then runs of blank lines, keeping paragraphs.
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n\s*", "\n\n", text)
    return text.strip()


def fetch_text(url: str) -> str:
    """Download one page and return its text, or "" for anything unusable."""
    try:
        with requests.get(
            url,
            timeout=config.FETCH_TIMEOUT_S,
            headers={"User-Agent": _UA, "Accept": "text/html,application/xhtml+xml"},
            stream=True,
        ) as response:
            response.raise_for_status()
            # PDFs, images and JSON APIs are not worth an HTML parser.
            if "text/html" not in response.headers.get("Content-Type", ""):
                return ""
            # Bounded read: a 40MB page must not become a 40MB string, and
            # Content-Length is a claim, not a guarantee.
            raw = response.raw.read(config.MAX_PAGE_BYTES, decode_content=True)
        html = raw.decode(response.encoding or "utf-8", errors="replace")
        return extract(html)
    except Exception as exc:  # DNS, TLS, 403, timeout - all mean "use the snippet"
        print(f"[ingest] fetch {url!r} failed: {exc}", file=sys.stderr)
        return ""


def chunk(text: str, size: int | None = None, overlap: int | None = None) -> list[str]:
    """Split text into overlapping chunks, preferring paragraph boundaries.

    The overlap is what stops a fact that straddles a boundary from being cut in
    half and retrieved by neither chunk.
    """
    size = size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP
    text = text.strip()
    if not text:
        return []

    # Paragraphs first; any single paragraph longer than `size` is then cut on
    # length, since there is no smaller natural boundary to use.
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        while len(para) > size:
            units.append(para[:size])
            para = para[size - overlap :]
        if para:
            units.append(para)

    chunks: list[str] = []
    current = ""
    for unit in units:
        if current and len(current) + len(unit) + 2 > size:
            chunks.append(current)
            # Carry the tail of the finished chunk into the next one.
            current = (current[-overlap:] + "\n\n" + unit) if overlap else unit
        else:
            current = f"{current}\n\n{unit}" if current else unit
    if current:
        chunks.append(current)
    return chunks


def ingest(state: ResearchState) -> dict[str, Any]:
    """Fetch and chunk every doc not already ingested."""
    docs = state.get("docs", [])
    chunks = list(state.get("chunks", []))
    ingested = set(state.get("ingested", []))

    # Only new URLs, and only the top few: fetching is the slow step, and the
    # tail of a results page rarely justifies the latency.
    pending = [d for d in docs if d["url"] not in ingested][: config.MAX_DOCS_TO_FETCH]
    if not pending:
        return {}

    with ThreadPoolExecutor(max_workers=config.SEARCH_WORKERS) as pool:
        texts = list(pool.map(fetch_text, [d["url"] for d in pending]))

    fetched = 0
    for doc, text in zip(pending, texts, strict=True):
        ingested.add(doc["url"])
        pieces = chunk(text)
        if pieces:
            fetched += 1
            source = "page"
        else:
            # The floor: no page text means we still have what we always had.
            pieces = [doc.get("snippet", "")] if doc.get("snippet") else []
            source = "snippet"
        for piece in pieces:
            chunks.append(
                {"title": doc["title"], "url": doc["url"], "text": piece, "source": source}
            )

    if state.get("debug"):
        print(
            f"[ingest] {len(pending)} pages -> {fetched} extracted, "
            f"{len(pending) - fetched} snippet fallbacks, {len(chunks)} chunks total",
            file=sys.stderr,
        )
    return {"chunks": chunks, "ingested": sorted(ingested)}
