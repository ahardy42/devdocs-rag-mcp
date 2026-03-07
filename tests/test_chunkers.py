"""Tests for chunking strategies (Step 6)."""

import pytest

import devdocs_rag.config as _config
from devdocs_rag.ingest.chunkers import _chunk_id, chunk_document
from devdocs_rag.ingest.loaders import Document
from devdocs_rag.store import Chunk

_HTML = """\
<html>
<head><title>Samsung TV Guide</title></head>
<body>
<h1>Introduction</h1>
<p>Handle remote input events on Samsung TV.</p>
<h2>Key Events</h2>
<p>Use addEventListener to capture keydown events from the remote control.</p>
<h3>Example</h3>
<p>window.addEventListener("keydown", handler);</p>
</body>
</html>"""

_MARKDOWN = """\
# Getting Started

Install the Tizen SDK to start developing Samsung TV apps.

## Configuration

Edit config.xml to configure your web application manifest.

### Example

Set the required permissions in the feature elements."""


def _html_doc(source: str = "guide.html") -> Document:
    return Document(content=_HTML, metadata={"source": source}, format="html")


def _markdown_doc(source: str = "guide.md") -> Document:
    return Document(content=_MARKDOWN, metadata={"source": source}, format="markdown")


# ---------------------------------------------------------------------------
# chunk_document — HTML
# ---------------------------------------------------------------------------

def test_html_produces_chunks():
    assert len(chunk_document(_html_doc())) > 0


def test_html_chunks_have_content():
    assert all(c.content.strip() for c in chunk_document(_html_doc()))


def test_html_chunk_has_required_fields():
    chunk = chunk_document(_html_doc())[0]
    assert isinstance(chunk, Chunk)
    assert chunk.id
    assert chunk.content
    assert isinstance(chunk.metadata, dict)


def test_html_chunk_metadata_has_chunk_index():
    assert all("chunk_index" in c.metadata for c in chunk_document(_html_doc()))


def test_html_chunk_metadata_preserves_source():
    chunks = chunk_document(_html_doc(source="guide.html"))
    assert all(c.metadata.get("source") == "guide.html" for c in chunks)


# ---------------------------------------------------------------------------
# chunk_document — Markdown
# ---------------------------------------------------------------------------

def test_markdown_produces_chunks():
    assert len(chunk_document(_markdown_doc())) > 0


def test_markdown_chunks_have_content():
    assert all(c.content.strip() for c in chunk_document(_markdown_doc()))


def test_markdown_chunk_metadata_has_chunk_index():
    assert all("chunk_index" in c.metadata for c in chunk_document(_markdown_doc()))


def test_markdown_chunk_metadata_preserves_source():
    chunks = chunk_document(_markdown_doc(source="guide.md"))
    assert all(c.metadata.get("source") == "guide.md" for c in chunks)


# ---------------------------------------------------------------------------
# chunk_document — unknown format (recursive fallback)
# ---------------------------------------------------------------------------

def test_unknown_format_produces_chunks():
    doc = Document(
        content="Plain text content for testing the recursive fallback path.",
        metadata={"source": "file.txt"},
        format="unknown",
    )
    assert len(chunk_document(doc)) > 0


# ---------------------------------------------------------------------------
# Deterministic IDs
# ---------------------------------------------------------------------------

def test_chunk_id_is_deterministic():
    assert _chunk_id("source.html", 0) == _chunk_id("source.html", 0)


def test_chunk_id_differs_by_index():
    assert _chunk_id("source.html", 0) != _chunk_id("source.html", 1)


def test_chunk_id_differs_by_source():
    assert _chunk_id("a.html", 0) != _chunk_id("b.html", 0)


def test_chunk_id_length():
    """IDs are the first 16 hex characters of a SHA256 hash."""
    assert len(_chunk_id("source.html", 0)) == 16


def test_same_doc_produces_same_ids():
    doc = _html_doc()
    assert [c.id for c in chunk_document(doc)] == [c.id for c in chunk_document(doc)]


# ---------------------------------------------------------------------------
# Second pass: large chunks are split further
# ---------------------------------------------------------------------------

def test_large_chunk_is_split(monkeypatch):
    """A single section exceeding CHUNK_SIZE words must be split in the second pass."""
    monkeypatch.setattr(_config, "CHUNK_SIZE", 10)
    monkeypatch.setattr(_config, "CHUNK_OVERLAP", 2)

    # 60 words in one markdown section — well above the patched limit of 10
    body = " ".join(["word"] * 60)
    doc = Document(
        content=f"# Section\n\n{body}",
        metadata={"source": "big.md"},
        format="markdown",
    )
    assert len(chunk_document(doc)) > 1
