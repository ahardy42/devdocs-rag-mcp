"""End-to-end ingestion pipeline tests (Step 7)."""

import pytest

from devdocs_rag.ingest.pipeline import IngestResult, ingest
from devdocs_rag.store import DocStore

_DIM = 8

_HTML = """\
<html>
<head><title>Samsung TV Guide</title></head>
<body>
<h1>Remote Control</h1>
<p>Handle remote input events on Samsung TV.</p>
<h2>Key Events</h2>
<p>Use addEventListener to capture keydown events.</p>
</body>
</html>"""

_MARKDOWN = """\
# Getting Started

Install the Tizen SDK to start developing Samsung TV apps.

## Configuration

Edit config.xml to configure your web application."""


class _FakeEmbeddingModel:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(i) / 10.0 for i in range(_DIM)] for _ in texts]

    def embed_query(self, query: str) -> list[float]:
        return [float(i) / 10.0 for i in range(_DIM)]


@pytest.fixture()
def model():
    return _FakeEmbeddingModel()


@pytest.fixture()
def store(tmp_path):
    return DocStore(db_path=str(tmp_path / "chroma"), embedding_model=_FakeEmbeddingModel())


@pytest.fixture()
def html_file(tmp_path):
    f = tmp_path / "guide.html"
    f.write_text(_HTML, encoding="utf-8")
    return f


@pytest.fixture()
def markdown_file(tmp_path):
    f = tmp_path / "guide.md"
    f.write_text(_MARKDOWN, encoding="utf-8")
    return f


# ---------------------------------------------------------------------------
# Single file ingestion — HTML
# ---------------------------------------------------------------------------

def test_ingest_html_processes_one_file(store, model, html_file):
    assert ingest(html_file, "col", store, model).files_processed == 1


def test_ingest_html_creates_chunks(store, model, html_file):
    assert ingest(html_file, "col", store, model).chunks_created > 0


def test_ingest_html_no_errors(store, model, html_file):
    assert ingest(html_file, "col", store, model).errors == []


# ---------------------------------------------------------------------------
# Single file ingestion — Markdown
# ---------------------------------------------------------------------------

def test_ingest_markdown_processes_one_file(store, model, markdown_file):
    assert ingest(markdown_file, "col", store, model).files_processed == 1


def test_ingest_markdown_creates_chunks(store, model, markdown_file):
    assert ingest(markdown_file, "col", store, model).chunks_created > 0


# ---------------------------------------------------------------------------
# Result fields
# ---------------------------------------------------------------------------

def test_ingest_returns_ingest_result(store, model, html_file):
    assert isinstance(ingest(html_file, "col", store, model), IngestResult)


def test_ingest_result_has_time(store, model, html_file):
    assert ingest(html_file, "col", store, model).time_seconds >= 0


# ---------------------------------------------------------------------------
# Directory ingestion
# ---------------------------------------------------------------------------

def test_ingest_directory_processes_supported_files(tmp_path, model):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.html").write_text(_HTML, encoding="utf-8")
    (docs / "b.md").write_text(_MARKDOWN, encoding="utf-8")
    (docs / "ignore.txt").write_text("plain text", encoding="utf-8")

    store = DocStore(db_path=str(tmp_path / "chroma"), embedding_model=model)
    result = ingest(docs, "col", store, model)
    assert result.files_processed == 2  # .txt is ignored


def test_ingest_directory_creates_chunks(tmp_path, model):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.html").write_text(_HTML, encoding="utf-8")
    (docs / "b.md").write_text(_MARKDOWN, encoding="utf-8")

    store = DocStore(db_path=str(tmp_path / "chroma"), embedding_model=model)
    assert ingest(docs, "col", store, model).chunks_created > 0


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

def test_ingest_is_idempotent(store, model, html_file):
    ingest(html_file, "col", store, model)
    first_count = store.collection_stats("col").doc_count

    ingest(html_file, "col", store, model)
    second_count = store.collection_stats("col").doc_count

    assert first_count == second_count


# ---------------------------------------------------------------------------
# Extra metadata
# ---------------------------------------------------------------------------

def test_extra_metadata_applied_to_chunks(store, model, html_file):
    ingest(html_file, "col", store, model, extra_metadata={"doc_type": "guide"})
    assert "guide" in store.collection_stats("col").doc_types


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_unsupported_file_records_error(tmp_path, model):
    f = tmp_path / "notes.txt"
    f.write_text("some text", encoding="utf-8")

    store = DocStore(db_path=str(tmp_path / "chroma"), embedding_model=model)
    result = ingest(f, "col", store, model)

    assert result.files_processed == 0
    assert len(result.errors) == 1
