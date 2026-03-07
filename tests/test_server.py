"""MCP tool integration tests (Steps 8 and 9).

Tool functions are called directly with fake dependencies injected via
monkeypatch so no real embedding model or ChromaDB path is needed.
"""

import pytest

import devdocs_rag.server as server_mod
from devdocs_rag.server import (
    collection_stats,
    get_doc_context,
    ingest_docs,
    list_collections,
    search_docs,
)
from devdocs_rag.store import Chunk, DocStore

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


class _FakeEmbeddingModel:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(i) / 10.0 for i in range(_DIM)] for _ in texts]

    def embed_query(self, query: str) -> list[float]:
        return [float(i) / 10.0 for i in range(_DIM)]


def _make_embeddings(n: int) -> list[list[float]]:
    return [[float(i) / 10.0 for i in range(_DIM)] for _ in range(n)]


@pytest.fixture()
def populated_store(tmp_path, monkeypatch):
    """Real DocStore pre-loaded with two chunks, injected into the server module."""
    model = _FakeEmbeddingModel()
    store = DocStore(db_path=str(tmp_path / "chroma"), embedding_model=model)
    chunks = [
        Chunk(
            id="c1",
            content="Remote control API for Samsung TV",
            metadata={"source": "api.html", "doc_type": "api_reference", "section": "Input"},
        ),
        Chunk(
            id="c2",
            content="Tizen SDK setup and configuration guide",
            metadata={"source": "guide.html", "doc_type": "guide", "section": "Setup"},
        ),
    ]
    store.add_documents("samsung_tv", chunks, _make_embeddings(len(chunks)))
    monkeypatch.setattr(server_mod, "_embedding_model", model)
    monkeypatch.setattr(server_mod, "_store", store)
    return store


@pytest.fixture()
def empty_store(tmp_path, monkeypatch):
    """Empty DocStore injected into the server module."""
    model = _FakeEmbeddingModel()
    store = DocStore(db_path=str(tmp_path / "chroma"), embedding_model=model)
    monkeypatch.setattr(server_mod, "_embedding_model", model)
    monkeypatch.setattr(server_mod, "_store", store)
    return store


# ---------------------------------------------------------------------------
# search_docs (Step 8)
# ---------------------------------------------------------------------------

def test_search_docs_returns_list(populated_store):
    results = search_docs(query="remote control")
    assert isinstance(results, list)


def test_search_docs_result_has_required_fields(populated_store):
    results = search_docs(query="remote control", n_results=1)
    assert len(results) == 1
    r = results[0]
    assert "content" in r
    assert "source" in r
    assert "section" in r
    assert "url" in r
    assert "relevance_score" in r


def test_search_docs_respects_n_results(populated_store):
    assert len(search_docs(query="samsung tv", n_results=1)) == 1


def test_search_docs_doc_type_filter(populated_store):
    results = search_docs(query="samsung", n_results=5, doc_type="guide")
    assert all(r["source"] == "guide.html" for r in results)


def test_search_docs_returns_empty_list_for_unknown_collection(empty_store):
    # collection doesn't exist yet — search should return [] not raise
    results = search_docs(query="anything", collection="nonexistent", n_results=3)
    assert results == []


# ---------------------------------------------------------------------------
# list_collections (Step 8)
# ---------------------------------------------------------------------------

def test_list_collections_returns_list(populated_store):
    assert isinstance(list_collections(), list)


def test_list_collections_contains_populated_collection(populated_store):
    names = [c["name"] for c in list_collections()]
    assert "samsung_tv" in names


def test_list_collections_result_has_required_fields(populated_store):
    col = next(c for c in list_collections() if c["name"] == "samsung_tv")
    assert "name" in col
    assert "doc_count" in col


def test_list_collections_doc_count(populated_store):
    col = next(c for c in list_collections() if c["name"] == "samsung_tv")
    assert col["doc_count"] == 2


def test_list_collections_empty(empty_store):
    assert list_collections() == []


# ---------------------------------------------------------------------------
# collection_stats (Step 8)
# ---------------------------------------------------------------------------

def test_collection_stats_returns_dict(populated_store):
    assert isinstance(collection_stats(collection="samsung_tv"), dict)


def test_collection_stats_has_required_fields(populated_store):
    stats = collection_stats(collection="samsung_tv")
    assert "name" in stats
    assert "doc_count" in stats
    assert "doc_types" in stats
    assert "sections" in stats


def test_collection_stats_doc_count(populated_store):
    assert collection_stats(collection="samsung_tv")["doc_count"] == 2


def test_collection_stats_doc_types(populated_store):
    types = set(collection_stats(collection="samsung_tv")["doc_types"])
    assert types == {"api_reference", "guide"}


def test_collection_stats_sections(populated_store):
    sections = set(collection_stats(collection="samsung_tv")["sections"])
    assert sections == {"Input", "Setup"}


# ---------------------------------------------------------------------------
# get_doc_context (Step 9)
# ---------------------------------------------------------------------------

def test_get_doc_context_returns_content(populated_store):
    result = get_doc_context(chunk_id="c1", collection="samsung_tv")
    assert "content" in result
    assert result["content"] == "Remote control API for Samsung TV"


def test_get_doc_context_returns_metadata(populated_store):
    result = get_doc_context(chunk_id="c1", collection="samsung_tv")
    assert "metadata" in result
    assert result["metadata"]["doc_type"] == "api_reference"


def test_get_doc_context_unknown_id_returns_error(populated_store):
    result = get_doc_context(chunk_id="does-not-exist", collection="samsung_tv")
    assert "error" in result


# ---------------------------------------------------------------------------
# ingest_docs (Step 9)
# ---------------------------------------------------------------------------

def test_ingest_docs_returns_dict(empty_store, tmp_path):
    f = tmp_path / "guide.html"
    f.write_text(_HTML, encoding="utf-8")
    result = ingest_docs(path=str(f), collection="samsung_tv")
    assert isinstance(result, dict)


def test_ingest_docs_result_has_required_fields(empty_store, tmp_path):
    f = tmp_path / "guide.html"
    f.write_text(_HTML, encoding="utf-8")
    result = ingest_docs(path=str(f), collection="samsung_tv")
    assert "files_processed" in result
    assert "chunks_created" in result
    assert "time_seconds" in result
    assert "errors" in result


def test_ingest_docs_processes_file(empty_store, tmp_path):
    f = tmp_path / "guide.html"
    f.write_text(_HTML, encoding="utf-8")
    result = ingest_docs(path=str(f), collection="samsung_tv")
    assert result["files_processed"] == 1
    assert result["chunks_created"] > 0
    assert result["errors"] == []


def test_ingest_docs_with_doc_type(empty_store, tmp_path):
    f = tmp_path / "guide.html"
    f.write_text(_HTML, encoding="utf-8")
    ingest_docs(path=str(f), collection="samsung_tv", doc_type="guide")
    stats = collection_stats(collection="samsung_tv")
    assert "guide" in stats["doc_types"]
