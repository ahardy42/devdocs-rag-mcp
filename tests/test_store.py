"""Tests for DocStore (Step 4).

Uses a lightweight fake embedding model and a temporary ChromaDB directory
so these tests run fast without requiring the real model to be loaded.
"""

import pytest

from devdocs_rag.store import Chunk, DocStore

_DIM = 8


class _FakeEmbeddingModel:
    """Deterministic fake model that returns fixed-dimension vectors."""

    def embed_query(self, query: str) -> list[float]:
        # Vary by query length so different queries produce different vectors
        base = float(len(query) % 10) / 10.0
        return [base + i * 0.01 for i in range(_DIM)]


def _make_embeddings(n: int) -> list[list[float]]:
    return [[float(i + j * 0.1) for i in range(_DIM)] for j in range(n)]


@pytest.fixture()
def store(tmp_path) -> DocStore:
    return DocStore(db_path=str(tmp_path / "chroma"), embedding_model=_FakeEmbeddingModel())


@pytest.fixture()
def chunks() -> list[Chunk]:
    return [
        Chunk(
            id="chunk-001",
            content="Samsung TV remote control API reference",
            metadata={"source": "api.html", "doc_type": "api_reference", "section": "Input"},
        ),
        Chunk(
            id="chunk-002",
            content="Tizen web application setup and configuration guide",
            metadata={"source": "guide.html", "doc_type": "guide", "section": "Setup"},
        ),
        Chunk(
            id="chunk-003",
            content="config.xml schema for Samsung Smart TV applications",
            metadata={"source": "config.html", "doc_type": "api_reference", "section": "Config"},
        ),
    ]


# ---------------------------------------------------------------------------
# add_documents
# ---------------------------------------------------------------------------

def test_add_documents_returns_count(store: DocStore, chunks: list[Chunk]) -> None:
    count = store.add_documents("col", chunks, _make_embeddings(len(chunks)))
    assert count == len(chunks)


def test_add_documents_persists_to_collection(store: DocStore, chunks: list[Chunk]) -> None:
    store.add_documents("col", chunks, _make_embeddings(len(chunks)))
    stats = store.collection_stats("col")
    assert stats.doc_count == len(chunks)


def test_add_documents_is_idempotent(store: DocStore, chunks: list[Chunk]) -> None:
    """Re-ingesting the same chunk IDs should upsert, not duplicate."""
    embeddings = _make_embeddings(len(chunks))
    store.add_documents("col", chunks, embeddings)
    store.add_documents("col", chunks, embeddings)
    assert store.collection_stats("col").doc_count == len(chunks)


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def test_search_returns_results(store: DocStore, chunks: list[Chunk]) -> None:
    store.add_documents("col", chunks, _make_embeddings(len(chunks)))
    results = store.search("col", "remote control", n_results=2)
    assert len(results) == 2


def test_search_result_has_required_fields(store: DocStore, chunks: list[Chunk]) -> None:
    store.add_documents("col", chunks, _make_embeddings(len(chunks)))
    results = store.search("col", "samsung tv", n_results=1)
    assert len(results) == 1
    r = results[0]
    assert isinstance(r.content, str)
    assert isinstance(r.metadata, dict)
    assert isinstance(r.relevance_score, float)


def test_search_respects_n_results(store: DocStore, chunks: list[Chunk]) -> None:
    store.add_documents("col", chunks, _make_embeddings(len(chunks)))
    results = store.search("col", "query", n_results=1)
    assert len(results) == 1


def test_search_with_doc_type_filter(store: DocStore, chunks: list[Chunk]) -> None:
    store.add_documents("col", chunks, _make_embeddings(len(chunks)))
    results = store.search("col", "api", n_results=5, filters={"doc_type": "guide"})
    assert all(r.metadata["doc_type"] == "guide" for r in results)


# ---------------------------------------------------------------------------
# list_collections
# ---------------------------------------------------------------------------

def test_list_collections_returns_all(store: DocStore, chunks: list[Chunk]) -> None:
    store.add_documents("col_a", chunks[:1], _make_embeddings(1))
    store.add_documents("col_b", chunks[1:2], _make_embeddings(1))
    names = {c.name for c in store.list_collections()}
    assert {"col_a", "col_b"}.issubset(names)


def test_list_collections_doc_count(store: DocStore, chunks: list[Chunk]) -> None:
    store.add_documents("col", chunks, _make_embeddings(len(chunks)))
    info = next(c for c in store.list_collections() if c.name == "col")
    assert info.doc_count == len(chunks)


# ---------------------------------------------------------------------------
# collection_stats
# ---------------------------------------------------------------------------

def test_collection_stats_doc_count(store: DocStore, chunks: list[Chunk]) -> None:
    store.add_documents("col", chunks, _make_embeddings(len(chunks)))
    stats = store.collection_stats("col")
    assert stats.name == "col"
    assert stats.doc_count == len(chunks)


def test_collection_stats_doc_types(store: DocStore, chunks: list[Chunk]) -> None:
    store.add_documents("col", chunks, _make_embeddings(len(chunks)))
    stats = store.collection_stats("col")
    assert set(stats.doc_types) == {"api_reference", "guide"}


def test_collection_stats_sections(store: DocStore, chunks: list[Chunk]) -> None:
    store.add_documents("col", chunks, _make_embeddings(len(chunks)))
    stats = store.collection_stats("col")
    assert set(stats.sections) == {"Input", "Setup", "Config"}


# ---------------------------------------------------------------------------
# delete_collection
# ---------------------------------------------------------------------------

def test_delete_collection_removes_it(store: DocStore, chunks: list[Chunk]) -> None:
    store.add_documents("to_delete", chunks[:1], _make_embeddings(1))
    assert store.delete_collection("to_delete") is True
    names = {c.name for c in store.list_collections()}
    assert "to_delete" not in names
