"""Tests for EmbeddingModel (Step 3).

Tests run against sentence-transformers/all-MiniLM-L6-v2, a small public model,
to verify the embedding infrastructure without requiring HuggingFace auth.

"""

import pytest

from devdocs_rag.embedding import EmbeddingModel

# Small, public model used for tests — no HuggingFace auth required.
_TEST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_EXPECTED_DIM = 384


@pytest.fixture(scope="module")
def model() -> EmbeddingModel:
    return EmbeddingModel(model_name=_TEST_MODEL)


def test_dimension(model: EmbeddingModel) -> None:
    assert model.dimension == _EXPECTED_DIM


def test_embed_single_document(model: EmbeddingModel) -> None:
    result = model.embed(["Samsung TV remote control API"])
    assert len(result) == 1
    assert len(result[0]) == _EXPECTED_DIM
    assert all(isinstance(v, float) for v in result[0])


def test_embed_multiple_documents(model: EmbeddingModel) -> None:
    texts = ["Samsung TV remote control", "Tizen web application", "config.xml schema"]
    result = model.embed(texts)
    assert len(result) == 3
    assert all(len(vec) == _EXPECTED_DIM for vec in result)


def test_embed_query_returns_vector(model: EmbeddingModel) -> None:
    result = model.embed_query("how do I handle focus navigation?")
    assert len(result) == _EXPECTED_DIM
    assert all(isinstance(v, float) for v in result)


def test_document_and_query_embeddings_differ(model: EmbeddingModel) -> None:
    """The search_document: / search_query: prefixes must produce different vectors."""
    text = "Samsung remote control input handling"
    doc_vec = model.embed([text])[0]
    query_vec = model.embed_query(text)
    assert doc_vec != query_vec


def test_embed_returns_distinct_vectors_for_distinct_texts(model: EmbeddingModel) -> None:
    texts = ["Samsung TV", "React Native mobile app"]
    result = model.embed(texts)
    assert result[0] != result[1]


def test_embed_is_deterministic(model: EmbeddingModel) -> None:
    text = "Tizen application lifecycle"
    first = model.embed([text])[0]
    second = model.embed([text])[0]
    assert first == second
