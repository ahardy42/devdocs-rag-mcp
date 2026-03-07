from __future__ import annotations

from fastmcp import FastMCP

from devdocs_rag import config
from devdocs_rag.embedding import EmbeddingModel
from devdocs_rag.store import DocStore
from devdocs_rag.utils.logging import get_logger

logger = get_logger(__name__)

mcp = FastMCP("devdocs-rag")

_embedding_model: EmbeddingModel | None = None
_store: DocStore | None = None


def _get_deps() -> tuple[EmbeddingModel, DocStore]:
    global _embedding_model, _store
    if _embedding_model is None:
        _embedding_model = EmbeddingModel()
    if _store is None:
        _store = DocStore(embedding_model=_embedding_model)
    return _embedding_model, _store


@mcp.tool()
def search_docs(
    query: str,
    collection: str = "samsung_tv",
    n_results: int = config.DEFAULT_N_RESULTS,
    doc_type: str | None = None,
    section: str | None = None,
) -> list[dict]:
    """Search indexed documentation semantically.

    Args:
        query: Natural language search query.
        collection: Which documentation collection to search.
        n_results: Number of results to return.
        doc_type: Optional filter by document type (e.g. "api_reference", "guide").
        section: Optional filter by section name.
    """
    _, store = _get_deps()
    filters: dict = {}
    if doc_type:
        filters["doc_type"] = doc_type
    if section:
        filters["section"] = section
    results = store.search(collection, query, n_results=n_results, filters=filters or None)
    return [
        {
            "content": r.content,
            "source": r.metadata.get("source", ""),
            "section": r.metadata.get("section", ""),
            "url": r.metadata.get("url", ""),
            "relevance_score": r.relevance_score,
        }
        for r in results
    ]


@mcp.tool()
def list_collections() -> list[dict]:
    """List all available documentation collections and their document counts."""
    _, store = _get_deps()
    return [{"name": c.name, "doc_count": c.doc_count} for c in store.list_collections()]


@mcp.tool()
def collection_stats(collection: str) -> dict:
    """Detailed stats for a specific documentation collection.

    Args:
        collection: Collection name.
    """
    _, store = _get_deps()
    stats = store.collection_stats(collection)
    return {
        "name": stats.name,
        "doc_count": stats.doc_count,
        "doc_types": stats.doc_types,
        "sections": stats.sections,
    }


@mcp.tool()
def get_doc_context(chunk_id: str, collection: str = "samsung_tv") -> dict:
    """Retrieve the full stored chunk for a given chunk ID.

    Args:
        chunk_id: The chunk ID returned in a search result.
        collection: Collection the chunk belongs to.
    """
    _, store = _get_deps()
    chroma_collection = store.get_or_create_collection(collection)
    result = chroma_collection.get(ids=[chunk_id], include=["documents", "metadatas"])
    if not result["documents"]:
        return {"error": "Chunk not found"}
    return {"content": result["documents"][0], "metadata": result["metadatas"][0]}


@mcp.tool()
def ingest_docs(
    path: str,
    collection: str,
    doc_type: str | None = None,
) -> dict:
    """Index new documentation from the local filesystem into a collection.

    Args:
        path: Absolute path to a file or directory of documentation.
        collection: Collection name to ingest into.
        doc_type: Optional document type label (e.g. "api_reference", "guide").
    """
    from devdocs_rag.ingest.pipeline import ingest

    embedding_model, store = _get_deps()
    extra: dict = {}
    if doc_type:
        extra["doc_type"] = doc_type
    result = ingest(path, collection, store, embedding_model, extra_metadata=extra or None)
    return {
        "files_processed": result.files_processed,
        "chunks_created": result.chunks_created,
        "time_seconds": round(result.time_seconds, 2),
        "errors": result.errors,
    }


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
