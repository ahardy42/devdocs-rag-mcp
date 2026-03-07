from __future__ import annotations

from dataclasses import dataclass, field

from devdocs_rag import config
from devdocs_rag.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Chunk:
    id: str
    content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    content: str
    metadata: dict
    relevance_score: float


@dataclass
class CollectionInfo:
    name: str
    doc_count: int


@dataclass
class CollectionStats:
    name: str
    doc_count: int
    doc_types: list[str]
    sections: list[str]


class DocStore:
    """ChromaDB wrapper for collection management, upsert, and search."""

    def __init__(self, db_path: str = config.CHROMA_DB_PATH, embedding_model=None) -> None:
        import chromadb
        self.embedding_model = embedding_model
        logger.info("Opening ChromaDB at %s", db_path)
        self._client = chromadb.PersistentClient(path=db_path)

    def get_or_create_collection(self, name: str):
        return self._client.get_or_create_collection(name)

    def add_documents(
        self, collection_name: str, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> int:
        collection = self.get_or_create_collection(collection_name)
        collection.upsert(
            ids=[c.id for c in chunks],
            documents=[c.content for c in chunks],
            embeddings=embeddings,
            metadatas=[c.metadata for c in chunks],
        )
        return len(chunks)

    def search(
        self,
        collection_name: str,
        query: str,
        n_results: int = config.DEFAULT_N_RESULTS,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        collection = self.get_or_create_collection(collection_name)
        query_embedding = self.embedding_model.embed_query(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filters or None,
            include=["documents", "metadatas", "distances"],
        )
        return [
            SearchResult(
                content=doc,
                metadata=meta,
                relevance_score=1.0 - dist,
            )
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def list_collections(self) -> list[CollectionInfo]:
        return [
            CollectionInfo(name=c.name, doc_count=c.count())
            for c in self._client.list_collections()
        ]

    def collection_stats(self, collection_name: str) -> CollectionStats:
        collection = self.get_or_create_collection(collection_name)
        all_meta = collection.get(include=["metadatas"])["metadatas"]
        doc_types = list({m.get("doc_type", "unknown") for m in all_meta})
        sections = list({m.get("section", "") for m in all_meta if m.get("section")})
        return CollectionStats(
            name=collection_name,
            doc_count=collection.count(),
            doc_types=doc_types,
            sections=sections,
        )

    def delete_collection(self, collection_name: str) -> bool:
        self._client.delete_collection(collection_name)
        return True
