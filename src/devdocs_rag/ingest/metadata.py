from __future__ import annotations

from devdocs_rag.store import Chunk


def enrich(chunks: list[Chunk], extra: dict) -> list[Chunk]:
    """Merge extra metadata (e.g. doc_type, url) into each chunk's metadata."""
    for chunk in chunks:
        chunk.metadata.update(extra)
    return chunks
