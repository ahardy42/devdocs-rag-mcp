from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from devdocs_rag.embedding import EmbeddingModel
from devdocs_rag.ingest.chunkers import chunk_document
from devdocs_rag.ingest.loaders import load_file
from devdocs_rag.ingest.metadata import enrich
from devdocs_rag.store import DocStore
from devdocs_rag.utils.logging import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".html", ".htm", ".md", ".pdf"}


@dataclass
class IngestResult:
    files_processed: int = 0
    chunks_created: int = 0
    time_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


def ingest(
    path: str | Path,
    collection_name: str,
    store: DocStore,
    embedding_model: EmbeddingModel,
    extra_metadata: dict | None = None,
) -> IngestResult:
    """Ingest a file or directory into a named collection. Idempotent."""
    path = Path(path)
    files = (
        [f for f in path.rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS]
        if path.is_dir()
        else [path]
    )

    result = IngestResult()
    start = time.monotonic()

    for file in files:
        try:
            doc = load_file(file)
            chunks = chunk_document(doc)
            if not chunks:
                logger.debug("Skipping %s — no chunks produced", file.name)
                continue
            if extra_metadata:
                enrich(chunks, extra_metadata)
            embeddings = embedding_model.embed([c.content for c in chunks])
            store.add_documents(collection_name, chunks, embeddings)
            result.files_processed += 1
            result.chunks_created += len(chunks)
            logger.info("Ingested %s → %d chunks", file.name, len(chunks))
        except Exception as exc:
            logger.error("Failed to ingest %s: %s", file, exc)
            result.errors.append(f"{file}: {exc}")

    result.time_seconds = time.monotonic() - start
    return result
