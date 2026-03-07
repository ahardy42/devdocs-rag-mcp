from __future__ import annotations

import hashlib
from pathlib import Path

from devdocs_rag import config
from devdocs_rag.ingest.loaders import Document
from devdocs_rag.store import Chunk


def chunk_document(doc: Document) -> list[Chunk]:
    """Two-pass hybrid chunking: structural split then recursive fallback."""
    if doc.format == "html":
        raw_chunks = _split_html(doc.content)
    elif doc.format == "markdown":
        raw_chunks = _split_markdown(doc.content)
    else:
        raw_chunks = _split_recursive(doc.content)

    # Second pass: recursively split any chunks that are too large
    final_texts: list[str] = []
    for chunk in raw_chunks:
        if len(chunk.split()) > config.CHUNK_SIZE:
            final_texts.extend(_split_recursive(chunk))
        else:
            final_texts.append(chunk)

    source = doc.metadata.get("source", "unknown")
    return [
        Chunk(
            id=_chunk_id(source, i),
            content=text,
            metadata={**doc.metadata, "chunk_index": i},
        )
        for i, text in enumerate(final_texts)
        if text.strip()
    ]


def _chunk_id(source: str, index: int) -> str:
    key = f"{source}:{index}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _split_html(content: str) -> list[str]:
    from langchain_text_splitters import HTMLHeaderTextSplitter
    splitter = HTMLHeaderTextSplitter(headers_to_split_on=[
        ("h1", "h1"), ("h2", "h2"), ("h3", "h3"),
    ])
    docs = splitter.split_text(content)
    return [d.page_content for d in docs if d.page_content.strip()]


def _split_markdown(content: str) -> list[str]:
    from langchain_text_splitters import MarkdownHeaderTextSplitter
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
        ("#", "h1"), ("##", "h2"), ("###", "h3"),
    ])
    docs = splitter.split_text(content)
    return [d.page_content for d in docs if d.page_content.strip()]


def _split_recursive(content: str) -> list[str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE * 5,  # approximate chars from token estimate
        chunk_overlap=config.CHUNK_OVERLAP * 5,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(content)
