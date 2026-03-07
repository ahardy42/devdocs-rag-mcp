# DevDocs RAG MCP Server

## What This Is

A local MCP server that provides RAG-powered documentation search to Claude Code. Uses ChromaDB for vector storage and Nomic Embed V2 for embeddings.

## Commands

### Run the MCP server (for development/testing)

fastmcp dev src/devdocs_rag/server.py

### Ingest documentation

python scripts/ingest.py <path-to-docs> --collection <collection-name>

### Run tests

uv run pytest tests/

### Install dependencies

uv sync

## Architecture

- `src/devdocs_rag/server.py` — MCP tool definitions (the entry point)
- `src/devdocs_rag/embedding.py` — Embedding model wrapper
- `src/devdocs_rag/store.py` — ChromaDB wrapper
- `src/devdocs_rag/ingest/` — Document processing pipeline
- `scripts/` — CLI utilities for ingestion and crawling
- `data/chroma/` — Vector database (gitignored)
- `data/raw/` — Raw documentation files (gitignored)

## Conventions

- All logging goes to stderr (stdout is reserved for MCP protocol)
- Configuration via environment variables prefixed with DEVDOCS\_
- Chunk IDs are deterministic (based on source file + position)
- Collections map 1:1 to documentation sets (e.g., "samsung_tv", "react_native")

## Testing

Run `uv run pytest tests/` before committing. Tests use a temporary ChromaDB instance, not the production data directory.
