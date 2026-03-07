# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A local MCP server that provides RAG-powered documentation search to Claude Code. Uses ChromaDB for vector storage and Nomic Embed Text V2 for embeddings. The first documentation set is Samsung Smart TV developer docs, but the architecture supports any documentation corpus via named collections.

## Commands

```bash
# Install dependencies
uv sync

# Run the MCP server (interactive inspector for development/testing)
fastmcp dev src/devdocs_rag/server.py

# Ingest documentation into a named collection
python scripts/ingest.py <path-to-docs> --collection <collection-name>

# Run all tests
uv run pytest tests/

# Run a single test file
uv run pytest tests/test_store.py

# Quick embedding model verification (prints the model's output dimension)
python -c "from devdocs_rag.embedding import EmbeddingModel; m = EmbeddingModel(); print(m.dimension)"

# Browse a ChromaDB collection interactively (terminal UI — arrow keys, s to search)
uv run chroma browse samsung_tv --path data/chroma

# Run the RAG accuracy evaluation
uv run python evals/run_eval.py
```

## Architecture

The system has three distinct phases:

1. **Ingestion (batch)** — `scripts/ingest.py` → `ingest/pipeline.py` → loaders → chunkers → `embedding.py` → `store.py` → ChromaDB
2. **Serving (runtime)** — `server.py` starts a FastMCP stdio server; each tool call embeds the query and searches ChromaDB
3. **Integration** — Claude Code spawns the server as a subprocess via `.mcp.json`; communication is over stdio (no network)

### Key Modules

- `src/devdocs_rag/server.py` — FastMCP tool definitions; the MCP entry point
- `src/devdocs_rag/config.py` — All configuration, env var overrides prefixed `DEVDOCS_`
- `src/devdocs_rag/embedding.py` — `EmbeddingModel` class with `embed(texts)` and `embed_query(query)` methods (distinction matters: Nomic uses different prefixes for documents vs queries)
- `src/devdocs_rag/store.py` — `DocStore` wraps ChromaDB; handles collection management, upsert, and search
- `src/devdocs_rag/ingest/pipeline.py` — Orchestrates load → chunk → embed → store; supports single file or directory; idempotent upserts via deterministic chunk IDs
- `src/devdocs_rag/ingest/loaders.py` — Returns `Document(content, metadata, format)` for HTML, PDF, Markdown
- `src/devdocs_rag/ingest/chunkers.py` — Two-pass hybrid: split by HTML/markdown headers first, then `RecursiveCharacterTextSplitter` on any chunk exceeding `CHUNK_SIZE` (~800 tokens)
- `scripts/crawl_samsung_docs.py` — Crawler for Samsung developer portal

### MCP Tools Exposed

| Tool | Purpose |
|---|---|
| `search_docs` | Primary tool — semantic search with optional `collection`, `doc_type`, `section` filters |
| `list_collections` | Shows what documentation sets are indexed |
| `collection_stats` | Detailed stats for a specific collection |
| `get_doc_context` | Retrieves full parent section for a chunk ID |
| `ingest_docs` | Indexes new docs from the filesystem (write operation) |

### Claude Code Integration

The server is registered via `.mcp.json` at the project root (project scope) or `~/.claude.json` (user scope). It runs as a stdio subprocess — no network exposure, no ports:

```json
{
  "mcpServers": {
    "devdocs-rag": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "--directory", "/path/to/devdocs-rag-mcp", "devdocs-rag-server"],
      "env": {
        "DEVDOCS_CHROMA_DB_PATH": "/path/to/devdocs-rag-mcp/data/chroma"
      }
    }
  }
}
```

After registering: `claude mcp list` and `claude mcp get devdocs-rag` to verify.

## Conventions

- All logging goes to stderr (stdout is reserved for MCP protocol)
- Configuration via environment variables prefixed with `DEVDOCS_` (see `config.py` for defaults)
- Chunk IDs are deterministic (based on source file + position) — re-ingesting a file replaces existing chunks
- Collections map 1:1 to documentation sets (e.g., `samsung_tv`, `react_native`)
- `scripts/` contains CLI entry points; `ingest/` contains the library code they call — the same library code is also called by the `ingest_docs` MCP tool

## Testing

Tests use a temporary ChromaDB instance, not `data/chroma/`. Run `uv run pytest tests/` before committing. The `data/` directory (ChromaDB and raw docs) is gitignored.
