# DevDocs RAG MCP Server — Project Specification

## Purpose

This document is the architectural blueprint and implementation guide for building a local MCP server that provides RAG-powered documentation retrieval to Claude Code (and other MCP-compatible clients). It is written to serve dual purposes: as a reference for a human developer, and as context for an AI coding assistant that will help build the system.

The first documentation set will be Samsung Smart TV development docs, but the architecture is designed to support any documentation corpus.

---

## System Overview

The system has three distinct components that work together:

1. **Ingestion Pipeline** — Crawls/loads documentation, chunks it, generates embeddings, and stores everything in a local vector database.
2. **RAG MCP Server** — A FastMCP server that exposes semantic search over the indexed documentation as MCP tools. Runs locally via stdio transport.
3. **Client Integration** — Configuration that registers the MCP server with Claude Code so its tools are available during coding sessions.

```
Documentation Sources           Ingestion Pipeline              MCP Server + Vector Store
(HTML, PDF, Markdown)           (one-time or scheduled)         (always-on during sessions)

  Samsung TV Docs ──┐                                          ┌─────────────────────────┐
  React Native   ──┤           ┌──────────────────┐           │    FastMCP RAG Server    │
  Tizen API      ──┤──────────▶│  Crawl / Parse   │           │                         │
  Custom Docs    ──┘           │  Chunk            │           │  Tools:                 │
                               │  Embed            │           │  - search_docs          │
                               │  Store            │           │  - list_collections     │
                               └────────┬─────────┘           │  - get_doc_context      │
                                        │                      │  - ingest_docs          │
                                        ▼                      │  - collection_stats     │
                               ┌──────────────────┐           │                         │
                               │    ChromaDB       │◀──────────│  Embedding Model:       │
                               │  ./data/chroma/   │           │  Nomic Embed V2         │
                               │                   │           │  (via sentence-         │
                               │  Collections:     │           │   transformers or       │
                               │  - samsung_tv     │           │   Ollama)               │
                               │  - react_native   │           └────────────┬────────────┘
                               │  - ...            │                        │ stdio
                               └──────────────────┘           ┌────────────▼────────────┐
                                                               │      Claude Code        │
                                                               │  (MCP Client)           │
                                                               └─────────────────────────┘
```

---

## Technology Stack

| Component          | Choice                                                  | Rationale                                                                                                 |
| ------------------ | ------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Language           | Python 3.11+                                            | Strongest ecosystem for embeddings, vector stores, and document processing                                |
| MCP Framework      | FastMCP (latest)                                        | Decorator-based tool definitions, built-in inspector, stdio transport, ~70% market share                  |
| Vector Store       | ChromaDB                                                | Simplest API, built-in metadata filtering, excellent RAG framework integration, Rust core for performance |
| Embedding Model    | Nomic Embed Text V2 (305M params)                       | Best quality/resource ratio for local deployment, MoE architecture, Matryoshka dimensions                 |
| Embedding Runtime  | sentence-transformers (primary) or Ollama (alternative) | sentence-transformers for direct Python integration; Ollama if you want a shared model server             |
| Document Parsing   | unstructured + BeautifulSoup                            | unstructured handles mixed formats (HTML, PDF, MD); BeautifulSoup for fine-grained HTML control           |
| Chunking           | LangChain text splitters                                | HTMLHeaderTextSplitter + RecursiveCharacterTextSplitter for hybrid approach                               |
| Package Management | uv                                                      | Fast, modern Python package manager with lockfile support                                                 |

---

## Project Structure

```
devdocs-rag-mcp/
├── pyproject.toml                  # Project metadata, dependencies, entry points
├── uv.lock                         # Locked dependency versions
├── README.md                       # Setup and usage instructions
├── CLAUDE.md                       # Context for Claude Code when working on this project
├── .mcp.json                       # MCP server registration for Claude Code (project-level)
│
├── src/
│   └── devdocs_rag/
│       ├── __init__.py
│       │
│       ├── server.py               # FastMCP server entry point — tool definitions, startup
│       │
│       ├── config.py               # Configuration management (env vars, defaults, paths)
│       │
│       ├── embedding.py            # Embedding model wrapper (supports sentence-transformers + Ollama)
│       │
│       ├── store.py                # ChromaDB wrapper — collection management, search, CRUD
│       │
│       ├── ingest/
│       │   ├── __init__.py
│       │   ├── pipeline.py         # Orchestrates the full ingestion flow: load → chunk → embed → store
│       │   ├── loaders.py          # Document loaders for HTML, PDF, Markdown files
│       │   ├── chunkers.py         # Chunking strategies (HTML-aware, recursive, hybrid)
│       │   └── metadata.py         # Metadata extraction and enrichment for chunks
│       │
│       └── utils/
│           ├── __init__.py
│           └── logging.py          # Logging setup (stderr only — stdout reserved for MCP protocol)
│
├── scripts/
│   ├── ingest.py                   # CLI script to run ingestion: python scripts/ingest.py ./docs/samsung-tv/
│   └── crawl_samsung_docs.py       # Crawler for Samsung developer portal
│
├── data/
│   ├── chroma/                     # ChromaDB persistent storage (gitignored)
│   └── raw/                        # Raw downloaded documentation (gitignored)
│       └── samsung-tv/             # Samsung TV docs (HTML/PDF files)
│
├── tests/
│   ├── __init__.py
│   ├── test_embedding.py           # Embedding model tests
│   ├── test_store.py               # ChromaDB wrapper tests
│   ├── test_chunkers.py            # Chunking strategy tests
│   ├── test_pipeline.py            # End-to-end ingestion tests
│   └── test_server.py              # MCP tool integration tests
│
├── evals/
│   └── samsung_tv_eval.xml         # Evaluation questions for Samsung TV docs (MCP eval format)
│
└── .gitignore
```

### Key Design Decisions in the Structure

**`src/devdocs_rag/` layout.** The server, embedding, and store modules are separate so they can be tested and modified independently. The `ingest/` subpackage is isolated because ingestion is a batch process separate from the server's runtime path.

**`scripts/` vs `ingest/`.** The `scripts/` directory contains CLI entry points (things you run directly). The `ingest/` package contains the library code they call. This separation means ingestion logic can also be called from MCP tools (the `ingest_docs` tool) without duplicating code.

**`data/` directory.** Holds both raw documentation and the ChromaDB database. Both are gitignored — the raw docs may be large and the database is a build artifact. The ingestion pipeline reads from `data/raw/` and writes to `data/chroma/`.

**`CLAUDE.md`.** This file provides context to Claude Code when working within this repository. It should describe the project structure, common commands, and conventions.

---

## Module Specifications

### `config.py` — Configuration

Centralizes all configuration with sensible defaults and environment variable overrides.

```python
# Key configuration values:
CHROMA_DB_PATH = "./data/chroma"           # Where ChromaDB stores its files
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v2"  # HuggingFace model ID
EMBEDDING_BACKEND = "sentence-transformers" # or "ollama"
OLLAMA_BASE_URL = "http://localhost:11434"  # If using Ollama backend
CHUNK_SIZE = 800                           # Target tokens per chunk
CHUNK_OVERLAP = 100                        # Overlap between consecutive chunks
DEFAULT_N_RESULTS = 5                      # Default number of search results
LOG_LEVEL = "INFO"
```

All values should be configurable via environment variables (prefixed with `DEVDOCS_`) so the server can be tuned without code changes.

### `embedding.py` — Embedding Model Wrapper

Provides a unified interface regardless of backend:

```python
class EmbeddingModel:
    def __init__(self, model_name: str, backend: str = "sentence-transformers"):
        ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts. Returns list of float vectors."""
        ...

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query. May apply query-specific prefixes."""
        ...

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...
```

The distinction between `embed` (for documents) and `embed_query` (for queries) matters because some models (including Nomic) use different prefixes for documents vs queries to improve retrieval quality.

### `store.py` — Vector Store Wrapper

Wraps ChromaDB with application-specific logic:

```python
class DocStore:
    def __init__(self, db_path: str, embedding_model: EmbeddingModel):
        ...

    def get_or_create_collection(self, name: str) -> Collection:
        ...

    def search(self, collection_name: str, query: str,
               n_results: int = 5, filters: dict | None = None) -> list[SearchResult]:
        ...

    def add_documents(self, collection_name: str,
                      chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        ...

    def list_collections(self) -> list[CollectionInfo]:
        ...

    def collection_stats(self, collection_name: str) -> CollectionStats:
        ...

    def delete_collection(self, collection_name: str) -> bool:
        ...
```

### `ingest/pipeline.py` — Ingestion Pipeline

Orchestrates the full flow:

```
Input path (file or directory)
    │
    ▼
Loader (detect format, extract text + structure)
    │
    ▼
Chunker (HTML-aware split → recursive fallback for oversized chunks)
    │
    ▼
Metadata enrichment (source URL, doc type, section hierarchy, chunk index)
    │
    ▼
Embedding (batch encode via EmbeddingModel)
    │
    ▼
Storage (batch upsert into ChromaDB collection)
    │
    ▼
Return stats (chunks created, time taken, errors)
```

The pipeline should support:

- Single file ingestion (for testing or incremental updates)
- Directory ingestion (recursively process all supported files)
- Idempotent upserts (re-ingesting the same file replaces existing chunks)

### `ingest/loaders.py` — Document Loaders

Each loader returns a standardized `Document` object:

```python
@dataclass
class Document:
    content: str           # The extracted text
    metadata: dict         # Source file, title, URL, format, etc.
    format: str            # "html", "pdf", "markdown"
```

Supported formats:

- **HTML**: Use `unstructured` or `BeautifulSoup` to extract text while preserving heading structure
- **PDF**: Use `unstructured` with PDF partitioning (handles tables, headers, etc.)
- **Markdown**: Parse with heading-awareness, preserving structure

### `ingest/chunkers.py` — Chunking Strategies

Implements the hybrid two-pass approach:

1. **First pass**: Split by structural boundaries (HTML headers, markdown headers)
2. **Second pass**: Apply `RecursiveCharacterTextSplitter` to any chunks exceeding `CHUNK_SIZE`
3. **Overlap**: Add `CHUNK_OVERLAP` tokens of overlap between consecutive sub-chunks

Each chunk gets a deterministic ID based on its source file + position, enabling idempotent re-ingestion.

### `server.py` — MCP Server (Tools)

The FastMCP server exposes these tools:

#### `search_docs`

```
Purpose: Semantic search across indexed documentation
Params:  query (str), collection (str, optional), n_results (int, default 5),
         doc_type (str, optional), section (str, optional)
Returns: List of {content, source, section, url, relevance_score}
Notes:   Primary tool. Claude will call this most often.
         Filters are AND-combined when multiple are provided.
```

#### `list_collections`

```
Purpose: Show what documentation sets are available
Params:  None
Returns: List of {name, doc_count, description}
Notes:   Helps Claude know what's indexed before searching.
```

#### `collection_stats`

```
Purpose: Detailed stats about a specific collection
Params:  collection (str)
Returns: {name, doc_count, doc_types, sections, last_updated}
Notes:   Useful for understanding what's in a collection before querying.
```

#### `get_doc_context`

```
Purpose: Retrieve the full parent section for a chunk
Params:  chunk_id (str)
Returns: {content, metadata, neighboring_chunks}
Notes:   For when a search result is relevant but Claude needs more surrounding text.
```

#### `ingest_docs`

```
Purpose: Index new documentation from the local filesystem
Params:  path (str), collection (str), doc_type (str, optional)
Returns: {chunks_created, files_processed, time_seconds, errors}
Notes:   Calls the ingestion pipeline. Enables Claude to help you add new docs.
         This is a write operation — annotate with readOnlyHint: false.
```

---

## Implementation Steps

These steps are ordered to build up the system incrementally, with a testable milestone at each stage.

### Step 1: Project Scaffolding

Set up the project with `uv`, create the directory structure above, and configure `pyproject.toml` with all dependencies.

**Dependencies:**

```toml
[project]
name = "devdocs-rag-mcp"
version = "0.1.0"
requires-python = ">=3.11"

dependencies = [
    "fastmcp>=2.0",
    "chromadb>=1.0",
    "sentence-transformers>=3.0",
    "langchain-text-splitters>=0.3",
    "unstructured[html,pdf]>=0.16",
    "beautifulsoup4>=4.12",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
]

[project.scripts]
devdocs-rag-server = "devdocs_rag.server:main"
```

**Milestone:** `uv sync` succeeds, project structure exists, imports work.

### Step 2: Configuration and Logging

Implement `config.py` and `utils/logging.py`. Logging must write to stderr (stdout is reserved for MCP protocol messages over stdio).

**Milestone:** Config loads from environment variables with defaults. Logger outputs to stderr.

### Step 3: Embedding Module

Implement `embedding.py` with the `sentence-transformers` backend first. Test that it can embed text and return vectors of the correct dimension.

```bash
# Quick verification
python -c "from devdocs_rag.embedding import EmbeddingModel; m = EmbeddingModel(); print(m.dimension)"
# Should print: 768
```

**Milestone:** `EmbeddingModel` can embed text and queries. Unit tests pass.

### Step 4: Vector Store Module

Implement `store.py`. Test CRUD operations: create collection, add documents, search, delete.

**Milestone:** `DocStore` can create collections, store embeddings, and return search results with metadata. Unit tests pass.

### Step 5: Document Loaders

Implement `ingest/loaders.py` for HTML and Markdown first. PDF can come later. Test with a few sample Samsung TV doc pages saved to `data/raw/samsung-tv/`.

**Milestone:** Loaders can parse HTML and Markdown files into `Document` objects with correct metadata.

### Step 6: Chunking

Implement `ingest/chunkers.py` with the hybrid strategy (structural split → recursive fallback). Test chunk sizes, overlap, and deterministic IDs.

**Milestone:** Chunker produces correctly-sized chunks with overlap and stable IDs. Unit tests pass.

### Step 7: Ingestion Pipeline

Wire up `ingest/pipeline.py` to connect loaders → chunkers → embedding → storage. Implement the `scripts/ingest.py` CLI.

```bash
# Ingest a directory of Samsung TV docs
python scripts/ingest.py ./data/raw/samsung-tv/ --collection samsung_tv
```

**Milestone:** Can ingest a directory of HTML files and verify chunks appear in ChromaDB.

### Step 8: MCP Server (Core Tools)

Implement `server.py` with `search_docs`, `list_collections`, and `collection_stats`. Test with FastMCP's inspector.

```bash
# Launch the inspector to test tools interactively
fastmcp dev src/devdocs_rag/server.py
```

**Milestone:** All three tools work in the MCP Inspector. `search_docs` returns relevant results from ingested Samsung TV docs.

### Step 9: MCP Server (Extended Tools)

Add `get_doc_context` and `ingest_docs` tools. Test through the inspector.

**Milestone:** Full tool suite works. `ingest_docs` can add new files through the MCP interface.

### Step 10: Claude Code Integration

Create the `.mcp.json` config and register the server. Test in a live Claude Code session.

```json
{
	"mcpServers": {
		"devdocs-rag": {
			"type": "stdio",
			"command": "uv",
			"args": [
				"run",
				"--directory",
				"/path/to/devdocs-rag-mcp",
				"devdocs-rag-server"
			],
			"env": {
				"DEVDOCS_CHROMA_DB_PATH": "/path/to/devdocs-rag-mcp/data/chroma",
				"DEVDOCS_LOG_LEVEL": "INFO"
			}
		}
	}
}
```

**Milestone:** Claude Code can discover and call the RAG tools. Asking "search the Samsung TV docs for how to handle remote control input" returns relevant results.

### Step 11: Samsung TV Documentation Crawl

Implement `scripts/crawl_samsung_docs.py` to download the Samsung Smart TV developer documentation. Target pages:

- Web app development guides
- Tizen .NET API references
- Samsung Product API docs (TV-specific extensions)
- Configuration file specs (config.xml schema)
- SDK setup and device management guides

Run the crawler, then ingest the results.

**Milestone:** Samsung TV docs are crawled, ingested, and searchable through Claude Code.

### Step 12: Evaluation

Create `evals/samsung_tv_eval.xml` with 10 questions that test whether Claude can use the RAG tools to answer Samsung TV development questions correctly. Follow the MCP evaluation format.

**Milestone:** Evaluation questions created. Run them to establish a baseline retrieval accuracy score.

---

## CLAUDE.md Template for the Project

When the project is initialized, include this `CLAUDE.md` at the root so Claude Code has context when working on the codebase:

```markdown
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
```

---

## Future Enhancements (Post-MVP)

These are not part of the initial build but are worth keeping in mind architecturally:

- **Re-ranking**: Add a cross-encoder re-ranking step (cross-encoder/ms-marco-MiniLM-L-6-v2) to improve precision on the top-N results
- **Hybrid search**: Switch to LanceDB for combined dense vector + keyword search
- **Query expansion**: Allow Claude to pass multiple query variants for broader recall
- **Incremental re-indexing**: Detect changed files and only re-embed modified chunks
- **PDF loader**: Add PDF support via unstructured's PDF partitioner
- **Ollama backend**: Add Ollama as an alternative embedding backend for shared model serving
- **Multi-server mode**: Split into per-collection MCP servers if memory becomes an issue

---

## Resources

- [FastMCP GitHub](https://github.com/jlowin/fastmcp)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Nomic Embed Text V2](https://huggingface.co/nomic-ai/nomic-embed-text-v2)
- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [Model Context Protocol Specification](https://modelcontextprotocol.io/)
- [Claude Code MCP Integration](https://code.claude.com/docs/en/mcp)
- [Samsung Smart TV Developer Hub](https://developer.samsung.com/smarttv/develop)
- [mcp-local-rag (reference implementation)](https://github.com/shinpr/mcp-local-rag)
- [mcp-rag-server (reference implementation)](https://github.com/kwanLeeFrmVi/mcp-rag-server)
