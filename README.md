# devdocs-rag-mcp

A local MCP server that provides RAG-powered documentation search to Claude Code. Index any documentation corpus into a local vector database, then search it semantically from within Claude Code sessions — no external APIs, no network exposure.

The first documentation set is Samsung Smart TV developer docs, but the architecture supports any collection.

---

## How it works

```
Documentation (HTML, PDF, Markdown)
        |
        v
  Ingestion pipeline
  (parse → chunk → embed → store)
        |
        v
   ChromaDB (local)
        |
        v
  FastMCP RAG Server  <-- stdio -->  Claude Code
```

The server runs as a subprocess of Claude Code over stdio. When Claude needs documentation, it calls one of the MCP tools (`search_docs`, `list_collections`, etc.), which embeds the query locally using Nomic Embed Text V2 and returns ranked chunks from ChromaDB.

---

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (package manager)

---

## Setup

```bash
git clone https://github.com/your-username/devdocs-rag-mcp.git
cd devdocs-rag-mcp
uv sync --extra dev
```

Verify the install:

```bash
uv run python -c "from devdocs_rag.server import mcp; print('OK')"
```

---

## Ingesting documentation

```bash
# Ingest a directory of HTML/Markdown/PDF files into a named collection
uv run python scripts/ingest.py ./data/raw/samsung-tv/ --collection samsung_tv

# Optionally tag documents with a type
uv run python scripts/ingest.py ./data/raw/samsung-tv/api/ --collection samsung_tv --doc-type api_reference
```

Ingestion is idempotent — re-running replaces existing chunks for the same files.

---

## Running the server

For interactive development and tool testing via the MCP Inspector:

```bash
uv run fastmcp dev inspector src/devdocs_rag/server.py
```

---

## Connecting to Claude Code

Edit `.mcp.json` at the project root to set the correct absolute paths for your machine, then register with Claude Code:

```bash
claude mcp add devdocs-rag --scope local
```

Or point Claude Code at the `.mcp.json` directly by opening this directory as your project root. Claude Code will pick it up automatically.

Verify the server is registered:

```bash
claude mcp list
claude mcp get devdocs-rag
```

Once connected, Claude Code can call `search_docs`, `list_collections`, `collection_stats`, `get_doc_context`, and `ingest_docs` directly during sessions.

---

## Configuration

All settings are controlled via environment variables prefixed with `DEVDOCS_`. Set them in `.mcp.json` under `env`, or export them before running scripts.

| Variable                    | Default                            | Description                         |
| --------------------------- | ---------------------------------- | ----------------------------------- |
| `DEVDOCS_CHROMA_DB_PATH`    | `./data/chroma`                    | Where ChromaDB stores its files     |
| `DEVDOCS_EMBEDDING_MODEL`   | `nomic-ai/nomic-embed-text-v2-moe` | HuggingFace model ID                |
| `DEVDOCS_EMBEDDING_BACKEND` | `sentence-transformers`            | `sentence-transformers` or `ollama` |
| `DEVDOCS_CHUNK_SIZE`        | `800`                              | Target tokens per chunk             |
| `DEVDOCS_CHUNK_OVERLAP`     | `100`                              | Overlap between consecutive chunks  |
| `DEVDOCS_DEFAULT_N_RESULTS` | `5`                                | Default number of search results    |
| `DEVDOCS_LOG_LEVEL`         | `INFO`                             | Log level (stderr only)             |

---

## Project structure

```
devdocs-rag-mcp/
├── pyproject.toml                  # Dependencies and entry points
├── .mcp.json                       # MCP server registration for Claude Code
│
├── src/devdocs_rag/
│   ├── server.py                   # FastMCP server — all tool definitions
│   ├── config.py                   # Configuration (env vars + defaults)
│   ├── embedding.py                # EmbeddingModel wrapper
│   ├── store.py                    # DocStore — ChromaDB wrapper
│   ├── ingest/
│   │   ├── pipeline.py             # Orchestrates load → chunk → embed → store
│   │   ├── loaders.py              # Document loaders (HTML, PDF, Markdown)
│   │   ├── chunkers.py             # Two-pass hybrid chunking strategy
│   │   └── metadata.py             # Metadata enrichment
│   └── utils/
│       └── logging.py              # Stderr-only logging setup
│
├── scripts/
│   ├── ingest.py                   # CLI for ingestion
│   └── crawl_samsung_docs.py       # Crawler for Samsung developer portal
│
├── tests/                          # pytest test suite
├── data/chroma/                    # ChromaDB persistent storage (gitignored)
└── data/raw/                       # Raw documentation files (gitignored)
```

---

## Inspecting the database

Browse a collection interactively in the terminal (arrow keys to navigate, `s` to search):

```bash
uv run chroma browse samsung_tv --path data/chroma
```

Or query directly from Python:

```bash
# Collection summary
uv run python -c "
from devdocs_rag.embedding import EmbeddingModel
from devdocs_rag.store import DocStore
store = DocStore(embedding_model=EmbeddingModel())
s = store.collection_stats('samsung_tv')
print('docs:', s.doc_count, '| types:', s.doc_types)
" 2>/dev/null

# Manual search
uv run python -c "
from devdocs_rag.embedding import EmbeddingModel
from devdocs_rag.store import DocStore
store = DocStore(embedding_model=EmbeddingModel())
for r in store.search('samsung_tv', 'remote control key events', n_results=3):
    print(f'score={r.relevance_score:.3f}', r.content[:200])
" 2>/dev/null
```

---

## Adding a new documentation set

Each documentation set is an independent named **collection** in ChromaDB. Collections are isolated — adding React Native docs has no effect on the Samsung TV collection, and Claude can search either or both.

### Step 1 — Get the docs

You need the documentation as local files (HTML, Markdown, or PDF). How you get them depends on the source:

**Option A: Download a static site**

Use `wget` to mirror a documentation site:

```bash
wget --mirror --convert-links --adjust-extension --no-parent \
     --directory-prefix=data/raw/react-native \
     https://reactnative.dev/docs/getting-started
```

**Option B: Write a crawler**

Use `scripts/crawl_samsung_docs.py` as a template. The key parts to adapt are:
- `SEED_URLS` — one entry point per major section of the docs
- `_ALLOWED_PREFIX` — the URL prefix used to stay within the site

**Option C: Clone a docs repo**

Many projects publish their docs as Markdown in a GitHub repo:

```bash
git clone --depth=1 https://github.com/sveltejs/svelte.dev data/raw/svelte
```

### Step 2 — Ingest into a named collection

```bash
# Ingest with per-file doc_type inference (recommended)
uv run python scripts/ingest.py data/raw/react-native/ \
    --collection react_native \
    --infer-doc-type

# Or apply a single doc_type to everything
uv run python scripts/ingest.py data/raw/svelte/documentation/ \
    --collection svelte \
    --doc-type guide
```

The `--infer-doc-type` flag classifies each file as `api_reference`, `guide`, `spec`, etc. based on its path — useful when the source site uses a standard directory structure. Use `--doc-type` when all files are the same type or the paths aren't structured.

Ingestion is idempotent — re-running updates existing chunks in place.

### Step 3 — Verify the collection

```bash
# Check what was indexed
uv run python -c "
from devdocs_rag.embedding import EmbeddingModel
from devdocs_rag.store import DocStore
store = DocStore(embedding_model=EmbeddingModel())
s = store.collection_stats('react_native')
print('docs:', s.doc_count, '| types:', s.doc_types)
" 2>/dev/null

# Test a search
uv run python -c "
from devdocs_rag.embedding import EmbeddingModel
from devdocs_rag.store import DocStore
store = DocStore(embedding_model=EmbeddingModel())
for r in store.search('react_native', 'how to use FlatList', n_results=3):
    print(f'score={r.relevance_score:.3f}', r.content[:200])
" 2>/dev/null
```

### Step 4 — Use it in Claude Code

No server restart needed. The new collection is immediately available via the existing MCP tools:

```
search_docs("how do I handle navigation?", collection="react_native")
list_collections()   ← confirms the new collection is present
```

You can search a specific collection with the `collection` argument, or omit it to search across all indexed collections at once.

---

## Running tests

```bash
uv run pytest tests/

# RAG accuracy evaluation against the indexed samsung_tv collection
uv run python evals/run_eval.py
```

---

## Stack

| Component         | Choice                                   |
| ----------------- | ---------------------------------------- |
| MCP framework     | FastMCP                                  |
| Vector store      | ChromaDB                                 |
| Embedding model   | Nomic Embed Text V2 (305M params, local) |
| Embedding runtime | sentence-transformers                    |
| Document parsing  | unstructured + BeautifulSoup             |
| Chunking          | LangChain text splitters                 |
| Package manager   | uv                                       |
