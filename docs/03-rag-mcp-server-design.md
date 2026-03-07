# RAG Retrieval via Local MCP Server

## Overview

The core of this system is an MCP server that exposes RAG retrieval as tools that Claude Code (or any MCP-compatible client) can call. The server handles the "last mile" — receiving a query, searching the vector store, and returning relevant documentation chunks with context.

This document covers the MCP server architecture, tool design, and how to wire up RAG retrieval using FastMCP.

---

## Why FastMCP

FastMCP is the dominant framework for building MCP servers in Python. As of early 2026, some version of FastMCP powers roughly 70% of MCP servers across all languages. Key reasons to use it:

- **Decorator-based tool definition** — write a Python function, add `@mcp.tool()`, and FastMCP generates the JSON schema, validates inputs, and handles the protocol automatically
- **FastMCP 3.0** (released January 2026) added component versioning, authorization controls, OpenTelemetry tracing, and multiple provider types
- **Built-in debugging** via the MCP Inspector
- **stdio transport** works natively with Claude Code's local MCP server support

### Alternatives Considered

- **MCP Python SDK (official)** — lower-level, more boilerplate. FastMCP wraps this SDK, so you get the same protocol compliance with less code.
- **Node/TypeScript MCP SDK** — viable if you prefer JS. The official MCP SDK has solid TypeScript support. However, since the embedding/vector tooling ecosystem is stronger in Python, FastMCP is the better fit here.

---

## Server Architecture

```
┌─────────────────────────────────────────────┐
│                Claude Code                   │
│         (MCP Client, stdio transport)        │
└──────────────────┬──────────────────────────┘
                   │ stdio
┌──────────────────▼──────────────────────────┐
│          FastMCP RAG Server                  │
│                                              │
│  Tools:                                      │
│  ┌────────────────────────────────────────┐  │
│  │ search_docs(query, filters?)           │  │
│  │ → Embed query → Search ChromaDB        │  │
│  │ → Return ranked chunks + metadata      │  │
│  ├────────────────────────────────────────┤  │
│  │ list_topics()                          │  │
│  │ → Return available doc categories      │  │
│  ├────────────────────────────────────────┤  │
│  │ get_doc_context(doc_id)                │  │
│  │ → Return full section for a chunk      │  │
│  ├────────────────────────────────────────┤  │
│  │ ingest_docs(path)                      │  │
│  │ → Process & embed new documentation    │  │
│  └────────────────────────────────────────┘  │
│                                              │
│  Resources:                                  │
│  ┌────────────────────────────────────────┐  │
│  │ docs://status                          │  │
│  │ → Index stats, last update, doc count  │  │
│  └────────────────────────────────────────┘  │
│                                              │
│  ┌──────────────┐    ┌──────────────────┐   │
│  │ Embedding    │    │ ChromaDB         │   │
│  │ Model        │    │ (Persistent)     │   │
│  │ (Ollama /    │    │                  │   │
│  │  local)      │    │                  │   │
│  └──────────────┘    └──────────────────┘   │
└─────────────────────────────────────────────┘
```

---

## Tool Design

### Core Tool: `search_docs`

This is the primary tool Claude Code will call. It should be designed to give Claude maximum flexibility in how it queries.

```python
from fastmcp import FastMCP
import chromadb
from sentence_transformers import SentenceTransformer

mcp = FastMCP("samsung-tv-docs")

# Initialize on startup
model = SentenceTransformer("nomic-ai/nomic-embed-text-v2", trust_remote_code=True)
chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_collection("samsung_tv_docs")

@mcp.tool()
def search_docs(
    query: str,
    n_results: int = 5,
    doc_type: str | None = None,
    section: str | None = None,
) -> list[dict]:
    """Search Samsung Smart TV documentation.

    Args:
        query: Natural language search query about Samsung TV development.
        n_results: Number of results to return (default 5).
        doc_type: Optional filter - "api_reference", "guide", "tutorial", "config".
        section: Optional filter - specific documentation section.

    Returns:
        List of relevant documentation chunks with source metadata.
    """
    # Build metadata filter
    where_filter = {}
    if doc_type:
        where_filter["doc_type"] = doc_type
    if section:
        where_filter["section"] = section

    results = collection.query(
        query_embeddings=[model.encode(query).tolist()],
        n_results=n_results,
        where=where_filter if where_filter else None,
        include=["documents", "metadatas", "distances"]
    )

    return [
        {
            "content": doc,
            "source": meta.get("source", "unknown"),
            "section": meta.get("section", ""),
            "url": meta.get("url", ""),
            "relevance_score": 1 - dist,  # Convert distance to similarity
        }
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )
    ]
```

### Supporting Tools

```python
@mcp.tool()
def list_topics() -> list[dict]:
    """List all available documentation topics and their document counts."""
    # Query ChromaDB metadata to get unique categories
    all_meta = collection.get(include=["metadatas"])
    topics = {}
    for meta in all_meta["metadatas"]:
        topic = meta.get("doc_type", "unknown")
        topics[topic] = topics.get(topic, 0) + 1
    return [{"topic": k, "doc_count": v} for k, v in sorted(topics.items())]

@mcp.tool()
def get_doc_context(doc_id: str) -> dict:
    """Get the full context around a specific document chunk.

    Useful when a search result is relevant but you need more surrounding context.

    Args:
        doc_id: The document chunk ID from search results.
    """
    result = collection.get(ids=[doc_id], include=["documents", "metadatas"])
    if not result["documents"]:
        return {"error": "Document not found"}

    return {
        "content": result["documents"][0],
        "metadata": result["metadatas"][0],
    }

@mcp.tool()
def ingest_docs(path: str) -> dict:
    """Ingest new documentation files into the index.

    Supports HTML, Markdown, and PDF files. Processes them into chunks
    and adds embeddings to the vector store.

    Args:
        path: Path to a file or directory of documentation to ingest.
    """
    # Implementation would use the chunking pipeline from 01-embedding-extraction.md
    # Returns stats about what was ingested
    pass
```

### Resource: Index Status

```python
@mcp.resource("docs://status")
def index_status() -> str:
    """Current state of the documentation index."""
    count = collection.count()
    return f"Samsung TV Docs Index: {count} chunks indexed"
```

---

## Retrieval Quality Patterns

### Re-ranking

For better retrieval quality, add a re-ranking step after the initial vector search. The pattern:

1. Retrieve top-K results from ChromaDB (e.g., K=20)
2. Re-rank using a cross-encoder model (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`)
3. Return top-N re-ranked results (e.g., N=5)

Cross-encoders are slower but more accurate than bi-encoders for relevance scoring because they process the query and document together.

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def search_with_rerank(query: str, n_results: int = 5):
    # Retrieve more candidates than needed
    candidates = collection.query(
        query_embeddings=[model.encode(query).tolist()],
        n_results=n_results * 4,
        include=["documents", "metadatas", "distances"]
    )

    # Re-rank
    pairs = [(query, doc) for doc in candidates["documents"][0]]
    scores = reranker.predict(pairs)

    # Sort by re-rank score and return top N
    ranked = sorted(
        zip(scores, candidates["documents"][0], candidates["metadatas"][0]),
        key=lambda x: x[0],
        reverse=True
    )[:n_results]

    return [{"content": doc, "metadata": meta, "score": float(score)}
            for score, doc, meta in ranked]
```

### Contextual Retrieval

When a chunk is retrieved, Claude often benefits from knowing the surrounding context. Two approaches:

1. **Parent-child chunking:** Store both large "parent" chunks and smaller "child" chunks. Search over child chunks, but return the parent chunk for context.
2. **Sliding window:** Store overlapping chunks. When a chunk is retrieved, also return its neighbors.

### Query Expansion

For technical documentation, the user's query might use different terminology than the docs. Consider having Claude rephrase queries or use multiple query variants:

```python
@mcp.tool()
def multi_search(queries: list[str], n_results: int = 5) -> list[dict]:
    """Search with multiple query variants for better recall.

    Claude can generate alternative phrasings of the same question
    and pass them all for broader coverage.
    """
    all_results = {}
    for q in queries:
        results = search_docs(q, n_results=n_results)
        for r in results:
            key = r["source"] + r.get("section", "")
            if key not in all_results or r["relevance_score"] > all_results[key]["relevance_score"]:
                all_results[key] = r

    return sorted(all_results.values(), key=lambda x: x["relevance_score"], reverse=True)[:n_results]
```

---

## Existing Open-Source Implementations

Several projects have already built RAG MCP servers and are worth studying:

| Project | Stack | Notable Features |
|---|---|---|
| [mcp-rag-server](https://github.com/kwanLeeFrmVi/mcp-rag-server) | Python, SQLite vectors | Simple architecture, good reference |
| [mcp-local-rag (shinpr)](https://github.com/shinpr/mcp-local-rag) | Python, local embeddings | 6 MCP tools, fully local |
| [mcp-ragdocs](https://github.com/hannesrudolph/mcp-ragdocs) | Node.js, Qdrant | Documentation-focused |
| [mcp-crawl4ai-rag](https://github.com/coleam00/mcp-crawl4ai-rag) | Python, Supabase | Web crawling + RAG |
| [UltraRAG](https://github.com/OpenBMB/UltraRAG) | Python, YAML config | MCP-native RAG framework |

The **shinpr/mcp-local-rag** and **kwanLeeFrmVi/mcp-rag-server** projects are the closest to what this system needs and are worth forking or studying closely.

---

## Sources

- [FastMCP GitHub](https://github.com/jlowin/fastmcp)
- [MCPcat - Building MCP Server with FastMCP](https://mcpcat.io/guides/building-mcp-server-python-fastmcp/)
- [Firecrawl - FastMCP Tutorial](https://www.firecrawl.dev/blog/fastmcp-tutorial-building-mcp-servers-python)
- [DataCamp - FastMCP 2.0 Tutorial](https://www.datacamp.com/tutorial/building-mcp-server-client-fastmcp)
- [DEV Community - Local RAG with MCP for VS Code](https://dev.to/lord_magus/building-a-local-rag-system-with-mcp-for-vs-code-ai-agents-a-technical-deep-dive-29ac)
- [Medium - RAG MCP Server Tutorial](https://medium.com/data-science-in-your-pocket/rag-mcp-server-tutorial-89badff90c00)
- [Analytics Vidhya - RAG using MCP](https://www.analyticsvidhya.com/blog/2025/06/rag-with-mcp/)
