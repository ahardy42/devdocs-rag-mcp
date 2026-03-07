# Local Vector Storage Options

## Overview

Once documentation is chunked and embedded, the vectors need to be stored and indexed for fast similarity search. For a fully local system, the key contenders are ChromaDB, LanceDB, and FAISS, each with distinct trade-offs around ease of use, performance, and features.

---

## Option Comparison

### ChromaDB

ChromaDB is purpose-built for RAG applications with the simplest developer experience of any vector store.

**Architecture:** Embedded database that runs in-process (no separate server needed). Originally Python, the core was rewritten in Rust in 2025 delivering 4x faster writes and queries.

**Key strengths:**
- Easiest API in the category — feels like working with a Python dictionary
- Built-in metadata filtering (filter by document source, section, etc.)
- Native integration with LangChain, LlamaIndex, and most RAG frameworks
- Supports both in-memory and persistent (on-disk) modes
- Automatic embedding via built-in embedding functions (can pass text directly)

**Limitations:**
- Best for datasets under ~10M vectors
- No built-in hybrid search (dense + sparse)

**Quick example:**
```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="samsung_tv_docs",
    metadata={"hnsw:space": "cosine"}
)

collection.add(
    documents=["Your chunked text here"],
    metadatas=[{"source": "quick-start-guide", "section": "setup"}],
    ids=["doc_001"]
)

results = collection.query(
    query_texts=["How do I configure config.xml?"],
    n_results=5
)
```

### LanceDB

LanceDB is an embedded, serverless vector database built in Rust on the Apache Arrow columnar format.

**Architecture:** Stores vectors and metadata together on disk using the Lance format. Memory-mapped file access with SIMD optimizations means it can query vectors directly from disk at near in-memory speeds. No separate process or server.

**Key strengths:**
- Handles tens of millions of vectors efficiently
- Native multimodal support (text, image, audio embeddings in the same table)
- Built-in full-text search (hybrid dense + sparse retrieval)
- Apache Arrow integration — excellent for data science workflows
- Tiny resource footprint (great for developer machines)
- Default vector store for AnythingLLM

**Limitations:**
- Smaller community/ecosystem than ChromaDB
- Fewer pre-built integrations with popular frameworks (though LangChain support exists)

**Quick example:**
```python
import lancedb
from lancedb.embeddings import get_registry

db = lancedb.connect("./lance_db")

table = db.create_table("samsung_tv_docs", data=[
    {"text": "Your chunked text", "source": "quick-start-guide", "section": "setup"},
], mode="overwrite")

# Create vector index
table.create_index(metric="cosine")

# Query
results = table.search("How do I configure config.xml?").limit(5).to_list()
```

### FAISS (Facebook AI Similarity Search)

FAISS is a low-level vector search library — the fastest raw similarity search engine available, but not a database.

**Architecture:** In-memory index with optional disk-backed storage. Supports multiple index types (Flat, IVF, HNSW, PQ) for different speed/accuracy/memory trade-offs.

**Key strengths:**
- Fastest raw search performance — often 100-1000x faster than database solutions for large-scale nearest-neighbor search
- Multiple index types allow fine-tuning the speed/accuracy trade-off
- GPU acceleration support
- Battle-tested at Meta's scale

**Limitations:**
- No built-in metadata storage or filtering — you manage that yourself
- No built-in persistence (you serialize/deserialize indexes manually)
- More code to write for a complete RAG pipeline
- Index building can be slow for large datasets

**Quick example:**
```python
import faiss
import numpy as np

# Build index
dimension = 768
index = faiss.IndexFlatIP(dimension)  # Inner product (cosine with normalized vectors)

# Add vectors
vectors = np.array(embeddings, dtype='float32')
faiss.normalize_L2(vectors)
index.add(vectors)

# Search
query_vec = np.array([query_embedding], dtype='float32')
faiss.normalize_L2(query_vec)
distances, indices = index.search(query_vec, k=5)

# Save/load
faiss.write_index(index, "samsung_tv.faiss")
```

---

## Head-to-Head Comparison

| Feature | ChromaDB | LanceDB | FAISS |
|---|---|---|---|
| **Type** | Embedded DB | Embedded DB | Search library |
| **Ease of use** | Excellent | Very good | Moderate |
| **Metadata support** | Built-in | Built-in (Arrow) | Manual |
| **Hybrid search** | No (dense only) | Yes (dense + FTS) | No |
| **Persistence** | Built-in | Built-in | Manual |
| **Scale** | <10M vectors | Tens of millions | Billions (in-memory) |
| **Framework integrations** | Excellent | Good | Moderate |
| **Memory footprint** | Low-moderate | Very low | Depends on index |
| **License** | Apache 2.0 | Apache 2.0 | MIT |

---

## Recommendation

**For this project, ChromaDB is the recommended starting point** for several reasons:

1. **Fastest path to a working prototype.** The API is minimal and the built-in embedding functions mean you can pass raw text and let Chroma handle embedding.
2. **Best RAG framework integration.** If you use LangChain or LlamaIndex for the retrieval pipeline, ChromaDB has first-class support.
3. **Metadata filtering is essential** for documentation RAG — you'll want to filter by doc type, API version, section, etc. ChromaDB handles this natively.
4. **Scale is sufficient.** Samsung TV documentation is thousands of pages, not millions. ChromaDB's <10M vector limit is more than adequate.

**Consider LanceDB** if you want hybrid search (combining vector similarity with keyword matching) or plan to expand to multimodal content (screenshots, diagrams from docs). Its disk-efficiency is also compelling for keeping the system lightweight.

**Skip FAISS** unless you have a specific performance bottleneck that ChromaDB/LanceDB can't handle. The extra plumbing to manage metadata and persistence isn't worth it for a documentation RAG system.

---

## Storage Architecture

```
Embedding Model (Ollama / sentence-transformers)
    │
    ▼
ChromaDB (PersistentClient)
    ├── Collection: "samsung_tv_docs"
    │   ├── Vectors (768-dim, cosine similarity)
    │   ├── Documents (original chunk text)
    │   └── Metadata (source, section, url, doc_type)
    │
    └── Stored at: ./chroma_db/  (portable directory)
```

---

## Sources

- [Zilliz - Chroma vs LanceDB Comparison](https://zilliz.com/comparison/chroma-vs-lancedb)
- [AIMultiple - Top Open-Source Vector Databases](https://research.aimultiple.com/open-source-vector-databases/)
- [Firecrawl - Best Vector Databases 2026](https://www.firecrawl.dev/blog/best-vector-databases)
- [LiquidMetal AI - Vector Database Comparison](https://liquidmetal.ai/casesAndBlogs/vector-comparison/)
- [Towards AI - ChromaDB vs Pinecone vs FAISS Benchmarks](https://pub.towardsai.net/vector-databases-performance-comparison-chromadb-vs-pinecone-vs-faiss-real-benchmarks-that-will-3eb83027c584)
- [LangCopilot - Best Vector Databases for RAG](https://langcopilot.com/posts/2025-10-14-best-vector-databases-milvus-vs-pinecone)
