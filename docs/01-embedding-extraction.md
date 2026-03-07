# Embedding Extraction from Documentation

## Overview

The first stage of a local RAG system is converting documentation into vector embeddings that can be searched semantically. This involves three steps: document ingestion (parsing raw docs into text), chunking (splitting text into retrieval-friendly segments), and embedding (converting chunks into dense vectors). Each step has meaningful trade-offs.

---

## Document Ingestion

Samsung Smart TV documentation lives primarily on [developer.samsung.com](https://developer.samsung.com/smarttv/develop) as HTML pages, with some PDF guides and API references. To build a local knowledge base, you'll need to:

1. **Crawl or download** the documentation site. Tools like [Crawl4AI](https://github.com/coleam00/mcp-crawl4ai-rag) or Scrapy can traverse and save HTML pages. For a more targeted approach, `wget --mirror` or `httrack` can snapshot the site structure.
2. **Parse HTML to clean text.** Libraries like `BeautifulSoup`, `trafilatura`, or `unstructured` can strip navigation, ads, and boilerplate, preserving the meaningful content. The `unstructured` library is particularly good at handling mixed-format docs (HTML, PDF, markdown).
3. **Preserve structure metadata.** Keep track of which page/section each chunk comes from. This metadata is critical for citation and context during retrieval.

For Samsung TV docs specifically, you're dealing with:
- Web app development guides (HTML/CSS/JS focused)
- Tizen .NET API references
- Samsung Product API docs (TV-specific extensions)
- Configuration file specs (config.xml schema)
- SDK setup and device management guides

---

## Chunking Strategies

How you split documents has a major impact on retrieval quality. The main approaches:

### Recursive Character Splitting
Splits at natural boundaries using a hierarchy of separators: paragraphs → newlines → sentences → words. This is the **recommended default** for technical documentation because it respects document structure.

- Typical chunk size: 512–1000 tokens with 50–200 token overlap
- Works well with LangChain's `RecursiveCharacterTextSplitter`
- A February 2026 benchmark found recursive 512-token splitting achieved 69% accuracy across academic papers, outperforming other strategies

### Semantic Chunking
Groups consecutive sentences by embedding similarity — when similarity drops below a threshold, a new chunk starts. Better for dense technical writing where topics shift without clear formatting cues.

- Higher computational cost (requires embedding every sentence first)
- Can produce inconsistent chunk sizes
- Best reserved for content where context coherence is critical

### Markdown/HTML-Aware Splitting
Splits at heading boundaries (h1, h2, h3, etc.), keeping each section as a chunk. Ideal for well-structured documentation like API references.

- LangChain provides `MarkdownHeaderTextSplitter` and `HTMLHeaderTextSplitter`
- Preserves the logical grouping that documentation authors intended
- **Recommended for Samsung TV docs** given their structured format

### Practical Recommendation

For technical documentation like Samsung Smart TV docs, use a **two-pass approach**:

1. First pass: Split by HTML/markdown headers to create section-level chunks
2. Second pass: Apply recursive splitting to any chunks that exceed ~800 tokens
3. Add contextual metadata: page title, section hierarchy, URL, doc type

This hybrid approach preserves logical structure while keeping chunks within the embedding model's sweet spot.

---

## Embedding Models

For a fully local system with no API calls, these are the top options:

### Recommended: Nomic Embed Text V2
- **Parameters:** 305M active (475M total, MoE architecture)
- **Why:** Best balance of quality and resource efficiency for local deployment. The MoE design means only 305M parameters are active during inference, keeping it fast on consumer hardware. Supports Matryoshka dimensions (truncation from 768 to 256 dims).
- **Run locally via:** Ollama (`ollama pull nomic-embed-text`), or Hugging Face `sentence-transformers`

### Runner-up: Snowflake Arctic-Embed-L-v2.0
- **Parameters:** ~568M
- **Why:** Explicitly designed for code retrieval alongside natural language. Supports 32K token context windows and Matryoshka dimensions (32–4096). Apache 2.0 license.
- **Run locally via:** Hugging Face, or via Ollama

### High-accuracy option: Qwen3-Embedding-8B
- **Parameters:** 8B
- **Why:** Top of the MTEB leaderboard (70.58 score). User-defined dimensions from 32 to 4096. Requires more GPU memory (~16GB+).
- **Run locally via:** Ollama, vLLM, or Hugging Face

### Lightweight option: BGE-M3 (BAAI)
- **Parameters:** ~568M
- **Why:** Strong all-rounder with MIT license. Supports dense, sparse, and multi-vector retrieval in a single model. Well-established in the RAG ecosystem.
- **Run locally via:** Hugging Face `sentence-transformers`, Ollama

### Comparison Table

| Model | Active Params | Dimensions | Code Support | License | Local Runtime |
|---|---|---|---|---|---|
| Nomic Embed V2 | 305M | 256–768 | Good | Open | Ollama, HF |
| Arctic-Embed-L-v2 | 568M | 32–4096 | Excellent | Apache 2.0 | HF, Ollama |
| Qwen3-Embedding-8B | 8B | 32–4096 | Good | Open | Ollama, vLLM |
| BGE-M3 | 568M | 1024 | Good | MIT | HF, Ollama |

### Running Embeddings Locally

**Ollama** is the simplest path to running embedding models locally:
```bash
ollama pull nomic-embed-text
# Then use via API at localhost:11434
```

For Python integration, `sentence-transformers` provides a clean interface:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("nomic-ai/nomic-embed-text-v2", trust_remote_code=True)
embeddings = model.encode(["Your documentation chunk here"])
```

---

## Processing Pipeline Summary

```
Samsung TV Docs (HTML/PDF)
    │
    ▼
Document Ingestion (BeautifulSoup / unstructured / Crawl4AI)
    │
    ▼
Chunking (HTML-aware split → recursive fallback, ~512-800 tokens)
    │
    ▼
Embedding (Nomic Embed V2 via Ollama or sentence-transformers)
    │
    ▼
Vector Store (see: 02-vector-storage.md)
```

---

## Sources

- [KDnuggets - Top 5 Embedding Models](https://www.kdnuggets.com/top-5-embedding-models-for-your-rag-pipeline)
- [BentoML - Best Open-Source Embedding Models 2026](https://www.bentoml.com/blog/a-guide-to-open-source-embedding-models)
- [Openxcell - 10 Best Embedding Models 2026](https://www.openxcell.com/blog/best-embedding-models/)
- [Firecrawl - Best Chunking Strategies for RAG](https://www.firecrawl.dev/blog/best-chunking-strategies-rag)
- [LangCopilot - Document Chunking Tested](https://langcopilot.com/posts/2025-10-11-document-chunking-for-rag-practical-guide)
- [Unstructured - Chunking Best Practices](https://unstructured.io/blog/chunking-for-rag-best-practices)
- [Databricks - Mastering Chunking Strategies](https://community.databricks.com/t5/technical-blog/the-ultimate-guide-to-chunking-strategies-for-rag-applications/ba-p/113089)
