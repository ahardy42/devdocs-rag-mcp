# Connecting the RAG MCP Server to Claude Code

## Overview

Once the RAG MCP server is built, it needs to be registered with Claude Code so that Claude can discover and call its tools during coding sessions. Claude Code supports local MCP servers via stdio transport, making this straightforward.

---

## How Claude Code Connects to MCP Servers

Claude Code acts as an MCP client. When you add a server, Claude Code:

1. Spawns the server process (via the command you specify)
2. Communicates over **stdio** (stdin/stdout) — no network, no ports
3. Discovers the server's tools, resources, and prompts via the MCP protocol
4. Makes those tools available for Claude to call during your session

This means your RAG server runs as a subprocess of Claude Code, with zero network exposure and no authentication needed.

---

## Registration Methods

### Method 1: CLI (Quick Setup)

```bash
# Add the RAG server to Claude Code
claude mcp add samsung-tv-rag \
  --transport stdio \
  --scope user \
  -- python /path/to/your/rag_server.py
```

Flags:
- `--transport stdio` — local process communication
- `--scope user` — available across all your projects (alternatives: `local` for current project only, `project` for shared team config)

### Method 2: Direct Config Edit (Recommended for Complex Setups)

Edit `~/.claude.json` (user scope) or `.mcp.json` (project scope):

```json
{
  "mcpServers": {
    "samsung-tv-rag": {
      "type": "stdio",
      "command": "python",
      "args": ["/path/to/your/rag_server.py"],
      "env": {
        "CHROMA_DB_PATH": "/path/to/chroma_db",
        "EMBEDDING_MODEL": "nomic-embed-text"
      }
    }
  }
}
```

For a **uvx-managed** server (recommended for dependency isolation):

```json
{
  "mcpServers": {
    "samsung-tv-rag": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--from", "/path/to/your/rag-server-package", "samsung-tv-rag-server"]
    }
  }
}
```

### Method 3: Project-Level Config (`.mcp.json`)

For sharing the RAG server config with a team, create `.mcp.json` in the project root:

```json
{
  "mcpServers": {
    "samsung-tv-rag": {
      "type": "stdio",
      "command": "python",
      "args": ["./tools/rag_server.py"],
      "env": {
        "CHROMA_DB_PATH": "${PROJECT_ROOT}/data/chroma_db"
      }
    }
  }
}
```

Environment variables can use `${VAR}` syntax to keep paths and secrets out of version control.

---

## Verifying the Connection

After adding the server:

```bash
# List registered MCP servers
claude mcp list

# Test a specific server (check tool discovery)
claude mcp get samsung-tv-rag
```

When you start a Claude Code session, you should see the server's tools listed in the available tools. Claude will be able to call `search_docs`, `list_topics`, etc. directly.

---

## Server Entry Point

Your RAG server's main script needs to start the FastMCP server in stdio mode:

```python
# rag_server.py
from fastmcp import FastMCP
import chromadb
from sentence_transformers import SentenceTransformer
import os

mcp = FastMCP("samsung-tv-docs")

# Configuration via environment variables
db_path = os.environ.get("CHROMA_DB_PATH", "./chroma_db")
model_name = os.environ.get("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v2")

# Initialize components
model = SentenceTransformer(model_name, trust_remote_code=True)
chroma = chromadb.PersistentClient(path=db_path)
collection = chroma.get_or_create_collection("samsung_tv_docs")

# ... tool definitions (from 03-rag-mcp-server-design.md) ...

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

---

## Making It Reusable Across Documentation Sets

The system should be designed to support multiple documentation sets beyond Samsung TV. A few architectural patterns:

### Multiple Collections
```python
# One ChromaDB instance, multiple collections
collections = {
    "samsung_tv": chroma.get_or_create_collection("samsung_tv_docs"),
    "react_native": chroma.get_or_create_collection("react_native_docs"),
    "tizen_native": chroma.get_or_create_collection("tizen_native_docs"),
}

@mcp.tool()
def search_docs(query: str, collection_name: str = "samsung_tv", n_results: int = 5):
    """Search documentation. Available collections: samsung_tv, react_native, tizen_native."""
    coll = collections.get(collection_name)
    if not coll:
        return {"error": f"Unknown collection: {collection_name}"}
    # ... search logic ...
```

### Multiple MCP Servers
Alternatively, run separate MCP servers per documentation set. This keeps each server focused and lets you register/remove them independently:

```json
{
  "mcpServers": {
    "samsung-tv-rag": {
      "type": "stdio",
      "command": "python",
      "args": ["./rag_server.py"],
      "env": { "COLLECTION": "samsung_tv" }
    },
    "react-native-rag": {
      "type": "stdio",
      "command": "python",
      "args": ["./rag_server.py"],
      "env": { "COLLECTION": "react_native" }
    }
  }
}
```

### Recommendation
Start with **multiple collections in one server**. It's simpler to manage and lets Claude search across all docs when needed. Split into separate servers only if startup time or memory becomes an issue.

---

## Practical Considerations

### Startup Time
The embedding model loads when the server starts. With Nomic Embed V2, expect ~2-5 seconds on first launch. ChromaDB persistent client initializes nearly instantly. If startup time is an issue, consider pre-loading via Ollama (the model stays warm in the Ollama process).

### Memory Usage
- Nomic Embed V2: ~600MB RAM
- ChromaDB with 10K chunks: ~50-100MB
- Total: ~700MB-1GB for a working system

### Error Handling
MCP servers that crash will disconnect from Claude Code. Build in robust error handling:

```python
@mcp.tool()
def search_docs(query: str, n_results: int = 5) -> list[dict]:
    try:
        # ... search logic ...
    except Exception as e:
        return [{"error": str(e), "suggestion": "Try rephrasing your query or check server logs."}]
```

### Logging
FastMCP supports standard Python logging. Log to stderr (stdout is reserved for MCP protocol messages):

```python
import logging
import sys

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger("samsung-tv-rag")
```

---

## Development Workflow

1. **Develop with MCP Inspector:** FastMCP includes a debugging tool — run `fastmcp dev rag_server.py` to test tools interactively before connecting to Claude Code.
2. **Register with Claude Code:** Use `claude mcp add` to connect.
3. **Test in a session:** Start Claude Code and verify the tools appear. Try queries like "search the Samsung TV docs for how to handle remote control input."
4. **Iterate:** Update the server code, restart Claude Code to pick up changes.

---

## Full System Diagram

```
┌──────────────────────────────────────────────────────────┐
│                    Your Machine                           │
│                                                          │
│  ┌─────────────────────────────────────────────┐        │
│  │              Claude Code                     │        │
│  │                                              │        │
│  │  "How do I handle focus navigation           │        │
│  │   on Samsung TV?"                            │        │
│  │       │                                      │        │
│  │       │ calls search_docs(                   │        │
│  │       │   query="focus navigation Samsung TV" │       │
│  │       │ )                                    │        │
│  │       ▼                                      │        │
│  └───────┬──────────────────────────────────────┘        │
│          │ stdio                                         │
│  ┌───────▼──────────────────────────────────────┐        │
│  │         FastMCP RAG Server                    │        │
│  │                                               │        │
│  │  1. Embed query via Nomic Embed V2            │        │
│  │  2. Search ChromaDB (cosine similarity)       │        │
│  │  3. Return top 5 chunks + metadata            │        │
│  │                                               │        │
│  │  ┌─────────────┐    ┌────────────────────┐   │        │
│  │  │ Ollama /    │    │ ChromaDB           │   │        │
│  │  │ sentence-   │    │ ./chroma_db/       │   │        │
│  │  │ transformers│    │ samsung_tv_docs    │   │        │
│  │  └─────────────┘    └────────────────────┘   │        │
│  └──────────────────────────────────────────────┘        │
│                                                          │
│  ┌──────────────────────────────────────────────┐        │
│  │  Documentation Source                         │        │
│  │  Samsung TV Docs (crawled HTML/PDF)           │        │
│  │  → Chunked → Embedded → Stored in ChromaDB   │        │
│  └──────────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────────┘
```

---

## Sources

- [Claude Code MCP Docs](https://code.claude.com/docs/en/mcp)
- [Scott Spence - Configuring MCP in Claude Code](https://scottspence.com/posts/configuring-mcp-tools-in-claude-code)
- [Builder.io - Claude Code MCP Servers](https://www.builder.io/blog/claude-code-mcp-servers)
- [MCPcat - Adding MCP Server to Claude Code](https://mcpcat.io/guides/adding-an-mcp-server-to-claude-code/)
- [Cloud Artisan - Claude Code MCP Tips](https://cloudartisan.com/posts/2025-04-12-adding-mcp-servers-claude-code/)
- [Model Context Protocol - Connect Local Servers](https://modelcontextprotocol.io/docs/develop/connect-local-servers)
- [Samsung Smart TV Development Hub](https://developer.samsung.com/smarttv/develop)
