import os
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent

CHROMA_DB_PATH = os.environ.get("DEVDOCS_CHROMA_DB_PATH", str(_ROOT / "data" / "chroma"))
EMBEDDING_MODEL = os.environ.get("DEVDOCS_EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v2")
EMBEDDING_BACKEND = os.environ.get("DEVDOCS_EMBEDDING_BACKEND", "sentence-transformers")
OLLAMA_BASE_URL = os.environ.get("DEVDOCS_OLLAMA_BASE_URL", "http://localhost:11434")
CHUNK_SIZE = int(os.environ.get("DEVDOCS_CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.environ.get("DEVDOCS_CHUNK_OVERLAP", "100"))
DEFAULT_N_RESULTS = int(os.environ.get("DEVDOCS_DEFAULT_N_RESULTS", "5"))
LOG_LEVEL = os.environ.get("DEVDOCS_LOG_LEVEL", "INFO")
