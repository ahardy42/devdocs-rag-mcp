from __future__ import annotations

from devdocs_rag import config
from devdocs_rag.utils.logging import get_logger

logger = get_logger(__name__)


class EmbeddingModel:
    """Unified embedding interface supporting sentence-transformers and Ollama backends."""

    def __init__(
        self,
        model_name: str = config.EMBEDDING_MODEL,
        backend: str = config.EMBEDDING_BACKEND,
    ) -> None:
        self.model_name = model_name
        self.backend = backend
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        if self.backend == "sentence-transformers":
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model %s", self.model_name)
            self._model = SentenceTransformer(self.model_name, trust_remote_code=True)
        else:
            raise NotImplementedError(f"Backend '{self.backend}' not yet implemented")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents. Applies document prefix for Nomic models."""
        self._load()
        prefixed = [f"search_document: {t}" for t in texts]
        return self._model.encode(prefixed, convert_to_numpy=True).tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query. Applies query prefix for Nomic models."""
        self._load()
        return self._model.encode(f"search_query: {query}", convert_to_numpy=True).tolist()

    @property
    def dimension(self) -> int:
        self._load()
        return self._model.get_sentence_embedding_dimension()
