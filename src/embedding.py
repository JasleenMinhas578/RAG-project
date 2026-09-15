import logging
from collections.abc import Callable
from functools import cache
from typing import Any

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from src import config

logger = logging.getLogger(__name__)


@cache
def get_embedding_model(model_name: str) -> SentenceTransformer:
    """Load an embedding model once per process. Every pipeline, vector store, and app
    session shares the same instance instead of reloading it from disk."""
    logger.info("Loading embedding model: %s", model_name)
    return SentenceTransformer(model_name)


class EmbeddingPipeline:
    def __init__(
        self,
        model_name: str = config.DEFAULT_EMBEDDING_MODEL,
        chunk_size: int = config.DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = config.DEFAULT_CHUNK_OVERLAP,
    ):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @property
    def model(self) -> SentenceTransformer:
        return get_embedding_model(self.model_name)

    def chunk_documents(self, documents: list[Any]) -> list[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        logger.info("Split %d documents into %d chunks", len(documents), len(chunks))
        return chunks

    def embed_chunks(
        self,
        chunks: list[Any],
        batch_size: int = 32,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        """Embed chunk texts. If progress_callback is given, it's called as
        progress_callback(done, total) after each batch, so a UI can show live progress."""
        texts = [chunk.page_content for chunk in chunks]
        logger.info("Generating embeddings for %d chunks", len(texts))

        if progress_callback is None:
            embeddings = self.model.encode(texts, show_progress_bar=logger.isEnabledFor(logging.INFO))
        else:
            batches = []
            for start in range(0, len(texts), batch_size):
                batches.append(self.model.encode(texts[start:start + batch_size]))
                progress_callback(min(start + batch_size, len(texts)), len(texts))
            embeddings = np.vstack(batches) if batches else np.empty((0, self.model.get_sentence_embedding_dimension()))

        logger.info("Embeddings shape: %s", embeddings.shape)
        return embeddings


if __name__ == "__main__":
    from src.data_loader import load_all_documents

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    pipe = EmbeddingPipeline()
    vectors = pipe.embed_chunks(pipe.chunk_documents(load_all_documents("data")))
    print("Example embedding:", vectors[0] if len(vectors) else None)
