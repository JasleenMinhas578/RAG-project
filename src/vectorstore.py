import logging
import os
import pickle
from typing import Any, List

import faiss
import numpy as np

from src import config
from src.embedding import EmbeddingPipeline, get_embedding_model

logger = logging.getLogger(__name__)

# Only the embedding model has to match for an existing index to be queryable; chunk size and
# overlap are baked into the vectors and just describe how the saved index was built.
CRITICAL_SETTING = "embedding_model"


def manifest_mismatches(saved: dict, current: dict) -> list:
    """Plain-language warnings about a saved index built with settings other than `current`.

    Returns an empty list for a matching index, and for one saved before manifests existed
    (nothing was recorded, so there is nothing to compare).
    """
    if not saved:
        return []
    warnings = []
    for key, value in current.items():
        was = saved.get(key)
        if was is None or was == value:
            continue
        if key == CRITICAL_SETTING:
            warnings.append(
                f"This index was built with embedding model {was!r} but is being queried with "
                f"{value!r}. The two models produce unrelated vectors, so results will look "
                f"plausible but be wrong. Rebuild the index, or query it with {was!r}."
            )
        else:
            warnings.append(
                f"This index was built with {key}={was}, not {value}. Existing vectors keep the "
                f"setting they were built with; {value} applies only after a rebuild."
            )
    return warnings


class FaissVectorStore:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = config.DEFAULT_EMBEDDING_MODEL,
        chunk_size: int = config.DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = config.DEFAULT_CHUNK_OVERLAP,
    ):
        self.persist_dir = persist_dir
        self.index = None
        self.metadata = []
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @property
    def model(self):
        return get_embedding_model(self.embedding_model)

    def build_from_documents(self, documents: List[Any]):
        logger.info("Building vector store from %d raw documents", len(documents))
        emb_pipe = EmbeddingPipeline(model_name=self.embedding_model, chunk_size=self.chunk_size,
                                     chunk_overlap=self.chunk_overlap)
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)
        metadatas = [{"text": c.page_content, "source": os.path.basename(c.metadata.get("source", "unknown"))}
                     for c in chunks]
        self.add_embeddings(np.asarray(embeddings, dtype="float32"), metadatas)
        self.save()

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)
        logger.info("Added %d vectors to the FAISS index", embeddings.shape[0])

    @property
    def manifest(self) -> dict:
        """The settings that produced the vectors. Querying an index with a different
        embedding model compares numbers from two different vector spaces, which returns
        confident-looking nonsense rather than an error -- so it is recorded and checked."""
        return {
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }

    def save(self):
        os.makedirs(self.persist_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(self.persist_dir, "faiss.index"))
        with open(os.path.join(self.persist_dir, "metadata.pkl"), "wb") as f:
            pickle.dump({"manifest": self.manifest, "metadata": self.metadata}, f)
        logger.info("Saved FAISS index and metadata to %s", self.persist_dir)

    def load(self):
        """Load a saved index, warning if it was built with different settings than this store's."""
        self.index = faiss.read_index(os.path.join(self.persist_dir, "faiss.index"))
        with open(os.path.join(self.persist_dir, "metadata.pkl"), "rb") as f:
            payload = pickle.load(f)
        # Stores saved before the manifest existed pickled the bare metadata list.
        if isinstance(payload, dict):
            self.metadata = payload["metadata"]
            saved = payload.get("manifest", {})
        else:
            self.metadata = payload
            saved = {}
        for warning in manifest_mismatches(saved, self.manifest):
            logger.warning("%s", warning)
        logger.info("Loaded FAISS index and metadata from %s", self.persist_dir)

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        distances, indices = self.index.search(query_embedding, top_k)
        results = []
        query_vec = query_embedding[0]
        query_norm = np.linalg.norm(query_vec)
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            stored_vec = self.index.reconstruct(int(idx))
            similarity = float(np.dot(query_vec, stored_vec) / (query_norm * np.linalg.norm(stored_vec) + 1e-12))
            results.append({"index": int(idx), "distance": float(dist), "similarity": similarity, "metadata": meta})
        return results

    def query(self, query_text: str, top_k: int = 5):
        logger.info("Querying vector store for: %r", query_text)
        query_emb = self.model.encode([query_text]).astype("float32")
        return self.search(query_emb, top_k=top_k)

    def neighbors(self, position: int, k: int = 5):
        """The k stored vectors most similar to the one at `position`, excluding that vector itself."""
        vector = self.index.reconstruct(int(position)).reshape(1, -1)
        hits = self.search(vector, top_k=min(k + 1, self.ntotal))
        return [hit for hit in hits if hit["index"] != position][:k]

    @property
    def ntotal(self) -> int:
        return self.index.ntotal if self.index is not None else 0

    def reset(self):
        """Drop the in-memory index/metadata so the store can be rebuilt from scratch."""
        self.index = None
        self.metadata = []


if __name__ == "__main__":
    from src.data_loader import load_all_documents

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    store = FaissVectorStore("faiss_store")
    store.build_from_documents(load_all_documents("data"))
    print(store.query("What is attention mechanism?", top_k=3))
