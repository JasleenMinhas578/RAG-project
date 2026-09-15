import numpy as np
import pytest
from langchain_core.documents import Document

import src.embedding as embedding
from src.embedding import EmbeddingPipeline, get_embedding_model
from src.vectorstore import FaissVectorStore
from src.visuals import find_overlap


class FakeModel:
    def encode(self, texts, **kwargs):
        return np.zeros((len(texts), 4), dtype="float32")

    def get_sentence_embedding_dimension(self):
        return 4


@pytest.fixture
def model_loads(monkeypatch):
    loads = []

    def fake_sentence_transformer(name):
        loads.append(name)
        return FakeModel()

    monkeypatch.setattr(embedding, "SentenceTransformer", fake_sentence_transformer)
    get_embedding_model.cache_clear()
    yield loads
    get_embedding_model.cache_clear()


def test_the_embedding_model_is_loaded_once_and_shared(model_loads):
    first = EmbeddingPipeline(model_name="m").model
    second = EmbeddingPipeline(model_name="m").model
    store_model = FaissVectorStore(embedding_model="m").model
    assert first is second is store_model
    assert model_loads == ["m"]


def test_creating_pipelines_and_stores_does_not_load_a_model(model_loads):
    EmbeddingPipeline(model_name="m")
    FaissVectorStore(embedding_model="m")
    assert model_loads == []


def test_chunks_respect_the_size_limit_and_overlap(model_loads):
    text = " ".join(f"word{i}" for i in range(400))
    chunks = EmbeddingPipeline(chunk_size=200, chunk_overlap=50).chunk_documents([Document(page_content=text)])
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 200 for c in chunks)
    assert all(find_overlap(a.page_content, b.page_content) > 0 for a, b in zip(chunks, chunks[1:], strict=False))


def test_embed_chunks_reports_progress_after_each_batch(model_loads):
    chunks = [Document(page_content=f"chunk {i}") for i in range(5)]
    calls = []
    vectors = EmbeddingPipeline(model_name="m").embed_chunks(
        chunks, batch_size=2, progress_callback=lambda done, total: calls.append((done, total)))
    assert vectors.shape == (5, 4)
    assert calls == [(2, 5), (4, 5), (5, 5)]
