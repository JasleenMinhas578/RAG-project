import numpy as np
import pytest

import src.vectorstore as vectorstore
from src.vectorstore import FaissVectorStore


@pytest.fixture
def store(monkeypatch, tmp_path):
    def no_model(name):
        raise AssertionError("these tests must not load an embedding model")

    monkeypatch.setattr(vectorstore, "get_embedding_model", no_model)
    s = FaissVectorStore(persist_dir=str(tmp_path / "store"))
    vectors = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype="float32")
    s.add_embeddings(vectors, [{"text": t, "source": "doc.txt"} for t in ("x axis", "y axis", "diagonal")])
    return s


def query(store, top_k=3):
    return store.search(np.array([[1, 0, 0]], dtype="float32"), top_k=top_k)


def test_similarity_is_cosine_similarity(store):
    results = query(store)
    similarity = {r["metadata"]["text"]: r["similarity"] for r in results}
    assert results[0]["metadata"]["text"] == "x axis"
    assert similarity["x axis"] == pytest.approx(1.0)
    assert similarity["diagonal"] == pytest.approx(1 / np.sqrt(2), abs=1e-6)
    assert similarity["y axis"] == pytest.approx(0.0, abs=1e-6)


def test_top_k_larger_than_the_index_returns_only_real_hits(store):
    assert len(query(store, top_k=10)) == 3


def test_save_and_load_round_trip(store, tmp_path):
    store.save()
    loaded = FaissVectorStore(persist_dir=str(tmp_path / "store"))
    loaded.load()
    assert loaded.ntotal == 3
    assert loaded.metadata == store.metadata


def test_creating_a_store_does_not_create_its_directory(tmp_path):
    FaissVectorStore(persist_dir=str(tmp_path / "unused"))
    assert not (tmp_path / "unused").exists()


def test_neighbors_ranks_other_chunks_and_excludes_the_chunk_itself(store):
    neighbors = store.neighbors(0, k=5)
    assert [n["metadata"]["text"] for n in neighbors] == ["diagonal", "y axis"]
    assert neighbors[0]["similarity"] == pytest.approx(1 / np.sqrt(2), abs=1e-6)


def test_reset_empties_the_store(store):
    store.reset()
    assert store.ntotal == 0
    assert store.metadata == []
