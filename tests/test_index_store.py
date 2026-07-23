import numpy as np
import pytest

from src.index_store import VectorIndex


def _dummy_data(n=6, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    paragraphs = [f"paragraph number {i}" for i in range(n)]
    embeddings = rng.random((n, dim)).astype("float32")
    return paragraphs, embeddings


def test_build_and_search_returns_k_results():
    paragraphs, embeddings = _dummy_data(n=6, dim=16)
    index = VectorIndex(dimension=16)
    index.build(paragraphs, embeddings)

    query = embeddings[0:1]  # search with an exact corpus vector
    results, scores = index.search(query, k=3)

    assert len(results) == 3
    assert len(scores) == 3
    # The nearest neighbor of a vector identical to itself should be itself.
    assert results[0] == paragraphs[0]
    assert scores[0] < 1e-4


def test_build_raises_on_mismatched_lengths():
    paragraphs, embeddings = _dummy_data(n=6, dim=16)
    index = VectorIndex(dimension=16)
    with pytest.raises(ValueError):
        index.build(paragraphs[:5], embeddings)  # 5 paragraphs, 6 embeddings


def test_search_on_empty_index_raises():
    index = VectorIndex(dimension=16)
    query = np.random.rand(1, 16).astype("float32")
    with pytest.raises(RuntimeError):
        index.search(query, k=3)


def test_search_k_larger_than_corpus_is_clamped():
    paragraphs, embeddings = _dummy_data(n=3, dim=8)
    index = VectorIndex(dimension=8)
    index.build(paragraphs, embeddings)

    query = np.random.rand(1, 8).astype("float32")
    results, scores = index.search(query, k=10)  # only 3 items exist

    assert len(results) == 3
    assert len(scores) == 3


def test_save_and_load_round_trip(tmp_path):
    paragraphs, embeddings = _dummy_data(n=4, dim=12)
    index = VectorIndex(dimension=12)
    index.build(paragraphs, embeddings)
    index.save(tmp_path, "test_index")

    assert VectorIndex.exists(tmp_path, "test_index")

    reloaded = VectorIndex.load(tmp_path, "test_index")
    assert reloaded.paragraphs == paragraphs
    assert reloaded.index.ntotal == 4


def test_exists_false_when_no_cache(tmp_path):
    assert VectorIndex.exists(tmp_path, "nonexistent") is False


def test_load_missing_cache_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        VectorIndex.load(tmp_path, "nonexistent")
