import pytest

from src.retrieval import Retriever


class FakeEmbedder:
    """Deterministic fake embedder so retrieval logic can be tested without
    downloading any real model -- keeps the test suite fast and offline."""

    name = "fake-test-embedder"
    dimension = 8

    def encode(self, texts):
        return [[float(len(t) % 10 + 1)] * self.dimension for t in texts]

    def encode_query(self, text):
        return self.encode([text])


@pytest.fixture
def corpus_file(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "Paragraph about cats and their behavior in the wild outdoors.\n\n"
        "Paragraph about dogs and how they are trained for service work.\n\n"
        "Paragraph about retrieval augmented generation and vector search."
    )
    return corpus


def _to_array(embedder_output):
    import numpy as np
    return np.array(embedder_output, dtype="float32")


class NumpyFakeEmbedder(FakeEmbedder):
    """Same as FakeEmbedder but returns real numpy arrays, matching what
    the real BERT/DPR embedders return."""

    def encode(self, texts):
        return _to_array(super().encode(texts))

    def encode_query(self, text):
        return _to_array(super().encode_query(text))


def test_retriever_builds_index_and_retrieves(tmp_path, corpus_file):
    retriever = Retriever(NumpyFakeEmbedder(), corpus_path=corpus_file, cache_dir=tmp_path)
    results, scores = retriever.retrieve("some query", k=2)
    assert len(results) == 2
    assert len(scores) == 2


def test_retriever_uses_cache_on_second_init(tmp_path, corpus_file):
    r1 = Retriever(NumpyFakeEmbedder(), corpus_path=corpus_file, cache_dir=tmp_path)
    cache_file = tmp_path / f"{NumpyFakeEmbedder.name}.faiss"
    assert cache_file.exists()

    mtime_before = cache_file.stat().st_mtime
    r2 = Retriever(NumpyFakeEmbedder(), corpus_path=corpus_file, cache_dir=tmp_path)
    mtime_after = cache_file.stat().st_mtime

    assert mtime_before == mtime_after
    assert len(r2.index.paragraphs) == len(r1.index.paragraphs)


def test_retrieve_empty_query_raises(tmp_path, corpus_file):
    retriever = Retriever(NumpyFakeEmbedder(), corpus_path=corpus_file, cache_dir=tmp_path)
    with pytest.raises(ValueError):
        retriever.retrieve("   ")


def test_rebuild_forces_reindex(tmp_path, corpus_file):
    retriever = Retriever(NumpyFakeEmbedder(), corpus_path=corpus_file, cache_dir=tmp_path)
    original_count = len(retriever.index.paragraphs)

    with open(corpus_file, "a") as f:
        f.write("\n\nA brand new paragraph that was not there before this edit.")

    retriever.rebuild()
    assert len(retriever.index.paragraphs) == original_count + 1
