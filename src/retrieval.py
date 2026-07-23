"""
Retrieval orchestration: builds (or loads) a FAISS index for a given
embedder, and exposes a simple `retrieve(query, k)` interface.

This is the layer that the Flask app and eval script actually talk to --
they shouldn't need to know about FAISS, tokenizers, or caching details.
"""

import logging
from typing import List, Tuple

import numpy as np

from src import config
from src.chunking import read_and_split_text
from src.index_store import VectorIndex

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, embedder, corpus_path=config.CORPUS_PATH, cache_dir=config.CACHE_DIR):
        self.embedder = embedder
        self.corpus_path = corpus_path
        self.cache_dir = cache_dir
        self.index = self._load_or_build_index()

    def _load_or_build_index(self) -> VectorIndex:
        cache_name = self.embedder.name

        if VectorIndex.exists(self.cache_dir, cache_name):
            logger.info("Loading cached FAISS index for '%s'", cache_name)
            return VectorIndex.load(self.cache_dir, cache_name)

        logger.info("Building FAISS index for '%s' (no cache found)", cache_name)
        paragraphs = read_and_split_text(self.corpus_path)
        embeddings = self._encode_corpus(paragraphs)

        index = VectorIndex(dimension=self.embedder.dimension)
        index.build(paragraphs, embeddings)
        index.save(self.cache_dir, cache_name)
        return index

    def _encode_corpus(self, paragraphs: List[str]) -> np.ndarray:
        # Batch in chunks of 16 to keep memory bounded on CPU while still
        # being much faster than one-at-a-time encoding for large corpora.
        batch_size = 16
        all_embeddings = []
        for i in range(0, len(paragraphs), batch_size):
            batch = paragraphs[i : i + batch_size]
            all_embeddings.append(self.embedder.encode(batch))
        return np.vstack(all_embeddings)

    def retrieve(self, query: str, k: int = config.DEFAULT_TOP_K) -> Tuple[List[str], List[float]]:
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        query_embedding = self.embedder.encode_query(query)
        return self.index.search(query_embedding, k=k)

    def rebuild(self) -> None:
        """Force a rebuild of the index, bypassing the cache. Useful after
        editing corpus.txt."""
        paragraphs = read_and_split_text(self.corpus_path)
        embeddings = self._encode_corpus(paragraphs)
        self.index = VectorIndex(dimension=self.embedder.dimension)
        self.index.build(paragraphs, embeddings)
        self.index.save(self.cache_dir, self.embedder.name)
