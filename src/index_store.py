"""
Thin wrapper around a FAISS flat L2 index, with disk caching.

Re-embedding the whole corpus on every process start is wasteful and
slow (BERT/DPR forward passes aren't free). This caches the built index
and the paragraph list together, keyed by embedder name, so subsequent
runs load instantly unless the corpus or embedder changes.
"""

import json
import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from src import config


class VectorIndex:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.paragraphs: List[str] = []

    def build(self, paragraphs: List[str], embeddings: np.ndarray) -> None:
        if embeddings.shape[0] != len(paragraphs):
            raise ValueError(
                f"Mismatch between {len(paragraphs)} paragraphs and "
                f"{embeddings.shape[0]} embeddings."
            )
        self.paragraphs = list(paragraphs)
        self.index.add(embeddings)

    def search(self, query_embedding: np.ndarray, k: int = config.DEFAULT_TOP_K) -> Tuple[List[str], List[float]]:
        if self.index.ntotal == 0:
            raise RuntimeError("Index is empty. Call build() before search().")
        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding.astype("float32"), k)
        results = [self.paragraphs[i] for i in indices[0]]
        scores = [float(d) for d in distances[0]]
        return results, scores

    def save(self, cache_dir: Path, name: str) -> None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(cache_dir / f"{name}.faiss"))
        with open(cache_dir / f"{name}.meta.json", "w") as f:
            json.dump({"paragraphs": self.paragraphs, "dimension": self.dimension}, f)

    @classmethod
    def load(cls, cache_dir: Path, name: str) -> "VectorIndex":
        cache_dir = Path(cache_dir)
        index_path = cache_dir / f"{name}.faiss"
        meta_path = cache_dir / f"{name}.meta.json"
        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"No cached index found for '{name}' in {cache_dir}")

        with open(meta_path) as f:
            meta = json.load(f)

        instance = cls(dimension=meta["dimension"])
        instance.index = faiss.read_index(str(index_path))
        instance.paragraphs = meta["paragraphs"]
        return instance

    @staticmethod
    def exists(cache_dir: Path, name: str) -> bool:
        cache_dir = Path(cache_dir)
        return (cache_dir / f"{name}.faiss").exists() and (cache_dir / f"{name}.meta.json").exists()
