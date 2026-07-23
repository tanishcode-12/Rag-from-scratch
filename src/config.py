"""
Central configuration for the RAG-from-scratch engine.

Keeping every model name, path, and hyperparameter in one place means
swapping an embedding model or generator later is a one-line change,
not a hunt through the codebase.
"""

import os
from pathlib import Path

# --- Paths -------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
CACHE_DIR: Path = BASE_DIR / "cache"

CORPUS_PATH: Path = Path(
    os.getenv("RAG_CORPUS_PATH", str(DATA_DIR / "corpus.txt")))

# --- Embedding models ----------------------------------------------------
# Two interchangeable embedding backends, matching the two approaches
# used in the source material: a plain BERT mean-pooled embedding, and
# a DPR (Dense Passage Retrieval) dual-encoder embedding.
BERT_MODEL_NAME: str = os.getenv("BERT_MODEL_NAME", "bert-base-uncased")
DPR_CONTEXT_MODEL_NAME: str = os.getenv(
    "DPR_CONTEXT_MODEL_NAME", "facebook/dpr-ctx_encoder-single-nq-base"
)
DPR_QUESTION_MODEL_NAME: str = os.getenv(
    "DPR_QUESTION_MODEL_NAME", "facebook/dpr-question_encoder-single-nq-base"
)

# --- Generation model ------------------------------------------------------
# GPT-2 is used because it runs on CPU with no API key, which keeps the
# whole demo self-contained and free to run. See README "Tradeoffs" section
# for what a production system would use instead.
GENERATOR_MODEL_NAME: str = os.getenv("GENERATOR_MODEL_NAME", "gpt2")

MAX_INPUT_TOKENS: int = 512
MAX_NEW_TOKENS: int = 80

# --- Retrieval -------------------------------------------------------------
DEFAULT_TOP_K: int = 5

# --- Misc --------------------------------------------------------------
RANDOM_SEED: int = 42
