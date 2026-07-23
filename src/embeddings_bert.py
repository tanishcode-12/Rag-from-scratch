"""
Embedding backend 1: BERT mean-pooled embeddings.

This is the "from scratch" embedding approach -- there is no dedicated
sentence-embedding head here. A plain bert-base-uncased model is run
over the text, and the token-level outputs are mean-pooled (excluding
padding positions) into a single fixed-size vector per input. This is
the same technique used in the source RAG-with-PyTorch notebook.
"""

from typing import List

import numpy as np
import torch
from transformers import BertModel, BertTokenizer

from src import config


class BertMeanPoolingEmbedder:
    """Encodes text into fixed-size vectors via mean-pooled BERT hidden states."""

    name = "bert-mean-pooling"
    dimension = 768

    def __init__(self, model_name: str = config.BERT_MODEL_NAME, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: List[str], max_length: int = config.MAX_INPUT_TOKENS) -> np.ndarray:
        """Encode a batch of texts into an (N, 768) numpy array of embeddings."""
        if isinstance(texts, str):
            texts = [texts]

        tokens = self.tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)

        hidden_states = self.model(input_ids, attention_mask=attention_mask)[0]  # (N, T, 768)

        # Mean-pool over real tokens only -- padding positions must not
        # dilute the average, otherwise short sequences get worse vectors.
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        summed = torch.sum(hidden_states * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        mean_pooled = summed / counts

        return mean_pooled.cpu().numpy().astype("float32")

    def encode_query(self, question: str) -> np.ndarray:
        """Encode a single question. BERT mean-pooling is symmetric, so this
        just reuses `encode` -- unlike DPR, there's no separate query encoder."""
        return self.encode([question])
