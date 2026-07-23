"""
Embedding backend 2: Dense Passage Retrieval (DPR).

Unlike the plain BERT mean-pooling approach, DPR uses two separate
encoders trained specifically for retrieval: one for passages/contexts,
one for questions, projected into a shared embedding space via their
pooled [CLS] output. This is what real production retrieval systems
are closer to, which is why it's worth comparing against the
from-scratch BERT approach rather than just picking one.
"""

from typing import List

import numpy as np
import torch
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizer,
)

from src import config


class DPREmbedder:
    """Encodes passages and questions separately using DPR's dual encoders."""

    name = "dpr-dual-encoder"
    dimension = 768

    def __init__(
        self,
        context_model_name: str = config.DPR_CONTEXT_MODEL_NAME,
        question_model_name: str = config.DPR_QUESTION_MODEL_NAME,
        device: str | None = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(context_model_name)
        self.context_encoder = DPRContextEncoder.from_pretrained(context_model_name).to(self.device)
        self.context_encoder.eval()

        self.question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(question_model_name)
        self.question_encoder = DPRQuestionEncoder.from_pretrained(question_model_name).to(self.device)
        self.question_encoder.eval()

    @torch.no_grad()
    def encode(self, texts: List[str], max_length: int = 256) -> np.ndarray:
        """Encode a batch of passages/contexts. Use `encode_query` for questions."""
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        # DPR's own tokenizer doesn't batch cleanly with variable-length
        # padding across very different passage lengths, so this encodes
        # one passage at a time -- slower, but avoids padding pollution
        # visible in the source notebook when batching mixed-length text.
        for text in texts:
            inputs = self.context_tokenizer(
                text, return_tensors="pt", padding=True, truncation=True, max_length=max_length
            ).to(self.device)
            output = self.context_encoder(**inputs)
            embeddings.append(output.pooler_output)

        return torch.cat(embeddings).cpu().numpy().astype("float32")

    @torch.no_grad()
    def encode_query(self, question: str) -> np.ndarray:
        """Encode a single question using the question encoder (not the context encoder)."""
        inputs = self.question_tokenizer(question, return_tensors="pt").to(self.device)
        output = self.question_encoder(**inputs)
        return output.pooler_output.cpu().numpy().astype("float32")
