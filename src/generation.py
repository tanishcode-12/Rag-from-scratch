"""
Generation layer: produces an answer either purely from the language
model's parametric memory (no retrieval) or grounded in retrieved
context (RAG). Comparing these two side by side is the entire point
of this project's demo.

GPT-2 is used deliberately instead of a hosted API model: it runs
fully offline/on-CPU with no API key, which keeps this project
self-contained and free to run for anyone who clones the repo. See
the README "Design Decisions & Tradeoffs" section for what a
production system would use instead (an instruction-tuned model
behind an API, at minimum).
"""

import logging

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from src import config

logger = logging.getLogger(__name__)


class Generator:
    def __init__(self, model_name: str = config.GENERATOR_MODEL_NAME, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        # GPT-2 has no pad token by default; reuse eos so batched generation
        # doesn't crash on padding.
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def _generate(self, prompt: str, max_new_tokens: int = config.MAX_NEW_TOKENS) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=config.MAX_INPUT_TOKENS - max_new_tokens,
        ).to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,       # deterministic output -- easier to demo and evaluate
            num_beams=3,
            no_repeat_ngram_size=3,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return generated.strip()

    def answer_without_retrieval(self, question: str) -> str:
        """Answer using only the model's parametric knowledge -- no context."""
        prompt = f"Question: {question}\nAnswer:"
        answer = self._generate(prompt)
        return answer or "(model produced no output)"

    def answer_with_retrieval(self, question: str, context_chunks: list[str]) -> str:
        """Answer grounded in retrieved context chunks (standard RAG prompting)."""
        if not context_chunks:
            return "(no relevant context was retrieved)"

        context = " ".join(context_chunks)
        prompt = (
            f"Context: {context}\n\n"
            f"Question: {question}\n"
            f"Answer based only on the context above:"
        )
        answer = self._generate(prompt)
        return answer or "(model produced no output)"
