"""
Retrieval evaluation harness.

Runs a fixed set of test questions, each with a manually-labeled keyword
that should appear in a correctly-retrieved passage, against both
embedder backends. Reports Hit Rate@k and Mean Reciprocal Rank (MRR) so
"BERT vs DPR" is backed by numbers instead of a vibe.

Usage:
    python scripts/evaluate_retrieval.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings_bert import BertMeanPoolingEmbedder
from src.embeddings_dpr import DPREmbedder
from src.retrieval import Retriever

# Each test case: a question, and a keyword that must appear in the
# retrieved passage for that passage to count as "relevant." This is a
# simple but legitimate proxy for relevance judgment at small scale.
TEST_CASES = [
    {"question": "What does the RAG and LangChain course cover?", "expect_keyword": "LangChain"},
    {"question": "What is Dense Passage Retrieval?", "expect_keyword": "DPR"},
    {"question": "What is FAISS used for?", "expect_keyword": "FAISS"},
    {"question": "What did the aircraft damage detection project achieve?", "expect_keyword": "Aircraft Damage"},
    {"question": "What does the Meta database course teach?", "expect_keyword": "Database Engineer"},
    {"question": "What is JobGenie?", "expect_keyword": "JobGenie"},
    {"question": "What is the Model Context Protocol course about?", "expect_keyword": "MCP"},
    {"question": "What car price prediction project was built?", "expect_keyword": "CarIQ"},
    {"question": "What does the prompt engineering course teach?", "expect_keyword": "Prompt Engineering"},
    {"question": "What is the transformer self-attention course about?", "expect_keyword": "Transformers"},
    {"question": "What is Escanor?", "expect_keyword": "Escanor"},
    {"question": "What pricing models were used in the parking hackathon project?", "expect_keyword": "Parking"},
]


def evaluate(embedder_name: str, embedder, k: int = 5) -> dict:
    retriever = Retriever(embedder)

    hits = 0
    reciprocal_ranks = []

    for case in TEST_CASES:
        results, _ = retriever.retrieve(case["question"], k=k)
        keyword = case["expect_keyword"].lower()

        rank = None
        for i, passage in enumerate(results, start=1):
            if keyword in passage.lower():
                rank = i
                break

        if rank is not None:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    hit_rate = hits / len(TEST_CASES)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

    return {"embedder": embedder_name, "hit_rate_at_k": hit_rate, "mrr": mrr, "k": k, "n_queries": len(TEST_CASES)}


def main():
    print(f"Running retrieval evaluation on {len(TEST_CASES)} labeled test questions...\n")

    results = []
    for name, embedder_cls in [("bert", BertMeanPoolingEmbedder), ("dpr", DPREmbedder)]:
        print(f"Evaluating '{name}' embedder...")
        embedder = embedder_cls()
        results.append(evaluate(name, embedder))

    print("\n" + "=" * 60)
    print(f"{'Embedder':<12}{'Hit Rate@5':<15}{'MRR':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['embedder']:<12}{r['hit_rate_at_k']:<15.2%}{r['mrr']:<10.3f}")
    print("=" * 60)
    print(
        "\nHit Rate@5: fraction of questions where a relevant passage appeared "
        "in the top 5 retrieved results.\n"
        "MRR: Mean Reciprocal Rank -- rewards relevant passages appearing "
        "higher in the ranking, not just present somewhere in top-k."
    )


if __name__ == "__main__":
    main()
