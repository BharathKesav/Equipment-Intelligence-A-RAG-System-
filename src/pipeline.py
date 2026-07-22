# src/pipeline.py

from src.retrieval.fusion import HybridRetriever
from src.retrieval.reranker import Reranker
from src.generation.scope_guard import should_decline, build_decline_response
from src.generation.generator import generate_answer
import yaml


def load_config(path: str = "configs/prompts.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# Initialise the heavy models once at startup — not on every query
# These stay in memory while the app runs
print("Initialising RAG pipeline...")
_hybrid_retriever = HybridRetriever()
_reranker         = Reranker()
print("Pipeline ready.\n")


def run_pipeline(query: str) -> dict:
    """
    Runs a user query through the full RAG pipeline:
    Hybrid retrieval → Reranking → Scope guard → LLM generation

    Args:
        query: the user's question as a plain string

    Returns:
        A dict with answer, citations, declined flag, confidence score
    """

    config = load_config()

    print(f"\nQuery: '{query}'")

    # ── Step 1: Hybrid retrieval (BM25 + vector + RRF) ──
    top_k = config.get("top_k_retrieval", 20)
    candidates = _hybrid_retriever.search(query, top_k=top_k)
    print(f"Retrieved {len(candidates)} candidates from hybrid retrieval")

    # ── Step 2: Rerank with BGE cross-encoder ──
    top_n = config.get("top_n_rerank", 3)
    top_chunks, top_scores = _reranker.rerank(query, candidates, top_n=top_n)
    print(f"Reranked to top {len(top_chunks)} chunks")
    print(f"Top reranker score: {max(top_scores):.3f}" if top_scores else "No scores")

    # ── Step 3: Scope guard ──
    if should_decline(top_scores):
        print("Scope guard triggered — declining to answer")
        return build_decline_response()

    # ── Step 4: LLM generation with citations ──
    print("Generating answer...")
    result = generate_answer(query, top_chunks, top_scores)
    print(f"Answer generated (prompt v{result['prompt_ver']})")

    return result