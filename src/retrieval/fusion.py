# src/retrieval/fusion.py

from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever


class HybridRetriever:
    """
    Combines dense (vector) and sparse (BM25) retrieval
    using Reciprocal Rank Fusion (RRF).

    This is the single retriever the rest of the pipeline uses.
    It replaces calling DenseRetriever or SparseRetriever directly.

    Usage:
        retriever = HybridRetriever()
        results = retriever.search("max hydraulic pressure CAT320", top_k=20)
    """

    def __init__(self, rrf_k: int = 60):
        """
        Args:
            rrf_k: the RRF smoothing constant.
                   60 is the standard value from the original RRF paper.
                   Higher values = more weight to lower-ranked results.
        """
        print("Initialising hybrid retriever...")
        self.dense   = DenseRetriever()
        self.sparse  = SparseRetriever()
        self.rrf_k   = rrf_k
        print("Hybrid retriever ready (dense + BM25 + RRF fusion)\n")

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """
        Runs both retrievers and fuses results using RRF.

        Args:
            query:  the user's question
            top_k:  how many fused results to return (before reranking)

        Returns:
            A list of dicts with text, metadata, rrf_score, and id,
            sorted by RRF score descending
        """
        print("  [DEBUG] Starting hybrid search...")

        # ── Step 1: Run both retrievers independently ──
        dense_results  = self.dense.search(query,  top_k=top_k)
        print("  [DEBUG] Vector search complete.")
        sparse_results = self.sparse.search(query, top_k=top_k)
        print("  [DEBUG] BM25 search complete.")

        # ── Step 2: Build RRF score for every chunk ──

        # We use a dict keyed by chunk ID to accumulate scores
        # chunk_id → {"rrf_score": float, "text": str, "metadata": dict}
        fused: dict[str, dict] = {}

        # Score dense results — enumerate gives us (rank_index, result)
        for rank, result in enumerate(dense_results):
            chunk_id = result["id"]

            # Initialise this chunk in the dict if first time we see it
            if chunk_id not in fused:
                fused[chunk_id] = {
                    "rrf_score": 0.0,
                    "text":      result["text"],
                    "metadata":  result["metadata"],
                    "id":        chunk_id,
                }

            # Add this retriever's RRF contribution
            # rank is 0-indexed, formula uses 1-indexed, hence rank+1
            fused[chunk_id]["rrf_score"] += 1 / (rank + 1 + self.rrf_k)

        # Score sparse results — same logic
        for rank, result in enumerate(sparse_results):
            chunk_id = result["id"]

            if chunk_id not in fused:
                fused[chunk_id] = {
                    "rrf_score": 0.0,
                    "text":      result["text"],
                    "metadata":  result["metadata"],
                    "id":        chunk_id,
                }

            fused[chunk_id]["rrf_score"] += 1 / (rank + 1 + self.rrf_k)
            print("  [DEBUG] Fusing results (RRF)...")
        # ── Step 3: Sort by RRF score descending ──
        sorted_results = sorted(
            fused.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
        )

        # Return the top_k fused results
        return sorted_results[:top_k]
