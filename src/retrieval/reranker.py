# src/retrieval/reranker.py

from sentence_transformers import CrossEncoder


# The reranker model — free, local, no API key needed
RERANKER_MODEL = "BAAI/bge-reranker-base"

# If the best chunk scores below this, the system declines to answer
# Start at 0.0 and tune upward once you see real queries
CONFIDENCE_THRESHOLD = 0.35


class Reranker:
    """
    Uses BAAI/bge-reranker-base (a cross-encoder) to re-score
    the top candidates from hybrid retrieval.

    Far more accurate than bi-encoder scores because it sees
    the query and document together in one pass.

    Usage:
        reranker = Reranker()
        top_chunks, scores = reranker.rerank(query, candidates, top_n=3)
    """

    def __init__(self):
        print(f"Loading reranker model: {RERANKER_MODEL}")
        print("(First run downloads ~280MB — subsequent runs are instant)")

        # CrossEncoder loads the model locally via sentence-transformers
        self.model = CrossEncoder(RERANKER_MODEL)

        print("Reranker model loaded\n")

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_n: int = 3,
    ) -> tuple[list[dict], list[float]]:
        """
        Reranks hybrid retrieval candidates using the cross-encoder.

        Args:
            query:      the user's original question
            candidates: list of dicts from HybridRetriever.search()
            top_n:      how many top results to return after reranking

        Returns:
            A tuple of:
                - top_chunks: list of top_n chunk dicts, best first
                - scores:     list of cross-encoder scores for those chunks
        """

        if not candidates:
            return [], []

        # Build (query, document) pairs — the cross-encoder needs both together
        pairs = [(query, candidate["text"]) for candidate in candidates]

        # Score every pair — returns a numpy array of floats
        # Higher score = more relevant to this specific query
        scores = self.model.predict(pairs)

        # Pair each candidate with its score
        scored_candidates = list(zip(scores, candidates))

        # Sort by score descending — best match first
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        # Separate back into scores and chunks
        top_scores = [float(sc[0]) for sc in scored_candidates[:top_n]]
        top_chunks = [sc[1] for sc in scored_candidates[:top_n]]

        return top_chunks, top_scores