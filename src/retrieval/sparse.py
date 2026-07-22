# src/retrieval/sparse.py

import pickle
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from src.ingestion.indexer import BM25_INDEX_PATH


class SparseRetriever:
    """
    Handles BM25 keyword search.
    Complements DenseRetriever — best for exact terms,
    model numbers, part codes, specific identifiers.

    Usage:
        retriever = SparseRetriever()
        results = retriever.search("CAT320 hydraulic pressure relief valve", top_k=20)
    """

    def __init__(self, index_path: str = BM25_INDEX_PATH):

        if not Path(index_path).exists():
            raise FileNotFoundError(
                f"BM25 index not found at '{index_path}'.\n"
                f"Run python ingest.py first to build the index."
            )

        print("Loading BM25 index from disk...")

        # Load the pickle file — gives us back the dict we saved
        with open(index_path, "rb") as f:  # "rb" = read binary
            data = pickle.load(f)

        self.bm25: BM25Okapi = data["bm25"]
        self.chunks: list[Document] = data["chunks"]

        print(f"BM25 index loaded — {len(self.chunks)} chunks indexed")

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """
        Finds the top_k chunks with the best keyword match to the query.

        Args:
            query:  the user's question
            top_k:  how many results to return

        Returns:
            Same format as DenseRetriever.search() so fusion can treat
            both retrievers identically
        """

        # Tokenize the query the same way we tokenized the chunks
        # MUST be identical preprocessing — same .lower().split()
        tokenized_query = query.lower().split()

        # Get a BM25 score for every chunk in the index
        # Returns a numpy array of length = total chunks
        scores = self.bm25.get_scores(tokenized_query)

        # Get the indices of the top_k highest scores
        # np.argsort sorts ascending, so we take the last top_k and reverse
        top_indices = np.argsort(scores)[-top_k:][::-1]

        # Build results in the same format as DenseRetriever
        results = []
        for idx in top_indices:
            # Only include chunks with a score above 0
            # A score of 0 means no query words appeared at all
            if scores[idx] > 0:
                results.append({
                    "text":     self.chunks[idx].page_content,
                    "metadata": self.chunks[idx].metadata,
                    "score":    float(scores[idx]),
                    "id":       f"chunk_{idx}",
                })

        return results