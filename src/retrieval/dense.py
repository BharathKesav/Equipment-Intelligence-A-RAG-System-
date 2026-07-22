# src/retrieval/dense.py

from sentence_transformers import SentenceTransformer
import chromadb
from src.ingestion.indexer import (
    EMBEDDING_MODEL_NAME,
    CHROMA_PERSIST_PATH,
    COLLECTION_NAME,
)


class DenseRetriever:
    """
    Handles semantic (vector) search against ChromaDB.

    Usage:
        retriever = DenseRetriever()
        results = retriever.search("what is the max hydraulic pressure?", top_k=20)
    """

    def __init__(self):
        # Load the same embedding model used during ingestion
        # This MUST be the same model — you can't embed queries with a
        # different model than you used for the chunks
        print("Initialising dense retriever...")
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        # Connect to the existing ChromaDB collection
        # (it was created during ingestion — we just reconnect here)
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_PATH)
        self.collection = client.get_collection(COLLECTION_NAME)

        print(f"Dense retriever ready — {self.collection.count()} chunks indexed")

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """
        Embeds the query and finds the top_k most similar chunks.

        Args:
            query:  the user's question as a plain string
            top_k:  how many results to return (we retrieve 20, reranker cuts to 3)

        Returns:
            A list of dicts, each containing:
                - text:     the chunk content
                - metadata: source, page, chunk_index, paragraph_ref
                - score:    similarity score (higher = more similar)
                - id:       the chunk's unique ID
        """

        # Embed the query the same way we embedded the chunks
        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True,
        ).tolist()

        # Query ChromaDB — it returns the top_k most similar entries
        results = self.collection.query(
            query_embeddings=[query_embedding],  # list of lists (we only have 1 query)
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        # ChromaDB returns results in a nested structure — let's flatten it
        # results["documents"][0] is the list of texts for our single query
        # results["distances"][0] is the list of distances (lower = more similar)
        formatted = []
        for i in range(len(results["documents"][0])):
            formatted.append({
                "text":     results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score":    1 - results["distances"][0][i],  # convert distance → similarity
                "id":       results["ids"][0][i],
            })

        return formatted