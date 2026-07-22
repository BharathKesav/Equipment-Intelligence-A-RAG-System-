# src/ingestion/indexer.py

import os
import pickle                        # saves Python objects to disk (for BM25 index)
from pathlib import Path
from langchain_core.documents import Document

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
import pickle
from rank_bm25 import BM25Okapi



# ─────────────────────────────────────────────
# CONSTANTS — change these in one place if needed
# ─────────────────────────────────────────────

# The embedding model — runs locally, free, excellent quality
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"

# Where ChromaDB saves its data on disk
CHROMA_PERSIST_PATH = "storage/chroma_db"

# The name of our collection inside ChromaDB
COLLECTION_NAME = "rag_documents"

# Where we save the BM25 index (filled in Phase 5)
BM25_INDEX_PATH = "storage/bm25_index.pkl"

# How many chunks to embed and upload at once
# Larger = faster, but uses more RAM. 100 is safe for most laptops.
BATCH_SIZE = 100


def get_embedding_model() -> SentenceTransformer:
    """
    Loads the BGE embedding model.
    First call downloads ~440MB to your local cache.
    Every subsequent call loads instantly from cache.
    """
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    print("(First run downloads ~440MB — subsequent runs are instant)")

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Embedding model loaded successfully")
    return model


def get_chroma_collection(persist_path: str = CHROMA_PERSIST_PATH):
    """
    Creates or connects to a persistent ChromaDB collection.

    If the collection already exists (from a previous ingest run),
    it connects to it. If not, it creates a fresh one.

    Returns the ChromaDB collection object.
    """
    # Make sure the storage folder exists
    Path(persist_path).mkdir(parents=True, exist_ok=True)

    # Create a ChromaDB client that saves to disk
    # PersistentClient means data survives after your Python process ends
    client = chromadb.PersistentClient(path=persist_path)

    # get_or_create_collection — safe to call whether collection exists or not
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,

        # This tells ChromaDB how to measure similarity between vectors
        # cosine is standard for text embeddings
        metadata={"hnsw:space": "cosine"}
    )

    print(f"ChromaDB collection '{COLLECTION_NAME}' ready")
    print(f"  Current document count: {collection.count()}")
    return collection


def embed_and_store_chunks(
    chunks: list[Document],
    model: SentenceTransformer,
    collection,
    batch_size: int = BATCH_SIZE,
) -> None:
    """
    Takes all chunks, embeds them using the BGE model,
    and stores them in ChromaDB with their metadata.

    Does this in batches so your laptop doesn't run out of RAM.

    Args:
        chunks:     list of Document objects from Phase 3
        model:      the loaded SentenceTransformer embedding model
        collection: the ChromaDB collection to store into
        batch_size: how many chunks to process at once
    """

    print(f"\nEmbedding {len(chunks)} chunks and storing in ChromaDB...")
    print(f"Processing in batches of {batch_size}")

    # Process chunks in batches
    for batch_start in tqdm(range(0, len(chunks), batch_size), desc="Embedding batches"):

        # Slice out this batch
        batch = chunks[batch_start : batch_start + batch_size]

        # ── Extract the three things ChromaDB needs ──

        # 1. IDs — must be unique strings
        ids = [f"chunk_{batch_start + i}" for i in range(len(batch))]

        # 2. Raw text — the actual content of each chunk
        texts = [chunk.page_content for chunk in batch]

        # 3. Metadata — the breadcrumb trail (source, page, etc.)
        metadatas = [chunk.metadata for chunk in batch]

        # ── Embed the texts into vectors ──
        # normalize_embeddings=True makes cosine similarity work correctly
        # The model returns a numpy array of shape (batch_size, 768)
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,    # tqdm above is already showing progress
        )

        # Convert numpy array to a plain Python list — ChromaDB requires this
        embeddings_list = embeddings.tolist()

        # ── Store in ChromaDB ──
        # upsert = "insert if new, update if ID already exists"
        # This makes re-running ingestion safe — it won't create duplicates
        collection.upsert(
            ids=ids,
            embeddings=embeddings_list,
            documents=texts,
            metadatas=metadatas,
        )

    print(f"\nChromaDB storage complete")
    print(f"Total chunks in collection: {collection.count()}")



def build_and_save_bm25_index(
    chunks: list[Document],
    index_path: str = BM25_INDEX_PATH,
) -> None:
    """
    Builds a BM25 keyword index from all chunks and saves it to disk.

    Args:
        chunks:     the same list of chunks used for ChromaDB
        index_path: where to save the .pkl file
    """

    print(f"\nBuilding BM25 keyword index from {len(chunks)} chunks...")

    # BM25 works on tokenized text — we split each chunk into individual words
    # .lower() ensures "Hydraulic" and "hydraulic" match the same
    # .split() splits on whitespace — simple but effective
    tokenized_chunks = [chunk.page_content.lower().split() for chunk in chunks]

    # Build the BM25 index — this is fast, takes a few seconds at most
    bm25_index = BM25Okapi(tokenized_chunks)

    # Save two things together in one pickle file:
    # 1. The BM25 index object itself
    # 2. The original chunks — so we can retrieve full text + metadata by index
    # We need both because BM25 only returns index positions, not the text itself
    save_data = {
        "bm25": bm25_index,
        "chunks": chunks,
    }

    # Make sure the storage directory exists
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)

    # Write to disk using pickle
    with open(index_path, "wb") as f:  # "wb" = write binary
        pickle.dump(save_data, f)

    print(f"BM25 index saved to '{index_path}'")
    print(f"Index size: {Path(index_path).stat().st_size / 1024 / 1024:.1f} MB")