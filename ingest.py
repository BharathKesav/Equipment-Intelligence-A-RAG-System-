# ingest.py — updated version

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from dotenv import load_dotenv
load_dotenv()

from src.ingestion.loader import load_pdfs_from_folder
from src.ingestion.chunker import chunk_documents
from src.ingestion.indexer import (
    get_embedding_model,
    get_chroma_collection,
    embed_and_store_chunks,
    build_and_save_bm25_index,   # ← new import
)

PDF_FOLDER = "data/pdfs"

def main():
    print("=" * 50)
    print("RAG INGESTION PIPELINE")
    print("=" * 50)

    print("\n[Phase 2] Loading PDFs...")
    documents = load_pdfs_from_folder(PDF_FOLDER)

    print("\n[Phase 3] Chunking documents...")
    chunks = chunk_documents(documents)

    print("\n[Phase 4] Embedding chunks into ChromaDB...")
    model      = get_embedding_model()
    collection = get_chroma_collection()
    embed_and_store_chunks(chunks, model, collection)

    print("\n[Phase 5] Building BM25 keyword index...")  # ← new
    build_and_save_bm25_index(chunks)                   # ← new

    print("\n" + "=" * 50)
    print("INGESTION COMPLETE")
    print(f"  {len(documents)} pages → {len(chunks)} chunks")
    print("  → ChromaDB vector index")
    print("  → BM25 keyword index")
    print("=" * 50)

if __name__ == "__main__":
    main()