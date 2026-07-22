# test_loader.py  — run this to verify Phase 2 works
# Delete this file after you're satisfied everything works

from src.ingestion.loader import load_pdfs_from_folder, preview_documents

# Load all PDFs from your data/pdfs folder
documents = load_pdfs_from_folder("data/pdfs")

# Preview the first 3 pages to make sure the text looks right
preview_documents(documents, num_docs=3)