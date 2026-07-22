import os                          # built-in Python — lets us work with file paths and folders
from pathlib import Path           # a cleaner way to handle file paths than raw strings
from langchain_community.document_loaders import PyMuPDFLoader  # the PDF reader
from langchain_core.documents import Document

from tqdm import tqdm              # progress bar — shows "3/5 files loaded" while running


def load_pdfs_from_folder(folder_path: str) -> list[Document]:
    """
    Scans a folder for all PDF files, reads each one,
    and returns a flat list of Document objects — one per page.

    Args:
        folder_path: path to the folder containing your PDFs
                     e.g. "data/pdfs"

    Returns:
        A list of Document objects, each representing one page
        of one PDF file.
    """

    # Convert the folder path string into a Path object
    # Path objects are easier to work with than raw strings
    folder = Path(folder_path)

    # Safety check — if the folder doesn't exist, tell the user clearly
    # instead of giving a confusing Python error
    if not folder.exists():
        raise FileNotFoundError(
            f"PDF folder not found: {folder_path}\n"
            f"Create it and add your PDFs there."
        )

    # Find every file in the folder that ends with .pdf
    # rglob means "search recursively" — it also finds PDFs in subfolders
    pdf_files = list(folder.rglob("*.pdf"))

    # If no PDFs found, tell the user clearly
    if not pdf_files:
        raise ValueError(
            f"No PDF files found in {folder_path}\n"
            f"Add at least one PDF to this folder before running ingestion."
        )

    print(f"Found {len(pdf_files)} PDF file(s) in '{folder_path}'")

    # This will hold all Document objects from all PDFs
    all_documents = []

    # Loop through every PDF file found
    # tqdm wraps the list to show a progress bar in the terminal
    for pdf_path in tqdm(pdf_files, desc="Loading PDFs"):

        # PyMuPDFLoader reads one PDF file
        # We pass the full file path as a string
        loader = PyMuPDFLoader(str(pdf_path))

        # .load() returns a list of Documents — one Document per page
        # So a 50-page PDF gives you a list of 50 Documents
        documents = loader.load()

        # PyMuPDF already adds some metadata, but we want to make sure
        # we have clean, consistent metadata across all our documents.
        # We add/overwrite the fields we care about.
        for doc in documents:
            doc.metadata["source"] = pdf_path.name        # e.g. "cat_320_manual.pdf"
            doc.metadata["file_path"] = str(pdf_path)     # full path for debugging
            # page is already added by PyMuPDF as doc.metadata["page"]
            # it's 0-indexed, so page 0 = first page of the PDF

        # Add this PDF's documents to our running total
        all_documents.extend(documents)

    print(f"Loaded {len(all_documents)} pages from {len(pdf_files)} PDF(s)")
    return all_documents
def preview_documents(documents: list[Document], num_docs: int = 3) -> None:
    """
    Prints a preview of the first few documents.
    Use this to verify loading worked correctly.

    Args:
        documents: the list returned by load_pdfs_from_folder()
        num_docs:  how many documents to preview (default 3)
    """
    print(f"\n--- Document Preview ({num_docs} of {len(documents)} total) ---\n")

    for i, doc in enumerate(documents[:num_docs]):
        print(f"Document {i + 1}")
        print(f"  Source  : {doc.metadata.get('source', 'unknown')}")
        print(f"  Page    : {doc.metadata.get('page', '?') + 1}")  # +1 because 0-indexed
        print(f"  Length  : {len(doc.page_content)} characters")

        # Show first 200 characters of text, replacing newlines with spaces
        preview_text = doc.page_content[:200].replace("\n", " ").strip()
        print(f"  Preview : {preview_text}...")
        print()