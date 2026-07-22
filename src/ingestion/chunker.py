# src/ingestion/chunker.py

from langchain_text_splitters import RecursiveCharacterTextSplitter


from langchain_core.documents import Document

from tqdm import tqdm


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Document]:
    """
    Takes a list of page-level Documents (from Phase 2)
    and splits them into smaller overlapping chunks.

    Args:
        documents:     list of Documents from load_pdfs_from_folder()
        chunk_size:    max characters per chunk (512 is a solid default)
        chunk_overlap: how many characters to repeat between chunks (64 default)

    Returns:
        A new, larger list of Documents — now chunk-level, not page-level
    """

    # Create the splitter with our settings
    # We use characters here, not tokens, because it's simpler to reason
    # about and works well for most documents. 512 chars ≈ 100-130 words.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,

        # These are the separators it tries IN ORDER
        # It always tries the first one first, falls back if chunk still too large
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],

        # This tells the splitter to measure length in characters
        length_function=len,

        # This tells it to keep the separator attached to the chunk
        # so chunks don't start with random whitespace
        is_separator_regex=False,
    )

    print(f"Chunking {len(documents)} pages into smaller pieces...")
    print(f"Settings: chunk_size={chunk_size}, overlap={chunk_overlap}")

    # Split all documents in one call
    # LangChain handles the loop internally
    chunks = splitter.split_documents(documents)

    # After splitting, each chunk still has the original metadata
    # from the parent page (source, page number, file_path).
    # We add two more fields:
    #   chunk_index    — the position of this chunk in the full list (0, 1, 2...)
    #   paragraph_ref  — a human-readable citation string for the final answer
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["paragraph_ref"] = (
            f"{chunk.metadata.get('source', 'unknown')}, "
            f"page {chunk.metadata.get('page', 0) + 1}, "  # +1 for human-readable
            f"chunk {i}"
        )

    print(f"Created {len(chunks)} chunks from {len(documents)} pages")
    print(f"Average chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} characters")
    return chunks


def preview_chunks(chunks: list[Document], num_chunks: int = 3) -> None:
    """
    Prints a preview of chunks side by side so you can see the overlap.
    Useful for verifying your chunking settings look right.
    """
    print(f"\n--- Chunk Preview (showing {num_chunks} consecutive chunks) ---\n")

    for i, chunk in enumerate(chunks[:num_chunks]):
        print(f"Chunk {i}")
        print(f"  Source : {chunk.metadata.get('source')}")
        print(f"  Page   : {chunk.metadata.get('page', 0) + 1}")
        print(f"  Length : {len(chunk.page_content)} characters")
        print(f"  Text   : {chunk.page_content[:300].replace(chr(10), ' ').strip()}...")
        print()

    # Show the overlap between chunk 0 and chunk 1 explicitly
    if len(chunks) >= 2:
        text0 = chunks[0].page_content
        text1 = chunks[1].page_content

        # Find the last 100 characters of chunk 0 in chunk 1
        tail = text0[-100:].strip()
        if tail in text1:
            print(f"✓ Overlap confirmed — last part of Chunk 0 appears in Chunk 1:")
            print(f"  '{tail[:80]}...'")
        else:
            print("(Overlap check: chunks may be from different pages — that's normal)")