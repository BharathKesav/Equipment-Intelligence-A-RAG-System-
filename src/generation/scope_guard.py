# src/generation/scope_guard.py

from src.retrieval.reranker import CONFIDENCE_THRESHOLD


def should_decline(top_scores: list[float]) -> bool:
    """
    Decides whether the system should decline to answer.

    Returns True (decline) if:
      - No chunks were retrieved at all
      - The best reranker score is below the confidence threshold

    Args:
        top_scores: list of cross-encoder scores from Reranker.rerank()

    Returns:
        True  → decline (no relevant information found)
        False → proceed with answer generation
    """

    # No candidates at all
    if not top_scores:
        return True

    # Best score below threshold — documents don't contain the answer
    if max(top_scores) < CONFIDENCE_THRESHOLD:
        return True

    return False


def build_decline_response() -> dict:
    """
    Returns a structured decline response.
    The generator returns this instead of calling the LLM.
    """
    return {
        "answer": (
            "I don't have enough information in the available documents "
            "to answer this question confidently. Please rephrase your "
            "question or check if the relevant documentation has been "
            "added to the system."
        ),
        "citations": [],
        "declined":  True,
        "confidence": 0.0,
    }