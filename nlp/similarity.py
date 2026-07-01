"""
nlp/similarity.py

Cosine similarity against the Embio reference embedding,
and keyword matching against a caller-supplied keyword list.
"""

import numpy as np

from config.relevance_profile import EMBIO_REFERENCE, PRIORITY_KEYWORDS

_REFERENCE_EMBEDDING: np.ndarray | None = None


def get_reference_embedding() -> np.ndarray:
    global _REFERENCE_EMBEDDING
    if _REFERENCE_EMBEDDING is None:
        from nlp.embedder import get_model
        _REFERENCE_EMBEDDING = get_model().encode(EMBIO_REFERENCE, normalize_embeddings=True)
    return _REFERENCE_EMBEDDING


def cosine_similarity(embedding_bytes: bytes) -> float:
    """
    Cosine similarity between a stored embedding blob and the Embio reference.
    Returns float in [0, 1] for normalised vectors. Higher = more relevant.
    """
    vector    = np.frombuffer(embedding_bytes, dtype=np.float32)
    reference = get_reference_embedding()
    return float(np.dot(vector, reference))


def keyword_score(
    text: str,
    keywords: list[str] | None = None,
) -> tuple[float, list[str]]:
    """
    Count keyword hits in lowercased text.

    Args:
        text:     Title + abstract/body concatenated.
        keywords: List of keywords to check. Defaults to PRIORITY_KEYWORDS.
                  scorer.py passes only the active subset.

    Returns:
        (normalised_score 0-1, list of matched keywords)
    """
    if keywords is None:
        keywords = PRIORITY_KEYWORDS
    lowered = text.lower()
    matched = [kw for kw in keywords if kw.lower() in lowered]
    score   = min(len(matched) / 5.0, 1.0)
    return score, matched
