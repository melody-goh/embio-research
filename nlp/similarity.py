import numpy as np

from config.relevance_profile import EMBIO_REFERENCE, PRIORITY_KEYWORDS
from nlp.embedder import get_model

_REFERENCE_EMBEDDING: np.ndarray | None = None


def get_reference_embedding() -> np.ndarray:
    global _REFERENCE_EMBEDDING
    if _REFERENCE_EMBEDDING is None:
        _REFERENCE_EMBEDDING = get_model().encode(EMBIO_REFERENCE, normalize_embeddings=True)
    return _REFERENCE_EMBEDDING


def cosine_similarity(embedding_bytes: bytes) -> float:
    """
    Compute cosine similarity between a stored embedding blob and the
    Embio reference.
    """
    vector = np.frombuffer(embedding_bytes, dtype=np.float32)
    reference = get_reference_embedding()
    return float(np.dot(vector, reference))


def keyword_score(text: str) -> tuple[float, list[str]]:
    """
    Count priority keyword hits in lowercased text.
    Returns (normalised_score 0-1, list of matched keywords).
    """
    lowered = (text or "").lower()
    matches = [keyword for keyword in PRIORITY_KEYWORDS if keyword.lower() in lowered]
    score = min(len(matches) / 5.0, 1.0)
    return score, matches
