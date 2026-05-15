import numpy as np

from config.relevance_profile import PRIORITY_KEYWORDS
from nlp.embedder import reference_embedding


def semantic_similarity(embedding: np.ndarray) -> float:
    reference = reference_embedding()
    if embedding.size == 0 or reference.size == 0:
        return 0.0
    return float(np.dot(embedding, reference))


def keyword_hits(text: str) -> tuple[int, list[str]]:
    lowered = (text or "").lower()
    matches = [keyword for keyword in PRIORITY_KEYWORDS if keyword.lower() in lowered]
    return len(matches), matches
