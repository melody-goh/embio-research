import logging

import numpy as np
from sentence_transformers import SentenceTransformer

from config.relevance_profile import EMBIO_REFERENCE
from config.settings import EMBEDDING_ALLOW_DOWNLOAD, EMBEDDING_MODEL
from storage.db import get_connection

LOGGER = logging.getLogger(__name__)
_MODEL: SentenceTransformer | None = None
_REFERENCE_EMBEDDING: np.ndarray | None = None


def get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        if not EMBEDDING_ALLOW_DOWNLOAD:
            raise RuntimeError(
                "Embedding model not loaded and EMBEDDING_ALLOW_DOWNLOAD=0. "
                "Set EMBEDDING_ALLOW_DOWNLOAD=1 in your .env file."
            )
        LOGGER.info("Loading embedding model: %s", EMBEDDING_MODEL)
        _MODEL = SentenceTransformer(EMBEDDING_MODEL)
    return _MODEL


def get_reference_embedding() -> np.ndarray:
    global _REFERENCE_EMBEDDING
    if _REFERENCE_EMBEDDING is None:
        _REFERENCE_EMBEDDING = get_model().encode(EMBIO_REFERENCE, normalize_embeddings=True)
    return _REFERENCE_EMBEDDING


def embed_text(text: str) -> np.ndarray:
    return get_model().encode(text or "", normalize_embeddings=True)


def embed_all_pending() -> int:
    """
    Embed every article and trial that has no entry in the embeddings table.
    Returns count of newly embedded documents.
    """
    model = get_model()
    embedded = 0

    with get_connection() as con:
        article_rows = con.execute(
            """
            SELECT a.id, a.title, a.abstract
            FROM articles a
            LEFT JOIN embeddings e ON e.source_id = a.id AND e.source_type = 'article'
            WHERE e.source_id IS NULL OR e.model_name != ?
            """,
            [EMBEDDING_MODEL],
        ).fetchall()

    for source_id, title, abstract in article_rows:
        text = f"{title or ''}. {abstract or ''}".strip()
        vector = model.encode(text, normalize_embeddings=True)
        _store_embedding(source_id, "article", vector)
        embedded += 1

    with get_connection() as con:
        trial_rows = con.execute(
            """
            SELECT t.id, t.title, t.conditions, t.interventions
            FROM trials t
            LEFT JOIN embeddings e ON e.source_id = t.id AND e.source_type = 'trial'
            WHERE e.source_id IS NULL OR e.model_name != ?
            """,
            [EMBEDDING_MODEL],
        ).fetchall()

    for source_id, title, conditions, interventions in trial_rows:
        text = f"{title or ''}. {conditions or ''} {interventions or ''}".strip()
        vector = model.encode(text, normalize_embeddings=True)
        _store_embedding(source_id, "trial", vector)
        embedded += 1

    LOGGER.info("Embedded %s new documents", embedded)
    return embedded


def _store_embedding(source_id: str, source_type: str, vector: np.ndarray) -> None:
    with get_connection() as con:
        con.execute(
            """
            INSERT OR REPLACE INTO embeddings
                (source_id, source_type, embedding, model_name)
            VALUES (?, ?, ?, ?)
            """,
            [source_id, source_type, vector.astype(np.float32).tobytes(), EMBEDDING_MODEL],
        )


if __name__ == "__main__":
    import logging

    from storage.db import init_db

    logging.basicConfig(level=logging.INFO)
    init_db()
    n = embed_all_pending()
    print(f"Done. {n} documents embedded.")
