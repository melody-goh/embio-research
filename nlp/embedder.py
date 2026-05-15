import hashlib
import logging
import os

import numpy as np

from config.relevance_profile import EMBIO_REFERENCE
from config.settings import EMBEDDING_ALLOW_DOWNLOAD, EMBEDDING_MODEL

LOGGER = logging.getLogger(__name__)
_MODEL = None
_MODEL_FAILED = False
_REFERENCE_EMBEDDING = None
FALLBACK_DIMENSIONS = 384


def embed_text(text: str) -> np.ndarray:
    model = _load_model()
    if model is not None:
        return model.encode(text or "", normalize_embeddings=True)
    return _hash_embedding(text or "")


def reference_embedding() -> np.ndarray:
    global _REFERENCE_EMBEDDING
    if _REFERENCE_EMBEDDING is None:
        _REFERENCE_EMBEDDING = embed_text(EMBIO_REFERENCE)
    return _REFERENCE_EMBEDDING


def model_name() -> str:
    return EMBEDDING_MODEL if _load_model() is not None else "hashing-fallback"


def to_blob(embedding: np.ndarray) -> bytes:
    return np.asarray(embedding, dtype=np.float32).tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def _load_model():
    global _MODEL, _MODEL_FAILED
    if _MODEL is not None or _MODEL_FAILED:
        return _MODEL
    try:
        if not EMBEDDING_ALLOW_DOWNLOAD:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from sentence_transformers import SentenceTransformer

        kwargs = {} if EMBEDDING_ALLOW_DOWNLOAD else {"local_files_only": True}
        _MODEL = SentenceTransformer(EMBEDDING_MODEL, **kwargs)
    except Exception as exc:
        _MODEL_FAILED = True
        LOGGER.warning("Using deterministic fallback embeddings: %s", exc)
    return _MODEL


def _hash_embedding(text: str) -> np.ndarray:
    vector = np.zeros(FALLBACK_DIMENSIONS, dtype=np.float32)
    for token in text.lower().replace("/", " ").replace("-", " ").split():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % FALLBACK_DIMENSIONS
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm
