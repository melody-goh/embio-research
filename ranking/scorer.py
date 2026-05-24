import logging
from datetime import date

import pandas as pd

from config.relevance_profile import SCORING_WEIGHTS
from nlp.similarity import cosine_similarity, keyword_score
from storage.db import feedback_weight, get_connection

LOGGER = logging.getLogger(__name__)
_RECENCY_DECAY_DAYS = 730


def score_all() -> pd.DataFrame:
    """
    Score every document that has an embedding.
    """
    with get_connection() as con:
        rows = con.execute(
            """
            SELECT
                e.source_id,
                e.source_type,
                e.embedding,
                COALESCE(a.title, t.title) AS title,
                COALESCE(
                    a.abstract,
                    concat_ws(' ', t.conditions, t.interventions)
                ) AS body,
                COALESCE(a.pub_date, t.start_date) AS item_date
            FROM embeddings e
            LEFT JOIN articles a ON a.id = e.source_id AND e.source_type = 'article'
            LEFT JOIN trials t ON t.id = e.source_id AND e.source_type = 'trial'
            """
        ).fetchall()

    if not rows:
        return pd.DataFrame()

    records = []
    today = date.today()

    for source_id, source_type, emb_bytes, title, body, item_date in rows:
        semantic = cosine_similarity(emb_bytes)
        kw_score, matched = keyword_score(f"{title or ''} {body or ''}")
        recency = _recency_score(item_date, today)
        feedback = feedback_weight(source_id, source_type)
        feedback_norm = (feedback + 1.0) / 2.0

        weights = SCORING_WEIGHTS
        composite = (
            weights["semantic"] * semantic
            + weights["keyword"] * kw_score
            + weights["recency"] * recency
            + weights["feedback"] * feedback_norm
        )

        records.append(
            {
                "id": source_id,
                "source_type": source_type,
                "title": title or "",
                "body": body or "",
                "item_date": item_date,
                "score": round(float(composite), 4),
                "semantic_score": round(float(semantic), 4),
                "keyword_score": round(float(kw_score), 4),
                "recency_score": round(float(recency), 4),
                "feedback_score": round(float(feedback_norm), 4),
                "matched_keywords": matched,
            }
        )

    return pd.DataFrame(records).sort_values("score", ascending=False).reset_index(drop=True)


def _recency_score(item_date, today: date) -> float:
    if item_date is None or pd.isna(item_date):
        return 0.5
    try:
        parsed = item_date if isinstance(item_date, date) else date.fromisoformat(str(item_date)[:10])
        days_old = (today - parsed).days
        return max(0.0, 1.0 - days_old / _RECENCY_DECAY_DAYS)
    except (ValueError, TypeError):
        return 0.5
