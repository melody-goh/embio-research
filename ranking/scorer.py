from datetime import date, datetime

import pandas as pd

from config.relevance_profile import SCORING_WEIGHTS
from nlp.embedder import embed_text, model_name, to_blob
from nlp.similarity import keyword_hits, semantic_similarity
from storage.db import feedback_weight, get_connection


def score_item(source_id: str, source_type: str, title: str, body: str, item_date) -> dict:
    text = f"{title}\n\n{body or ''}".strip()
    embedding = embed_text(text)
    semantic = max(0.0, semantic_similarity(embedding))
    hit_count, matched_keywords = keyword_hits(text)
    keyword_score = min(hit_count / 5.0, 1.0)
    recency_score = _recency_score(item_date)
    feedback = (feedback_weight(source_id, source_type) + 1.0) / 2.0

    weights = SCORING_WEIGHTS
    score = (
        weights["semantic"] * semantic
        + weights["keyword"] * keyword_score
        + weights["recency"] * recency_score
        + weights["feedback"] * feedback
    )
    return {
        "score": round(float(score), 4),
        "semantic_score": round(float(semantic), 4),
        "keyword_score": round(float(keyword_score), 4),
        "recency_score": round(float(recency_score), 4),
        "feedback_score": round(float(feedback), 4),
        "keyword_hits": hit_count,
        "matched_keywords": matched_keywords,
        "embedding": embedding,
    }


def score_all(limit: int | None = None) -> pd.DataFrame:
    con = get_connection()
    rows = con.execute(
        """
        SELECT id, 'article' AS source_type, title, abstract AS body, pub_date AS item_date
        FROM articles
        UNION ALL
        SELECT id, 'trial' AS source_type, title,
               concat_ws(' ', conditions, interventions, sponsor, phase, status) AS body,
               start_date AS item_date
        FROM trials
        """
    ).fetchdf()

    output = []
    for _, row in rows.iterrows():
        result = score_item(row.id, row.source_type, row.title, row.body, row.item_date)
        con.execute(
            """
            INSERT OR REPLACE INTO embeddings (source_id, source_type, embedding, model_name)
            VALUES (?, ?, ?, ?)
            """,
            [row.id, row.source_type, to_blob(result["embedding"]), model_name()],
        )
        result.pop("embedding")
        output.append({**row.to_dict(), **result})

    con.close()
    frame = pd.DataFrame(output).sort_values("score", ascending=False) if output else pd.DataFrame()
    if limit is not None and not frame.empty:
        return frame.head(limit)
    return frame


def _recency_score(value) -> float:
    parsed = _parse_date(value)
    if parsed is None:
        return 0.0
    days = max(0, (date.today() - parsed).days)
    return max(0.0, 1.0 - (days / 730.0))


def _parse_date(value) -> date | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return datetime.fromisoformat(str(value)).date()
    except ValueError:
        return None
