"""
ranking/scorer.py

Composite relevance scoring

Respects the user's keyword profile saved in user_profile:
  active  — keyword contributes to keyword_score as normal
  muted   — keyword is ignored in scoring (not counted as a hit)
  removed — keyword match applies a 0.3× penalty to the composite score

If no profile is saved, all keywords default to active.
"""

import logging
from datetime import date

import pandas as pd

from config.relevance_profile import PRIORITY_KEYWORDS, SCORING_WEIGHTS
from nlp.similarity import cosine_similarity, keyword_score
from storage.db import feedback_weight, get_connection, load_user_profile

LOGGER = logging.getLogger(__name__)
_RECENCY_DECAY_DAYS = 730


def score_all() -> pd.DataFrame:
    """
    Score every document that has an embedding

    returns a DataFrame with columns:
        id, source_type, title, body, item_date, score,
        semantic_score, keyword_score, recency_score, matched_keywords
    """
    profile = load_user_profile()
    kw_states   = profile.get("keyword_states", {})
    saved_weights = profile.get("scoring_weights", {})

    weights = {**SCORING_WEIGHTS, **{k: v for k, v in saved_weights.items() if v is not None}}

    # Build active and removed keyword lists from user profile.
    # Keywords not in kw_states default to active.
    active_keywords  = [kw for kw in PRIORITY_KEYWORDS if kw_states.get(kw, "active") == "active"]
    removed_keywords = [kw for kw in PRIORITY_KEYWORDS if kw_states.get(kw, "active") == "removed"]

    with get_connection() as con:
        rows = con.execute("""
            SELECT
                e.source_id,
                e.source_type,
                e.embedding,
                COALESCE(a.title,  t.title)                                        AS title,
                COALESCE(a.abstract,
                         concat_ws(' ', t.conditions, t.interventions))            AS body,
                COALESCE(a.pub_date, t.start_date)                                 AS item_date
            FROM embeddings e
            LEFT JOIN articles a ON a.id = e.source_id AND e.source_type = 'article'
            LEFT JOIN trials   t ON t.id = e.source_id AND e.source_type = 'trial'
        """).fetchall()

    if not rows:
        return pd.DataFrame()

    records = []
    today = date.today()

    for source_id, source_type, emb_bytes, title, body, item_date in rows:
        text = f"{title or ''} {body or ''}"

        sem              = cosine_similarity(emb_bytes)
        kw_sc, matched   = keyword_score(text, active_keywords)
        rec              = _recency_score(item_date, today)
        fb               = feedback_weight(source_id, source_type)
        fb_norm          = (fb + 1.0) / 2.0

        composite = (
            weights["semantic"]  * sem    +
            weights["keyword"]   * kw_sc  +
            weights["recency"]   * rec    +
            weights["feedback"]  * fb_norm
        )

        # Apply removal penalty if any removed keyword appears in the text
        lowered = text.lower()
        if any(kw.lower() in lowered for kw in removed_keywords):
            composite *= 0.3

        records.append({
            "id":               source_id,
            "source_type":      source_type,
            "title":            title or "",
            "body":             body or "",
            "item_date":        item_date,
            "score":            round(composite, 4),
            "semantic_score":   round(sem,   4),
            "keyword_score":    round(kw_sc, 4),
            "recency_score":    round(rec,   4),
            "matched_keywords": matched,
        })

    df = pd.DataFrame(records)
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def _recency_score(item_date, today: date) -> float:
    if item_date is None:
        return 0.5
    try:
        d = item_date if isinstance(item_date, date) else date.fromisoformat(str(item_date)[:10])
        return max(0.0, 1.0 - (today - d).days / _RECENCY_DECAY_DAYS)
    except (ValueError, TypeError):
        return 0.5
