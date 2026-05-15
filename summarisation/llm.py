import json
import logging
import re

from config.settings import OPENAI_API_KEY, SUMMARY_MODEL
from storage.db import get_connection

LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are a research analyst assistant for Embio Medical AB, a Swedish medtech
startup developing a flexible catheter platform for electroporation and
electrochemotherapy in pancreatic cancer applications.

Return strict JSON with:
summary: 2-3 plain-language sentences.
relevance_note: 1-2 sentences explaining why this matters to Embio.
tags: 3-5 concise topic tags.
""".strip()


def get_cached_summary(source_id: str, source_type: str) -> dict | None:
    con = get_connection()
    row = con.execute(
        """
        SELECT summary_text, relevance_note, tags
        FROM summaries
        WHERE source_id = ? AND source_type = ?
        """,
        [source_id, source_type],
    ).fetchone()
    con.close()
    if not row:
        return None
    return {"summary": row[0], "relevance_note": row[1], "tags": _parse_tags(row[2])}


def summarise(title: str, body: str, source_id: str, source_type: str, relevance_score: float) -> dict:
    cached = get_cached_summary(source_id, source_type)
    if cached:
        return cached

    if not OPENAI_API_KEY:
        summary = _fallback_summary(title, body)
    else:
        summary = _openai_summary(title, body)

    store_summary(source_id, source_type, summary, relevance_score)
    return summary


def store_summary(source_id: str, source_type: str, summary: dict, relevance_score: float) -> None:
    con = get_connection()
    con.execute(
        """
        INSERT OR REPLACE INTO summaries
        (source_id, source_type, summary_text, relevance_note, relevance_score, tags)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            source_id,
            source_type,
            summary.get("summary", ""),
            summary.get("relevance_note", ""),
            relevance_score,
            json.dumps(summary.get("tags", [])),
        ],
    )
    con.close()


def _openai_summary(title: str, body: str) -> dict:
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=SUMMARY_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Title: {title}\n\nDocument:\n{body}"},
        ],
        max_tokens=450,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or "{}"
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        LOGGER.warning("Model returned non-JSON summary; using fallback parser.")
        return _fallback_summary(title, content)


def _fallback_summary(title: str, body: str) -> dict:
    clean_body = re.sub(r"\s+", " ", body or "").strip()
    first_sentence = clean_body.split(". ")[0][:350] if clean_body else "No abstract or study summary was available."
    return {
        "summary": f"{title}. {first_sentence}".strip(),
        "relevance_note": "Potentially relevant because it overlaps Embio's electroporation, pancreatic cancer, catheter, ablation, or drug delivery focus areas.",
        "tags": _infer_tags(f"{title} {body}"),
    }


def _infer_tags(text: str) -> list[str]:
    lowered = text.lower()
    candidates = {
        "electroporation": "Electroporation",
        "electrochemotherapy": "Electrochemotherapy",
        "pancreatic": "Pancreatic cancer",
        "catheter": "Catheter",
        "ablation": "Ablation",
        "drug delivery": "Drug delivery",
        "trial": "Clinical trial",
    }
    tags = [label for needle, label in candidates.items() if needle in lowered]
    return tags[:5] or ["Research watch"]


def _parse_tags(raw: str) -> list[str]:
    try:
        value = json.loads(raw or "[]")
        return value if isinstance(value, list) else []
    except json.JSONDecodeError:
        return []
