"""
storage/db.py

All database access for Embio Intelligence

Tables:
    articles        Research papers from all sources
    trials          Clinical trials
    embeddings      Embedding vectors
    summaries       LLM-generated summaries (cached)
    feedback        Raw thumbs up/down signals
    ml_labels       Versioned training labels for the classifier
    ingestion_log   Per-run audit log
    user_profile    Keyword states (active/muted/removed) and scoring weights
                    saved from the Relevance profile page.
"""

import json
from contextlib import contextmanager

import duckdb

from config.settings import DB_PATH


@contextmanager
def get_connection():
    """
    Gives a DuckDB connection guaranteed to close on exit.
    DuckDB allows only one writer at a time - leaked connections cause
    'Could not set lock' errors.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    try:
        yield con
    finally:
        con.close()


def init_db() -> None:
    """Create all tables, sequences, and indexes. Safe to call multiple times."""
    with get_connection() as con:

        con.execute("CREATE SEQUENCE IF NOT EXISTS feedback_id_seq      START 1")
        con.execute("CREATE SEQUENCE IF NOT EXISTS ingestion_log_id_seq START 1")
        con.execute("CREATE SEQUENCE IF NOT EXISTS ml_labels_id_seq     START 1")

        con.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                id          VARCHAR PRIMARY KEY,
                title       TEXT,
                abstract    TEXT,
                authors     TEXT,
                journal     TEXT,
                pub_date    DATE,
                url         TEXT,
                raw_json    TEXT,
                source      VARCHAR DEFAULT 'pubmed',
                fetched_at  TIMESTAMP DEFAULT now()
            )
        """)
        con.execute("ALTER TABLE articles ADD COLUMN IF NOT EXISTS source VARCHAR DEFAULT 'pubmed'")
        con.execute("ALTER TABLE articles ADD COLUMN IF NOT EXISTS fetched_at TIMESTAMP DEFAULT now()")

        con.execute("""
            CREATE TABLE IF NOT EXISTS trials (
                id              VARCHAR PRIMARY KEY,
                title           TEXT,
                status          TEXT,
                phase           TEXT,
                conditions      TEXT,
                interventions   TEXT,
                sponsor         TEXT,
                start_date      DATE,
                url             TEXT,
                raw_json        TEXT,
                source          VARCHAR DEFAULT 'clinicaltrials',
                fetched_at      TIMESTAMP DEFAULT now()
            )
        """)
        con.execute("ALTER TABLE trials ADD COLUMN IF NOT EXISTS source VARCHAR DEFAULT 'clinicaltrials'")
        con.execute("ALTER TABLE trials ADD COLUMN IF NOT EXISTS fetched_at TIMESTAMP DEFAULT now()")

        con.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                source_id    VARCHAR,
                source_type  VARCHAR,
                embedding    BLOB,
                model_name   VARCHAR,
                created_at   TIMESTAMP DEFAULT now(),
                PRIMARY KEY (source_id, source_type)
            )
        """)

        con.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                source_id       VARCHAR,
                source_type     VARCHAR,
                summary_text    TEXT,
                relevance_note  TEXT,
                relevance_score FLOAT,
                tags            TEXT,
                generated_at    TIMESTAMP DEFAULT now(),
                PRIMARY KEY (source_id, source_type)
            )
        """)

        con.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id          INTEGER DEFAULT nextval('feedback_id_seq') PRIMARY KEY,
                source_id   VARCHAR,
                source_type VARCHAR,
                signal      INTEGER,
                notes       TEXT,
                created_at  TIMESTAMP DEFAULT now()
            )
        """)

        con.execute("""
            CREATE TABLE IF NOT EXISTS ml_labels (
                id              INTEGER DEFAULT nextval('ml_labels_id_seq') PRIMARY KEY,
                source_id       VARCHAR,
                source_type     VARCHAR,
                label           INTEGER,
                label_version   INTEGER DEFAULT 1,
                label_source    VARCHAR,
                confidence      FLOAT DEFAULT 1.0,
                created_at      TIMESTAMP DEFAULT now()
            )
        """)

        con.execute("""
            CREATE TABLE IF NOT EXISTS ingestion_log (
                id              INTEGER DEFAULT nextval('ingestion_log_id_seq') PRIMARY KEY,
                source          VARCHAR,
                query           TEXT,
                new_records     INTEGER DEFAULT 0,
                updated_records INTEGER DEFAULT 0,
                error           TEXT,
                run_at          TIMESTAMP DEFAULT now()
            )
        """)

        # user_profile stores the keyword filter states and scoring weights saved from the Relevance profile page
        #
        # keyword_states: JSON dict {keyword: "active"|"muted"|"removed"}
        # scoring_weights: JSON dict {semantic, keyword, recency, feedback} floats 0-1
        #
        # Single row with id=1. Use save_user_profile() / load_user_profile()
        con.execute("""
            CREATE TABLE IF NOT EXISTS user_profile (
                id              INTEGER PRIMARY KEY,
                keyword_states  TEXT DEFAULT '{}',
                scoring_weights TEXT DEFAULT '{}',
                updated_at      TIMESTAMP DEFAULT now()
            )
        """)

        # Indexes
        con.execute("CREATE INDEX IF NOT EXISTS idx_articles_pub_date    ON articles(pub_date)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_articles_fetched     ON articles(fetched_at)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_articles_source      ON articles(source)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_trials_start_date    ON trials(start_date)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_trials_fetched       ON trials(fetched_at)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_feedback_source      ON feedback(source_id, source_type)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_summaries_source     ON summaries(source_id, source_type)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_source    ON embeddings(source_id, source_type)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_ml_labels_source     ON ml_labels(source_id, source_type)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_ml_labels_version    ON ml_labels(label_version)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_ingestion_log_run    ON ingestion_log(source, run_at)")


# ---------------------------------------------------------------------------
# Article / trial upserts
# ---------------------------------------------------------------------------

def upsert_article(article: dict) -> bool:
    """Insert or replace an article. Returns True if new record."""
    raw = article.get("raw_json", "")
    if isinstance(raw, dict):
        raw_str = json.dumps(raw, default=str)
    elif isinstance(raw, str):
        raw_str = raw
    else:
        raw_str = json.dumps(article, default=str)

    with get_connection() as con:
        existing = con.execute(
            "SELECT 1 FROM articles WHERE id = ?", [article["id"]]
        ).fetchone()
        con.execute(
            """
            INSERT OR REPLACE INTO articles
                (id, title, abstract, authors, journal, pub_date, url, raw_json, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                article["id"],
                article.get("title", ""),
                article.get("abstract", ""),
                article.get("authors", ""),
                article.get("journal", ""),
                article.get("pub_date"),
                article.get("url", ""),
                raw_str,
                article.get("source", "pubmed"),
            ],
        )
    return existing is None


def upsert_trial(trial: dict) -> bool:
    """Insert or replace a trial. Returns True if new record."""
    raw = trial.get("raw_json", "")
    if isinstance(raw, dict):
        raw_str = json.dumps(raw, default=str)
    elif isinstance(raw, str):
        raw_str = raw
    else:
        raw_str = json.dumps(trial, default=str)

    with get_connection() as con:
        existing = con.execute(
            "SELECT 1 FROM trials WHERE id = ?", [trial["id"]]
        ).fetchone()
        con.execute(
            """
            INSERT OR REPLACE INTO trials
                (id, title, status, phase, conditions, interventions,
                 sponsor, start_date, url, raw_json, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                trial["id"],
                trial.get("title", ""),
                trial.get("status", ""),
                trial.get("phase", ""),
                trial.get("conditions", ""),
                trial.get("interventions", ""),
                trial.get("sponsor", ""),
                trial.get("start_date"),
                trial.get("url", ""),
                raw_str,
                trial.get("source", "clinicaltrials"),
            ],
        )
    return existing is None


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

def store_feedback(source_id: str, source_type: str, signal: int, notes: str = "") -> None:
    with get_connection() as con:
        con.execute(
            "INSERT INTO feedback (source_id, source_type, signal, notes) VALUES (?, ?, ?, ?)",
            [source_id, source_type, 1 if signal > 0 else -1, notes],
        )


def feedback_weight(source_id: str, source_type: str) -> float:
    with get_connection() as con:
        rows = con.execute(
            "SELECT signal FROM feedback WHERE source_id = ? AND source_type = ?",
            [source_id, source_type],
        ).fetchall()
    if not rows:
        return 0.0
    total = sum(r[0] for r in rows)
    return max(-1.0, min(1.0, total / 3.0))


# ---------------------------------------------------------------------------
# User profile — keyword states and scoring weights
# ---------------------------------------------------------------------------

def load_user_profile() -> dict:
    """
    Return the saved user profile as:
        {
            "keyword_states":  {keyword: "active"|"muted"|"removed"},
            "scoring_weights": {semantic, keyword, recency, feedback}
        }
    Returns empty dicts if no profile has been saved yet.
    """
    with get_connection() as con:
        row = con.execute(
            "SELECT keyword_states, scoring_weights FROM user_profile WHERE id = 1"
        ).fetchone()
    if not row:
        return {"keyword_states": {}, "scoring_weights": {}}
    return {
        "keyword_states":  json.loads(row[0] or "{}"),
        "scoring_weights": json.loads(row[1] or "{}"),
    }


def save_user_profile(keyword_states: dict, scoring_weights: dict) -> None:
    """
    Persist the user's keyword filter states and scoring weight preferences.
    Upserts into the single-row user_profile table (id=1).
    """
    with get_connection() as con:
        con.execute(
            """
            INSERT OR REPLACE INTO user_profile (id, keyword_states, scoring_weights, updated_at)
            VALUES (1, ?, ?, now())
            """,
            [json.dumps(keyword_states), json.dumps(scoring_weights)],
        )


# ---------------------------------------------------------------------------
# ML labels
# ---------------------------------------------------------------------------

def upsert_ml_label(
    source_id: str,
    source_type: str,
    label: int,
    label_version: int = 1,
    label_source: str = "feedback_aggregate",
    confidence: float = 1.0,
) -> None:
    with get_connection() as con:
        con.execute(
            "DELETE FROM ml_labels WHERE source_id = ? AND source_type = ? AND label_version = ?",
            [source_id, source_type, label_version],
        )
        con.execute(
            """
            INSERT INTO ml_labels
                (source_id, source_type, label, label_version, label_source, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [source_id, source_type, label, label_version, label_source, confidence],
        )


def get_labelled_examples(label_version: int = 1) -> list[dict]:
    with get_connection() as con:
        rows = con.execute(
            "SELECT source_id, source_type, label, confidence FROM ml_labels WHERE label_version = ? ORDER BY created_at",
            [label_version],
        ).fetchall()
    return [{"source_id": r[0], "source_type": r[1], "label": r[2], "confidence": r[3]} for r in rows]


# ---------------------------------------------------------------------------
# Ingestion log
# ---------------------------------------------------------------------------

def log_ingestion(
    source: str,
    query: str,
    new_records: int = 0,
    updated_records: int = 0,
    error: str | None = None,
) -> None:
    with get_connection() as con:
        con.execute(
            "INSERT INTO ingestion_log (source, query, new_records, updated_records, error) VALUES (?, ?, ?, ?, ?)",
            [source, query, new_records, updated_records, error],
        )


def ingestion_summary(days: int = 30) -> list[dict]:
    with get_connection() as con:
        rows = con.execute(
            """
            SELECT source,
                   COUNT(*)             AS runs,
                   SUM(new_records)     AS total_new,
                   SUM(updated_records) AS total_updated,
                   MAX(run_at)          AS last_run
            FROM ingestion_log
            WHERE run_at >= now() - INTERVAL (?) DAY
              AND error IS NULL
            GROUP BY source
            ORDER BY total_new DESC
            """,
            [days],
        ).fetchall()
    return [
        {"source": r[0], "runs": r[1], "total_new": r[2], "total_updated": r[3], "last_run": r[4]}
        for r in rows
    ]


if __name__ == "__main__":
    init_db()
    print("Database initialised at", DB_PATH)
