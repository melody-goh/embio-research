"""
storage/db.py

All database access for Embio Intelligence.

SCHEMA OVERVIEW — designed to support a future ML relevance classifier:

    articles        Raw research papers from all sources (PubMed, Europe PMC, bioRxiv)
    trials          Clinical trials from ClinicalTrials.gov (+ ICTRP when enabled)
    embeddings      Embedding vectors for every article and trial
    summaries       LLM-generated summaries and relevance notes (cached — one per doc)
    feedback        Raw user signals: thumbs up/down per document
    ml_labels       Derived training labels for the ML classifier.
                    Separate from feedback because labels are processed/versioned,
                    feedback is raw. One row per (source_id, source_type, label_version).
    ingestion_log   One row per ingestion run per source. Tracks how much data
                    came in, from where, and when. Essential for debugging and
                    for the dashboard's data-health panel.

WHY ml_labels IS SEPARATE FROM feedback:
    feedback holds raw signals that may be noisy (misclicks, changed opinions).
    ml_labels holds processed, versioned labels used for training. When you
    retrain the classifier with a better labelling strategy, you bump the version
    and keep old labels for comparison. You cannot reconstruct versioned
    training sets from raw feedback alone.

STORAGE FORMAT FOR EMBEDDINGS:
    numpy arrays stored as BLOB via array.tobytes().
    Retrieve with: np.frombuffer(blob, dtype=np.float32)
    Faster and more compact than JSON-encoding float arrays.
"""

import json
from contextlib import contextmanager

import duckdb

from config.settings import DB_PATH


@contextmanager
def get_connection():
    """
    Yields a DuckDB connection guaranteed to close on exit, even on exception.
    DuckDB allows only one writer at a time — leaked connections cause
    cryptic 'Could not set lock' errors.

    Usage:
        with get_connection() as con:
            con.execute(...)
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

        # DuckDB does not auto-increment INTEGER PRIMARY KEY like SQLite.
        # Sequences are required for auto-incrementing integer PKs.
        con.execute("CREATE SEQUENCE IF NOT EXISTS feedback_id_seq      START 1")
        con.execute("CREATE SEQUENCE IF NOT EXISTS ingestion_log_id_seq START 1")
        con.execute("CREATE SEQUENCE IF NOT EXISTS ml_labels_id_seq     START 1")

        # --- Articles ---
        # source column: 'pubmed' | 'europepmc' | 'biorxiv' | 'medrxiv'
        # Tracks which ingestion pipeline produced each record.
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

        # --- Trials ---
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

        # --- Embeddings ---
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

        # --- Summaries (LLM cache) ---
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

        # --- Feedback (raw user signals) ---
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

        # --- ML training labels ---
        # Populated by running: python -m ml.prepare_labels
        # NOT written to during normal ingestion or dashboard use.
        #
        # label:         1 = relevant, 0 = not relevant
        # label_version: bump when changing labelling strategy, preserving history
        # label_source:  'feedback_aggregate' | 'manual' | 'heuristic'
        # confidence:    0.0-1.0. Feedback-derived labels get lower confidence
        #                than manually verified ones. Used as sample weights
        #                during classifier training.
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

        # --- Ingestion audit log ---
        # One row per (source, query, run). Answers:
        #   "Did anything come in today?"
        #   "Which source produces the most new content?"
        #   "Did a query fail silently?"
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

        # --- Indexes ---
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
# Article helpers
# ---------------------------------------------------------------------------

def upsert_article(article: dict) -> bool:
    """
    Insert or replace an article.
    Returns True if new record, False if existing record was refreshed.
    """
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
    """Insert or replace a trial. Returns True if new."""
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
# Feedback helpers
# ---------------------------------------------------------------------------

def store_feedback(source_id: str, source_type: str, signal: int, notes: str = "") -> None:
    """Record a thumbs-up (+1) or thumbs-down (-1) signal."""
    with get_connection() as con:
        con.execute(
            """
            INSERT INTO feedback (source_id, source_type, signal, notes)
            VALUES (?, ?, ?, ?)
            """,
            [source_id, source_type, 1 if signal > 0 else -1, notes],
        )


def feedback_weight(source_id: str, source_type: str) -> float:
    """
    Aggregate feedback into a weight in [-1.0, 1.0].
    Three unanimous votes saturate the scale. No feedback returns 0.0.
    """
    with get_connection() as con:
        rows = con.execute(
            "SELECT signal FROM feedback WHERE source_id = ? AND source_type = ?",
            [source_id, source_type],
        ).fetchall()

    if not rows:
        return 0.0
    total = sum(row[0] for row in rows)
    return max(-1.0, min(1.0, total / 3.0))


# ---------------------------------------------------------------------------
# ML label helpers
# ---------------------------------------------------------------------------

def upsert_ml_label(
    source_id: str,
    source_type: str,
    label: int,
    label_version: int = 1,
    label_source: str = "feedback_aggregate",
    confidence: float = 1.0,
) -> None:
    """
    Insert or replace a training label for the ML classifier.

    Called by ml/prepare_labels.py — NOT called during normal ingestion.
    Labels are derived from aggregated feedback, not written directly by users.
    """
    with get_connection() as con:
        con.execute(
            """
            DELETE FROM ml_labels
            WHERE source_id = ? AND source_type = ? AND label_version = ?
            """,
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
    """
    Return all labelled examples for a given label version.
    Used by ml/train.py to build the training set.
    """
    with get_connection() as con:
        rows = con.execute(
            """
            SELECT source_id, source_type, label, confidence
            FROM ml_labels
            WHERE label_version = ?
            ORDER BY created_at
            """,
            [label_version],
        ).fetchall()

    return [
        {"source_id": r[0], "source_type": r[1], "label": r[2], "confidence": r[3]}
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Ingestion log helpers
# ---------------------------------------------------------------------------

def log_ingestion(
    source: str,
    query: str,
    new_records: int = 0,
    updated_records: int = 0,
    error: str | None = None,
) -> None:
    """Record the outcome of one ingestion query run."""
    with get_connection() as con:
        con.execute(
            """
            INSERT INTO ingestion_log (source, query, new_records, updated_records, error)
            VALUES (?, ?, ?, ?, ?)
            """,
            [source, query, new_records, updated_records, error],
        )


def ingestion_summary(days: int = 30) -> list[dict]:
    """
    Per-source ingestion totals for the last N days.
    Used in the dashboard's data-health panel.
    """
    with get_connection() as con:
        rows = con.execute(
            """
            SELECT source,
                   COUNT(*)               AS runs,
                   SUM(new_records)       AS total_new,
                   SUM(updated_records)   AS total_updated,
                   MAX(run_at)            AS last_run
            FROM ingestion_log
            WHERE run_at >= now() - INTERVAL (?) DAY
              AND error IS NULL
            GROUP BY source
            ORDER BY total_new DESC
            """,
            [days],
        ).fetchall()

    return [
        {
            "source": r[0], "runs": r[1],
            "total_new": r[2], "total_updated": r[3],
            "last_run": r[4],
        }
        for r in rows
    ]


if __name__ == "__main__":
    init_db()
    print("Database initialised at", DB_PATH)
