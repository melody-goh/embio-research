import duckdb
import json

from config.settings import DB_PATH


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(DB_PATH))

def init_db():
    con = get_connection()

    con.execute("""
    CREATE TABLE IF NOT EXISTS articles (
        id VARCHAR PRIMARY KEY,
        title TEXT,
        abstract TEXT,
        authors TEXT,
        journal TEXT,
        pub_date DATE,
        url TEXT,
        raw_json TEXT,
        fetched_at TIMESTAMP DEFAULT now()
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS trials (
        id VARCHAR PRIMARY KEY,
        title TEXT,
        status TEXT,
        phase TEXT,
        conditions TEXT,
        interventions TEXT,
        sponsor TEXT,
        start_date DATE,
        url TEXT,
        raw_json TEXT,
        fetched_at TIMESTAMP DEFAULT now()
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS embeddings (
        source_id VARCHAR,
        source_type VARCHAR,
        embedding BLOB,
        model_name VARCHAR,
        created_at TIMESTAMP DEFAULT now(),
        PRIMARY KEY (source_id, source_type)
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY,
        source_id VARCHAR,
        source_type VARCHAR,
        signal INTEGER,
        notes TEXT,
        created_at TIMESTAMP DEFAULT now()
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS summaries (
        source_id VARCHAR,
        source_type VARCHAR,
        summary_text TEXT,
        relevance_note TEXT,
        relevance_score FLOAT,
        tags TEXT,
        generated_at TIMESTAMP DEFAULT now(),
        PRIMARY KEY (source_id, source_type)
    );
    """)

    con.close()


def upsert_article(article: dict) -> None:
    con = get_connection()
    con.execute(
        """
        INSERT OR REPLACE INTO articles
        (id, title, abstract, authors, journal, pub_date, url, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            article["id"],
            article.get("title", ""),
            article.get("abstract", ""),
            article.get("authors", ""),
            article.get("journal", ""),
            article.get("pub_date"),
            article.get("url", ""),
            json.dumps(article.get("raw_json", article), default=str),
        ],
    )
    con.close()


def upsert_trial(trial: dict) -> None:
    con = get_connection()
    con.execute(
        """
        INSERT OR REPLACE INTO trials
        (id, title, status, phase, conditions, interventions, sponsor, start_date, url, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            json.dumps(trial.get("raw_json", trial), default=str),
        ],
    )
    con.close()


def store_feedback(source_id: str, source_type: str, signal: int, notes: str = "") -> None:
    con = get_connection()
    con.execute(
        """
        INSERT INTO feedback (source_id, source_type, signal, notes)
        VALUES (?, ?, ?, ?)
        """,
        [source_id, source_type, 1 if signal > 0 else -1, notes],
    )
    con.close()


def feedback_weight(source_id: str, source_type: str) -> float:
    con = get_connection()
    rows = con.execute(
        "SELECT signal FROM feedback WHERE source_id = ? AND source_type = ?",
        [source_id, source_type],
    ).fetchall()
    con.close()
    if not rows:
        return 0.0
    total = sum(row[0] for row in rows)
    return max(-1.0, min(1.0, total / 3.0))

if __name__ == "__main__":
    init_db()
    print("Database initialised.")
