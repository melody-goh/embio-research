# Embio Intelligence

Embio Intelligence is a local research radar for Embio Medical. It ingests PubMed papers and
ClinicalTrials.gov studies, ranks them against Embio's electroporation and pancreatic cancer focus,
summarises high-value items, and captures feedback so the signal can improve over time.

## What It Does

- Fetches focused evidence from PubMed and ClinicalTrials.gov.
- Stores raw records, embeddings, summaries, and feedback in one DuckDB file.
- Scores each document with semantic similarity, priority keyword hits, recency, and feedback.
- Presents a Streamlit dashboard with filters, highlights, summaries, source links, and relevance votes.
- Works without cloud infrastructure. OpenAI is optional and used only for cached summaries.

## Quick Start

```bash
python -m storage.db
python -m ingestion.scheduler --once
streamlit run dashboard/app.py
```

The dashboard will be available at the local URL printed by Streamlit.

## Environment

Create a `.env` file when you want API-backed summaries or higher PubMed limits:

```bash
OPENAI_API_KEY=...
NCBI_API_KEY=...
NCBI_EMAIL=you@example.com
```

Optional settings:

```bash
PUBMED_MAX_RESULTS_PER_QUERY=20
TRIALS_MAX_RESULTS_PER_QUERY=20
SUMMARY_MIN_SCORE=0.55
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_ALLOW_DOWNLOAD=1
```

By default, embedding model downloads are disabled. If the SentenceTransformer model is not already
cached locally, the app uses a deterministic hashing fallback so scoring still works offline.

## Architecture

The project follows the deck's layered architecture. Modules communicate through DuckDB rather than
calling across layers unnecessarily:

- `config/`: settings and Embio-specific relevance profile.
- `ingestion/`: PubMed and ClinicalTrials.gov clients.
- `storage/`: DuckDB schema and write helpers.
- `nlp/`: embeddings, semantic similarity, and keyword matching.
- `ranking/`: composite relevance scoring.
- `summarisation/`: cached OpenAI summaries with a no-key fallback.
- `feedback/`: relevance signal capture.
- `dashboard/`: Streamlit product surface.

## Development Workflow

```bash
# Compile/import sanity check
python -m compileall -q config storage ingestion nlp ranking summarisation feedback dashboard

# Run ingestion once
python -m ingestion.scheduler --once

# Launch product UI
streamlit run dashboard/app.py
```

The database lives at `data/embio.db`.
