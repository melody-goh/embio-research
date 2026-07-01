import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data"
DB_PATH      = DATA_DIR / "embio.db"

NCBI_API_KEY   = os.getenv("NCBI_API_KEY",   "")
NCBI_EMAIL     = os.getenv("NCBI_EMAIL",     "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

PUBMED_MAX_RESULTS_PER_QUERY          = int(os.getenv("PUBMED_MAX_RESULTS_PER_QUERY",          "100"))
CLINICALTRIALS_MAX_RESULTS_PER_QUERY  = int(os.getenv("CLINICALTRIALS_MAX_RESULTS_PER_QUERY",  "30"))
EUROPEPMC_MAX_RESULTS_PER_QUERY       = int(os.getenv("EUROPEPMC_MAX_RESULTS_PER_QUERY",       "100"))
BIORXIV_MAX_RESULTS_PER_QUERY         = int(os.getenv("BIORXIV_MAX_RESULTS_PER_QUERY",         "25"))

BIORXIV_ENABLED = os.getenv("BIORXIV_ENABLED", "1") == "1"
ICTRP_ENABLED   = os.getenv("ICTRP_ENABLED",   "0") == "1"

EMBEDDING_MODEL          = os.getenv("EMBEDDING_MODEL",          "all-MiniLM-L6-v2")
EMBEDDING_ALLOW_DOWNLOAD = os.getenv("EMBEDDING_ALLOW_DOWNLOAD", "1") == "1"  # fixed from "0"

SUMMARY_MODEL     = os.getenv("SUMMARY_MODEL",     "gpt-4o-mini")
SUMMARY_MIN_SCORE = float(os.getenv("SUMMARY_MIN_SCORE", "0.55"))

# Minimum number of feedback signals before the ML classifier is trained.
# Below this the system uses rule-based scoring only.
ML_MIN_FEEDBACK_SAMPLES = int(os.getenv("ML_MIN_FEEDBACK_SAMPLES", "30"))

HTTP_TIMEOUT_SECONDS = int(os.getenv("HTTP_TIMEOUT_SECONDS", "30"))
