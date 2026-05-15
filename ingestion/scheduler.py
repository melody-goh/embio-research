import argparse
import logging
import time

from config.relevance_profile import CLINICALTRIALS_QUERIES, PUBMED_QUERIES
from config.settings import PUBMED_MAX_RESULTS_PER_QUERY, TRIALS_MAX_RESULTS_PER_QUERY
from ingestion.clinicaltrials import ingest_trials_query
from ingestion.pubmed import ingest_pubmed_query
from storage.db import init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def run_once() -> dict[str, int]:
    init_db()
    counts = {"articles": 0, "trials": 0}

    for query in PUBMED_QUERIES:
        try:
            counts["articles"] += ingest_pubmed_query(query, max_results=PUBMED_MAX_RESULTS_PER_QUERY)
        except Exception:
            LOGGER.exception("PubMed ingestion failed for query: %s", query)

    for query in CLINICALTRIALS_QUERIES:
        try:
            counts["trials"] += ingest_trials_query(query, max_results=TRIALS_MAX_RESULTS_PER_QUERY)
        except Exception:
            LOGGER.exception("ClinicalTrials ingestion failed for query: %s", query)

    LOGGER.info("Ingestion complete: %s", counts)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run ingestion once and exit.")
    parser.add_argument("--every-hours", type=float, default=24.0, help="Interval for continuous mode.")
    args = parser.parse_args()

    if args.once:
        run_once()
        return

    while True:
        run_once()
        time.sleep(args.every_hours * 3600)


if __name__ == "__main__":
    main()
