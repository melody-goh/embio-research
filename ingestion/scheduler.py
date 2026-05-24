import argparse
import logging
import time

from config.relevance_profile import CLINICALTRIALS_QUERIES, PUBMED_QUERIES
from config.settings import (
    BIORXIV_MAX_RESULTS_PER_QUERY,
    CLINICALTRIALS_MAX_RESULTS_PER_QUERY,
    PUBMED_MAX_RESULTS_PER_QUERY,
)
from ingestion.biorxiv import ingest_biorxiv
from ingestion.clinicaltrials import ingest_trials_query
from ingestion.pubmed import ingest_pubmed_query
from storage.db import init_db, log_ingestion


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def run_once() -> dict[str, int]:
    init_db()
    totals = {"articles": 0, "trials": 0}

    for query in PUBMED_QUERIES:
        try:
            counts = ingest_pubmed_query(query, max_results=PUBMED_MAX_RESULTS_PER_QUERY)
            totals["articles"] += counts["total"]
            log_ingestion("pubmed", query, counts["new"], counts["updated"])
        except Exception as e:
            LOGGER.exception("PubMed ingestion failed for query: %s", query)
            log_ingestion("pubmed", query, error=str(e))

    for query in CLINICALTRIALS_QUERIES:
        try:
            result = ingest_trials_query(query, max_results=CLINICALTRIALS_MAX_RESULTS_PER_QUERY)
            counts = result if isinstance(result, dict) else {"new": result, "updated": 0}
            totals["trials"] += counts["new"] + counts["updated"]
            log_ingestion("clinicaltrials", query, counts["new"], counts["updated"])
        except Exception as e:
            LOGGER.exception("ClinicalTrials ingestion failed for query: %s", query)
            log_ingestion("clinicaltrials", query, error=str(e))

    try:
        biorxiv_counts = ingest_biorxiv(max_results=BIORXIV_MAX_RESULTS_PER_QUERY)
        totals["articles"] += biorxiv_counts["total"]
        log_ingestion("biorxiv", "date-window", biorxiv_counts["new"], biorxiv_counts["updated"])
    except Exception as e:
        LOGGER.exception("bioRxiv/medRxiv ingestion failed")
        log_ingestion("biorxiv", "date-window", error=str(e))

    LOGGER.info("Ingestion complete: %s", totals)
    return totals


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
