"""
ingestion/biorxiv.py

bioRxiv and medRxiv preprint ingestion via the official bioRxiv REST API

WHY PREPRINTS MATTER FOR EMBIO:
    In fast-moving fields like electroporation oncology, preprints appear
    6-18 months before peer-reviewed publication. A competitor filing a patent
    or running a trial often publishes a preprint first
    -> this source gives Embio early-warning signal that PubMed cannot

    medRxiv specifically covers clinical medicine — exactly where
    electroporation trials will first appear in the literature.

API reference: https://api.biorxiv.org/
- No API key required
- Rate limit: not formally published; 1 req/s is safe and respectful
- 2 servers: biorxiv.org (life sciences) and medrxiv.org (clinical medicine)
  both r queried, but medRxiv is higher priority for Embio.

ID FORMAT:
    Preprints don't have PMIDs. We use "biorxiv_{doi_suffix}" as the ID,
    where doi_suffix is the numeric part of the DOI (e.g. "10.1101/2024.03.15.123456"
    becomes "biorxiv_2024.03.15.123456"). This is stable across preprint versions.
"""

import logging
import time
from datetime import date, timedelta

import requests

from config.settings import BIORXIV_MAX_RESULTS_PER_QUERY, HTTP_TIMEOUT_SECONDS
from storage.db import upsert_article

LOGGER = logging.getLogger(__name__)
BASE_URL = "https://api.biorxiv.org/details"
_RATE_SLEEP = 1.0  # conservative — API has no published limit


# ---------------------------------------------------------------------------
# Search terms mapped to server
# ---------------------------------------------------------------------------
# bioRxiv API does not support keyword search 
#  — it only supports date-range fetches
#  -> fetch recent content and filter by keyword locally
# This is intentional API design on their part, not a missing feature
#
# Strategy: fetch the last N days from medRxiv (clinical) and bioRxiv (bio), then keep only records whose title or abstract contains at least one of our keywords
# -> efficient because we only do it for recent content.

BIORXIV_FILTER_KEYWORDS = [
    "electroporation",
    "electrochemotherapy",
    "irreversible electroporation",
    "pulsed electric field",
    "calcium electroporation",
    "pancreatic cancer",
    "pancreatic adenocarcinoma",
    "pdac",
    "tumor ablation",
    "catheter ablation",
]


def fetch_recent(
    server: str = "medrxiv",
    days_back: int = 90,
    max_results: int = 50,
) -> list[dict]:
    """
    Fetch recent preprints from bioRxiv or medRxiv for a date window,
    then filter locally by keyword relevance.

    Args:
        server:      "biorxiv" or "medrxiv"
        days_back:   How many days of preprints to fetch (rolling window).
        max_results: Maximum number of filtered results to return.

    Returns:
        List of parsed article dicts ready for upsert_article().
    """
    end_date   = date.today()
    start_date = end_date - timedelta(days=days_back)
    interval   = f"{start_date.isoformat()}/{end_date.isoformat()}"

    # API paginates in batches of 100 (their max per call)
    # -> fetch pages until we have enough keyword-matching results or run out
    results = []
    offset  = 0
    page_size = 100

    while len(results) < max_results:
        url = f"{BASE_URL}/{server}/{interval}/{offset}/json"
        try:
            response = _get_with_retry(url, {})
        except requests.RequestException:
            LOGGER.warning("bioRxiv fetch stopped at offset %s due to request error", offset)
            break

        data = response.json()
        collection = data.get("collection", [])
        if not collection:
            break  # no more results in this date window

        for item in collection:
            if _matches_keywords(item):
                parsed = _parse_preprint(item, server)
                if parsed["id"]:
                    results.append(parsed)
                    if len(results) >= max_results:
                        break

        total_available = int(data.get("messages", [{}])[0].get("total", 0))
        offset += page_size
        if offset >= total_available:
            break

        time.sleep(_RATE_SLEEP)

    LOGGER.info(
        "bioRxiv/%s: found %s relevant preprints in last %s days",
        server, len(results), days_back,
    )
    return results


def ingest_biorxiv(days_back: int = 90, max_results: int = 50) -> dict:
    """
    Ingest recent preprints from both medRxiv and bioRxiv.
    medRxiv is fetched first as it's higher priority for clinical content.

    Returns:
        {"new": int, "updated": int, "total": int}
    """
    new = updated = 0

    for server in ("medrxiv", "biorxiv"):
        articles = fetch_recent(server=server, days_back=days_back, max_results=max_results)
        for article in articles:
            is_new = upsert_article(article)
            if is_new:
                new += 1
            else:
                updated += 1
        time.sleep(_RATE_SLEEP)

    counts = {"new": new, "updated": updated, "total": new + updated}
    LOGGER.info("bioRxiv/medRxiv ingestion: %s new, %s updated", new, updated)
    return counts


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _matches_keywords(item: dict) -> bool:
    """Return True if title or abstract contains at least one filter keyword."""
    text = (
        (item.get("title") or "") + " " + (item.get("abstract") or "")
    ).lower()
    return any(kw.lower() in text for kw in BIORXIV_FILTER_KEYWORDS)


def _parse_preprint(item: dict, server: str) -> dict:
    doi = item.get("doi", "")
    # Strip the "10.1101/" prefix to get the stable numeric suffix
    doi_suffix = doi.replace("10.1101/", "").strip("/")
    record_id  = f"biorxiv_{doi_suffix}" if doi_suffix else ""

    # Use latest version's date if available
    pub_date = item.get("date") or item.get("date_revised") or None

    return {
        "id":       record_id,
        "title":    (item.get("title") or "").strip(),
        "abstract": (item.get("abstract") or "").strip(),
        "authors":  (item.get("authors") or "").strip(),
        "journal":  f"{server.capitalize()} preprint",
        "pub_date": pub_date,
        "url":      f"https://doi.org/{doi}" if doi else "",
        "raw_json": item,  # dict — db.py handles serialisation
        "source": server,
    }


def _get_with_retry(url: str, params: dict, retries: int = 3) -> requests.Response:
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=HTTP_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            if attempt == retries - 1:
                LOGGER.error("bioRxiv request failed after %s attempts: %s", retries, exc)
                raise
            wait = 2 ** attempt
            LOGGER.warning("bioRxiv request failed (%s). Retrying in %ss...", exc, wait)
            time.sleep(wait)
    raise RuntimeError("unreachable")
