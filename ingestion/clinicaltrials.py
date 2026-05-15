import logging

import requests

from config.settings import HTTP_TIMEOUT_SECONDS
from storage.db import upsert_trial

LOGGER = logging.getLogger(__name__)
CT_BASE = "https://clinicaltrials.gov/api/v2/studies"


def search_trials(query: str, max_results: int = 20) -> list[dict]:
    response = requests.get(
        CT_BASE,
        params={"query.term": query, "pageSize": max_results, "format": "json"},
        timeout=HTTP_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json().get("studies", [])


def ingest_trials_query(query: str, max_results: int = 20) -> int:
    studies = search_trials(query, max_results=max_results)
    trials = [_parse_trial(study) for study in studies]
    for trial in trials:
        upsert_trial(trial)
    LOGGER.info("Stored %s ClinicalTrials studies for query: %s", len(trials), query)
    return len(trials)


def _parse_trial(study: dict) -> dict:
    protocol = study.get("protocolSection", {})
    identification = protocol.get("identificationModule", {})
    status = protocol.get("statusModule", {})
    design = protocol.get("designModule", {})
    conditions = protocol.get("conditionsModule", {})
    arms = protocol.get("armsInterventionsModule", {})
    sponsors = protocol.get("sponsorCollaboratorsModule", {})

    nct_id = identification.get("nctId", "")
    interventions = arms.get("interventions", [])
    lead_sponsor = sponsors.get("leadSponsor", {})

    return {
        "id": nct_id,
        "title": identification.get("briefTitle") or identification.get("officialTitle") or "",
        "status": status.get("overallStatus", ""),
        "phase": ", ".join(design.get("phases", [])),
        "conditions": ", ".join(conditions.get("conditions", [])),
        "interventions": ", ".join(item.get("name", "") for item in interventions if item.get("name")),
        "sponsor": lead_sponsor.get("name", ""),
        "start_date": _normalise_date(status.get("startDateStruct", {}).get("date")),
        "url": f"https://clinicaltrials.gov/study/{nct_id}" if nct_id else "",
        "raw_json": study,
    }


def _normalise_date(value: str | None) -> str | None:
    if not value:
        return None
    parts = value.split("-")
    if len(parts) == 1:
        return f"{parts[0]}-01-01"
    if len(parts) == 2:
        return f"{parts[0]}-{parts[1]}-01"
    return value
