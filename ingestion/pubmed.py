import logging
import time
import xml.etree.ElementTree as ET
from datetime import date

import requests

from config.settings import HTTP_TIMEOUT_SECONDS, NCBI_API_KEY, NCBI_EMAIL
from storage.db import upsert_article

LOGGER = logging.getLogger(__name__)
BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _base_params() -> dict:
    params = {"tool": "embio_intelligence"}
    if NCBI_EMAIL:
        params["email"] = NCBI_EMAIL
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    return params


def search_pubmed(query: str, max_results: int = 20) -> list[str]:
    params = {
        **_base_params(),
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "sort": "pub_date",
        "retmode": "json",
    }
    response = requests.get(f"{BASE_URL}/esearch.fcgi", params=params, timeout=HTTP_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()["esearchresult"].get("idlist", [])


def fetch_articles(pmids: list[str]) -> list[dict]:
    if not pmids:
        return []
    params = {
        **_base_params(),
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
    }
    response = requests.get(f"{BASE_URL}/efetch.fcgi", params=params, timeout=HTTP_TIMEOUT_SECONDS)
    response.raise_for_status()
    root = ET.fromstring(response.text)
    return [_parse_article(node) for node in root.findall(".//PubmedArticle")]


def ingest_pubmed_query(query: str, max_results: int = 20) -> int:
    pmids = search_pubmed(query, max_results=max_results)
    time.sleep(0.12 if NCBI_API_KEY else 0.35)
    articles = fetch_articles(pmids)
    for article in articles:
        upsert_article(article)
    LOGGER.info("Stored %s PubMed articles for query: %s", len(articles), query)
    return len(articles)


def _parse_article(node: ET.Element) -> dict:
    pmid = _text(node, ".//PMID")
    title = " ".join(_text(node, ".//ArticleTitle").split())
    abstract_parts = [
        " ".join(part.itertext()).strip()
        for part in node.findall(".//Abstract/AbstractText")
        if " ".join(part.itertext()).strip()
    ]
    abstract = "\n".join(abstract_parts)
    authors = _parse_authors(node)
    journal = _text(node, ".//Journal/Title")
    pub_date = _parse_pub_date(node)
    return {
        "id": pmid,
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "journal": journal,
        "pub_date": pub_date,
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        "raw_json": ET.tostring(node, encoding="unicode"),
    }


def _parse_authors(node: ET.Element) -> str:
    names = []
    for author in node.findall(".//AuthorList/Author"):
        last = _text(author, "LastName")
        fore = _text(author, "ForeName")
        collective = _text(author, "CollectiveName")
        if collective:
            names.append(collective)
        elif last or fore:
            names.append(f"{fore} {last}".strip())
    return ", ".join(names)


def _parse_pub_date(node: ET.Element) -> str | None:
    date_node = node.find(".//Article/Journal/JournalIssue/PubDate")
    if date_node is None:
        return None
    year = _text(date_node, "Year")
    if not year:
        return None
    month = _month_to_int(_text(date_node, "Month")) or 1
    day = int(_text(date_node, "Day") or 1)
    try:
        return date(int(year), month, day).isoformat()
    except ValueError:
        return date(int(year), 1, 1).isoformat()


def _month_to_int(value: str) -> int | None:
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        months = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        }
        return months.get(value[:3].lower())


def _text(node: ET.Element, path: str) -> str:
    found = node.find(path)
    return "".join(found.itertext()).strip() if found is not None else ""
