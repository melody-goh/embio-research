from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Article:
    id: str
    title: str
    abstract: str
    authors: str
    journal: str
    pub_date: date | None
    url: str


@dataclass(frozen=True)
class Trial:
    id: str
    title: str
    status: str
    phase: str
    conditions: str
    interventions: str
    sponsor: str
    start_date: date | None
    url: str
