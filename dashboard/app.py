"""
dashboard/app.py  —  Embio Intelligence
Clean, purposeful design using brand palette:
  Periwinkle  #C5CAFB
  Wisteria    #9CA1FF
  Blue Slate  #396070
  Black       #000000
  White       #FFFFFF
"""

import sys
from collections import Counter
from datetime import date, timedelta
from html import escape
from pathlib import Path
from urllib.parse import quote

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
LOGO_PATH = PROJECT_ROOT / "dashboard" / "assets" / "embio-black-logo.png"
ICON_PATH = PROJECT_ROOT / "dashboard" / "assets" / "embio-black-icon.png"

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config.relevance_profile import (
    DEFAULT_MIN_RELEVANCE,
    PRIORITY_KEYWORDS,
    SCORING_WEIGHTS,
)
from dashboard.reporting import build_weekly_pdf
from feedback.store import record_feedback
from ingestion.scheduler import run_once
from nlp.embedder import embed_all_pending
from ranking.scorer import score_all
from storage.db import (
    get_connection,
    init_db,
    ingestion_summary,
    load_user_profile,
    save_user_profile,
)
from summarisation.llm import get_cached_summary, summarise


# ─────────────────────────────────────────────
# KEYWORD CLUSTERS
# ─────────────────────────────────────────────

KEYWORD_CLUSTERS = {
    "Electroporation core": [
        "electroporation","irreversible electroporation","reversible electroporation",
        "electrochemotherapy","calcium electroporation","pulsed electric field","pef",
        "non-thermal ablation","low voltage electroporation","electroporation parameters","nanoknife",
    ],
    "Catheter / intraductal": [
        "electroporation catheter","catheter-based electroporation","flexible catheter","catheter",
        "intraductal catheter","bipolar electrode catheter","ring electrode catheter","microcatheter",
        "intraductal electroporation","pancreatic duct electroporation","ercp electroporation","ercp guided ablation",
    ],
    "Oncology / clinical": [
        "pancreatic cancer","pancreatic ductal adenocarcinoma","pdac","pancreatic adenocarcinoma",
        "locally advanced pancreatic cancer","cholangiocarcinoma","pancreatic intraepithelial neoplasia",
        "panin","ablation","tumor ablation","minimally invasive","first-in-human",
        "interventional oncology","locoregional therapy","endoscopic ablation","biliary electroporation",
    ],
    "Drug delivery": [
        "drug delivery","localized drug delivery","intratumoral drug delivery",
        "pancreatic duct drug delivery","catheter-based drug delivery","endoluminal drug delivery",
    ],
    "Biomarkers / diagnostics": [
        "pancreatic juice biomarker","pancreatic liquid biopsy","pancreatic juice","pancreatic cyst fluid",
        "kras pancreatic","tp53 pancreatic","circulating tumor dna","ctdna","exosomes pancreatic",
        "early pancreatic cancer detection","pdac early diagnosis","ercp pancreatic juice","pancreatic duct sampling",
    ],
    "Adjacent / strategic": [
        "eus-guided","endoscopic ultrasound","clinical feasibility","safety","organoid pancreatic cancer",
    ],
}

TAG_COLORS = {
    "pancreatic cancer":"red","pancreatic adenocarcinoma":"red","pancreatic ductal adenocarcinoma":"red",
    "pdac":"red","locally advanced pancreatic cancer":"red","cholangiocarcinoma":"red","panin":"red",
    "electroporation":"blue","electrochemotherapy":"blue","irreversible electroporation":"blue",
    "calcium electroporation":"blue","pulsed electric field":"blue","nanoknife":"blue",
    "ablation":"blue","tumor ablation":"blue",
    "electroporation catheter":"green","flexible catheter":"green","catheter":"green",
    "intraductal":"green","ercp":"green","eus-guided":"green","endoscopic ultrasound":"green","minimally invasive":"green",
    "first-in-human":"orange","clinical feasibility":"orange","safety":"orange",
    "interventional oncology":"orange","locoregional therapy":"orange",
    "drug delivery":"violet","localized drug delivery":"violet","intratumoral drug delivery":"violet",
    "pancreatic juice":"orange","pancreatic juice biomarker":"orange","liquid biopsy":"orange","ctdna":"orange",
}

STATE_CYCLE  = {"active":"muted","muted":"removed","removed":"active"}


# ─────────────────────────────────────────────
# DESIGN TOKENS
# ─────────────────────────────────────────────

# Brand colours from image 10
C_PERI   = "#C5CAFB"   # Periwinkle — light accent fills
C_WIST   = "#9CA1FF"   # Wisteria Blue — primary accent
C_SLATE  = "#396070"   # Blue Slate — text, borders, icons
C_BLACK  = "#0A0C10"   # Near-black backgrounds
C_WHITE  = "#FFFFFF"

# Surface levels
C_BG     = "#F8F9FF"   # Page background — very light periwinkle tint
C_CARD   = "#FFFFFF"   # Card background
C_CARD2  = "#F3F4FD"   # Slightly tinted card (secondary)
C_BORDER = "#E4E6F4"   # Dividers and card borders
C_MUTED  = "#7C8599"   # Secondary text

# Semantic
C_HIGH   = "#0E7B55"   # High relevance text
C_HIGH_BG= "#E6F5EF"
C_MID    = "#7A5200"
C_MID_BG = "#FFF5DC"
C_LOW    = "#9B2020"
C_LOW_BG = "#FDEAEA"

# Chart colours — using brand palette
CH_PRIMARY   = C_SLATE   # #396070  — main bars
CH_ACCENT    = C_WIST    # #9CA1FF  — highlight / second series
CH_LIGHT     = C_PERI    # #C5CAFB  — muted / background bars
CH_CORAL     = "#6B82A8" # Cool blue-gray for 4th series

PLOTLY_LAYOUT = dict(
    margin=dict(l=0, r=0, t=8, b=0),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(size=11, family="Inter, sans-serif", color=C_MUTED),
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

/* ── Reset & base ── */
*, *::before, *::after {{ box-sizing: border-box; }}
html, body, [class*="css"],
[data-testid="stAppViewContainer"] {{
    font-family: "Inter", sans-serif !important;
    color: {C_BLACK};
}}

/* ── Page background — flat tinted canvas ── */
.stApp {{
    background: #FFFFFF !important;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: {C_WHITE} !important;
    border-right: 1px solid {C_BORDER} !important;
}}
[data-testid="stSidebar"] h1 {{
    font-size: 1rem !important;
    font-weight: 700 !important;
    color: {C_BLACK} !important;
    margin-bottom: 0.1rem !important;
}}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {{
    font-family: "Inter", sans-serif !important;
    font-size: 0.84rem !important;
}}
[data-testid="stSidebar"] .stSlider > label {{
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    color: {C_MUTED} !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
}}

/* Sidebar slider accent */
[data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] {{
    background: {C_WIST} !important;
    border-color: {C_WIST} !important;
}}
[data-testid="stSidebar"] [data-baseweb="slider"] [data-testid="stSliderTrackFill"] {{
    background: {C_WIST} !important;
}}

/* ── Main content width ── */
.block-container {{
    background: transparent;
    max-width: 1280px !important;
    padding-top: 2.5rem !important;
    padding-bottom: 4rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}}

/* White card treatment for Streamlit bordered vertical blocks */
[data-testid="stVerticalBlockBorderWrapper"] {{
    background: #FFFFFF;
    border-radius: 14px;
    border: 1px solid #E2E4F3;
    box-shadow: 0 2px 12px rgba(100, 100, 180, 0.07);
    padding: 1.2rem 1.25rem !important;
}}

/* ── Typography ── */
h1 {{
    font-size: 2.6rem !important;
    font-weight: 900 !important;
    letter-spacing: -0.03em !important;
    color: {C_BLACK} !important;
    line-height: 1.1 !important;
    margin-bottom: 0.3rem !important;
}}
h2 {{
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    color: {C_BLACK} !important;
    letter-spacing: -0.01em !important;
}}
h3 {{
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    color: {C_BLACK} !important;
}}
[data-testid="stCaptionContainer"] p {{
    color: {C_MUTED} !important;
    font-size: 0.85rem !important;
}}

/* ── Metric cards ── */
[data-testid="stMetric"] {{
    background: {C_WHITE} !important;
    border: 1px solid {C_BORDER} !important;
    border-radius: 12px !important;
    padding: 1.1rem 1.25rem !important;
    box-shadow: 0 2px 8px rgba(57,96,112,0.06) !important;
    transition: box-shadow 0.2s ease !important;
}}
[data-testid="stMetric"]:hover {{
    box-shadow: 0 4px 16px rgba(57,96,112,0.12) !important;
}}
[data-testid="stMetricLabel"] p {{
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    color: {C_MUTED} !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}}
[data-testid="stMetricValue"] {{
    font-size: 2rem !important;
    font-weight: 800 !important;
    color: {C_BLACK} !important;
    letter-spacing: -0.02em !important;
    line-height: 1.15 !important;
}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0 !important;
    border-bottom: 1.5px solid {C_BORDER} !important;
    background: transparent !important;
}}
.stTabs [data-baseweb="tab"] {{
    padding: 0.75rem 1.4rem 0.6rem !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    color: {C_MUTED} !important;
    border-radius: 0 !important;
    background: transparent !important;
}}
.stTabs [aria-selected="true"] {{
    color: {C_SLATE} !important;
    background: transparent !important;
}}
.stTabs [aria-selected="true"]::after {{
    background-color: {C_WIST} !important;
    height: 2.5px !important;
}}

/* ── Buttons ── */
.stButton > button {{
    font-family: "Inter", sans-serif !important;
    font-size: 0.84rem !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    border: 1.5px solid {C_SLATE} !important;
    color: {C_SLATE} !important;
    background: transparent !important;
    padding: 0.45rem 1rem !important;
    transition: all 0.15s ease !important;
}}
.stButton > button:hover {{
    background: {C_SLATE} !important;
    color: {C_WHITE} !important;
}}
.stLinkButton > a {{
    font-family: "Inter", sans-serif !important;
    font-size: 0.84rem !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    border: 1.5px solid {C_BORDER} !important;
    color: {C_SLATE} !important;
    background: {C_WHITE} !important;
    text-decoration: none !important;
    padding: 0.45rem 1rem !important;
}}
.stDownloadButton > button {{
    font-family: "Inter", sans-serif !important;
    font-size: 0.84rem !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    border: 1.5px solid {C_BORDER} !important;
    color: {C_SLATE} !important;
    background: {C_WHITE} !important;
}}

/* ── Expander ── */
[data-testid="stExpander"] {{
    border: 1px solid {C_BORDER} !important;
    border-radius: 10px !important;
    background: {C_WHITE} !important;
    margin-bottom: 0 !important;
}}
[data-testid="stExpander"] summary {{
    font-size: 0.84rem !important;
    font-weight: 600 !important;
    color: {C_SLATE} !important;
    padding: 0.6rem 0.8rem !important;
}}

/* ── Divider ── */
hr {{
    border-color: {C_BORDER} !important;
    margin: 1rem 0 !important;
}}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {{
    border-radius: 10px !important;
    overflow: hidden !important;
    border: 1px solid {C_BORDER} !important;
}}

/* ── Radio (summary status) ── */
[data-testid="stSidebar"] .stRadio > label {{
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    color: {C_MUTED} !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
}}
[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {{
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    color: {C_BLACK} !important;
}}

/* ─────────────────────────────────────────
   SCORE CHIP
───────────────────────────────────────── */
.sc {{
    display: inline-flex;
    align-items: center;
    border-radius: 999px;
    padding: 0.18rem 0.62rem;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    margin-right: 0.3rem;
}}
.sc-high  {{ background:{C_HIGH_BG}; color:{C_HIGH}; }}
.sc-mid   {{ background:{C_MID_BG};  color:{C_MID};  }}
.sc-low   {{ background:{C_LOW_BG};  color:{C_LOW};  }}

/* ─────────────────────────────────────────
   TAG BADGES — using brand palette
───────────────────────────────────────── */
.tb {{
    display: inline-flex;
    align-items: center;
    border-radius: 999px;
    padding: 0.16rem 0.52rem;
    margin: 0.1rem 0.15rem 0.1rem 0;
    font-size: 0.71rem;
    font-weight: 600;
    line-height: 1.2;
    border: 1px solid transparent;
    white-space: nowrap;
}}
.tb-red    {{ background:#EEEEFF; border-color:{C_PERI};  color:#3A3FAA; }}
.tb-blue   {{ background:{C_CARD2}; border-color:{C_BORDER}; color:{C_SLATE}; }}
.tb-green  {{ background:#EBF0F3; border-color:#C4D3DA; color:#2B4C5A; }}
.tb-orange {{ background:#F5F5FF; border-color:{C_PERI}; color:#555A9E; }}
.tb-violet {{ background:#EDEDFF; border-color:{C_WIST}; color:#4346A8; }}
.tb-gray   {{ background:#F5F6FA; border-color:{C_BORDER}; color:{C_MUTED}; }}

/* ─────────────────────────────────────────
   META BADGES (source, date)
───────────────────────────────────────── */
.mb {{
    display: inline-flex;
    align-items: center;
    border-radius: 6px;
    padding: 0.16rem 0.5rem;
    margin: 0.06rem 0.15rem 0.1rem 0;
    font-size: 0.72rem;
    font-weight: 600;
    line-height: 1.2;
    border: 1px solid {C_BORDER};
    background: {C_WHITE};
    color: {C_SLATE};
}}
.mb-src {{ background:{C_CARD2}; border-color:{C_BORDER}; }}
.mb-date {{ background:{C_WHITE}; color:{C_MUTED}; }}

/* ─────────────────────────────────────────
   RESULT CARD (evidence feed)
───────────────────────────────────────── */
.rc {{
    background: {C_WHITE};
    border: 1px solid {C_BORDER};
    border-radius: 12px;
    padding: 1.1rem 1.25rem 1rem;
    margin-bottom: 0.65rem;
    transition: box-shadow 0.18s ease, border-color 0.18s ease;
}}
.rc:hover {{
    box-shadow: 0 4px 20px rgba(57,96,112,0.10);
    border-color: {C_PERI};
}}
.rc-priority {{
    border-left: 3px solid {C_WIST};
}}
.rc-score-line {{
    display: flex;
    align-items: center;
    gap: 0.4rem;
    margin-bottom: 0.5rem;
    flex-wrap: wrap;
}}
.rc-score-num {{
    font-size: 0.82rem;
    font-weight: 700;
    color: {C_SLATE};
    margin-right: 0.1rem;
}}
.rc-title {{
    font-size: 0.97rem;
    font-weight: 700;
    color: {C_BLACK};
    line-height: 1.45;
    margin: 0.35rem 0 0.3rem;
}}
.rc-note {{
    font-size: 0.84rem;
    color: {C_MUTED};
    line-height: 1.5;
    margin-top: 0.4rem;
}}
.rc-tags {{
    margin-top: 0.45rem;
}}

/* ─────────────────────────────────────────
   WATCHLIST — top 5 priority cards
───────────────────────────────────────── */
.wc {{
    background: {C_WHITE};
    border: 1px solid {C_BORDER};
    border-radius: 12px;
    padding: 1rem 1.15rem;
    margin-bottom: 0.65rem;
    box-shadow: 0 2px 10px rgba(100, 100, 180, 0.07);
    transition: box-shadow 0.18s ease;
}}
.wc:hover {{
    box-shadow: 0 6px 24px rgba(100, 100, 180, 0.14);
}}
.wc-priority {{
    background: linear-gradient(
        110deg,
        #ECEEFF 0%,
        #F4F5FF 40%,
        #FFFFFF 100%
    );
    border: 1px solid #C8D2FF;
    box-shadow: 0 4px 18px rgba(156, 161, 255, 0.18);
}}
.wc-rank {{
    font-size: 0.7rem;
    font-weight: 800;
    color: {C_WIST};
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
}}
.wc-title {{
    font-size: 0.97rem;
    font-weight: 700;
    color: {C_BLACK};
    line-height: 1.45;
    margin: 0.3rem 0 0.3rem;
}}
.wc-note {{
    font-size: 0.83rem;
    color: {C_MUTED};
    line-height: 1.5;
    margin-top: 0.3rem;
}}

/* ─────────────────────────────────────────
   CHART WRAPPER CARD
───────────────────────────────────────── */
.chart-card {{
    background: {C_WHITE};
    border: 1px solid {C_BORDER};
    border-radius: 12px;
    padding: 1.2rem 1.25rem 1rem;
    margin-bottom: 0.8rem;
}}
.chart-card-title {{
    font-size: 0.9rem;
    font-weight: 700;
    color: {C_BLACK};
    margin-bottom: 0.15rem;
}}
.chart-card-sub {{
    font-size: 0.78rem;
    color: {C_MUTED};
    margin-bottom: 0.6rem;
}}

/* ─────────────────────────────────────────
   SOURCE BAR (simple horizontal bars)
───────────────────────────────────────── */
.src-bar-wrap {{ margin-top: 0.3rem; }}
.src-row {{
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.55rem;
    font-size: 0.82rem;
}}
.src-label {{ width: 130px; flex-shrink: 0; color: {C_BLACK}; font-weight: 500; }}
.src-track {{
    flex: 1;
    height: 7px;
    background: {C_CARD2};
    border-radius: 4px;
    overflow: hidden;
}}
.src-fill {{ height: 100%; border-radius: 4px; }}
.src-count {{ width: 28px; text-align: right; color: {C_MUTED}; font-weight: 600; }}

/* ─────────────────────────────────────────
   KEYWORD PILLS (relevance profile tab)
───────────────────────────────────────── */
.kw-cluster-title {{
    font-size: 0.72rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: {C_WIST};
    margin: 1.4rem 0 0.6rem;
}}
.kw-cloud {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem 0.55rem;
    margin-bottom: 0.4rem;
}}
.kw-pill {{
    display: inline-flex;
    align-items: center;
    min-height: 2.1rem;
    border-radius: 999px;
    padding: 0.28rem 0.9rem;
    font-size: 0.82rem;
    font-weight: 600;
    text-decoration: none !important;
    transition: all 0.13s ease;
    cursor: pointer;
}}
.kw-pill:hover {{ transform: translateY(-1px); }}
.kw-pill-active  {{ background:{C_PERI}; color:#1A1E6A; border:1px solid {C_WIST}; }}
.kw-pill-muted   {{ background:{C_WHITE}; color:{C_MUTED}; border:1px dashed {C_BORDER}; }}
.kw-pill-removed {{ background:#FFF0F0; color:#9B2020; border:1px solid #F2B7B7; text-decoration:line-through !important; }}

.kw-legend-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-bottom: 1.2rem;
}}
.kw-legend-item {{
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.82rem;
    font-weight: 500;
    color: {C_MUTED};
    background: {C_WHITE};
    border: 1px solid {C_BORDER};
    border-radius: 8px;
    padding: 0.4rem 0.7rem;
}}
.kw-dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}}
.kw-dot-active  {{ background:{C_WIST}; }}
.kw-dot-muted   {{ background:{C_BORDER}; }}
.kw-dot-removed {{ background:#E05454; }}
</style>
"""


# ─────────────────────────────────────────────
# RENDER HELPERS
# ─────────────────────────────────────────────

def tag_badge(keyword: str) -> str:
    color = TAG_COLORS.get(keyword.lower(), "gray")
    return f'<span class="tb tb-{color}">{escape(keyword)}</span>'

def meta_badge(text: str, kind: str = "") -> str:
    cls = f" mb-{kind}" if kind else ""
    return f'<span class="mb{cls}">{escape(str(text))}</span>'

def score_chip(score: float) -> str:
    if score >= 0.80:   tone, label = "high", "High"
    elif score >= 0.65: tone, label = "mid",  "Medium"
    else:               tone, label = "low",  "Low"
    return f'<span class="sc sc-{tone}">{label}</span>'

def format_date(value) -> str:
    if pd.isna(value): return "Unknown"
    return str(value)[:10]

def source_meta(row) -> str:
    src_type  = str(getattr(row, "source_type", "") or "").upper()
    src_label = getattr(row, "source_label", "") or "Unknown"
    item_date = format_date(getattr(row, "item_date", None))
    return (meta_badge(src_type, "src")
            + meta_badge(src_label, "src")
            + meta_badge(item_date, "date"))

def result_card_html(row, rank: int = 0, priority: bool = False) -> str:
    matched = getattr(row, "matched_keywords", None) or []
    note    = getattr(row, "relevance_note", "") or getattr(row, "summary", "") or getattr(row, "body", "") or ""
    note    = " ".join(str(note).split())
    if len(note) > 200: note = note[:197].rstrip() + "…"
    tags    = "".join(tag_badge(kw) for kw in matched[:6])
    score   = float(getattr(row, "score", 0) or 0)
    title   = escape(getattr(row, "title", "") or "Untitled")
    pri_cls = " rc-priority" if priority else ""
    rank_str = f'<div class="wc-rank">#{rank}</div>' if priority else ""

    if priority:
        return f"""
        <div class="wc wc-priority">
            {rank_str}
            <div class="rc-score-line">
                {score_chip(score)}
                <span class="rc-score-num">{score:.2f}</span>
                {source_meta(row)}
            </div>
            <div class="rc-title">{title}</div>
            <div class="rc-tags">{tags}</div>
            <div class="rc-note">{escape(note or "No summary available yet.")}</div>
        </div>"""
    else:
        return f"""
        <div class="rc{pri_cls}">
            <div class="rc-score-line">
                {score_chip(score)}
                <span class="rc-score-num">{score:.2f}</span>
                {source_meta(row)}
            </div>
            <div class="rc-title">{title}</div>
            <div class="rc-tags">{tags}</div>
            <div class="rc-note">{escape(note or "No summary available yet.")}</div>
        </div>"""

def source_bars_html(results: pd.DataFrame) -> str:
    if "data_source" not in results.columns:
        return ""
    colors = {
        "pubmed":"#396070","clinicaltrials":"#9CA1FF",
        "europepmc":"#C5CAFB","biorxiv":"#6B82A8","medrxiv":"#8A9EC0",
    }
    labels = {
        "pubmed":"PubMed","clinicaltrials":"ClinicalTrials.gov",
        "europepmc":"Europe PMC","biorxiv":"bioRxiv","medrxiv":"medRxiv",
    }
    counts = results["data_source"].value_counts()
    if counts.empty: return ""
    max_c  = counts.max()
    rows   = ""
    for src, cnt in counts.items():
        color = colors.get(src, "#9CA1FF")
        label = labels.get(src, src)
        pct   = cnt / max_c * 100
        rows += f"""
        <div class="src-row">
            <div class="src-label">{label}</div>
            <div class="src-track"><div class="src-fill" style="width:{pct:.0f}%;background:{color}"></div></div>
            <div class="src-count">{cnt}</div>
        </div>"""
    return f'<div class="src-bar-wrap">{rows}</div>'

def keyword_pill_html(keyword: str, state: str) -> str:
    safe = escape(keyword)
    href = f"?cycle_kw={quote(keyword)}"
    return f'<a class="kw-pill kw-pill-{state}" href="{href}" target="_self">{safe}</a>'

def profile_keywords() -> list[str]:
    seen, kws = set(), []
    for ks in KEYWORD_CLUSTERS.values():
        for k in ks:
            if k not in seen:
                seen.add(k); kws.append(k)
    return kws


# ─────────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Embio Intelligence",
    page_icon=str(ICON_PATH),
    layout="wide",
)
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def ensure_database():
    init_db()

ensure_database()


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_results() -> pd.DataFrame:
    scored = score_all()
    if scored.empty:
        return scored
    with get_connection() as con:
        articles = con.execute("""
            SELECT id,'article' AS source_type,
                   journal AS source_label,authors,url,fetched_at,
                   COALESCE(source,'pubmed') AS data_source
            FROM articles
        """).fetchdf()
        trials = con.execute("""
            SELECT id,'trial' AS source_type,
                   sponsor AS source_label,status AS authors,url,fetched_at,
                   COALESCE(source,'clinicaltrials') AS data_source
            FROM trials
        """).fetchdf()
    metadata = pd.concat([articles, trials], ignore_index=True)
    results  = scored.merge(metadata, on=["id","source_type"], how="left")
    cached = []
    for _, row in results.iterrows():
        s = get_cached_summary(row.id, row.source_type) or {}
        cached.append({
            "summary":        s.get("summary",""),
            "relevance_note": s.get("relevance_note",""),
            "tags":           s.get("tags",[]),
        })
    return pd.concat([results.reset_index(drop=True), pd.DataFrame(cached)], axis=1)

def refresh_data():
    load_results.clear()
    st.rerun()

def refresh_pipeline():
    with st.spinner("Fetching new sources and embedding documents…"):
        run_once()
        embed_all_pending()
    refresh_data()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=140)
    st.title("Intelligence")
    st.caption("Research & market radar")
    st.divider()

    source_filter = st.multiselect(
        "Source type", ["article","trial"],
        default=["article","trial"]
    )
    min_score  = st.slider("Min relevance score", 0.0, 1.0, DEFAULT_MIN_RELEVANCE, 0.01)
    years_back = st.slider("Published within (years)", 1, 5, 2, 1)

    st.divider()
    summary_filter = st.radio(
        "Summary status",
        ["All records","Needs summary","Has AI summary"],
    )

    st.divider()
    if st.button("Refresh", use_container_width=True):
        refresh_pipeline()
    st.caption("Pulls new data from all sources and embeds new documents.")


# ─────────────────────────────────────────────
# LOAD + FILTER
# ─────────────────────────────────────────────

results = load_results()

st.title("Embio Intelligence")
st.caption(
    "Research and clinical-trial radar — electroporation catheters, "
    "intraductal & ERCP delivery, pancreatic cancer, and adjacent medtech signals."
)

if results.empty:
    st.info("No embedded records yet. Click **Refresh** in the sidebar.")
    st.stop()

cutoff   = date.today() - timedelta(days=years_back * 365)
filtered = results[results["source_type"].isin(source_filter)]
filtered = filtered[filtered["score"] >= min_score]
filtered = filtered[
    filtered["item_date"].isna()
    | (pd.to_datetime(filtered["item_date"], errors="coerce").dt.date >= cutoff)
]
if summary_filter == "Needs summary":
    filtered = filtered[filtered["summary"].fillna("") == ""]
elif summary_filter == "Has AI summary":
    filtered = filtered[filtered["summary"].fillna("") != ""]


# ─────────────────────────────────────────────
# METRICS ROW
# ─────────────────────────────────────────────

with get_connection() as con:
    feedback_count = con.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]

latest_fetch  = results["fetched_at"].dropna().max() if "fetched_at" in results else None
week_cutoff   = date.today() - timedelta(days=7)
new_this_week = int((pd.to_datetime(filtered["item_date"], errors="coerce").dt.date >= week_cutoff).sum())
high_relevance = int((filtered["score"] >= 0.65).sum())

m = st.columns(5)
m[0].metric("Total records",    f"{len(results):,}")
m[1].metric("New this week",    f"{new_this_week:,}")
m[2].metric("High relevance",   f"{high_relevance:,}")
m[3].metric("Feedback signals", f"{feedback_count:,}")
last_updated_str = (
    pd.Timestamp(latest_fetch).strftime("%-d %b, %H:%M")
    if latest_fetch is not None else "Never"
)
m[4].metric("Last updated", last_updated_str)

pdf_col, _ = st.columns([1.3, 5])
pdf_col.download_button(
    "Download weekly PDF",
    data=build_weekly_pdf(filtered),
    file_name=f"embio-weekly-{date.today().isoformat()}.pdf",
    mime="application/pdf",
    use_container_width=True,
)

st.divider()


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────

tab_overview, tab_feed, tab_profile = st.tabs(["Overview", "Evidence feed", "Relevance profile"])


# ══════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════

with tab_overview:

    col_l, col_r = st.columns(2, gap="medium")

    # ── Publications over time ──
    with col_l:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-card-title">Publications over time</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-card-sub">Monthly volume, last 3 years</div>', unsafe_allow_html=True)
        with get_connection() as con:
            trend = con.execute("""
                SELECT date_trunc('month', pub_date) AS month, COUNT(*) AS count
                FROM articles
                WHERE pub_date IS NOT NULL
                  AND pub_date >= now() - INTERVAL '3 years'
                GROUP BY 1 ORDER BY 1
            """).df()
        if not trend.empty:
            fig = go.Figure(go.Bar(
                x=trend["month"], y=trend["count"],
                marker_color=CH_PRIMARY,
                marker_line_width=0,
            ))
            fig.update_traces(marker_cornerradius=4)
            fig.update_layout(
                **PLOTLY_LAYOUT, height=200,
                xaxis=dict(showgrid=False, tickfont=dict(size=10, color=C_MUTED)),
                yaxis=dict(gridcolor="#EEF0FA", tickfont=dict(size=10, color=C_MUTED), zeroline=False),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No dated articles yet.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Score distribution — gradient bars light to dark by score value ──
    with col_r:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-card-title">Score distribution</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chart-card-sub">Dashed line = current threshold ({min_score:.2f})</div>', unsafe_allow_html=True)

        scores = results["score"].dropna().values
        hist_counts, bin_edges = np.histogram(scores, bins=25, range=(0, 1))
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

        def score_to_color(val: float) -> str:
            r = int(197 + (57 - 197) * val)
            g = int(202 + (96 - 202) * val)
            b = int(251 + (112 - 251) * val)
            return f"rgb({r},{g},{b})"

        bar_colors = [score_to_color(float(v)) for v in bin_centres]
        fig2 = go.Figure(go.Bar(
            x=bin_centres,
            y=hist_counts,
            width=(bin_edges[1] - bin_edges[0]) * 0.85,
            marker_color=bar_colors,
            marker_line_width=0,
        ))
        fig2.add_vline(
            x=min_score, line_dash="dot", line_color=C_SLATE, line_width=1.5,
            annotation_text=f"  threshold", annotation_font_size=10,
            annotation_font_color=C_SLATE,
        )
        fig2.update_traces(marker_cornerradius=4)
        fig2.update_layout(
            **PLOTLY_LAYOUT, height=200,
            xaxis=dict(showgrid=False, tickfont=dict(size=10, color=C_MUTED), title="relevance score"),
            yaxis=dict(gridcolor="#EEF0FA", tickfont=dict(size=10, color=C_MUTED), zeroline=False, title="records"),
            bargap=0.08,
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    col_l2, col_r2 = st.columns(2, gap="medium")

    # ── Top keyword frequency ──
    with col_l2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-card-title">Top keyword matches</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-card-sub">Across all records above threshold</div>', unsafe_allow_html=True)
        all_kws = [kw for kws in results["matched_keywords"].dropna() for kw in kws]
        if all_kws:
            top_kws = pd.DataFrame(Counter(all_kws).most_common(12), columns=["keyword","count"])
            # Colour bars by cluster membership
            def kw_color(kw):
                for cluster, kws in KEYWORD_CLUSTERS.items():
                    if kw in kws:
                        cmap = {
                            "Electroporation core": CH_PRIMARY,
                            "Catheter / intraductal": CH_ACCENT,
                            "Oncology / clinical": C_PERI,
                            "Drug delivery": "#7B86D4",
                            "Biomarkers / diagnostics": "#8A9EC0",
                            "Adjacent / strategic": C_MUTED,
                        }
                        return cmap.get(cluster, CH_PRIMARY)
                return CH_PRIMARY
            top_kws["color"] = top_kws["keyword"].apply(kw_color)
            fig3 = go.Figure(go.Bar(
                x=top_kws["count"], y=top_kws["keyword"],
                orientation="h",
                marker_color=top_kws["color"].tolist(),
                marker_line_width=0,
            ))
            fig3.update_traces(marker_cornerradius=3)
            fig3.update_layout(
                **PLOTLY_LAYOUT, height=320,
                xaxis=dict(showgrid=True, gridcolor="#EEF0FA", tickfont=dict(size=10, color=C_MUTED), zeroline=False),
                yaxis=dict(showgrid=False, tickfont=dict(size=10, color=C_BLACK), autorange="reversed"),
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.caption("No keyword matches yet.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Source breakdown + ingestion health ──
    with col_r2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-card-title">Source breakdown</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chart-card-sub">{len(results):,} total records</div>', unsafe_allow_html=True)
        st.markdown(source_bars_html(results), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="chart-card-title" style="font-size:0.82rem;color:{C_MUTED};font-weight:600;text-transform:uppercase;letter-spacing:0.05em">Ingestion log — last 30 days</div>', unsafe_allow_html=True)
        try:
            health = ingestion_summary(days=30)
            if health:
                hdf = pd.DataFrame(health)[["source","total_new","total_updated","last_run"]]
                hdf["last_run"] = pd.to_datetime(hdf["last_run"]).dt.strftime("%Y-%m-%d %H:%M")
                st.dataframe(hdf, hide_index=True, use_container_width=True)
            else:
                st.caption("No log entries yet.")
        except Exception:
            st.caption("Ingestion log not available.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Scoring diagnostics (collapsed) ──
    with st.expander("Scoring diagnostics — semantic vs keyword"):
        st.caption("Each dot is one document. Bubble size = composite score. Hover for title.")
        fig5 = px.scatter(
            filtered, x="semantic_score", y="keyword_score",
            color="source_type", hover_name="title",
            size="score", size_max=14,
            color_discrete_map={"article": CH_PRIMARY, "trial": CH_ACCENT},
            opacity=0.65,
        )
        fig5.update_layout(
            **PLOTLY_LAYOUT, height=300,
            xaxis=dict(title="semantic score", gridcolor="#EEF0FA", tickfont=dict(size=10, color=C_MUTED)),
            yaxis=dict(title="keyword score", gridcolor="#EEF0FA", tickfont=dict(size=10, color=C_MUTED)),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=11)),
        )
        st.plotly_chart(fig5, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 2 — EVIDENCE FEED
# ══════════════════════════════════════════════

with tab_feed:

    if filtered.empty:
        st.info("No results match the current filters. Lower the minimum relevance score to see more.")
        st.stop()

    # Priority watchlist — top 5 as rich cards
    st.subheader("Priority watchlist")
    st.caption("Top 5 highest-scoring results from current filters.")
    for rank, row in enumerate(filtered.head(5).itertuples(index=False), start=1):
        st.markdown(result_card_html(row, rank=rank, priority=True), unsafe_allow_html=True)

    st.divider()
    st.subheader(f"All results — {len(filtered):,}")

    for _, row in filtered.iterrows():
        matched = row.matched_keywords or []

        # Render the card header inline, then expander for details
        st.markdown(result_card_html(row, priority=False), unsafe_allow_html=True)

        with st.expander("View details"):
            left, right = st.columns([3, 1])

            with left:
                if row.summary:
                    st.markdown(f"**Summary**  \n{row.summary}")
                    st.markdown(f"**Why it matters**  \n{row.relevance_note}")
                    if row.tags:
                        st.markdown(
                            "**Tags:** " + " ".join(tag_badge(t) for t in row.tags),
                            unsafe_allow_html=True,
                        )
                else:
                    body_text = row.body or "No abstract or study details available."
                    st.markdown(body_text[:800] + ("…" if len(body_text) > 800 else ""))
                    if st.button("Generate summary", key=f"sum_{row.source_type}_{row.id}"):
                        with st.spinner("Generating…"):
                            summarise(row.title, row.body or "", row.id, row.source_type, float(row.score))
                        refresh_data()

            with right:
                st.metric("Semantic",  f"{row.semantic_score:.2f}")
                st.metric("Keywords",  f"{row.keyword_score:.2f}")
                st.metric("Recency",   f"{row.recency_score:.2f}")
                st.metric("Composite", f"{row.score:.2f}")

            st.divider()
            a = st.columns([1, 1, 1, 3])
            if a[0].button("👍 Relevant",     key=f"up_{row.source_type}_{row.id}"):
                record_feedback(row.id, row.source_type, 1)
                refresh_data()
            if a[1].button("👎 Not relevant", key=f"dn_{row.source_type}_{row.id}"):
                record_feedback(row.id, row.source_type, -1)
                refresh_data()
            a[2].link_button("Open source", row.url)


# ══════════════════════════════════════════════
# TAB 3 — RELEVANCE PROFILE
# ══════════════════════════════════════════════

with tab_profile:

    saved_profile = load_user_profile()
    saved_states  = saved_profile.get("keyword_states", {})
    saved_weights = saved_profile.get("scoring_weights", {})

    st.subheader("Keyword filter")
    st.markdown(f"""
    <div class="kw-legend-row">
        <div class="kw-legend-item"><span class="kw-dot kw-dot-active"></span> <strong>Active</strong> — boosts relevance score</div>
        <div class="kw-legend-item"><span class="kw-dot kw-dot-muted"></span> <strong>Muted</strong> — ignored in scoring</div>
        <div class="kw-legend-item"><span class="kw-dot kw-dot-removed"></span> <strong>Removed</strong> — suppresses matches</div>
        <span style="font-size:0.78rem;color:{C_MUTED};align-self:center;margin-left:0.3rem;font-style:italic">Click any pill to cycle its state</span>
    </div>
    """, unsafe_allow_html=True)

    if "kw_states" not in st.session_state:
        st.session_state.kw_states = {**{kw: "active" for kw in profile_keywords()}, **saved_states}

    cycle_kw = st.query_params.get("cycle_kw")
    if isinstance(cycle_kw, list):
        cycle_kw = cycle_kw[0] if cycle_kw else None
    if cycle_kw:
        cur = st.session_state.kw_states.get(cycle_kw, "active")
        st.session_state.kw_states[cycle_kw] = STATE_CYCLE[cur]
        st.query_params.clear()

    if "weights" not in st.session_state:
        st.session_state.weights = {
            "semantic":  saved_weights.get("semantic",  SCORING_WEIGHTS["semantic"]),
            "keyword":   saved_weights.get("keyword",   SCORING_WEIGHTS["keyword"]),
            "recency":   saved_weights.get("recency",   SCORING_WEIGHTS["recency"]),
            "feedback":  saved_weights.get("feedback",  SCORING_WEIGHTS["feedback"]),
        }

    for cluster_name, keywords in KEYWORD_CLUSTERS.items():
        pills = "".join(keyword_pill_html(kw, st.session_state.kw_states.get(kw, "active")) for kw in keywords)
        st.markdown(f"""
        <div class="kw-cluster-title">{escape(cluster_name)}</div>
        <div class="kw-cloud">{pills}</div>
        """, unsafe_allow_html=True)

    active_count  = sum(1 for s in st.session_state.kw_states.values() if s == "active")
    muted_count   = sum(1 for s in st.session_state.kw_states.values() if s == "muted")
    removed_count = sum(1 for s in st.session_state.kw_states.values() if s == "removed")
    st.caption(f"Active: {active_count}  ·  Muted: {muted_count}  ·  Removed: {removed_count}")

    st.divider()
    st.subheader("Scoring weights")
    st.caption("How much each signal contributes. Should sum to 1.0.")

    w_sem = st.slider("Semantic weight",  0.0, 0.9, float(st.session_state.weights["semantic"]),  0.05)
    w_kw  = st.slider("Keyword weight",   0.0, 0.9, float(st.session_state.weights["keyword"]),   0.05)
    w_rec = st.slider("Recency weight",   0.0, 0.5, float(st.session_state.weights["recency"]),   0.05)
    w_fb  = st.slider("Feedback weight",  0.0, 0.3, float(st.session_state.weights["feedback"]),  0.05)

    total = round(w_sem + w_kw + w_rec + w_fb, 2)
    if abs(total - 1.0) > 0.01:
        st.warning(f"Weights sum to {total:.2f} — adjust to reach 1.0.")
    else:
        st.success(f"Weights sum to {total:.2f} ✓")

    save_col, reset_col, _ = st.columns([1, 1, 3])

    if save_col.button("Save profile", use_container_width=True):
        new_weights = {"semantic": w_sem, "keyword": w_kw, "recency": w_rec, "feedback": w_fb}
        st.session_state.weights = new_weights
        save_user_profile(st.session_state.kw_states, new_weights)
        load_results.clear()
        st.success("Profile saved. Scores will update on next refresh.")

    if reset_col.button("Reset to defaults", use_container_width=True):
        st.session_state.kw_states = {kw: "active" for kw in profile_keywords()}
        st.session_state.weights   = dict(SCORING_WEIGHTS)
        save_user_profile(st.session_state.kw_states, st.session_state.weights)
        load_results.clear()
        st.rerun()
