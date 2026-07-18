"""
dashboard/app.py — Embio Intelligence
Professional analytical dashboard.

Design language:
  - Hero banner: periwinkle→wisteria gradient, contains title + key metrics
  - Content: white cards on #F4F5FF page background
  - Typography: Inter, tight letter-spacing, clear hierarchy
  - Charts: area (trend), gradient bars (score dist), horizontal bars (keywords)
  - Evidence feed: clean card anatomy, gradient accent on priority items
  - Colour palette: #C5CAFB / #9CA1FF / #396070 / #FFFFFF
"""

import base64
import fcntl
import sys
import tempfile
from collections import Counter
from contextlib import contextmanager
from datetime import date, timedelta
from html import escape
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
LOGO_PATH = PROJECT_ROOT / "dashboard" / "assets" / "embio-black-logo.png"
ICON_PATH = PROJECT_ROOT / "dashboard" / "assets" / "embio-black-icon.png"
REFRESH_LOCK_PATH = Path(tempfile.gettempdir()) / "embio-intelligence-refresh.lock"

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from config.relevance_profile import DEFAULT_MIN_RELEVANCE, PRIORITY_KEYWORDS, SCORING_WEIGHTS
from dashboard.reporting import build_weekly_pdf
from feedback.store import record_feedback
from ingestion.scheduler import run_once
from nlp.embedder import embed_all_pending
from ranking.scorer import score_all
from storage.db import get_connection, init_db, ingestion_summary, load_user_profile, save_user_profile
from summarisation.llm import get_cached_summary, summarise


# ─────────────────────────────────────────────────────────────
# KEYWORD CLUSTERS
# ─────────────────────────────────────────────────────────────

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

CLUSTER_COLORS = {
    "Electroporation core":     "#396070",
    "Catheter / intraductal":   "#9CA1FF",
    "Oncology / clinical":      "#C5CAFB",
    "Drug delivery":            "#7B86D4",
    "Biomarkers / diagnostics": "#8A9EC0",
    "Adjacent / strategic":     "#B0B6CC",
}

STATE_CYCLE = {"active": "muted", "muted": "removed", "removed": "active"}


# ─────────────────────────────────────────────────────────────
# DESIGN TOKENS
# ─────────────────────────────────────────────────────────────

C_PERI    = "#C5CAFB"
C_WIST    = "#9CA1FF"
C_SLATE   = "#396070"
C_INK     = "#111827"
C_WHITE   = "#FFFFFF"
C_PAGE    = "#F4F5FF"
C_CARD    = "#FFFFFF"
C_BORDER  = "#E4E6F4"
C_MUTED   = "#6B7280"
C_MUTED2  = "#9CA3AF"

C_HIGH    = "#065F46"; C_HIGH_BG = "#ECFDF5"
C_MID     = "#78350F"; C_MID_BG  = "#FFFBEB"
C_LOW     = "#7F1D1D"; C_LOW_BG  = "#FEF2F2"

CHART_SLATE = C_SLATE
CHART_WIST  = C_WIST
CHART_PERI  = C_PERI

BASE_LAYOUT = dict(
    margin=dict(l=0, r=0, t=4, b=0),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(size=11, family="Inter, sans-serif", color=C_MUTED),
)
GRID_COLOR = "#EAECF8"


# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,400&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}

html, body, [class*="css"], [data-testid="stAppViewContainer"] {{
    font-family: "Inter", -apple-system, sans-serif !important;
    color: {C_INK};
}}

.material-symbols-rounded,
.material-symbols-outlined,
.material-icons,
[class*="material-symbols"],
[class*="material-icons"] {{
    font-family: "Material Symbols Rounded" !important;
    font-weight: normal !important;
    font-style: normal !important;
    font-size: 1.25rem !important;
    line-height: 1 !important;
    letter-spacing: normal !important;
    text-transform: none !important;
    display: inline-block !important;
    white-space: nowrap !important;
    word-wrap: normal !important;
    direction: ltr !important;
    -webkit-font-feature-settings: "liga" !important;
    -webkit-font-smoothing: antialiased !important;
}}

/* ── Page: clean light background ── */
.stApp {{ background: {C_PAGE} !important; }}

/* ── Keep hero clear of Streamlit's top toolbar ── */
.block-container {{
    padding-top: 3rem !important;
    padding-bottom: 4rem !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
    max-width: 100% !important;
}}

/* ── Inner content wrapper (used manually below hero) ── */
.content-wrap {{
    max-width: 1280px;
    margin: 0 auto;
    padding: 0 2rem;
}}

/* ════════════════════════════════════
   HERO BANNER
════════════════════════════════════ */
.hero {{
    background: linear-gradient(135deg, {C_SLATE} 0%, #4A7A90 30%, {C_WIST} 75%, {C_PERI} 100%);
    padding: 2.4rem 2.5rem 2.2rem;
    margin-bottom: 0;
    position: relative;
}}
.hero-eyebrow {{
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.65);
    margin-bottom: 0.3rem;
}}
.hero-title-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1.25rem;
}}
.hero-title {{
    font-size: 2.1rem;
    font-weight: 900;
    color: {C_WHITE};
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}}
.hero-download-btn {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 12.5rem;
    min-height: 2.35rem;
    padding: 0.48rem 1.1rem;
    border-radius: 8px;
    border: 1.5px solid rgba(255,255,255,0.68);
    color: {C_WHITE} !important;
    background: rgba(255,255,255,0.18);
    backdrop-filter: blur(6px);
    font-size: 0.84rem;
    font-weight: 600;
    line-height: 1;
    text-decoration: none !important;
    white-space: nowrap;
    transition: background 0.15s ease, border-color 0.15s ease;
}}
.hero-download-btn:hover {{
    background: rgba(255,255,255,0.30);
    border-color: rgba(255,255,255,0.9);
}}
.hero-sub {{
    font-size: 0.88rem;
    color: rgba(255,255,255,0.72);
    margin-bottom: 1.8rem;
    line-height: 1.5;
}}
.hero-metrics {{
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0.9rem;
}}
.hm {{
    background: rgba(255,255,255,0.13);
    border: 1px solid rgba(255,255,255,0.22);
    border-radius: 12px;
    padding: 1rem 1.1rem;
    backdrop-filter: blur(8px);
    transition: background 0.2s ease;
}}
.hm:hover {{ background: rgba(255,255,255,0.2); }}
.hm-label {{
    font-size: 0.69rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: rgba(255,255,255,0.6);
    margin-bottom: 0.4rem;
}}
.hm-value {{
    font-size: 1.85rem;
    font-weight: 800;
    color: {C_WHITE};
    letter-spacing: -0.025em;
    line-height: 1.1;
}}
.hm-sub {{
    font-size: 0.72rem;
    color: rgba(255,255,255,0.55);
    margin-top: 0.2rem;
}}
.hm-accent {{ color: #86EFAC; font-weight: 600; }}

/* ── Hero action row ── */
.hero-actions {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-top: 1.5rem;
    flex-wrap: wrap;
}}
.hero-btn {{
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.82rem;
    font-weight: 600;
    padding: 0.48rem 1.1rem;
    border-radius: 8px;
    border: 1.5px solid rgba(255,255,255,0.5);
    color: {C_WHITE};
    background: rgba(255,255,255,0.15);
    cursor: pointer;
    text-decoration: none;
    transition: all 0.15s ease;
    backdrop-filter: blur(4px);
}}
.hero-btn:hover {{
    background: rgba(255,255,255,0.28);
    border-color: rgba(255,255,255,0.75);
}}
.hero-btn-primary {{
    background: {C_WHITE};
    color: {C_SLATE};
    border-color: {C_WHITE};
}}
.hero-btn-primary:hover {{
    background: rgba(255,255,255,0.9);
}}

@media (max-width: 760px) {{
    .hero-title-row {{
        align-items: flex-start;
        flex-direction: column;
        gap: 0.8rem;
    }}
    .hero-download-btn {{
        min-width: 0;
        width: 100%;
    }}
}}

/* ════════════════════════════════════
   TABS (inside content area)
════════════════════════════════════ */
.stTabs {{
    padding: 0 2rem;
    max-width: 1280px;
    margin: 0 auto;
}}
.stTabs [data-baseweb="tab-list"] {{
    gap: 0 !important;
    border-bottom: 1.5px solid {C_BORDER} !important;
    background: transparent !important;
    padding: 0 !important;
    margin-bottom: 1.5rem !important;
}}
.stTabs [data-baseweb="tab"] {{
    padding: 0.85rem 1.4rem 0.7rem !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    color: {C_MUTED} !important;
    border-radius: 0 !important;
    background: transparent !important;
    letter-spacing: 0.01em !important;
}}
.stTabs [aria-selected="true"] {{
    color: {C_SLATE} !important;
    background: transparent !important;
}}
.stTabs [aria-selected="true"]::after {{
    background-color: {C_WIST} !important;
    height: 2px !important;
}}
[data-testid="stTabsContent"] {{
    padding: 0 2rem;
    max-width: 1280px;
    margin: 0 auto;
}}

/* ════════════════════════════════════
   SIDEBAR
════════════════════════════════════ */
[data-testid="stSidebar"] {{
    background: {C_WHITE} !important;
    border-right: 1px solid {C_BORDER} !important;
}}
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}}
.sidebar-logo {{
    margin: 0.55rem 0 2rem;
}}
.sidebar-logo img {{
    display: block;
    width: 170px;
    max-width: 78%;
    height: auto;
    image-rendering: auto;
}}
[data-testid="stSidebar"] h1 {{
    font-size: 1rem !important;
    font-weight: 700 !important;
    color: {C_INK} !important;
    margin-bottom: 0.1rem !important;
}}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p {{
    font-family: "Inter", sans-serif !important;
    font-size: 0.84rem !important;
}}
/* Slider accent */
[data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] {{
    background: {C_WIST} !important;
    border-color: {C_WIST} !important;
}}

/* ════════════════════════════════════
   CARD COMPONENT
════════════════════════════════════ */
.card {{
    background: {C_WHITE};
    border: 1px solid {C_BORDER};
    border-radius: 14px;
    padding: 1.35rem 1.4rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 4px rgba(100,100,200,0.05), 0 4px 16px rgba(100,100,200,0.05);
    transition: box-shadow 0.2s ease;
}}
.card:hover {{ box-shadow: 0 2px 8px rgba(100,100,200,0.08), 0 8px 24px rgba(100,100,200,0.08); }}
.card-title {{
    font-size: 0.88rem;
    font-weight: 700;
    color: {C_INK};
    letter-spacing: -0.01em;
    margin-bottom: 0.15rem;
}}
.card-sub {{
    font-size: 0.76rem;
    color: {C_MUTED2};
    margin-bottom: 0.9rem;
}}

/* ════════════════════════════════════
   SCORE CHIPS
════════════════════════════════════ */
.sc {{
    display: inline-flex;
    align-items: center;
    border-radius: 999px;
    padding: 0.16rem 0.58rem;
    font-size: 0.71rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    margin-right: 0.3rem;
}}
.sc-high {{ background:{C_HIGH_BG}; color:{C_HIGH}; }}
.sc-mid  {{ background:{C_MID_BG};  color:{C_MID};  }}
.sc-low  {{ background:{C_LOW_BG};  color:{C_LOW};  }}

/* ════════════════════════════════════
   TAG BADGES
════════════════════════════════════ */
.tb {{
    display: inline-flex;
    align-items: center;
    border-radius: 999px;
    padding: 0.15rem 0.5rem;
    margin: 0.08rem 0.12rem 0.08rem 0;
    font-size: 0.69rem;
    font-weight: 600;
    line-height: 1.2;
    border: 1px solid transparent;
    white-space: nowrap;
}}
.tb-red    {{ background:#EEF0FF; border-color:#D8DAFF; color:#4B52C8; }}
.tb-blue   {{ background:#EBF1F4; border-color:#BACED8; color:{C_SLATE}; }}
.tb-green  {{ background:#EDF2F4; border-color:#C8D8DE; color:#2D5264; }}
.tb-orange {{ background:#F8F8FF; border-color:{C_PERI}; color:#575CA0; }}
.tb-violet {{ background:#EEEFFF; border-color:{C_WIST}; color:#4B50C0; }}
.tb-gray   {{ background:#F5F6FA; border-color:#E3E5EE; color:#5F6872; }}

/* ════════════════════════════════════
   META BADGES
════════════════════════════════════ */
.mb {{
    display: inline-flex;
    align-items: center;
    border-radius: 6px;
    padding: 0.14rem 0.48rem;
    margin: 0.05rem 0.12rem 0.08rem 0;
    font-size: 0.69rem;
    font-weight: 600;
    line-height: 1.2;
    background: #F0F1FA;
    border: 1px solid {C_BORDER};
    color: {C_SLATE};
}}
.mb-src  {{ background:#EBF1F4; border-color:#C4D6DD; }}
.mb-date {{ background:{C_WHITE}; border-color:{C_BORDER}; color:{C_MUTED}; }}

/* ════════════════════════════════════
   WATCHLIST CARDS (priority feed)
════════════════════════════════════ */
.wc {{
    background: {C_WHITE};
    border: 1px solid {C_BORDER};
    border-radius: 12px;
    padding: 1.05rem 1.2rem;
    margin-bottom: 0.6rem;
    transition: box-shadow 0.18s ease, border-color 0.18s ease;
}}
.wc:hover {{
    box-shadow: 0 4px 20px rgba(100,100,200,0.10);
    border-color: {C_PERI};
}}
/* Ranks 1-5 get gradient fill */
.wc-p {{
    background: linear-gradient(115deg, #EAECFF 0%, #F1F2FF 45%, {C_WHITE} 100%);
    border-color: #C8CCFF;
    border-left: 3px solid {C_WIST};
    box-shadow: 0 2px 14px rgba(156,161,255,0.15);
}}
.wc-rank {{
    font-size: 0.67rem;
    font-weight: 800;
    color: {C_WIST};
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.25rem;
}}
.wc-title {{
    font-size: 0.95rem;
    font-weight: 700;
    color: {C_INK};
    line-height: 1.42;
    margin: 0.28rem 0 0.28rem;
}}
.wc-note {{
    font-size: 0.82rem;
    color: {C_MUTED};
    line-height: 1.5;
    margin-top: 0.28rem;
}}

/* ════════════════════════════════════
   RESULT CARDS (evidence feed)
════════════════════════════════════ */
.rc {{
    background: {C_WHITE};
    border: 1px solid {C_BORDER};
    border-radius: 12px;
    padding: 1rem 1.2rem 0.9rem;
    margin-bottom: 0.55rem;
    transition: box-shadow 0.18s ease, border-color 0.18s ease;
}}
.rc:hover {{
    box-shadow: 0 4px 20px rgba(100,100,200,0.09);
    border-color: {C_PERI};
}}
.rc-title {{
    font-size: 0.94rem;
    font-weight: 700;
    color: {C_INK};
    line-height: 1.42;
    margin: 0.28rem 0 0.28rem;
}}
.rc-note {{
    font-size: 0.82rem;
    color: {C_MUTED};
    line-height: 1.5;
    margin-top: 0.3rem;
}}

/* ════════════════════════════════════
   SOURCE BREAKDOWN BARS
════════════════════════════════════ */
.src-wrap {{ padding-top: 0.2rem; }}
.src-row {{
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.6rem;
    font-size: 0.81rem;
}}
.src-label {{ width: 136px; flex-shrink:0; color:{C_INK}; font-weight:500; }}
.src-track {{
    flex: 1;
    height: 6px;
    background: #EAECF8;
    border-radius: 3px;
    overflow: hidden;
}}
.src-fill {{ height: 100%; border-radius: 3px; transition: width 0.4s ease; }}
.src-count {{ width: 30px; text-align:right; color:{C_MUTED}; font-weight:600; font-size:0.78rem; }}

/* ════════════════════════════════════
   KEYWORD PILLS (relevance profile)
════════════════════════════════════ */
.kw-cluster {{
    font-size: 0.7rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: {C_WIST};
    margin: 1.4rem 0 0.6rem;
}}
.kw-cloud {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem 0.5rem;
    margin-bottom: 0.3rem;
}}
.kw-pill {{
    display: inline-flex;
    align-items: center;
    min-height: 2rem;
    border-radius: 999px;
    padding: 0.24rem 0.85rem;
    font-size: 0.81rem;
    font-weight: 600;
    text-decoration: none !important;
    transition: all 0.12s ease;
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
    align-items: center;
}}
.kw-legend-item {{
    display: flex;
    align-items: center;
    gap: 0.38rem;
    font-size: 0.81rem;
    font-weight: 500;
    color: {C_MUTED};
    background: {C_WHITE};
    border: 1px solid {C_BORDER};
    border-radius: 8px;
    padding: 0.38rem 0.65rem;
}}
.kw-dot {{
    width: 7px; height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
}}
.kw-dot-active  {{ background:{C_WIST}; }}
.kw-dot-muted   {{ background:{C_BORDER}; }}
.kw-dot-removed {{ background:#E05454; }}

/* ════════════════════════════════════
   STREAMLIT OVERRIDES
════════════════════════════════════ */
[data-testid="stMetric"] {{
    background: {C_WHITE} !important;
    border: 1px solid {C_BORDER} !important;
    border-radius: 12px !important;
    padding: 0.9rem 1rem !important;
    box-shadow: 0 1px 4px rgba(100,100,200,0.06) !important;
}}
[data-testid="stMetricLabel"] p {{
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    color: {C_MUTED2} !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
}}
[data-testid="stMetricValue"] {{
    font-size: 1.75rem !important;
    font-weight: 800 !important;
    color: {C_INK} !important;
    letter-spacing: -0.02em !important;
    line-height: 1.2 !important;
}}
[data-testid="stExpander"] {{
    border: 1px solid {C_BORDER} !important;
    border-radius: 10px !important;
    background: {C_WHITE} !important;
}}
[data-testid="stExpander"] summary {{
    font-size: 0.84rem !important;
    font-weight: 600 !important;
    color: {C_SLATE} !important;
    padding: 0.55rem 0.75rem !important;
}}
[data-testid="stDataFrame"] {{
    border-radius: 10px !important;
    border: 1px solid {C_BORDER} !important;
    overflow: hidden !important;
}}
hr {{ border-color: {C_BORDER} !important; margin: 0.9rem 0 !important; }}
[data-testid="stCaptionContainer"] p {{
    color: {C_MUTED2} !important;
    font-size: 0.81rem !important;
}}
h2, h3 {{
    font-family: "Inter", sans-serif !important;
    font-weight: 700 !important;
    color: {C_INK} !important;
    letter-spacing: -0.01em !important;
}}
h2 {{ font-size: 1.05rem !important; }}
h3 {{ font-size: 0.92rem !important; }}
.stButton > button {{
    font-family: "Inter", sans-serif !important;
    font-size: 0.84rem !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    border: 1.5px solid {C_SLATE} !important;
    color: {C_SLATE} !important;
    background: transparent !important;
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
}}
.stDownloadButton > button {{
    font-family: "Inter", sans-serif !important;
    font-size: 0.84rem !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    border: 1.5px solid rgba(255,255,255,0.55) !important;
    color: {C_WHITE} !important;
    background: rgba(255,255,255,0.15) !important;
}}
.stDownloadButton > button:hover {{
    background: rgba(255,255,255,0.28) !important;
}}
/* Tighter column gaps */
[data-testid="column"] {{ padding: 0 0.4rem !important; }}
</style>
"""


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def tag_badge(kw: str) -> str:
    c = TAG_COLORS.get(kw.lower(), "gray")
    return f'<span class="tb tb-{c}">{escape(kw)}</span>'

def meta_badge(text: str, kind: str = "") -> str:
    cls = f" mb-{kind}" if kind else ""
    return f'<span class="mb{cls}">{escape(str(text))}</span>'

def score_chip(score: float) -> str:
    if score >= 0.80:   tone, label = "high", "High"
    elif score >= 0.65: tone, label = "mid",  "Medium"
    else:               tone, label = "low",  "Low"
    return f'<span class="sc sc-{tone}">{label}</span>'

def fmt_date(v) -> str:
    return "Unknown" if pd.isna(v) else str(v)[:10]

def source_meta(row) -> str:
    return (
        meta_badge(str(getattr(row,"source_type","") or "").upper(), "src")
        + meta_badge(getattr(row,"source_label","") or "Unknown", "src")
        + meta_badge(fmt_date(getattr(row,"item_date",None)), "date")
    )

def watchlist_card(row, rank: int) -> str:
    matched = getattr(row,"matched_keywords",None) or []
    note    = getattr(row,"relevance_note","") or getattr(row,"summary","") or getattr(row,"body","") or ""
    note    = " ".join(str(note).split())
    if len(note) > 210: note = note[:207].rstrip() + "…"
    score   = float(getattr(row,"score",0) or 0)
    title   = escape(getattr(row,"title","") or "Untitled")
    tags    = "".join(tag_badge(kw) for kw in matched[:5])
    pri     = " wc-p" if rank <= 5 else ""
    return f"""
    <div class="wc{pri}">
        <div class="wc-rank">#{rank}</div>
        <div style="display:flex;align-items:center;gap:0.35rem;flex-wrap:wrap;">
            {score_chip(score)}<strong style="font-size:0.8rem;color:{C_SLATE}">{score:.2f}</strong>
            {source_meta(row)}
        </div>
        <div class="wc-title">{title}</div>
        <div>{tags}</div>
        <div class="wc-note">{escape(note or "No summary available yet.")}</div>
    </div>"""

def result_card(row) -> str:
    matched = getattr(row,"matched_keywords",None) or []
    note    = getattr(row,"relevance_note","") or getattr(row,"summary","") or getattr(row,"body","") or ""
    note    = " ".join(str(note).split())
    if len(note) > 200: note = note[:197].rstrip() + "…"
    score   = float(getattr(row,"score",0) or 0)
    title   = escape(getattr(row,"title","") or "Untitled")
    tags    = "".join(tag_badge(kw) for kw in matched[:5])
    return f"""
    <div class="rc">
        <div style="display:flex;align-items:center;gap:0.35rem;flex-wrap:wrap;">
            {score_chip(score)}<strong style="font-size:0.8rem;color:{C_SLATE}">{score:.2f}</strong>
            {source_meta(row)}
        </div>
        <div class="rc-title">{title}</div>
        <div>{tags}</div>
        <div class="rc-note">{escape(note or "No summary available yet.")}</div>
    </div>"""

def source_bars_html(df: pd.DataFrame) -> str:
    if "data_source" not in df.columns: return ""
    colors = {
        "pubmed":"#396070","clinicaltrials":"#9CA1FF",
        "europepmc":"#C5CAFB","biorxiv":"#6B82A8","medrxiv":"#8A9EC0",
    }
    labels = {
        "pubmed":"PubMed","clinicaltrials":"ClinicalTrials.gov",
        "europepmc":"Europe PMC","biorxiv":"bioRxiv","medrxiv":"medRxiv",
    }
    counts = df["data_source"].value_counts()
    if counts.empty: return ""
    mx = counts.max()
    rows = ""
    for src, cnt in counts.items():
        color = colors.get(src,"#9CA1FF")
        label = labels.get(src, src)
        pct   = cnt / mx * 100
        rows += f"""
        <div class="src-row">
            <div class="src-label">{label}</div>
            <div class="src-track"><div class="src-fill" style="width:{pct:.0f}%;background:{color}"></div></div>
            <div class="src-count">{cnt}</div>
        </div>"""
    return f'<div class="src-wrap">{rows}</div>'

def profile_keywords() -> list[str]:
    seen, out = set(), []
    for ks in KEYWORD_CLUSTERS.values():
        for k in ks:
            if k not in seen:
                seen.add(k); out.append(k)
    return out

def keyword_button_key(kw: str) -> str:
    return f"kw_btn_{profile_keywords().index(kw)}"

def cycle_keyword_state(kw: str) -> None:
    cur = st.session_state.kw_states.get(kw, "active")
    st.session_state.kw_states[kw] = STATE_CYCLE.get(cur, "active")

def keyword_button_styles() -> str:
    tones = {
        "active":  dict(bg=C_PERI,   border=C_WIST,   color="#1A1E6A", style="solid", deco="none"),
        "muted":   dict(bg=C_WHITE,  border=C_BORDER, color=C_MUTED,   style="dashed", deco="none"),
        "removed": dict(bg="#FFF0F0", border="#F2B7B7", color="#9B2020", style="solid", deco="line-through"),
    }
    rules = []
    for kw in profile_keywords():
        state = st.session_state.kw_states.get(kw, "active")
        tone = tones.get(state, tones["active"])
        key = keyword_button_key(kw)
        rules.append(f"""
        [class*="st-key-{key}"] button {{
            min-height: 2rem !important;
            border-radius: 999px !important;
            padding: 0.24rem 0.85rem !important;
            font-size: 0.81rem !important;
            font-weight: 600 !important;
            line-height: 1.2 !important;
            background: {tone["bg"]} !important;
            border: 1px {tone["style"]} {tone["border"]} !important;
            color: {tone["color"]} !important;
            text-decoration: {tone["deco"]} !important;
            box-shadow: none !important;
            transition: transform 0.12s ease, box-shadow 0.12s ease !important;
        }}
        [class*="st-key-{key}"] button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 2px 8px rgba(57,96,112,0.08) !important;
        }}
        """)
    return "<style>" + "\n".join(rules) + "</style>"

def score_to_color(val: float) -> str:
    """Periwinkle #C5CAFB → Blue Slate #396070 by score."""
    r = int(197 + (57  - 197) * val)
    g = int(202 + (96  - 202) * val)
    b = int(251 + (112 - 251) * val)
    return f"rgb({r},{g},{b})"


# ─────────────────────────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Embio Intelligence",
    page_icon=str(ICON_PATH),
    layout="wide",
)
st.markdown(CSS, unsafe_allow_html=True)

@st.cache_resource
def ensure_db():
    init_db()

ensure_db()


# ─────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────

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
    meta    = pd.concat([articles, trials], ignore_index=True)
    results = scored.merge(meta, on=["id","source_type"], how="left")
    cached  = []
    for _, row in results.iterrows():
        s = get_cached_summary(row.id, row.source_type) or {}
        cached.append({"summary": s.get("summary",""), "relevance_note": s.get("relevance_note",""), "tags": s.get("tags",[])})
    return pd.concat([results.reset_index(drop=True), pd.DataFrame(cached)], axis=1)

def refresh_data():
    load_results.clear(); st.rerun()

@contextmanager
def refresh_lock():
    REFRESH_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REFRESH_LOCK_PATH.open("w") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)

def refresh_pipeline():
    with refresh_lock() as acquired:
        if not acquired:
            st.warning("Refresh is already running. Please wait for it to finish before starting another one.")
            return
        with st.spinner("Fetching sources and embedding…"):
            run_once()
            embed_all_pending()
    refresh_data()


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    if LOGO_PATH.exists():
        logo_src = "data:image/png;base64," + base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
        st.markdown(
            f'<div class="sidebar-logo"><img src="{logo_src}" alt="Embio Medical"></div>',
            unsafe_allow_html=True,
        )
    source_filter  = st.multiselect("Source type", ["article","trial"], default=["article","trial"])
    min_score      = st.slider("Min relevance score", 0.0, 1.0, DEFAULT_MIN_RELEVANCE, 0.01)
    years_back     = st.slider("Published within (years)", 1, 5, 2, 1)
    st.divider()
    summary_filter = st.radio("Summary status", ["All records","Needs summary","Has AI summary"])
    st.divider()
    if st.button("Refresh", width="stretch"):
        refresh_pipeline()
    st.caption("Fetches new data and embeds new documents.")


# ─────────────────────────────────────────────────────────────
# LOAD + FILTER
# ─────────────────────────────────────────────────────────────

results = load_results()

if results.empty:
    st.markdown('<div class="hero"><div class="hero-title">Embio Intelligence</div><div class="hero-sub">No embedded records yet. Click Refresh in the sidebar.</div></div>', unsafe_allow_html=True)
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

with get_connection() as con:
    feedback_count = con.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]

latest_fetch   = results["fetched_at"].dropna().max() if "fetched_at" in results else None
week_cutoff    = date.today() - timedelta(days=7)
new_this_week  = int((pd.to_datetime(filtered["item_date"], errors="coerce").dt.date >= week_cutoff).sum())
high_relevance = int((filtered["score"] >= 0.65).sum())
last_str       = pd.Timestamp(latest_fetch).strftime("%-d %b, %H:%M") if latest_fetch else "Never"
top_score      = float(filtered["score"].max()) if not filtered.empty else 0.0

pdf_bytes = build_weekly_pdf(filtered)
pdf_filename = f"embio-weekly-{date.today().isoformat()}.pdf"
pdf_href = "data:application/pdf;base64," + base64.b64encode(pdf_bytes).decode("ascii")


# ─────────────────────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="hero">
    <div class="hero-eyebrow">Embio Medical AB · Intelligence Platform</div>
    <div class="hero-title-row">
        <div class="hero-title">Research & Market Radar</div>
        <a class="hero-download-btn" href="{pdf_href}" download="{pdf_filename}">Download weekly PDF</a>
    </div>
    <div class="hero-sub">
        Electroporation catheters · intraductal &amp; ERCP delivery ·
        pancreatic cancer · adjacent medtech signals
    </div>
    <div class="hero-metrics">
        <div class="hm">
            <div class="hm-label">Total records</div>
            <div class="hm-value">{len(results):,}</div>
            <div class="hm-sub"><span class="hm-accent">+{new_this_week}</span> this week</div>
        </div>
        <div class="hm">
            <div class="hm-label">Visible</div>
            <div class="hm-value">{len(filtered):,}</div>
            <div class="hm-sub">above {min_score:.2f} threshold</div>
        </div>
        <div class="hm">
            <div class="hm-label">High relevance</div>
            <div class="hm-value">{high_relevance:,}</div>
            <div class="hm-sub">score &gt;= 0.65</div>
        </div>
        <div class="hm">
            <div class="hm-label">Feedback signals</div>
            <div class="hm-value">{feedback_count:,}</div>
            <div class="hm-sub">ML training labels</div>
        </div>
        <div class="hm">
            <div class="hm-label">Last updated</div>
            <div class="hm-value" style="font-size:1.3rem">{last_str}</div>
            <div class="hm-sub">top score {top_score:.2f}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────

tab_overview, tab_feed, tab_profile = st.tabs(["Overview", "Evidence feed", "Relevance profile"])


# ══════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════

with tab_overview:

    col_l, col_r = st.columns(2, gap="medium")

    # ── Publications over time — AREA chart with gradient fill ──
    with col_l:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Publication volume</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-sub">Monthly article count, last 3 years</div>', unsafe_allow_html=True)
        with get_connection() as con:
            trend = con.execute("""
                SELECT date_trunc('month', pub_date) AS month, COUNT(*) AS count
                FROM articles
                WHERE pub_date IS NOT NULL
                  AND pub_date >= now() - INTERVAL '3 years'
                GROUP BY 1 ORDER BY 1
            """).df()
        if not trend.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trend["month"], y=trend["count"],
                mode="lines",
                line=dict(color=C_WIST, width=2.5),
                fill="tozeroy",
                fillcolor="rgba(156,161,255,0.12)",
                hovertemplate="<b>%{x|%b %Y}</b><br>%{y} articles<extra></extra>",
            ))
            fig.update_layout(
                **BASE_LAYOUT, height=200,
                xaxis=dict(showgrid=False, tickfont=dict(size=10), zeroline=False),
                yaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(size=10), zeroline=False, title=""),
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.caption("No dated articles yet.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Score distribution — gradient bars ──
    with col_r:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Relevance score distribution</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="card-sub">All {len(results):,} records · threshold at {min_score:.2f}</div>', unsafe_allow_html=True)
        scores = results["score"].dropna().values
        hist_counts, bin_edges = np.histogram(scores, bins=25, range=(0, 1))
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
        bar_colors  = [score_to_color(float(v)) for v in bin_centres]
        fig2 = go.Figure(go.Bar(
            x=bin_centres, y=hist_counts,
            width=(bin_edges[1] - bin_edges[0]) * 0.82,
            marker_color=bar_colors, marker_line_width=0,
            hovertemplate="Score %{x:.2f}<br>%{y} records<extra></extra>",
        ))
        fig2.update_traces(marker_cornerradius=4)
        fig2.add_vline(
            x=min_score, line_dash="dot", line_color=C_SLATE, line_width=1.5,
            annotation_text="  threshold", annotation_font_size=10,
            annotation_font_color=C_SLATE,
        )
        fig2.update_layout(
            **BASE_LAYOUT, height=200,
            xaxis=dict(title="relevance score", showgrid=False, tickfont=dict(size=10)),
            yaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(size=10), zeroline=False),
            bargap=0.08,
        )
        st.plotly_chart(fig2, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    col_l2, col_r2 = st.columns(2, gap="medium")

    # ── Top keyword frequency — coloured by cluster ──
    with col_l2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Top keyword matches</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-sub">Frequency across all records · coloured by research cluster</div>', unsafe_allow_html=True)
        all_kws = [kw for kws in results["matched_keywords"].dropna() for kw in kws]
        if all_kws:
            top_kws = pd.DataFrame(Counter(all_kws).most_common(12), columns=["keyword","count"])
            def kw_color(kw):
                for cluster, ks in KEYWORD_CLUSTERS.items():
                    if kw in ks: return CLUSTER_COLORS.get(cluster, CHART_SLATE)
                return CHART_SLATE
            top_kws["color"] = top_kws["keyword"].apply(kw_color)
            fig3 = go.Figure(go.Bar(
                x=top_kws["count"], y=top_kws["keyword"],
                orientation="h",
                marker_color=top_kws["color"].tolist(),
                marker_line_width=0,
                hovertemplate="%{y}<br>%{x} matches<extra></extra>",
            ))
            fig3.update_traces(marker_cornerradius=4)
            fig3.update_layout(
                **BASE_LAYOUT, height=330,
                xaxis=dict(showgrid=True, gridcolor=GRID_COLOR, tickfont=dict(size=10), zeroline=False),
                yaxis=dict(showgrid=False, tickfont=dict(size=10), autorange="reversed"),
                bargap=0.28,
            )
            st.plotly_chart(fig3, width="stretch")
        else:
            st.caption("No keyword matches yet.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Source breakdown + ingestion log ──
    with col_r2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Source breakdown</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="card-sub">{len(results):,} total records across all ingestion sources</div>', unsafe_allow_html=True)
        st.markdown(source_bars_html(results), unsafe_allow_html=True)
        st.markdown('<hr style="margin:1rem 0">', unsafe_allow_html=True)
        st.markdown('<div class="card-title" style="font-size:0.8rem;letter-spacing:0.04em;text-transform:uppercase;color:#9CA3AF">Ingestion log — last 30 days</div>', unsafe_allow_html=True)
        try:
            health = ingestion_summary(days=30)
            if health:
                hdf = pd.DataFrame(health)[["source","total_new","total_updated","last_run"]]
                hdf["last_run"] = pd.to_datetime(hdf["last_run"]).dt.strftime("%d %b %H:%M")
                st.dataframe(hdf, hide_index=True, width="stretch")
            else:
                st.caption("No ingestion log entries yet.")
        except Exception:
            st.caption("Ingestion log not available.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Cluster activity — small multiples (articles per cluster over time) ──
    with st.expander("Research cluster activity — scoring diagnostics"):
        st.caption(
            "Semantic score vs keyword score for all visible records. "
            "Ideal results cluster top-right (high on both axes). "
            "Bubble size = composite score."
        )
        fig5 = px.scatter(
            filtered, x="semantic_score", y="keyword_score",
            color="source_type", hover_name="title",
            size="score", size_max=13,
            color_discrete_map={"article": CHART_SLATE, "trial": CHART_WIST},
            opacity=0.65,
        )
        fig5.update_layout(
            **BASE_LAYOUT, height=300,
            xaxis=dict(title="semantic score", gridcolor=GRID_COLOR, tickfont=dict(size=10), range=[0,1]),
            yaxis=dict(title="keyword score",  gridcolor=GRID_COLOR, tickfont=dict(size=10), range=[0,1]),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=11)),
        )
        # Draw quadrant lines
        for x_val in [0.5]:
            fig5.add_vline(x=x_val, line_dash="dot", line_color=C_BORDER, line_width=1)
        for y_val in [0.5]:
            fig5.add_hline(y=y_val, line_dash="dot", line_color=C_BORDER, line_width=1)
        st.plotly_chart(fig5, width="stretch")


# ══════════════════════════════════════════════════════════════
# TAB 2 — EVIDENCE FEED
# ══════════════════════════════════════════════════════════════

with tab_feed:

    if filtered.empty:
        st.info("No results match the current filters. Try lowering the minimum relevance score.")
        st.stop()

    st.subheader("Priority watchlist")
    st.caption("Top 5 highest-scoring results")
    for rank, row in enumerate(filtered.head(5).itertuples(index=False), start=1):
        st.markdown(watchlist_card(row, rank), unsafe_allow_html=True)

    st.divider()
    st.subheader(f"All results — {len(filtered):,}")

    for _, row in filtered.iterrows():
        st.markdown(result_card(row), unsafe_allow_html=True)

        with st.expander("View details"):
            left, right = st.columns([3, 1])
            with left:
                if row.summary:
                    st.markdown(f"**Summary**  \n{row.summary}")
                    st.markdown(f"**Why it matters**  \n{row.relevance_note}")
                    if row.tags:
                        st.markdown(
                            "**Tags:** " + "".join(tag_badge(t) for t in row.tags),
                            unsafe_allow_html=True,
                        )
                else:
                    body = row.body or "No abstract or study details available."
                    st.markdown(body[:800] + ("…" if len(body) > 800 else ""))
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
            a = st.columns([1,1,1,3])
            if a[0].button("👍 Relevant",     key=f"up_{row.source_type}_{row.id}"):
                record_feedback(row.id, row.source_type, 1); refresh_data()
            if a[1].button("👎 Not relevant", key=f"dn_{row.source_type}_{row.id}"):
                record_feedback(row.id, row.source_type, -1); refresh_data()
            a[2].link_button("Open source", row.url)

        st.divider()


# ══════════════════════════════════════════════════════════════
# TAB 3 — RELEVANCE PROFILE
# ══════════════════════════════════════════════════════════════

with tab_profile:

    saved   = load_user_profile()
    s_states  = saved.get("keyword_states", {})
    s_weights = saved.get("scoring_weights", {})

    st.subheader("Keyword filter")
    st.markdown(f"""
    <div class="kw-legend-row">
        <div class="kw-legend-item"><span class="kw-dot kw-dot-active"></span><strong>Active</strong> — boosts score</div>
        <div class="kw-legend-item"><span class="kw-dot kw-dot-muted"></span><strong>Muted</strong> — ignored</div>
        <div class="kw-legend-item"><span class="kw-dot kw-dot-removed"></span><strong>Removed</strong> — suppresses</div>
        <span style="font-size:0.77rem;color:{C_MUTED2};font-style:italic;align-self:center">Click any pill to cycle its state, then save to apply</span>
    </div>
    """, unsafe_allow_html=True)

    if "kw_states" not in st.session_state:
        st.session_state.kw_states = {**{kw: "active" for kw in profile_keywords()}, **s_states}
    if st.session_state.get("profile_saved"):
        st.success("Profile saved. Scores have been refreshed.")
        del st.session_state.profile_saved

    if "weights" not in st.session_state:
        st.session_state.weights = {
            "semantic": s_weights.get("semantic", SCORING_WEIGHTS["semantic"]),
            "keyword":  s_weights.get("keyword",  SCORING_WEIGHTS["keyword"]),
            "recency":  s_weights.get("recency",  SCORING_WEIGHTS["recency"]),
            "feedback": s_weights.get("feedback", SCORING_WEIGHTS["feedback"]),
        }

    st.markdown(keyword_button_styles(), unsafe_allow_html=True)

    for cluster_name, keywords in KEYWORD_CLUSTERS.items():
        st.markdown(f'<div class="kw-cluster">{escape(cluster_name)}</div>', unsafe_allow_html=True)
        for start in range(0, len(keywords), 4):
            cols = st.columns(4)
            for col, kw in zip(cols, keywords[start:start + 4]):
                state = st.session_state.kw_states.get(kw, "active")
                next_state = STATE_CYCLE.get(state, "active")
                col.button(
                    kw,
                    key=keyword_button_key(kw),
                    help=f"Current: {state}. Click to set {next_state}.",
                    on_click=cycle_keyword_state,
                    args=(kw,),
                    width="stretch",
                )

    a_cnt = sum(1 for s in st.session_state.kw_states.values() if s == "active")
    m_cnt = sum(1 for s in st.session_state.kw_states.values() if s == "muted")
    r_cnt = sum(1 for s in st.session_state.kw_states.values() if s == "removed")
    st.caption(f"Active: {a_cnt}  ·  Muted: {m_cnt}  ·  Removed: {r_cnt}")

    st.divider()
    st.subheader("Scoring weights")
    st.caption("Adjust how each signal contributes to the composite score. Must sum to 1.0.")

    w_sem = st.slider("Semantic weight",  0.0, 0.9, float(st.session_state.weights["semantic"]),  0.05)
    w_kw  = st.slider("Keyword weight",   0.0, 0.9, float(st.session_state.weights["keyword"]),   0.05)
    w_rec = st.slider("Recency weight",   0.0, 0.5, float(st.session_state.weights["recency"]),   0.05)
    w_fb  = st.slider("Feedback weight",  0.0, 0.3, float(st.session_state.weights["feedback"]),  0.05)

    total = round(w_sem + w_kw + w_rec + w_fb, 2)
    if abs(total - 1.0) > 0.01:
        st.warning(f"Weights sum to {total:.2f} — adjust to reach 1.0.")
    else:
        st.success(f"Weights sum to {total:.2f} ✓")

    sc, rc, _ = st.columns([1,1,3])
    if sc.button("Save profile", width="stretch"):
        nw = {"semantic": w_sem, "keyword": w_kw, "recency": w_rec, "feedback": w_fb}
        st.session_state.weights = nw
        save_user_profile(st.session_state.kw_states, nw)
        load_results.clear()
        st.session_state.profile_saved = True
        st.rerun()
    if rc.button("Reset to defaults", width="stretch"):
        st.session_state.kw_states = {kw: "active" for kw in profile_keywords()}
        st.session_state.weights   = dict(SCORING_WEIGHTS)
        save_user_profile(st.session_state.kw_states, st.session_state.weights)
        load_results.clear()
        st.rerun()
