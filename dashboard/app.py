"""
dashboard/app.py

Three-tab layout:
  Overview         — metrics, trend chart, source breakdown, keyword frequency
  Evidence feed    — filtered, scored result cards with feedback and summaries
  Relevance profile — keyword state editor (active/muted/removed) + weight sliders
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

import pandas as pd
import plotly.express as px
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


# ---------------------------------------------------------------------------
# Keyword clusters — drives the Relevance profile tab
# Each cluster is rendered as a labelled group of pills
# ---------------------------------------------------------------------------

KEYWORD_CLUSTERS = {
    "Electroporation core": [
        "electroporation",
        "irreversible electroporation",
        "reversible electroporation",
        "electrochemotherapy",
        "calcium electroporation",
        "pulsed electric field",
        "pef",
        "non-thermal ablation",
        "low voltage electroporation",
        "electroporation parameters",
        "nanoknife",
    ],
    "Catheter / intraductal": [
        "electroporation catheter",
        "catheter-based electroporation",
        "flexible catheter",
        "catheter",
        "intraductal catheter",
        "bipolar electrode catheter",
        "ring electrode catheter",
        "microcatheter",
        "intraductal electroporation",
        "pancreatic duct electroporation",
        "ercp electroporation",
        "ercp guided ablation",
    ],
    "Oncology / clinical": [
        "pancreatic cancer",
        "pancreatic ductal adenocarcinoma",
        "pdac",
        "pancreatic adenocarcinoma",
        "locally advanced pancreatic cancer",
        "cholangiocarcinoma",
        "pancreatic intraepithelial neoplasia",
        "panin",
        "ablation",
        "tumor ablation",
        "minimally invasive",
        "first-in-human",
        "interventional oncology",
        "locoregional therapy",
        "endoscopic ablation",
        "biliary electroporation",
    ],
    "Drug delivery": [
        "drug delivery",
        "localized drug delivery",
        "intratumoral drug delivery",
        "pancreatic duct drug delivery",
        "catheter-based drug delivery",
        "endoluminal drug delivery",
    ],
    "Biomarkers / diagnostics": [
        "pancreatic juice biomarker",
        "pancreatic liquid biopsy",
        "pancreatic juice",
        "pancreatic cyst fluid",
        "kras pancreatic",
        "tp53 pancreatic",
        "circulating tumor dna",
        "ctdna",
        "exosomes pancreatic",
        "early pancreatic cancer detection",
        "pdac early diagnosis",
        "ercp pancreatic juice",
        "pancreatic duct sampling",
    ],
    "Adjacent / strategic": [
        "eus-guided",
        "endoscopic ultrasound",
        "clinical feasibility",
        "safety",
        "organoid pancreatic cancer",
    ],
}

# Individual keyword overrides (used in the evidence feed badges)
TAG_COLORS = {
    "pancreatic cancer": "red",
    "pancreatic adenocarcinoma": "red",
    "pancreatic ductal adenocarcinoma": "red",
    "pdac": "red",
    "locally advanced pancreatic cancer": "red",
    "cholangiocarcinoma": "red",
    "panin": "red",
    "electroporation": "blue",
    "electrochemotherapy": "blue",
    "irreversible electroporation": "blue",
    "calcium electroporation": "blue",
    "pulsed electric field": "blue",
    "nanoknife": "blue",
    "ablation": "blue",
    "tumor ablation": "blue",
    "electroporation catheter": "green",
    "flexible catheter": "green",
    "catheter": "green",
    "intraductal": "green",
    "ercp": "green",
    "eus-guided": "green",
    "endoscopic ultrasound": "green",
    "minimally invasive": "green",
    "first-in-human": "orange",
    "clinical feasibility": "orange",
    "safety": "orange",
    "interventional oncology": "orange",
    "locoregional therapy": "orange",
    "drug delivery": "violet",
    "localized drug delivery": "violet",
    "intratumoral drug delivery": "violet",
    "pancreatic juice": "orange",
    "pancreatic juice biomarker": "orange",
    "liquid biopsy": "orange",
    "ctdna": "orange",
}

PLOTLY_LAYOUT = dict(
    margin=dict(l=0, r=0, t=4, b=0),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(size=11, family="Inter, sans-serif", color="#396070"),
)
CHART_GREEN  = "#396070"
CHART_BLUE   = "#9CA1FF"
CHART_PURPLE = "#C5CAFB"
CHART_CORAL  = "#6B7280"
CHART_MUTED  = "#C5CAFB"

DASHBOARD_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --embio-periwinkle: #C5CAFB;
    --embio-wisteria: #9CA1FF;
    --embio-slate: #396070;
    --embio-black: #000000;
    --embio-white: #FFFFFF;
    --ink: #121820;
    --muted: #68717A;
    --line: #E6E8F2;
    --soft: #F6F7FF;
}

html, body, [class*="css"], [data-testid="stAppViewContainer"] {
    font-family: "Inter", sans-serif;
    color: var(--ink);
}

.stApp {
    background: var(--embio-white);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #F7F8FF 0%, #FFFFFF 100%);
    border-right: 1px solid var(--line);
}

[data-testid="stSidebar"] [data-testid="stImage"] {
    margin: 0.35rem 0 1rem;
}

[data-testid="stSidebar"] h1 {
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 0;
    margin-bottom: 0.15rem;
}

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    font-family: "Inter", sans-serif;
}

.block-container {
    max-width: 1240px;
    padding-top: 3.4rem;
    padding-bottom: 4rem;
}

h1, h2, h3 {
    font-family: "Inter", sans-serif;
    color: var(--ink);
    letter-spacing: 0;
}

h1 {
    font-size: 2.45rem;
    font-weight: 800;
    margin-bottom: 0.45rem;
}

h2, h3 {
    font-weight: 750;
}

[data-testid="stCaptionContainer"] {
    color: var(--muted);
}

[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 1rem 1.05rem;
    box-shadow: 0 10px 28px rgba(57, 96, 112, 0.05);
}

[data-testid="stMetricLabel"] {
    color: var(--muted);
    font-size: 0.82rem;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: var(--embio-slate);
    font-weight: 750;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 1.1rem;
    border-bottom: 1px solid var(--line);
}

.stTabs [data-baseweb="tab"] {
    padding: 0.8rem 0 0.65rem;
    color: var(--muted);
    font-weight: 650;
}

.stTabs [aria-selected="true"] {
    color: var(--embio-slate);
}

.stTabs [aria-selected="true"]::after {
    background-color: var(--embio-slate);
}

.stButton > button,
.stLinkButton > a {
    border-radius: 8px;
    border-color: var(--embio-slate);
    color: var(--embio-slate);
    font-weight: 650;
}

.stButton > button[kind="primary"],
.stButton > button:hover,
.stLinkButton > a:hover {
    border-color: var(--embio-slate);
    background: var(--embio-slate);
    color: #FFFFFF;
}

.stSlider [data-baseweb="slider"] > div {
    color: var(--embio-wisteria);
}

hr {
    border-color: var(--line);
}

.score-chip {
    display: inline-flex;
    align-items: center;
    border-radius: 999px;
    padding: 0.17rem 0.55rem;
    margin-right: 0.35rem;
    font-size: 0.76rem;
    font-weight: 750;
    letter-spacing: 0.02em;
}

.score-chip-high {
    background: #E8F7EF;
    color: #087348;
}

.score-chip-mid {
    background: #FFF6D8;
    color: #8A5A00;
}

.score-chip-low {
    background: #FDECEC;
    color: #A32020;
}

.tag-badge {
    display: inline-flex;
    align-items: center;
    border-radius: 999px;
    padding: 0.18rem 0.5rem;
    margin: 0.1rem 0.18rem 0.1rem 0;
    font-size: 0.72rem;
    font-weight: 650;
    line-height: 1.2;
    border: 1px solid transparent;
}

.tag-badge-red {
    background: #F0F1FF;
    border-color: #D8DAFF;
    color: #5157C8;
}

.tag-badge-blue {
    background: #E7F0F3;
    border-color: #B8CDD5;
    color: var(--embio-slate);
}

.tag-badge-green {
    background: #EEF2F4;
    border-color: #CBD6DA;
    color: #2F5361;
}

.tag-badge-orange {
    background: #F8F8FF;
    border-color: var(--embio-periwinkle);
    color: #5C617C;
}

.tag-badge-violet {
    background: #F0F1FF;
    border-color: var(--embio-wisteria);
    color: #5055BF;
}

.tag-badge-gray {
    background: #F5F6FA;
    border-color: #E3E5EE;
    color: #5F6872;
}

.meta-badge {
    display: inline-flex;
    align-items: center;
    border-radius: 999px;
    padding: 0.18rem 0.52rem;
    margin: 0.08rem 0.18rem 0.12rem 0;
    font-size: 0.72rem;
    font-weight: 700;
    line-height: 1.2;
    background: #F6F7FF;
    border: 1px solid #E1E4F6;
    color: var(--embio-slate);
}

.meta-badge-source {
    background: #E7F0F3;
    border-color: #C7D8DE;
}

.meta-badge-date {
    background: #FFFFFF;
    border-color: #E3E5EE;
    color: var(--muted);
}

.watchlist-card {
    border: 1px solid #DDE3EF;
    border-left: 4px solid var(--embio-slate);
    border-radius: 8px;
    padding: 0.95rem 1rem;
    margin: 0.7rem 0;
    background: #FFFFFF;
    box-shadow: 0 10px 30px rgba(57, 96, 112, 0.08);
}

.watchlist-card-priority {
    border-color: #C8D2FF;
    box-shadow: 0 0 0 1px rgba(156, 161, 255, 0.16), 0 14px 34px rgba(57, 96, 112, 0.10);
}

.watchlist-title {
    margin: 0.35rem 0 0.25rem;
    font-size: 0.98rem;
    font-weight: 750;
    color: var(--ink);
}

.watchlist-note {
    color: var(--muted);
    font-size: 0.86rem;
    line-height: 1.45;
    margin-top: 0.25rem;
}

.kw-dot {
    width: 0.46rem;
    height: 0.46rem;
    border-radius: 999px;
    display: inline-block;
}
.kw-dot-active { background: #1D9E75; }
.kw-dot-muted { background: #94a3b8; }
.kw-dot-removed { background: #D64545; }
.kw-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 0.65rem;
    margin: 0.35rem 0 1.2rem;
}
.kw-legend-item {
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 0.56rem 0.7rem;
    background: #ffffff;
    color: var(--muted);
    font-size: 0.84rem;
}
.kw-legend-item strong {
    color: var(--ink);
}
.kw-click-note {
    margin-left: 0.6rem;
    color: var(--muted);
    font-style: italic;
    white-space: nowrap;
    align-self: center;
}
.kw-section-title {
    margin: 1.5rem 0 0.75rem;
    color: var(--embio-slate);
    font-size: 0.93rem;
    font-weight: 800;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.kw-pill-cloud {
    display: flex;
    flex-wrap: wrap;
    gap: 0.62rem 0.7rem;
    margin-bottom: 1.35rem;
}
.kw-pill {
    display: inline-flex;
    align-items: center;
    min-height: 2.26rem;
    border-radius: 999px;
    padding: 0.3rem 1rem;
    font-size: 0.98rem;
    font-weight: 650;
    line-height: 1.15;
    text-decoration: none !important;
    transition: border-color 120ms ease, background-color 120ms ease, transform 120ms ease;
}
.kw-pill:hover {
    transform: translateY(-1px);
}
.kw-pill-active {
    border: 1px solid #9BD9C7;
    background: #E1F7EF;
    color: #006C54 !important;
}
.kw-pill-muted {
    border: 1px dashed #C7CAD8;
    background: #ffffff;
    color: var(--muted) !important;
}
.kw-pill-removed {
    border: 1px solid #F2B7B7;
    background: #FFF0F0;
    color: #A71F1F !important;
    text-decoration: line-through !important;
}
</style>
"""

STATE_LABELS = {
    "active": "Active",
    "muted": "Muted",
    "removed": "Removed",
}
STATE_CYCLE = {"active": "muted", "muted": "removed", "removed": "active"}


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Embio Intelligence", page_icon=str(ICON_PATH), layout="wide")
st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)


@st.cache_resource
def ensure_database() -> None:
    init_db()


ensure_database()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_results() -> pd.DataFrame:
    scored = score_all()
    if scored.empty:
        return scored

    with get_connection() as con:
        articles = con.execute("""
            SELECT id, 'article' AS source_type,
                   journal AS source_label, authors, url, fetched_at,
                   COALESCE(source, 'pubmed') AS data_source
            FROM articles
        """).fetchdf()
        trials = con.execute("""
            SELECT id, 'trial' AS source_type,
                   sponsor AS source_label, status AS authors, url, fetched_at,
                   COALESCE(source, 'clinicaltrials') AS data_source
            FROM trials
        """).fetchdf()

    metadata = pd.concat([articles, trials], ignore_index=True)
    results  = scored.merge(metadata, on=["id", "source_type"], how="left")

    cached = []
    for _, row in results.iterrows():
        s = get_cached_summary(row.id, row.source_type) or {}
        cached.append({
            "summary":        s.get("summary", ""),
            "relevance_note": s.get("relevance_note", ""),
            "tags":           s.get("tags", []),
        })

    return pd.concat([results.reset_index(drop=True), pd.DataFrame(cached)], axis=1)


def refresh_data() -> None:
    load_results.clear()
    st.rerun()


def refresh_pipeline() -> None:
    with st.spinner("Fetching new sources and embedding documents..."):
        run_once()
        embed_all_pending()
    refresh_data()


def tag_badge(keyword: str) -> str:
    color = TAG_COLORS.get(keyword.lower(), "gray")
    return f'<span class="tag-badge tag-badge-{color}">{escape(keyword)}</span>'


def meta_badge(text: str, kind: str = "") -> str:
    cls = f" meta-badge-{kind}" if kind else ""
    return f'<span class="meta-badge{cls}">{escape(str(text))}</span>'


def score_chip(score: float) -> str:
    if score >= 0.80:
        tone = "high"
        label = "High"
    elif score >= 0.65:
        tone = "mid"
        label = "Medium"
    else:
        tone = "low"
        label = "Low"
    return f'<span class="score-chip score-chip-{tone}">{label}</span>'


def format_item_date(value) -> str:
    if pd.isna(value):
        return "Unknown date"
    return str(value)[:10]


def source_date_badges(row) -> str:
    source_type = str(getattr(row, "source_type", "") or "record").upper()
    source_label = getattr(row, "source_label", "") or "Unknown source"
    item_date = format_item_date(getattr(row, "item_date", None))
    return (
        meta_badge(source_type, "source")
        + meta_badge(source_label, "source")
        + meta_badge(item_date, "date")
    )


def watchlist_card(row, rank: int) -> str:
    matched = getattr(row, "matched_keywords", None) or []
    note = getattr(row, "relevance_note", "") or getattr(row, "summary", "") or getattr(row, "body", "")
    note = " ".join(str(note or "").split())
    if len(note) > 220:
        note = note[:217].rstrip() + "..."
    priority_class = " watchlist-card-priority" if rank <= 3 else ""
    tags = " ".join(tag_badge(kw) for kw in matched[:5])
    return f"""
    <div class="watchlist-card{priority_class}">
        <div>{score_chip(float(getattr(row, "score", 0) or 0))}<strong>{float(getattr(row, "score", 0) or 0):.2f}</strong> {source_date_badges(row)}</div>
        <div class="watchlist-title">{rank}. {escape(getattr(row, "title", "") or "Untitled")}</div>
        <div>{tags}</div>
        <div class="watchlist-note">{escape(note or "No summary available yet.")}</div>
    </div>
    """


def profile_keywords() -> list[str]:
    seen = set()
    keywords = []
    for cluster_keywords in KEYWORD_CLUSTERS.values():
        for keyword in cluster_keywords:
            if keyword not in seen:
                seen.add(keyword)
                keywords.append(keyword)
    return keywords


def keyword_pill(keyword: str, state: str) -> str:
    safe_keyword = escape(keyword)
    href = f"?cycle_kw={quote(keyword)}"
    return f'<a class="kw-pill kw-pill-{state}" href="{href}" target="_self">{safe_keyword}</a>'


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image(str(LOGO_PATH), width=150)
    st.title("Intelligence")
    st.caption("Research & market radar")
    st.divider()

    source_filter   = st.multiselect("Source type", ["article", "trial"], default=["article", "trial"])
    min_score       = st.slider("Min relevance score", 0.0, 1.0, DEFAULT_MIN_RELEVANCE, 0.01)
    years_back      = st.slider("Published within (years)", 1, 5, 2, 1)

    st.divider()
    summary_filter = st.radio(
        "Summary status",
        ["All records", "Needs summary", "Has AI summary"],
        horizontal=False,
    )

    st.divider()
    if st.button("Refresh", width="stretch"):
        refresh_pipeline()
    st.caption("Pulls new data from all sources and embeds new documents.")


# ---------------------------------------------------------------------------
# Load data + apply filters
# ---------------------------------------------------------------------------

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

# Metrics row — always visible above tabs
with get_connection() as con:
    feedback_count = con.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]

latest_fetch = results["fetched_at"].dropna().max() if "fetched_at" in results else None
week_cutoff = date.today() - timedelta(days=7)
filtered_dates = pd.to_datetime(filtered["item_date"], errors="coerce").dt.date
new_this_week = int((filtered_dates >= week_cutoff).sum())
high_relevance = int((filtered["score"] >= 0.65).sum())

m = st.columns(5)
m[0].metric("Total records",    f"{len(results):,}")
m[1].metric("New this week",    f"{new_this_week:,}")
m[2].metric("High relevance",   f"{high_relevance:,}")
m[3].metric("Feedback signals", f"{feedback_count:,}")
m[4].metric("Last updated",     str(latest_fetch)[:16] if latest_fetch is not None else "Never")

report_col, _ = st.columns([1.25, 4])
report_col.download_button(
    "Download weekly PDF",
    data=build_weekly_pdf(filtered),
    file_name=f"embio-weekly-insights-{date.today().isoformat()}.pdf",
    mime="application/pdf",
    width="stretch",
)

st.divider()


# ---------------------------------------------------------------------------
# Three tabs
# ---------------------------------------------------------------------------

tab_overview, tab_feed, tab_profile = st.tabs(["Overview", "Evidence feed", "Relevance profile"])


# ============================================================
# TAB 1 — OVERVIEW
# ============================================================

with tab_overview:

    col_l, col_r = st.columns(2)

    # Publications over time
    with col_l:
        st.subheader("Publications over time")
        with get_connection() as con:
            trend = con.execute("""
                SELECT date_trunc('month', pub_date) AS month, COUNT(*) AS count
                FROM articles
                WHERE pub_date IS NOT NULL
                  AND pub_date >= now() - INTERVAL '3 years'
                GROUP BY 1 ORDER BY 1
            """).df()
        if not trend.empty:
            fig = px.bar(trend, x="month", y="count", color_discrete_sequence=[CHART_GREEN])
            fig.update_traces(marker_cornerradius=3)
            fig.update_layout(**PLOTLY_LAYOUT, height=220, xaxis_title="", yaxis_title="articles")
            st.plotly_chart(fig, width="stretch")
        else:
            st.caption("No dated articles yet.")

    # Score distribution
    with col_r:
        st.subheader("Score distribution")
        fig2 = px.histogram(results, x="score", nbins=20, color_discrete_sequence=[CHART_GREEN])
        fig2.add_vline(x=min_score, line_dash="dash", line_color=CHART_MUTED,
                       annotation_text="threshold", annotation_font_size=10)
        fig2.update_traces(marker_cornerradius=3)
        fig2.update_layout(**PLOTLY_LAYOUT, height=220, xaxis_title="relevance score", yaxis_title="records")
        st.plotly_chart(fig2, width="stretch")

    col_l2, col_r2 = st.columns(2)

    # Top keyword frequency
    with col_l2:
        st.subheader("Top keyword matches")
        all_kws = [kw for kws in results["matched_keywords"].dropna() for kw in kws]
        if all_kws:
            top_kws = pd.DataFrame(Counter(all_kws).most_common(15), columns=["keyword", "count"])
            fig3 = px.bar(top_kws, x="count", y="keyword", orientation="h",
                          color_discrete_sequence=[CHART_GREEN])
            fig3.update_traces(marker_cornerradius=3)
            fig3.update_layout(**PLOTLY_LAYOUT, height=340,
                               xaxis_title="matches", yaxis_title="",
                               yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig3, width="stretch")
        else:
            st.caption("No keyword matches yet.")

    # Source breakdown + ingestion health
    with col_r2:
        st.subheader("Source breakdown")
        if "data_source" in results.columns:
            src = (results.groupby("data_source").size()
                   .reset_index(name="count")
                   .sort_values("count", ascending=False))
            color_map = {
                "pubmed": CHART_GREEN, "europepmc": CHART_PURPLE,
                "clinicaltrials": CHART_BLUE, "biorxiv": CHART_CORAL, "medrxiv": "#EF9F27",
            }
            fig4 = px.bar(src, x="data_source", y="count", color="data_source",
                          color_discrete_map=color_map)
            fig4.update_traces(marker_cornerradius=3)
            fig4.update_layout(**PLOTLY_LAYOUT, height=200,
                               xaxis_title="", yaxis_title="records", showlegend=False)
            st.plotly_chart(fig4, width="stretch")

        st.caption("Ingestion log — last 30 days")
        try:
            health = ingestion_summary(days=30)
            if health:
                hdf = pd.DataFrame(health)[["source", "total_new", "total_updated", "last_run"]]
                hdf["last_run"] = pd.to_datetime(hdf["last_run"]).dt.strftime("%Y-%m-%d %H:%M")
                st.dataframe(hdf, hide_index=True, width="stretch")
            else:
                st.caption("No log entries yet.")
        except Exception:
            st.caption("Ingestion log not available.")

    with st.expander("Scoring diagnostics"):
        st.caption("Semantic vs keyword score is mainly for checking how the ranking model behaves.")
        fig5 = px.scatter(
            filtered, x="semantic_score", y="keyword_score",
            color="source_type", hover_name="title", size="score", size_max=14,
            color_discrete_map={"article": CHART_GREEN, "trial": CHART_BLUE},
            opacity=0.7,
        )
        fig5.update_layout(**PLOTLY_LAYOUT, height=320,
                           xaxis_title="semantic score", yaxis_title="keyword score",
                           showlegend=True,
                           legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig5, width="stretch")


# ============================================================
# TAB 2 — EVIDENCE FEED
# ============================================================

with tab_feed:

    if filtered.empty:
        st.info("No results match the current filters. Try lowering the minimum relevance score.")
        st.stop()

    st.subheader("Priority watchlist")
    st.caption("Highest-priority signals from the current filters, with source, date, score, and matched themes.")
    for rank, row in enumerate(filtered.head(5).itertuples(index=False), start=1):
        st.markdown(watchlist_card(row, rank), unsafe_allow_html=True)

    st.divider()
    st.subheader(f"All results — {len(filtered):,}")

    for _, row in filtered.iterrows():
        matched = row.matched_keywords or []

        # Score dot + keyword badges
        badge_line = f"{score_chip(row.score)} **{row.score:.2f}**"
        if matched:
            badge_line += "  " + " ".join(tag_badge(kw) for kw in matched[:6])
        st.markdown(badge_line, unsafe_allow_html=True)
        st.markdown(source_date_badges(row), unsafe_allow_html=True)

        st.markdown(f"**{row.title}**")

        with st.expander("View details"):
            left, right = st.columns([3, 1])

            with left:
                if row.summary:
                    st.markdown(f"**Summary**  \n{row.summary}")
                    st.markdown(f"**Why it matters**  \n{row.relevance_note}")
                    if row.tags:
                        st.markdown("**Tags:** " + " ".join(tag_badge(t) for t in row.tags), unsafe_allow_html=True)
                else:
                    st.markdown(row.body or "No abstract or study details available.")
                    if st.button("Generate summary", key=f"sum_{row.source_type}_{row.id}"):
                        with st.spinner("Generating..."):
                            summarise(row.title, row.body or "", row.id, row.source_type, float(row.score))
                        refresh_data()

            with right:
                st.metric("Semantic",  f"{row.semantic_score:.2f}")
                st.metric("Keywords",  f"{row.keyword_score:.2f}")
                st.metric("Recency",   f"{row.recency_score:.2f}")
                st.metric("Composite", f"{row.score:.2f}")

            st.divider()
            actions = st.columns([1, 1, 1, 3])
            if actions[0].button("Relevant",     key=f"up_{row.source_type}_{row.id}"):
                record_feedback(row.id, row.source_type, 1)
                refresh_data()
            if actions[1].button("Not relevant", key=f"dn_{row.source_type}_{row.id}"):
                record_feedback(row.id, row.source_type, -1)
                refresh_data()
            actions[2].link_button("Open source", row.url)

        st.divider()


# ============================================================
# TAB 3 — RELEVANCE PROFILE
# ============================================================

with tab_profile:

    # Load saved profile from DB; fall back to all-active defaults
    saved_profile  = load_user_profile()
    saved_states   = saved_profile.get("keyword_states", {})
    saved_weights  = saved_profile.get("scoring_weights", {})

    st.subheader("Keyword filter")
    st.markdown(
        """
        <div class="kw-legend">
            <div class="kw-legend-item"><span class="kw-dot kw-dot-active"></span> <strong>Active</strong> boosts relevance</div>
            <div class="kw-legend-item"><span class="kw-dot kw-dot-muted"></span> <strong>Muted</strong> ignored in scoring</div>
            <div class="kw-legend-item"><span class="kw-dot kw-dot-removed"></span> <strong>Removed</strong> suppresses matches</div>
            <div class="kw-click-note">Click to cycle state</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "kw_states" not in st.session_state:
        st.session_state.kw_states = {**{kw: "active" for kw in profile_keywords()}, **saved_states}

    cycle_kw = st.query_params.get("cycle_kw")
    if isinstance(cycle_kw, list):
        cycle_kw = cycle_kw[0] if cycle_kw else None
    if cycle_kw:
        current = st.session_state.kw_states.get(cycle_kw, "active")
        st.session_state.kw_states[cycle_kw] = STATE_CYCLE[current]
        st.query_params.clear()

    if "weights" not in st.session_state:
        st.session_state.weights = {
            "semantic":  saved_weights.get("semantic",  SCORING_WEIGHTS["semantic"]),
            "keyword":   saved_weights.get("keyword",   SCORING_WEIGHTS["keyword"]),
            "recency":   saved_weights.get("recency",   SCORING_WEIGHTS["recency"]),
            "feedback":  saved_weights.get("feedback",  SCORING_WEIGHTS["feedback"]),
        }

    for cluster_name, keywords in KEYWORD_CLUSTERS.items():
        pills = []
        for keyword in keywords:
            state = st.session_state.kw_states.get(keyword, "active")
            pills.append(keyword_pill(keyword, state))
        st.markdown(
            f"""
            <div class="kw-section-title">{escape(cluster_name)}</div>
            <div class="kw-pill-cloud">{''.join(pills)}</div>
            """,
            unsafe_allow_html=True,
        )

    active_count = sum(1 for state in st.session_state.kw_states.values() if state == "active")
    muted_count = sum(1 for state in st.session_state.kw_states.values() if state == "muted")
    removed_count = sum(1 for state in st.session_state.kw_states.values() if state == "removed")
    st.caption(f"Active: {active_count}  ·  Muted: {muted_count}  ·  Removed: {removed_count}")

    st.divider()
    st.subheader("Scoring weights")
    st.caption("Adjust how much each signal contributes to the composite score. Values should sum to 1.0.")

    w_sem = st.slider("Semantic weight",  0.0, 0.9, st.session_state.weights["semantic"],  0.05)
    w_kw  = st.slider("Keyword weight",   0.0, 0.9, st.session_state.weights["keyword"],   0.05)
    w_rec = st.slider("Recency weight",   0.0, 0.5, st.session_state.weights["recency"],   0.05)
    w_fb  = st.slider("Feedback weight",  0.0, 0.3, st.session_state.weights["feedback"],  0.05)

    total = round(w_sem + w_kw + w_rec + w_fb, 2)
    if abs(total - 1.0) > 0.01:
        st.warning(f"Weights sum to {total:.2f} — they should sum to 1.0.")
    else:
        st.success(f"Weights sum to {total:.2f} ✓")

    save_col, reset_col, _ = st.columns([1, 1, 3])

    if save_col.button("Save profile", width="stretch"):
        new_weights = {"semantic": w_sem, "keyword": w_kw, "recency": w_rec, "feedback": w_fb}
        st.session_state.weights = new_weights
        save_user_profile(st.session_state.kw_states, new_weights)
        load_results.clear()
        st.success("Profile saved. Scores will update on next refresh.")

    if reset_col.button("Reset to defaults", width="stretch"):
        st.session_state.kw_states  = {kw: "active" for kw in profile_keywords()}
        st.session_state.weights    = dict(SCORING_WEIGHTS)
        save_user_profile(st.session_state.kw_states, st.session_state.weights)
        load_results.clear()
        st.rerun()
