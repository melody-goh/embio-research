import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from config.relevance_profile import DEFAULT_MIN_RELEVANCE
from feedback.store import record_feedback
from ingestion.scheduler import run_once
from nlp.embedder import embed_all_pending
from ranking.scorer import score_all
from storage.db import get_connection, init_db, ingestion_summary
from summarisation.llm import get_cached_summary, summarise


# ---------------------------------------------------------------------------
# Keyword → badge colour mapping
# Streamlit badge syntax: :colour-badge[text]
# ---------------------------------------------------------------------------

TAG_COLORS = {
    # Oncology — red
    "pancreatic cancer": "red",
    "pancreatic adenocarcinoma": "red",
    "pancreatic ductal adenocarcinoma": "red",
    "pdac": "red",
    "locally advanced pancreatic cancer": "red",
    "cholangiocarcinoma": "red",
    "pancreatic intraepithelial neoplasia": "red",
    "panin": "red",
    # Electroporation — blue
    "electroporation": "blue",
    "electrochemotherapy": "blue",
    "irreversible electroporation": "blue",
    "reversible electroporation": "blue",
    "calcium electroporation": "blue",
    "ire": "blue",
    "nanoknife": "blue",
    "pulsed electric field": "blue",
    "pef": "blue",
    "non-thermal ablation": "blue",
    "ablation": "blue",
    "tumor ablation": "blue",
    # Catheter / device — green
    "electroporation catheter": "green",
    "flexible catheter": "green",
    "catheter": "green",
    "intraductal catheter": "green",
    "bipolar electrode catheter": "green",
    "ring electrode catheter": "green",
    "eus-guided": "green",
    "endoscopic ultrasound": "green",
    "minimally invasive": "green",
    "intraductal": "green",
    "ercp": "green",
    # Clinical / strategic — orange
    "first-in-human": "orange",
    "clinical feasibility": "orange",
    "safety": "orange",
    "interventional oncology": "orange",
    "locoregional therapy": "orange",
    "organoid pancreatic cancer": "orange",
    # Drug delivery / biomarkers — violet
    "drug delivery": "violet",
    "electroporation parameters": "violet",
    "localized drug delivery": "violet",
    "intratumoral drug delivery": "violet",
    "liquid biopsy": "violet",
    "pancreatic juice": "violet",
    "pancreatic juice biomarker": "violet",
    "ctdna": "violet",
    "kras pancreatic": "violet",
}


# ---------------------------------------------------------------------------
# Plotly theme helpers — keeps all charts visually consistent
# ---------------------------------------------------------------------------

CHART_COLOR_PRIMARY   = "#1D9E75"
CHART_COLOR_SECONDARY = "#378ADD"
CHART_COLOR_MUTED     = "#B4B2A9"

PLOTLY_LAYOUT = dict(
    margin=dict(l=0, r=0, t=4, b=0),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(size=11),
)


def _score_color(score: float) -> str:
    """Return a coloured dot for the score tier."""
    if score >= 0.80:
        return "🟢"
    if score >= 0.65:
        return "🟡"
    return "🔴"


def tag_badge(keyword: str) -> str:
    color = TAG_COLORS.get(keyword.lower(), "gray")
    return f":{color}-badge[{keyword}]"


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Embio Intelligence", layout="wide")
init_db()


# ---------------------------------------------------------------------------
# Data loading — cached for 5 minutes
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_results() -> pd.DataFrame:
    scored = score_all()
    if scored.empty:
        return scored

    with get_connection() as con:
        articles = con.execute("""
            SELECT id, 'article' AS source_type, journal AS source_label,
                   authors, url, fetched_at,
                   COALESCE(source, 'pubmed') AS data_source
            FROM articles
        """).fetchdf()
        trials = con.execute("""
            SELECT id, 'trial' AS source_type, sponsor AS source_label,
                   status AS authors, url, fetched_at,
                   COALESCE(source, 'clinicaltrials') AS data_source
            FROM trials
        """).fetchdf()

    metadata = pd.concat([articles, trials], ignore_index=True)
    results = scored.merge(metadata, on=["id", "source_type"], how="left")

    cached = []
    for _, row in results.iterrows():
        summary = get_cached_summary(row.id, row.source_type) or {}
        cached.append({
            "summary":        summary.get("summary", ""),
            "relevance_note": summary.get("relevance_note", ""),
            "tags":           summary.get("tags", []),
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


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Embio Intelligence")
    st.caption("Research & market radar")

    st.divider()

    source_filter = st.multiselect(
        "Source type", ["article", "trial"],
        default=["article", "trial"]
    )
    min_score = st.slider("Minimum relevance score", 0.0, 1.0, DEFAULT_MIN_RELEVANCE, 0.01)
    years_back = st.slider("Published within (years)", 1, 5, 2, 1)

    st.divider()

    only_unsummarised = st.checkbox("Only unsummarised")
    only_with_summary = st.checkbox("Only with AI summary")

    st.divider()

    if st.button("🔄 Refresh", use_container_width=True):
        refresh_pipeline()

    st.caption("Pulls new data from all sources and re-embeds new documents.")


# ---------------------------------------------------------------------------
# Load and filter
# ---------------------------------------------------------------------------

results = load_results()

st.title("Embio Intelligence")
st.caption(
    "Research and clinical-trial radar — electroporation, pancreatic cancer, "
    "catheter platforms, intraductal delivery, and adjacent medtech signals."
)

if results.empty:
    st.info(
        "No embedded records yet. Click **Refresh** in the sidebar to fetch "
        "sources and embed new documents."
    )
    st.stop()

cutoff = date.today() - timedelta(days=years_back * 365)
filtered = results[results["source_type"].isin(source_filter)]
filtered = filtered[filtered["score"] >= min_score]
filtered = filtered[
    filtered["item_date"].isna()
    | (pd.to_datetime(filtered["item_date"], errors="coerce").dt.date >= cutoff)
]
if only_unsummarised:
    filtered = filtered[filtered["summary"].fillna("") == ""]
if only_with_summary:
    filtered = filtered[filtered["summary"].fillna("") != ""]


# ---------------------------------------------------------------------------
# Metrics row
# ---------------------------------------------------------------------------

with get_connection() as con:
    feedback_count = con.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]

top_score    = float(results["score"].max()) if not results.empty else 0.0
latest_fetch = results["fetched_at"].dropna().max() if "fetched_at" in results else None

m = st.columns(5)
m[0].metric("Total records",    f"{len(results):,}")
m[1].metric("Visible",          f"{len(filtered):,}")
m[2].metric("Top score",        f"{top_score:.2f}")
m[3].metric("Feedback signals", f"{feedback_count:,}")
m[4].metric(
    "Last updated",
    str(latest_fetch)[:16] if latest_fetch is not None else "Never"
)


# ---------------------------------------------------------------------------
# Analytics section — three tabs
# ---------------------------------------------------------------------------

st.divider()
analytics, feed_tab = st.tabs(["📊 Analytics", "📄 Evidence feed"])

with analytics:

    col_left, col_right = st.columns(2)

    # --- Publications over time ---
    with col_left:
        st.subheader("Publications over time")
        with get_connection() as con:
            trend = con.execute("""
                SELECT date_trunc('month', pub_date) AS month,
                       COUNT(*) AS count
                FROM articles
                WHERE pub_date IS NOT NULL
                  AND pub_date >= now() - INTERVAL '3 years'
                GROUP BY 1
                ORDER BY 1
            """).df()

        if not trend.empty:
            fig_trend = px.bar(
                trend, x="month", y="count",
                color_discrete_sequence=[CHART_COLOR_PRIMARY],
            )
            fig_trend.update_traces(marker_cornerradius=3)
            fig_trend.update_layout(
                **PLOTLY_LAYOUT, height=220,
                xaxis_title="", yaxis_title="articles",
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.caption("No dated articles yet.")

    # --- Score distribution ---
    with col_right:
        st.subheader("Score distribution")
        fig_dist = px.histogram(
            results, x="score", nbins=20,
            color_discrete_sequence=[CHART_COLOR_PRIMARY],
        )
        fig_dist.add_vline(
            x=min_score, line_dash="dash",
            line_color=CHART_COLOR_MUTED,
            annotation_text="threshold",
            annotation_font_size=10,
        )
        fig_dist.update_traces(marker_cornerradius=3)
        fig_dist.update_layout(
            **PLOTLY_LAYOUT, height=220,
            xaxis_title="relevance score", yaxis_title="records",
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    col_left2, col_right2 = st.columns(2)

    # --- Top keyword frequency ---
    with col_left2:
        st.subheader("Top keyword matches")
        all_kws = [
            kw for kws in results["matched_keywords"].dropna()
            for kw in kws
        ]
        if all_kws:
            top_kws = pd.DataFrame(
                Counter(all_kws).most_common(15),
                columns=["keyword", "count"]
            )
            fig_kw = px.bar(
                top_kws, x="count", y="keyword",
                orientation="h",
                color_discrete_sequence=[CHART_COLOR_PRIMARY],
            )
            fig_kw.update_traces(marker_cornerradius=3)
            fig_kw.update_layout(
                **PLOTLY_LAYOUT, height=320,
                xaxis_title="matches", yaxis_title="",
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_kw, use_container_width=True)
        else:
            st.caption("No keyword matches yet.")

    # --- Source breakdown + ingestion health ---
    with col_right2:
        st.subheader("Source breakdown")

        source_colors = {
            "pubmed":         CHART_COLOR_PRIMARY,
            "europepmc":      "#7F77DD",
            "clinicaltrials": CHART_COLOR_SECONDARY,
            "biorxiv":        "#D85A30",
            "medrxiv":        "#EF9F27",
        }

        if "data_source" in results.columns:
            src_counts = (
                results.groupby("data_source")
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
            )
            colors = [source_colors.get(s, CHART_COLOR_MUTED) for s in src_counts["data_source"]]
            fig_src = px.bar(
                src_counts, x="data_source", y="count",
                color="data_source",
                color_discrete_map=source_colors,
            )
            fig_src.update_traces(marker_cornerradius=3)
            fig_src.update_layout(
                **PLOTLY_LAYOUT, height=180,
                xaxis_title="", yaxis_title="records",
                showlegend=False,
            )
            st.plotly_chart(fig_src, use_container_width=True)
        else:
            # Fallback: article vs trial
            src_counts = results["source_type"].value_counts().reset_index()
            src_counts.columns = ["type", "count"]
            st.dataframe(src_counts, hide_index=True, use_container_width=True)

        # Ingestion health table
        st.caption("Ingestion log (last 30 days)")
        try:
            health = ingestion_summary(days=30)
            if health:
                health_df = pd.DataFrame(health)[["source", "total_new", "total_updated", "last_run"]]
                health_df["last_run"] = pd.to_datetime(health_df["last_run"]).dt.strftime("%Y-%m-%d %H:%M")
                st.dataframe(health_df, hide_index=True, use_container_width=True)
            else:
                st.caption("No ingestion log entries yet.")
        except Exception:
            st.caption("Ingestion log not available.")

    # --- Semantic vs keyword scatter ---
    st.subheader("Semantic vs keyword score")
    st.caption("Each dot is one document. Hover to see the title. High on both axes = strong signal.")
    fig_scatter = px.scatter(
        filtered,
        x="semantic_score",
        y="keyword_score",
        color="source_type",
        hover_name="title",
        size="score",
        size_max=14,
        color_discrete_map={
            "article": CHART_COLOR_PRIMARY,
            "trial":   CHART_COLOR_SECONDARY,
        },
        opacity=0.7,
    )
    fig_scatter.update_layout(
        **PLOTLY_LAYOUT, height=340,
        xaxis_title="semantic score",
        yaxis_title="keyword score",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)


# ---------------------------------------------------------------------------
# Evidence feed tab
# ---------------------------------------------------------------------------

with feed_tab:

    if filtered.empty:
        st.info("No results match the current filters. Try lowering the minimum relevance score.")
        st.stop()

    # Highlights table — top 5
    st.subheader("Highlights")
    highlights = filtered.head(5).copy()
    highlights["score"] = highlights["score"].round(2)
    st.dataframe(
        highlights[["score", "source_type", "title", "source_label", "item_date"]],
        hide_index=True,
        use_container_width=True,
    )

    st.divider()
    st.subheader(f"Evidence feed — {len(filtered):,} results")

    for _, row in filtered.iterrows():
        matched     = row.matched_keywords or []
        source_label = row.source_label if pd.notna(row.source_label) and row.source_label else "Unknown"
        item_date    = row.item_date if pd.notna(row.item_date) else "Unknown"

        # Score dot + badges above title
        badge_line = f"{_score_color(row.score)} **{row.score:.2f}**"
        if matched:
            badge_line += "  " + " ".join(tag_badge(kw) for kw in matched[:6])
        st.markdown(badge_line)

        st.markdown(f"**{row.title}**")
        st.caption(
            f"{row.source_type.upper()} · {source_label} · {item_date}"
        )

        with st.expander("View details"):

            detail_left, detail_right = st.columns([3, 1])

            with detail_left:
                if row.summary:
                    st.markdown(f"**Summary**  \n{row.summary}")
                    st.markdown(f"**Why it matters**  \n{row.relevance_note}")
                    if row.tags:
                        st.markdown("**Tags:** " + "  ".join(
                            tag_badge(t) for t in row.tags
                        ))
                else:
                    st.markdown(row.body or "No abstract or study details available.")
                    if st.button("Generate summary ✨", key=f"sum_{row.source_type}_{row.id}"):
                        with st.spinner("Generating..."):
                            summarise(
                                row.title, row.body or "",
                                row.id, row.source_type, float(row.score)
                            )
                        refresh_data()

            with detail_right:
                st.metric("Semantic",  f"{row.semantic_score:.2f}")
                st.metric("Keywords",  f"{row.keyword_score:.2f}")
                st.metric("Recency",   f"{row.recency_score:.2f}")
                st.metric("Composite", f"{row.score:.2f}")

            st.divider()

            action_cols = st.columns([1, 1, 1, 3])
            if action_cols[0].button("👍 Relevant",     key=f"up_{row.source_type}_{row.id}"):
                record_feedback(row.id, row.source_type, 1)
                refresh_data()
            if action_cols[1].button("👎 Not relevant", key=f"dn_{row.source_type}_{row.id}"):
                record_feedback(row.id, row.source_type, -1)
                refresh_data()
            action_cols[2].link_button("🔗 Open source", row.url)

        st.divider()
