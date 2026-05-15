import sys
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from config.relevance_profile import DEFAULT_MIN_RELEVANCE
from feedback.store import record_feedback
from ranking.scorer import score_all
from storage.db import get_connection, init_db
from summarisation.llm import get_cached_summary, summarise


st.set_page_config(page_title="Embio Intelligence", layout="wide")
init_db()


@st.cache_data(ttl=300)
def load_results() -> pd.DataFrame:
    scored = score_all()
    if scored.empty:
        return scored

    con = get_connection()
    articles = con.execute(
        """
        SELECT id, 'article' AS source_type, title, abstract AS body, journal AS source_label,
               authors, pub_date AS item_date, url, fetched_at
        FROM articles
        """
    ).fetchdf()
    trials = con.execute(
        """
        SELECT id, 'trial' AS source_type, title,
               concat_ws(' ', conditions, interventions, sponsor, phase, status) AS body,
               sponsor AS source_label, status AS authors, start_date AS item_date, url, fetched_at
        FROM trials
        """
    ).fetchdf()
    con.close()

    metadata = pd.concat([articles, trials], ignore_index=True)
    results = scored.merge(metadata, on=["id", "source_type", "title", "body", "item_date"], how="left")
    cached = []
    for _, row in results.iterrows():
        summary = get_cached_summary(row.id, row.source_type) or {}
        cached.append(
            {
                "summary": summary.get("summary", ""),
                "relevance_note": summary.get("relevance_note", ""),
                "tags": summary.get("tags", []),
            }
        )
    return pd.concat([results.reset_index(drop=True), pd.DataFrame(cached)], axis=1)


def refresh_data() -> None:
    load_results.clear()
    st.rerun()


with st.sidebar:
    st.title("Embio Intelligence")
    source_filter = st.multiselect("Source", ["article", "trial"], default=["article", "trial"])
    min_score = st.slider("Minimum relevance", 0.0, 1.0, DEFAULT_MIN_RELEVANCE, 0.01)
    days_back = st.slider("Published or started within", 30, 1825, 730, 30)
    only_unsummarised = st.checkbox("Needs summary")
    if st.button("Refresh scores", use_container_width=True):
        refresh_data()

results = load_results()

st.title("Embio Intelligence")
st.caption("Research and clinical-trial radar for electroporation, pancreatic cancer, catheter platforms, and adjacent medtech signals.")

if results.empty:
    st.info("No records yet. Run `python -m ingestion.scheduler --once` to fetch PubMed articles and clinical trials.")
    st.stop()

cutoff = date.today() - timedelta(days=days_back)
filtered = results[results["source_type"].isin(source_filter)]
filtered = filtered[filtered["score"] >= min_score]
filtered = filtered[
    filtered["item_date"].isna()
    | (pd.to_datetime(filtered["item_date"], errors="coerce").dt.date >= cutoff)
]
if only_unsummarised:
    filtered = filtered[filtered["summary"].fillna("") == ""]

top_score = float(results["score"].max()) if not results.empty else 0.0
latest_fetch = results["fetched_at"].dropna().max() if "fetched_at" in results else None
metric_cols = st.columns(4)
metric_cols[0].metric("Records", f"{len(results):,}")
metric_cols[1].metric("Visible", f"{len(filtered):,}")
metric_cols[2].metric("Top score", f"{top_score:.2f}")
metric_cols[3].metric("Last updated", str(latest_fetch)[:16] if latest_fetch is not None else "Never")

highlights = filtered.head(5)
if not highlights.empty:
    st.subheader("Highlights")
    st.dataframe(
        highlights[["score", "source_type", "title", "source_label", "item_date", "matched_keywords"]],
        width="stretch",
        hide_index=True,
    )

st.subheader("Evidence Feed")
for _, row in filtered.iterrows():
    label = f"{row.score:.2f} | {row.source_type.upper()} | {row.title}"
    with st.expander(label):
        top = st.columns([2, 1])
        with top[0]:
            st.markdown(f"**Source:** {row.source_label or 'Unknown'}")
            st.markdown(f"**Date:** {row.item_date or 'Unknown'}")
            st.markdown(f"**Matched keywords:** {', '.join(row.matched_keywords) if row.matched_keywords else 'None'}")
        with top[1]:
            st.metric("Semantic", f"{row.semantic_score:.2f}")
            st.metric("Keywords", f"{row.keyword_score:.2f}")
            st.metric("Recency", f"{row.recency_score:.2f}")

        if row.summary:
            st.markdown(f"**Summary:** {row.summary}")
            st.markdown(f"**Why it matters:** {row.relevance_note}")
            if row.tags:
                st.markdown("**Tags:** " + ", ".join(row.tags))
        else:
            st.markdown(row.body or "No abstract or study details available.")
            if st.button("Generate summary", key=f"sum_{row.source_type}_{row.id}"):
                summarise(row.title, row.body or "", row.id, row.source_type, float(row.score))
                refresh_data()

        actions = st.columns([1, 1, 1, 4])
        if actions[0].button("Relevant", key=f"up_{row.source_type}_{row.id}"):
            record_feedback(row.id, row.source_type, 1)
            refresh_data()
        if actions[1].button("Not relevant", key=f"down_{row.source_type}_{row.id}"):
            record_feedback(row.id, row.source_type, -1)
            refresh_data()
        actions[2].link_button("Open source", row.url)
