"""
Human-in-the-Loop tab — passkey-gated panel for rating retrieved pairs,
correcting answers, and injecting approved pairs into ChromaDB.
"""

from __future__ import annotations

import hashlib
import os

import pandas as pd
import streamlit as st

from backend.studybot import StudyBot
from ml.guardrails import InsufficientContextError


def render() -> None:
    st.header("Human-in-the-Loop Evaluation")
    st.caption(
        "Rate the quality of retrieved Q&A pairs and optionally inject "
        "corrected pairs into ChromaDB for the current session."
    )

    # ------------------------------------------------------------------
    # Authentication gate
    # ------------------------------------------------------------------
    if not os.getenv("HITL_PASSKEY"):
        st.error("HITL_PASSKEY environment variable is not configured.")
        return

    if not st.session_state.hitl_unlocked:
        passkey_input = st.text_input(
            "Enter HITL passkey", type="password", key="hitl_passkey_input"
        )
        if st.button("Verify", key="hitl_verify"):
            evaluator = st.session_state.hitl_evaluator
            if evaluator.check_passkey(passkey_input):
                st.session_state.hitl_unlocked = True
                st.rerun()
            else:
                st.error("Invalid passkey.")
        return

    # ------------------------------------------------------------------
    # Retrieve & Rate
    # ------------------------------------------------------------------
    st.subheader("Retrieve & Rate")

    query = st.text_input("Query", key="hitl_query_input")
    if st.button("Retrieve", key="hitl_retrieve"):
        if not query.strip():
            st.warning("Please enter a query.")
        else:
            bot: StudyBot = st.session_state.studybot
            try:
                result = bot.retrieve(query, k=5, mode="hitl")
                st.session_state.hitl_results = result.pairs
                st.session_state.hitl_result_obj = result
            except InsufficientContextError:
                st.warning("I don't have sufficient notes on that topic.")
                st.session_state.hitl_results = []

    pairs = st.session_state.hitl_results
    ratings: list[str] = []
    corrections: list[str] = []

    for i, pair in enumerate(pairs):
        with st.expander(f"Result {i + 1}: {pair.page_title} — {pair.section}", expanded=True):
            st.markdown(f"**Page:** {pair.page_title}  |  **Section:** `{pair.section}`")
            st.markdown(f"**Question:** {pair.question}")
            st.markdown(pair.answer)
            rating = st.radio(
                "Relevance",
                ["relevant", "partial", "not_relevant"],
                key=f"hitl_rating_{i}",
                horizontal=True,
            )
            correction = st.text_area(
                "Corrected answer (optional)", key=f"hitl_correction_{i}", height=80
            )
        ratings.append(rating)
        corrections.append(correction)

    if pairs and st.button("Save Ratings", key="hitl_save"):
        evaluator = st.session_state.hitl_evaluator
        result_obj = st.session_state.get("hitl_result_obj")
        if result_obj is not None:
            evaluator.save_ratings(
                query=query,
                result=result_obj,
                ratings=ratings,
                corrections=corrections,
                session_id=st.session_state.session_id,
            )
            st.toast(f"Saved {len(pairs)} ratings to CSV.")

    st.divider()

    # ------------------------------------------------------------------
    # Pending Injections
    # ------------------------------------------------------------------
    st.subheader("Pending Injections")
    evaluator = st.session_state.hitl_evaluator
    pending = evaluator.pending_rows()

    if not pending:
        st.caption("No ratings saved yet.")
    else:
        df = pd.DataFrame(pending)
        st.dataframe(df, use_container_width=True)

        selected_ids = st.multiselect(
            "Select QA Pair IDs to inject into ChromaDB",
            options=[r["qa_pair_id"] for r in pending],
            key="hitl_selected_ids",
        )

        if st.button("Add Selected to ChromaDB", key="hitl_inject"):
            if not selected_ids:
                st.warning("No pairs selected.")
            else:
                injected = evaluator.inject_to_chroma(selected_ids)
                st.success(f"{injected} pairs injected into ChromaDB for this session.")
                st.warning(
                    "Note: ChromaDB HITL data is cleared on server restart. CSV is permanent."
                )

        if df is not None:
            csv_bytes = df.to_csv(index=False).encode()
            st.download_button(
                "Download CSV",
                data=csv_bytes,
                file_name="human-in-the-loop-results.csv",
                mime="text/csv",
            )
