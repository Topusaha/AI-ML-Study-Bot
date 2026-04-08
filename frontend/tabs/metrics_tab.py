"""
Metrics & Logs tab — read the current session JSONL and display stats.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


def _load_events(log_path: Path) -> list[dict]:
    events = []
    if not log_path.exists():
        return events
    try:
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except Exception:
        pass
    return events


def render() -> None:
    st.header("Metrics & Logs")

    logger = st.session_state.logger
    log_path: Path = logger.current_log_path

    events = _load_events(log_path)

    retrievals = [e for e in events if e.get("event_type") == "retrieval"]
    refusals   = [e for e in events if e.get("event_type") == "refusal"]
    grades     = [e for e in events if e.get("event_type") == "quiz_grade"]

    # --- Session summary ---
    st.subheader("Session Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Session ID", st.session_state.session_id[:8])
    c2.metric("Total Queries", len(retrievals))
    c3.metric("Quiz Rounds", len(grades))
    c4.metric("Refusals", len(refusals))
    c5.metric("Session Start", logger.session_start.strftime("%H:%M:%S"))

    st.divider()

    # --- Quiz accuracy ---
    st.subheader("Quiz Accuracy by Topic")
    topic_data: dict[str, dict] = {}
    if grades:
        for g in grades:
            topic = g.get("page_title", "unknown")
            rec = topic_data.setdefault(topic, {"correct": 0, "total": 0})
            rec["total"] += 1
            if g.get("grade", "").lower() == "correct":
                rec["correct"] += 1
        rows = [
            {"topic": t, "accuracy": v["correct"] / v["total"] * 100}
            for t, v in topic_data.items()
        ]
        df = pd.DataFrame(rows).set_index("topic")
        st.bar_chart(df)
    else:
        st.caption("No quiz data yet.")

    st.divider()

    # --- Recent queries ---
    st.subheader("Recent Queries (last 10)")
    if retrievals:
        rows = [
            {
                "timestamp": e.get("timestamp", "")[:19],
                "query": e.get("query", ""),
                "pairs_returned": e.get("num_pairs_returned", ""),
                "cosine_distance": round(e.get("top_cosine_distance", 0), 3),
            }
            for e in retrievals[-10:]
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.caption("No queries yet.")

    st.divider()

    # --- Refusals ---
    st.subheader("Refusals Log")
    if refusals:
        rows = [
            {
                "timestamp": e.get("timestamp", "")[:19],
                "query": e.get("query", ""),
                "layer": e.get("layer", ""),
                "cosine_distance": round(e.get("top_cosine_distance", 0), 3),
                "bm25_score": round(e.get("top_bm25_score", 0), 3),
            }
            for e in refusals
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.caption("No refusals this session.")

    st.divider()

    # --- Weak topics ---
    st.subheader("Weak Topics")
    if grades and topic_data:
        weak = [
            t for t, v in topic_data.items()
            if v["total"] >= 3 and v["correct"] / v["total"] < 0.5
        ]
        if weak:
            st.warning("Weak topics (< 50% accuracy, ≥ 3 attempts):\n" + "\n".join(f"- {t}" for t in weak))
        else:
            st.success("No weak topics detected yet.")
    else:
        st.caption("No quiz data yet.")

    st.divider()

    # --- Download ---
    if log_path.exists():
        raw = log_path.read_bytes()
        st.download_button(
            label="Download Session Log (.jsonl)",
            data=raw,
            file_name=f"session_{st.session_state.session_id[:8]}.jsonl",
            mime="application/jsonl",
        )
