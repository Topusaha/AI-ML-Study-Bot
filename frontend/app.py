"""
Streamlit entry point for the ML Study Bot.

Initialises shared session state once per browser session, then delegates
rendering to one of four tab modules.  Nothing outside frontend/ imports
from this file; nothing inside frontend/ imports from ml/ directly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from uuid import uuid4

# Ensure repo root is on sys.path so backend/ and ml/ are importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from backend.studybot import StudyBot
from ml.llm_client import OllamaClient
from backend.logger import StudyLogger
from backend.evaluation import HITLEvaluator

from frontend.tabs import qa_tab, quiz_tab, metrics_tab, hitl_tab


st.set_page_config(
    page_title="ML Study Bot",
    page_icon="🤖",
    layout="wide",
)

# ---------------------------------------------------------------------------
# One-time session initialisation
# ---------------------------------------------------------------------------

if "studybot" not in st.session_state:
    with st.spinner("Loading and indexing your ML notes..."):
        bot = StudyBot(
            notion_dir="data/assets/notion",
            chroma_path="data/.chroma",
            log_dir="data/logs",
        )
        bot.load_and_index()
        st.session_state.studybot = bot

    st.session_state.ollama_client = OllamaClient(
        model=os.getenv("OLLAMA_MODEL", "llama3.2")
    )
    st.session_state.logger = bot.logger
    st.session_state.hitl_evaluator = HITLEvaluator(
        csv_path="data/human-in-the-loop-results.csv",
        retriever=bot._retriever,
        logger=bot.logger,
    )
    st.session_state.session_id = str(uuid4())
    st.session_state.hitl_unlocked = False
    st.session_state.query_count = 0
    st.session_state.refusal_count = 0
    st.session_state.rag_history = []
    st.session_state.current_quiz_snippets = []
    st.session_state.current_quiz_question = ""
    st.session_state.hitl_results = []
    st.session_state.reingest_confirm_pending = False

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("ML Study Bot")
    st.caption("v1.0.0")
    st.markdown(f"**Model:** `{os.getenv('OLLAMA_MODEL', 'llama3.2')}`")
    st.markdown(f"**Session:** `{st.session_state.session_id[:8]}`")
    st.metric("Queries this session", st.session_state.query_count)
    st.metric("Refusals this session", st.session_state.refusal_count)
    st.divider()

    # Two-step reingest confirmation
    if st.button("Reingest Notes", key="reingest_btn"):
        st.session_state.reingest_confirm_pending = True

    if st.session_state.reingest_confirm_pending:
        st.warning(
            "This re-parses all Notion files and rebuilds the ChromaDB index. "
            "Any HITL-injected pairs for this session will be lost."
        )
        if st.button("Confirm Reingest", key="reingest_confirm_btn"):
            with st.spinner("Re-ingesting..."):
                st.session_state.studybot.load_and_index(force_reingest=True)
            st.success("Reingest complete.")
            st.session_state.reingest_confirm_pending = False

    st.divider()
    with st.expander("Recent Queries (last 5)"):
        for q in st.session_state.rag_history:
            st.markdown(f"- {q}")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
    ["RAG Q&A", "Quiz Me", "Metrics & Logs", "Human-in-the-Loop"]
)

with tab1:
    qa_tab.render()

with tab2:
    quiz_tab.render()

with tab3:
    metrics_tab.render()

with tab4:
    hitl_tab.render()
