"""
RAG Q&A tab — user submits a free-form ML question, gets a grounded LLM answer
with source citations.
"""

from __future__ import annotations

import streamlit as st

from backend.studybot import StudyBot
from ml.guardrails import InsufficientContextError


def render() -> None:
    st.header("RAG Q&A")
    st.caption("Ask any question about your ML notes.")

    query = st.text_input(
        "Ask a question about ML",
        key="rag_query_input",
        placeholder="e.g. What is the CART algorithm?",
    )
    submitted = st.button("Submit", key="rag_submit")

    if not submitted:
        return

    if not query.strip():
        st.warning("Please enter a question before submitting.")
        return

    bot: StudyBot = st.session_state.studybot
    ollama = st.session_state.ollama_client
    logger = st.session_state.logger

    try:
        with st.spinner("Retrieving and generating answer..."):
            result = bot.retrieve(query, k=5, mode="rag")
            answer = ollama.answer_from_snippets(query, result.pairs)
            logger.log_llm_response(query=query, response=answer, mode="rag")

        st.success(answer)

        with st.expander(f"Sources ({len(result.pairs)} retrieved)"):
            for pair in result.pairs:
                st.markdown(
                    f"**{pair.page_title}** — _{pair.section}_ — {pair.question}"
                )

        st.session_state.query_count += 1
        st.session_state.rag_history = [query] + st.session_state.rag_history[:4]

    except InsufficientContextError:
        st.warning("I don't have sufficient notes on that topic.")
        st.markdown("**Topics I have notes on:**")
        st.markdown(
            "\n".join(f"- {t}" for t in StudyBot.AVAILABLE_TOPICS)
        )
        st.session_state.refusal_count += 1

    except Exception as exc:
        if "connection" in str(exc).lower() or "refused" in str(exc).lower():
            st.error("Could not reach Ollama. Is it running on localhost:11434?")
        else:
            st.error(f"Unexpected error: {exc}")
