"""
Quiz Me tab — generate a grounded quiz question, accept a student answer, grade it.
"""

from __future__ import annotations

import random
import re

import streamlit as st

from backend.studybot import StudyBot
from ml.guardrails import InsufficientContextError

def _question_only(text: str) -> str:
    """Strip any answer portion from quiz question text before displaying."""
    # Handle "Q: question text A: answer text" (inline or multiline)
    match = re.search(r'Q:\s*(.+?)(?=\s+A:|$)', text, re.DOTALL)
    if match:
        return "Q: " + match.group(1).strip()
    # Fallback: drop any line that starts with A:
    lines = text.split('\n')
    filtered = []
    for line in lines:
        if re.match(r'\s*A:', line):
            break
        filtered.append(line)
    return '\n'.join(filtered).strip() or text


_SECTION_MAP = {
    "All": None,
    "Main Ideas": "main_ideas",
    "Exercises": "exercises",
}


def render() -> None:
    st.header("Quiz Me")
    st.caption("Get quizzed on your ML notes.")

    topic_options = ["All Topics"] + StudyBot.AVAILABLE_TOPICS
    topic_choice = st.selectbox("Topic (optional)", topic_options, key="quiz_topic_filter")
    section_choice = st.radio(
        "Section", list(_SECTION_MAP.keys()), key="quiz_section_filter", horizontal=True
    )

    topic_filter = None if topic_choice == "All Topics" else topic_choice
    section_filter = _SECTION_MAP[section_choice]

    if st.button("Generate Question", key="quiz_generate"):
        bot: StudyBot = st.session_state.studybot
        ollama = st.session_state.ollama_client

        try:
            with st.spinner("Generating question..."):
                query = topic_filter or random.choice(StudyBot.AVAILABLE_TOPICS)
                result = bot.retrieve(
                    query,
                    k=8,
                    page_title_filter=topic_filter,
                    section_filter=section_filter,
                    mode="quiz",
                )
                st.session_state.current_quiz_snippets = result.pairs
                question = ollama.quiz_from_snippets(result.pairs)
                st.session_state.current_quiz_question = question

        except InsufficientContextError:
            st.warning("No questions available for that combination. Try a different filter.")
            return
        except Exception as exc:
            st.error(f"Failed to generate question: {exc}")
            return

    # Show question + answer form if a question has been generated
    if st.session_state.current_quiz_question:
        st.info(_question_only(st.session_state.current_quiz_question))
        student_answer = st.text_area("Your Answer", key="quiz_answer_input")

        col1, col2 = st.columns([1, 4])
        with col1:
            submit_answer = st.button("Submit Answer", key="quiz_submit")
        with col2:
            next_question = st.button("Next Question", key="quiz_next")

        if next_question:
            st.session_state.current_quiz_question = ""
            st.session_state.current_quiz_snippets = []
            st.rerun()

        if submit_answer:
            if not student_answer.strip():
                st.warning("Please write an answer before submitting.")
                return

            ollama = st.session_state.ollama_client
            logger = st.session_state.logger
            snippets = st.session_state.current_quiz_snippets
            question = st.session_state.current_quiz_question

            try:
                with st.spinner("Grading..."):
                    feedback_text = ollama.grade_student_answer(
                        question, student_answer, snippets
                    )

                first_line = feedback_text.split("\n")[0].strip().lower()
                if "correct" in first_line and "partial" not in first_line and "incorrect" not in first_line:
                    st.success("Correct")
                    grade = "correct"
                elif "partial" in first_line:
                    st.warning("Partial Credit")
                    grade = "partial"
                else:
                    st.error("Incorrect")
                    grade = "incorrect"

                # Render the reference answer and reasoning as distinct sections
                body = "\n".join(feedback_text.split("\n")[1:]).strip()
                ref_match = re.search(r'Reference Answer:\s*(.+?)(?=\n\s*Reasoning:|$)', body, re.DOTALL)
                reasoning_match = re.search(r'Reasoning:\s*(.+)', body, re.DOTALL)
                if ref_match:
                    st.markdown(f"**Reference Answer:** {ref_match.group(1).strip()}")
                if reasoning_match:
                    st.markdown(f"**Reasoning:** {reasoning_match.group(1).strip()}")
                if not ref_match and not reasoning_match:
                    st.markdown(body)

                logger.log_quiz_grade(
                    question=question,
                    student_answer=student_answer,
                    grade=grade,
                    feedback=feedback_text,
                    mode="quiz",
                    page_title=snippets[0].page_title if snippets else "",
                )

            except Exception as exc:
                st.error(f"Grading failed: {exc}")
