"""
OllamaClient — wraps ollama.chat() with three task-specific prompt methods.

All prompts include a Layer 2 grounding instruction: the model is told to
respond with exactly REFUSAL_PHRASE (and nothing else) if the provided notes
are insufficient.  After each call the response is checked and
InsufficientContextError is raised if the refusal phrase is detected.

No backend/ or frontend/ imports allowed in this module.
"""

from __future__ import annotations

import ollama

from ml.guardrails import OLLAMA_MODEL, REFUSAL_PHRASE, InsufficientContextError
from ml.models import QAPair

_GROUNDING_INSTRUCTION = (
    "You are an ML study assistant. Answer ONLY using the study notes provided below.\n"
    f'If the notes do not contain enough information to answer the question, '
    f'respond with exactly this phrase and nothing else:\n"{REFUSAL_PHRASE}"\n'
    "Do not use outside knowledge. Do not guess. Do not partially answer "
    "if the notes are insufficient."
)


def _format_snippets(snippets: list[QAPair]) -> str:
    """Render a list of QAPairs as a numbered block for inclusion in a prompt."""
    parts = []
    for i, pair in enumerate(snippets, start=1):
        parts.append(f"[{i}] Q: {pair.question}\n    A: {pair.answer}")
    return "\n\n".join(parts)


def _check_refusal(response_text: str, query: str) -> None:
    """Raise InsufficientContextError if the model returned the refusal phrase."""
    if response_text.strip().startswith(REFUSAL_PHRASE):
        raise InsufficientContextError(layer="llm", query=query)


class OllamaClient:
    """Thin wrapper around the Ollama Python client for Study Bot tasks."""

    def __init__(self, model: str = OLLAMA_MODEL) -> None:
        self.model = model

    # ------------------------------------------------------------------
    # RAG Q&A
    # ------------------------------------------------------------------

    def answer_from_snippets(self, query: str, snippets: list[QAPair]) -> str:
        """
        Generate a grounded answer to `query` using the provided Q&A snippets.

        Instructs the model to cite the source question number for each claim.
        Raises InsufficientContextError if the model issues a refusal.
        """
        snippet_text = _format_snippets(snippets)
        messages = [
            {
                "role": "system",
                "content": (
                    _GROUNDING_INSTRUCTION
                    + "\nCite the snippet number (e.g. [1], [2]) for each claim you make."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Study notes:\n{snippet_text}\n\nQuestion: {query}"
                ),
            },
        ]
        response = ollama.chat(model=self.model, messages=messages)
        text: str = response["message"]["content"]
        _check_refusal(text, query)
        return text

    # ------------------------------------------------------------------
    # Quiz generation
    # ------------------------------------------------------------------

    def quiz_from_snippets(self, snippets: list[QAPair]) -> str:
        """
        Generate ONE novel quiz question grounded strictly in the provided snippets.

        The model must not reproduce a question verbatim and must not invent facts
        outside the notes.  Returns the generated question as a plain string.
        """
        snippet_text = _format_snippets(snippets)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an ML study quiz generator. "
                    "Using ONLY the study notes below, generate exactly ONE novel "
                    "question that tests understanding of the material. "
                    "Do not reproduce a question verbatim from the notes. "
                    "Do not introduce facts not present in the notes. "
                    "Output the question only — no preamble, no answer."
                ),
            },
            {
                "role": "user",
                "content": f"Study notes:\n{snippet_text}",
            },
        ]
        response = ollama.chat(model=self.model, messages=messages)
        return response["message"]["content"].strip()

    # ------------------------------------------------------------------
    # Answer grading
    # ------------------------------------------------------------------

    def grade_student_answer(
        self,
        question: str,
        student_answer: str,
        snippets: list[QAPair],
    ) -> str:
        """
        Grade a student's free-text answer against the reference snippets.

        The first line of the response must be exactly one of:
          Correct | Partial | Incorrect
        Followed by 2–3 sentences of feedback grounded in the snippets.

        Returns the full response string (grade line + feedback).
        """
        snippet_text = _format_snippets(snippets)
        messages = [
            {
                "role": "system",
                "content": (
                    _GROUNDING_INSTRUCTION
                    + "\nYou are grading a student's answer. "
                    "Your response MUST follow this exact format:\n"
                    "Line 1: exactly one of: Correct, Partial, or Incorrect\n"
                    "Line 2: blank\n"
                    "Reference Answer: [the correct answer drawn from the study notes]\n"
                    "Line 4: blank\n"
                    "Reasoning: [2-3 sentences explaining why the student's answer is "
                    "correct, partially correct, or incorrect, grounded strictly in the study notes]"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Study notes:\n{snippet_text}\n\n"
                    f"Question: {question}\n\n"
                    f"Student answer: {student_answer}"
                ),
            },
        ]
        response = ollama.chat(model=self.model, messages=messages)
        text: str = response["message"]["content"]
        _check_refusal(text, question)
        return text
