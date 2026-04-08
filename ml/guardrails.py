"""
Guardrail constants and exceptions for the ML Study Bot.

This module is a leaf dependency — it imports nothing from the project.
All other modules that need guardrail behaviour import from here.

Layer 1 (retrieval gate) — enforced in backend/studybot.py
Layer 2 (LLM refusal)   — enforced in ml/llm_client.py
"""

import os


# ---------------------------------------------------------------------------
# Thresholds (overridable via .env)
# ---------------------------------------------------------------------------

COSINE_DISTANCE_THRESHOLD: float = float(
    os.getenv("COSINE_DISTANCE_THRESHOLD", "0.8")
)
# ChromaDB distance above which retrieval is considered too dissimilar.
# Range [0.0, 2.0]; lower = more similar.  Default 0.8 is deliberately
# conservative — tune downward after inspecting refusal logs.

BM25_SCORE_THRESHOLD: float = 0.0
# BM25 score at or below which sparse retrieval has zero term overlap.
# Both signals (BM25 AND cosine) must be weak simultaneously to trigger
# a low_confidence refusal; one strong signal is enough to proceed.

OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")

REFUSAL_PHRASE: str = "I don't have sufficient notes on that topic."
# The exact string the LLM is instructed to return (and nothing else) when
# the provided snippets are insufficient.  Checked with str.startswith —
# case-sensitive, no trailing whitespace expected.


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class InsufficientContextError(Exception):
    """
    Raised when the system cannot produce a grounded answer.

    layer values:
      "no_results"    — retrieval returned zero pairs
      "low_confidence" — both BM25 and cosine signals are weak
      "llm_refusal"   — LLM responded with REFUSAL_PHRASE
    """

    def __init__(self, layer: str, query: str) -> None:
        self.layer = layer
        self.query = query
        super().__init__(f"[{layer}] Insufficient context for query: {query!r}")
