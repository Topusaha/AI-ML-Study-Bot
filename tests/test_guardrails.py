"""
Tests for guardrail constants and InsufficientContextError (ml/guardrails.py).

Run with: pytest tests/test_guardrails.py -v
"""

import os
import pytest

from ml.guardrails import (
    COSINE_DISTANCE_THRESHOLD,
    BM25_SCORE_THRESHOLD,
    REFUSAL_PHRASE,
    InsufficientContextError,
)
from ml.models import QAPair, RetrievalResult
from ml.retriever import HybridRetriever
_TMPDIR = "/tmp/test_chroma_guardrails"


@pytest.fixture(scope="module")
def populated_retriever() -> HybridRetriever:
    r = HybridRetriever(collection_name="guardrail_test", chroma_persist_dir=_TMPDIR)
    r.index([
        QAPair(
            id="dt_1",
            question="What is Gini impurity?",
            answer="Measures misclassification probability at a node.",
            page_title="Decision Trees",
            section="main_ideas",
            has_diagram=False,
        )
    ])
    return r


class TestInsufficientContextError:
    def test_layer_attribute(self):
        err = InsufficientContextError("no_results", "some query")
        assert err.layer == "no_results"
        assert err.query == "some query"

    def test_is_exception(self):
        err = InsufficientContextError("llm", "q")
        assert isinstance(err, Exception)

    def test_message_contains_layer_and_query(self):
        err = InsufficientContextError("low_confidence", "test query")
        assert "low_confidence" in str(err)
        assert "test query" in str(err)


class TestThresholds:
    def test_cosine_threshold_is_float(self):
        assert isinstance(COSINE_DISTANCE_THRESHOLD, float)

    def test_cosine_threshold_range(self):
        assert 0.0 < COSINE_DISTANCE_THRESHOLD <= 2.0

    def test_bm25_threshold_is_zero(self):
        assert BM25_SCORE_THRESHOLD == 0.0

    def test_refusal_phrase_is_nonempty_string(self):
        assert isinstance(REFUSAL_PHRASE, str)
        assert len(REFUSAL_PHRASE) > 0


class TestEnvOverride:
    def test_threshold_can_be_overridden_by_env(self, monkeypatch):
        monkeypatch.setenv("COSINE_DISTANCE_THRESHOLD", "0.6")
        import importlib
        import ml.guardrails as g
        importlib.reload(g)
        assert g.COSINE_DISTANCE_THRESHOLD == pytest.approx(0.6)
        # Restore
        monkeypatch.delenv("COSINE_DISTANCE_THRESHOLD", raising=False)
        importlib.reload(g)
