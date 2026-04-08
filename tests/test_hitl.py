"""
Tests for HITLEvaluator (backend/evaluation.py).

Covers: CSV write/read round-trip, passkey gate, inject_to_chroma + cleanup.

Run with: pytest tests/test_hitl.py -v
"""

import os
import hashlib
import csv
import tempfile
from pathlib import Path

import pytest

from ml.models import QAPair, RetrievalResult
from ml.retriever import HybridRetriever
from backend.evaluation import HITLEvaluator

_TMPDIR = "/tmp/test_chroma_hitl"


@pytest.fixture
def tmp_csv(tmp_path) -> str:
    return str(tmp_path / "hitl_test.csv")


@pytest.fixture(scope="module")
def retriever() -> HybridRetriever:
    r = HybridRetriever(collection_name="hitl_test", chroma_persist_dir=_TMPDIR)
    r.index([
        QAPair(
            id="dt_1",
            question="What is Gini impurity?",
            answer="Measures misclassification probability.",
            page_title="Decision Trees",
            section="main_ideas",
            has_diagram=False,
        )
    ])
    return r


@pytest.fixture
def evaluator(tmp_csv, retriever) -> HITLEvaluator:
    return HITLEvaluator(csv_path=tmp_csv, retriever=retriever)


@pytest.fixture
def sample_result() -> RetrievalResult:
    pair = QAPair(
        id="dt_1",
        question="What is Gini impurity?",
        answer="Measures misclassification probability.",
        page_title="Decision Trees",
        section="main_ideas",
        has_diagram=False,
    )
    return RetrievalResult(pairs=[pair], top_cosine_distance=0.3, top_bm25_score=1.5)


class TestPasskeyGate:
    def test_correct_passkey(self, evaluator, monkeypatch):
        monkeypatch.setenv("HITL_PASSKEY", "mysecret")
        assert evaluator.check_passkey("mysecret") is True

    def test_wrong_passkey(self, evaluator, monkeypatch):
        monkeypatch.setenv("HITL_PASSKEY", "mysecret")
        assert evaluator.check_passkey("wrong") is False

    def test_empty_env_always_false(self, evaluator, monkeypatch):
        monkeypatch.setenv("HITL_PASSKEY", "")
        assert evaluator.check_passkey("anything") is False


class TestCSVRoundTrip:
    def test_save_creates_csv(self, evaluator, sample_result, tmp_csv):
        evaluator.save_ratings(
            query="What is Gini?",
            result=sample_result,
            ratings=["relevant"],
            corrections=[""],
            session_id="test-session-id",
        )
        assert Path(tmp_csv).exists()

    def test_saved_row_has_correct_fields(self, evaluator, sample_result, tmp_csv):
        evaluator.save_ratings(
            query="What is Gini?",
            result=sample_result,
            ratings=["partial"],
            corrections=["Better answer here."],
            session_id="test-session-id",
        )
        with open(tmp_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) >= 1
        row = rows[-1]
        assert row["query"] == "What is Gini?"
        assert row["human_rating"] == "partial"
        assert row["human_corrected_answer"] == "Better answer here."
        assert row["added_to_chroma"] == "False"

    def test_pending_rows_returned(self, evaluator, sample_result):
        evaluator.save_ratings(
            query="Pending query",
            result=sample_result,
            ratings=["not_relevant"],
            corrections=[""],
            session_id="test-session-id",
        )
        pending = evaluator.pending_rows()
        assert len(pending) >= 1
        assert all(r["added_to_chroma"] == "False" for r in pending)


class TestInjectToChroma:
    def test_inject_marks_row_as_added(self, evaluator, sample_result, tmp_csv):
        evaluator.save_ratings(
            query="inject test",
            result=sample_result,
            ratings=["relevant"],
            corrections=[""],
            session_id="test-session-id",
        )
        pending = evaluator.pending_rows()
        ids_to_inject = [r["qa_pair_id"] for r in pending]

        injected = evaluator.inject_to_chroma(ids_to_inject)
        assert injected == len(ids_to_inject)

        # Rows should now be marked added_to_chroma=True
        remaining_pending = evaluator.pending_rows()
        injected_set = set(ids_to_inject)
        for r in remaining_pending:
            assert r["qa_pair_id"] not in injected_set
