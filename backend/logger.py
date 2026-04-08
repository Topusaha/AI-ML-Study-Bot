"""
StudyLogger — append-only JSONL session logger.

One log file per CLI/Streamlit session under data/logs/.
Every log method is wrapped in try/except so a write failure never crashes
the main interaction — a warning is printed instead.

Import order: Level 3 (imports ml/retriever for type annotation only).
"""

from __future__ import annotations

import json
import os
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ml.models import RetrievalResult

_SENTINEL_DISTANCE = 1.0
_SENTINEL_BM25 = 0.0


class StudyLogger:
    """Write structured JSONL event records for every interaction in a session."""

    def __init__(self, log_dir: str = "data/logs") -> None:
        self.session_id: str = str(uuid.uuid4())
        self.session_start: datetime = datetime.now(timezone.utc)

        log_path = Path(log_dir)
        os.makedirs(log_path, exist_ok=True)

        filename = "session_" + self.session_start.strftime("%Y%m%d_%H%M%S") + ".jsonl"
        self.current_log_path: Path = log_path / filename

    # ------------------------------------------------------------------
    # Core write
    # ------------------------------------------------------------------

    def _write(self, record: dict) -> None:
        try:
            with open(self.current_log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"[StudyLogger] Failed to write log entry: {exc}")

    def _base(self, event_type: str, mode: str) -> dict:
        return {
            "session_id": self.session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": mode,
            "event_type": event_type,
        }

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def log_retrieval(
        self,
        query: str,
        result: "RetrievalResult",
        mode: str,
    ) -> None:
        record = self._base("retrieval", mode)
        record.update(
            {
                "query": query,
                "num_pairs_returned": len(result.pairs),
                "top_cosine_distance": result.top_cosine_distance,
                "top_bm25_score": result.top_bm25_score,
            }
        )
        self._write(record)

    def log_llm_response(
        self,
        query: str,
        response: str,
        mode: str,
    ) -> None:
        record = self._base("llm_response", mode)
        record.update(
            {
                "query": query,
                "response_length": len(response),
            }
        )
        self._write(record)

    def log_quiz_grade(
        self,
        question: str,
        student_answer: str,
        grade: str,
        feedback: str,
        mode: str,
        page_title: str = "",
    ) -> None:
        record = self._base("quiz_grade", mode)
        record.update(
            {
                "question": question,
                "student_answer": student_answer,
                "grade": grade,
                "feedback": feedback,
                "page_title": page_title,
            }
        )
        self._write(record)

    def log_hitl_rating(
        self,
        query: str,
        qa_pair_id: str,
        human_rating: str,
        mode: str,
    ) -> None:
        record = self._base("hitl_rating", mode)
        record.update(
            {
                "query": query,
                "qa_pair_id": qa_pair_id,
                "human_rating": human_rating,
            }
        )
        self._write(record)

    def log_refusal(
        self,
        query: str,
        layer: str,
        top_cosine_distance: float,
        top_bm25_score: float,
        mode: str,
    ) -> None:
        record = self._base("refusal", mode)
        record.update(
            {
                "query": query,
                "layer": layer,
                "top_cosine_distance": top_cosine_distance,
                "top_bm25_score": top_bm25_score,
            }
        )
        self._write(record)
