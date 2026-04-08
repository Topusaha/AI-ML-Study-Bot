"""
Evaluation module for the ML Study Bot.

Contains:
  - HITLEvaluator: passkey-gated human-in-the-loop ratings + ChromaDB injection
  - AutoEvaluator: keyword hit-rate regression testing
  - print_metrics_report(): aggregate stats from a JSONL session log

Import order: Level 4 (imports ml/* and backend/logger.py).
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from ml.models import QAPair, RetrievalResult
from ml.retriever import HybridRetriever
from backend.logger import StudyLogger


# ---------------------------------------------------------------------------
# HITLEvaluator
# ---------------------------------------------------------------------------

class HITLEvaluator:
    """
    Human-in-the-loop evaluation: rate retrieved pairs, save to CSV,
    and optionally inject approved pairs back into ChromaDB.
    """

    CSV_COLUMNS = [
        "timestamp",
        "session_id",
        "query",
        "qa_pair_id",
        "question",
        "answer",
        "page_title",
        "section",
        "human_rating",
        "human_corrected_answer",
        "added_to_chroma",
    ]

    def __init__(
        self,
        csv_path: str = "data/human-in-the-loop-results.csv",
        retriever: HybridRetriever | None = None,
        logger: StudyLogger | None = None,
    ) -> None:
        self._csv_path = Path(csv_path)
        self._retriever = retriever
        self._logger = logger
        self._ensure_csv_header()

    def _ensure_csv_header(self) -> None:
        if not self._csv_path.exists():
            os.makedirs(self._csv_path.parent, exist_ok=True)
            with open(self._csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS)
                writer.writeheader()

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def check_passkey(self, input_passkey: str) -> bool:
        """Compare SHA-256 hashes — never compares plaintext."""
        stored = os.getenv("HITL_PASSKEY", "")
        if not stored:
            return False
        return (
            hashlib.sha256(input_passkey.encode()).hexdigest()
            == hashlib.sha256(stored.encode()).hexdigest()
        )

    # ------------------------------------------------------------------
    # Rating persistence
    # ------------------------------------------------------------------

    def save_ratings(
        self,
        query: str,
        result: RetrievalResult,
        ratings: list[str],
        corrections: list[str],
        session_id: str,
    ) -> list[dict]:
        """
        Persist ratings to CSV and log each rating event.

        Args:
            query:       The query used for retrieval.
            result:      RetrievalResult containing the rated pairs.
            ratings:     List of rating strings aligned with result.pairs.
                         Values: "relevant" | "partial" | "not_relevant"
            corrections: List of corrected answer strings (empty string = no correction).
            session_id:  Current session UUID.

        Returns:
            List of row dicts written to the CSV.
        """
        rows = []
        now = datetime.now(timezone.utc).isoformat()

        for pair, rating, correction in zip(result.pairs, ratings, corrections):
            row = {
                "timestamp": now,
                "session_id": session_id,
                "query": query,
                "qa_pair_id": pair.id,
                "question": pair.question,
                "answer": pair.answer,
                "page_title": pair.page_title,
                "section": pair.section,
                "human_rating": rating,
                "human_corrected_answer": correction,
                "added_to_chroma": "False",
            }
            rows.append(row)

            if self._logger:
                self._logger.log_hitl_rating(
                    query=query,
                    qa_pair_id=pair.id,
                    human_rating=rating,
                    mode="hitl",
                )

        with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS)
            writer.writerows(rows)

        return rows

    # ------------------------------------------------------------------
    # ChromaDB injection
    # ------------------------------------------------------------------

    def inject_to_chroma(self, qa_pair_ids: list[str]) -> int:
        """
        Upsert selected CSV rows into ChromaDB and mark them as injected.

        Returns the number of pairs successfully injected.
        Raises RuntimeError if no retriever was provided at construction.
        """
        if self._retriever is None:
            raise RuntimeError("HITLEvaluator requires a HybridRetriever to inject.")

        rows = self._load_csv()
        to_inject = [r for r in rows if r["qa_pair_id"] in qa_pair_ids and r["added_to_chroma"] == "False"]
        injected_ids: set[str] = set()

        for row in to_inject:
            pair = QAPair(
                id=row["qa_pair_id"],
                question=row["question"],
                answer=row["answer"],
                page_title=row["page_title"],
                section=row["section"],
                has_diagram=False,
            )
            self._retriever.upsert_hitl_pair(pair, corrected_answer=row["human_corrected_answer"])
            injected_ids.add(row["qa_pair_id"])

        # Update CSV: mark injected rows
        if injected_ids:
            self._update_csv_added_flag(injected_ids)

        return len(injected_ids)

    def pending_rows(self) -> list[dict]:
        """Return CSV rows where added_to_chroma == 'False'."""
        return [r for r in self._load_csv() if r["added_to_chroma"] == "False"]

    def _load_csv(self) -> list[dict]:
        if not self._csv_path.exists():
            return []
        with open(self._csv_path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def _update_csv_added_flag(self, injected_ids: set[str]) -> None:
        rows = self._load_csv()
        for row in rows:
            if row["qa_pair_id"] in injected_ids and row["added_to_chroma"] == "False":
                row["added_to_chroma"] = "True"
        with open(self._csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)


# ---------------------------------------------------------------------------
# AutoEvaluator
# ---------------------------------------------------------------------------

class AutoEvaluator:
    """Keyword-based hit-rate evaluation for regression testing."""

    def __init__(self, retriever: HybridRetriever) -> None:
        self._retriever = retriever

    def run_keyword_hit_rate(
        self,
        eval_pairs: list[tuple[str, list[str]]],
        k: int = 5,
    ) -> dict:
        """
        For each (query, expected_keywords) pair, retrieve top-k pairs and
        check whether any returned pair's text contains all expected keywords
        (case-insensitive).

        Returns:
            {"hit_rate": float, "num_queries": int, "hits": int}
        """
        hits = 0
        for query, keywords in eval_pairs:
            result = self._retriever.retrieve(query, k=k)
            corpus_text = " ".join(
                p.question + " " + p.answer for p in result.pairs
            ).lower()
            if all(kw.lower() in corpus_text for kw in keywords):
                hits += 1

        n = len(eval_pairs)
        return {
            "hit_rate": hits / n if n > 0 else 0.0,
            "num_queries": n,
            "hits": hits,
        }


# ---------------------------------------------------------------------------
# Metrics report
# ---------------------------------------------------------------------------

def print_metrics_report(log_path: str) -> None:
    """
    Read a JSONL session log and print a formatted summary to stdout.

    Metrics computed:
      - Total queries / refusals / quiz rounds
      - Quiz accuracy by page_title
      - HITL hit-rate by page_title (y + partial / total)
      - Weak topics (quiz accuracy < 50%, min 3 attempts)
    """
    events: list[dict] = []
    try:
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
    except FileNotFoundError:
        print(f"[metrics] Log file not found: {log_path}")
        return

    retrievals = [e for e in events if e.get("event_type") == "retrieval"]
    refusals   = [e for e in events if e.get("event_type") == "refusal"]
    grades     = [e for e in events if e.get("event_type") == "quiz_grade"]
    hitl       = [e for e in events if e.get("event_type") == "hitl_rating"]

    print("\n=== Session Summary ===")
    print(f"  Total retrievals : {len(retrievals)}")
    print(f"  Total refusals   : {len(refusals)}")
    print(f"  Quiz rounds      : {len(grades)}")
    print(f"  HITL ratings     : {len(hitl)}")

    if grades:
        print("\nQuiz Accuracy by Topic:")
        topic_grades: dict[str, list[str]] = {}
        for g in grades:
            topic = g.get("page_title", "unknown")
            topic_grades.setdefault(topic, []).append(g.get("grade", ""))
        for topic, gs in sorted(topic_grades.items()):
            correct = sum(1 for g in gs if g.lower() == "correct")
            pct = correct / len(gs) * 100
            bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
            flag = " ← needs work" if pct < 50 and len(gs) >= 3 else ""
            print(f"  {topic:<40} {bar}  {pct:.0f}%  ({correct}/{len(gs)}){flag}")

    if refusals:
        print("\nRefusals:")
        for r in refusals:
            print(f"  [{r.get('layer')}] {r.get('query')}")
