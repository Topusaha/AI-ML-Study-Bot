"""
StudyBot — orchestrates ingestion, indexing, retrieval, and session logging.

This is the single entry point for all backend operations called by main.py
and the Streamlit frontend.  It does NOT call OllamaClient directly — the
caller is responsible for passing RetrievalResult to the appropriate prompt
method.

Import order: Level 3 (imports ml/* and backend/logger.py).
"""

from __future__ import annotations

from ml.guardrails import (
    BM25_SCORE_THRESHOLD,
    COSINE_DISTANCE_THRESHOLD,
    InsufficientContextError,
)
from ml.models import QAPair, RetrievalResult
from ml.parser import MarkdownParser
from ml.retriever import HybridRetriever
from backend.logger import StudyLogger

_AVAILABLE_TOPICS = [
    "Decision Trees",
    "Training Models",
    "Support Vector Machines",
    "Ensemble Learning and Random Forests",
    "Dimensionality Reduction",
    "Unsupervised Learning Techniques",
    "Introduction to Artificial Neural Networks",
    "Training Deep Neural Networks",
]


class StudyBot:
    """
    High-level orchestrator for the ML Study Bot.

    Typical lifecycle:
        bot = StudyBot()
        bot.load_and_index()          # idempotent — skips if already indexed
        result = bot.retrieve(query)  # raises InsufficientContextError if weak
    """

    AVAILABLE_TOPICS: list[str] = _AVAILABLE_TOPICS

    def __init__(
        self,
        notion_dir: str = "data/assets/notion",
        chroma_path: str = "data/.chroma",
        log_dir: str = "data/logs",
    ) -> None:
        self._notion_dir = notion_dir
        self._parser = MarkdownParser()
        self._retriever = HybridRetriever(chroma_persist_dir=chroma_path)
        self._logger = StudyLogger(log_dir=log_dir)
        self._pairs: list[QAPair] = []

    @property
    def logger(self) -> StudyLogger:
        return self._logger

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def load_and_index(self, force_reingest: bool = False) -> None:
        """
        Parse Notion markdown, embed with ChromaDB default embedder, build BM25.

        If force_reingest=False and the ChromaDB collection is already populated,
        this method skips parsing and embedding (fast startup).

        On every startup, HITL-injected documents (source="hitl") are purged
        from ChromaDB before the idempotency check so that only source="notion"
        documents count toward the skip condition.
        """
        # Always clean up HITL injections from the previous session
        self._retriever.delete_by_source("hitl")

        if not force_reingest and self._retriever.is_populated():
            self._pairs = self._parser.parse(self._notion_dir)
            self._retriever.rebuild_bm25(self._pairs)
            return

        self._pairs = self._parser.parse(self._notion_dir)
        print(f"[StudyBot] Parsed {len(self._pairs)} Q&A pairs from {self._notion_dir}")
        self._retriever.index(self._pairs)
        print(f"[StudyBot] Indexed {len(self._pairs)} pairs into ChromaDB + BM25.")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        k: int = 5,
        page_title_filter: str | None = None,
        section_filter: str | None = None,
        mode: str = "rag",
    ) -> RetrievalResult:
        """
        Hybrid BM25 + dense retrieval with RRF merge, then Layer 1 guardrail check.

        Args:
            query:              Natural-language query.
            k:                  Number of pairs to return.
            page_title_filter:  Restrict results to a specific page title.
            section_filter:     Restrict results to "main_ideas" or "exercises".
            mode:               Mode label for the session log ("rag", "quiz", "hitl").

        Returns:
            RetrievalResult

        Raises:
            InsufficientContextError: if retrieval quality is too low.
        """
        where = _build_where(page_title_filter, section_filter)

        result = self._retriever.retrieve(query, k=k, where=where)

        # Layer 1 guardrail
        if len(result.pairs) == 0:
            self._logger.log_refusal(
                query=query,
                layer="no_results",
                top_cosine_distance=1.0,
                top_bm25_score=0.0,
                mode=mode,
            )
            raise InsufficientContextError("no_results", query)

        if (
            result.top_bm25_score <= BM25_SCORE_THRESHOLD
            and result.top_cosine_distance > COSINE_DISTANCE_THRESHOLD
        ):
            self._logger.log_refusal(
                query=query,
                layer="low_confidence",
                top_cosine_distance=result.top_cosine_distance,
                top_bm25_score=result.top_bm25_score,
                mode=mode,
            )
            raise InsufficientContextError("low_confidence", query)

        self._logger.log_retrieval(query=query, result=result, mode=mode)
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def full_corpus_text(self) -> str:
        """Concatenate all Q&A pairs as plain text for naive (non-RAG) LLM mode."""
        parts = [f"Q: {p.question}\nA: {p.answer}" for p in self._pairs]
        return "\n\n".join(parts)


def _build_where(
    page_title: str | None,
    section: str | None,
) -> dict | None:
    """Build a ChromaDB where-filter dict from optional filter values."""
    conditions = []
    if page_title:
        conditions.append({"page_title": {"$eq": page_title}})
    if section:
        conditions.append({"section": {"$eq": section}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}
