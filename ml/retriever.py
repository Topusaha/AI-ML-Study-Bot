"""
HybridRetriever — ChromaDB dense retrieval + BM25 sparse retrieval fused via RRF.

Dependencies: chromadb, rank_bm25, ml/models.py
No backend/ or frontend/ imports allowed in this module.
"""

from __future__ import annotations

import re
from typing import Optional

import chromadb
from rank_bm25 import BM25Okapi

from ml.models import QAPair, RetrievalResult


_RRF_K: int = 60  # standard RRF constant; higher = smoother rank blending


def _tokenize(text: str) -> list[str]:
    """Lowercase and split on whitespace/punctuation."""
    return re.findall(r"[a-z0-9]+", text.lower())


class HybridRetriever:
    """
    Combines ChromaDB (dense, cosine-distance) and BM25 (sparse, keyword)
    retrieval via Reciprocal Rank Fusion.

    Usage:
        retriever = HybridRetriever()
        retriever.index(qa_pairs)
        result = retriever.retrieve("What is the CART algorithm?", k=5)
    """

    def __init__(
        self,
        collection_name: str = "ml_study_bot",
        chroma_persist_dir: str = "data/.chroma",
    ) -> None:
        self._collection_name = collection_name
        self._client = chromadb.PersistentClient(path=chroma_persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        # BM25 state — populated by index()
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_corpus: list[QAPair] = []  # parallel list to BM25 corpus

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(self, pairs: list[QAPair]) -> None:
        """
        Upsert all pairs into ChromaDB and rebuild the in-memory BM25 index.
        Safe to call multiple times; ChromaDB upsert is idempotent on pair.id.
        """
        if not pairs:
            return

        ids = [p.id for p in pairs]
        documents = [p.question + " " + p.answer for p in pairs]
        metadatas = [
            {
                "page_title": p.page_title,
                "section": p.section,
                "has_diagram": p.has_diagram,
                "source": "notion",
            }
            for p in pairs
        ]

        # Upsert in batches of 100 to stay within ChromaDB limits
        batch_size = 100
        for i in range(0, len(pairs), batch_size):
            self._collection.upsert(
                ids=ids[i : i + batch_size],
                documents=documents[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )

        # Build BM25 over the same text
        tokenized = [_tokenize(doc) for doc in documents]
        self._bm25 = BM25Okapi(tokenized)
        self._bm25_corpus = list(pairs)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        k: int = 5,
        where: Optional[dict] = None,
    ) -> RetrievalResult:
        """
        Hybrid retrieval: dense (ChromaDB) + sparse (BM25) fused with RRF.

        Args:
            query:  Natural-language query string.
            k:      Number of pairs to return.
            where:  Optional ChromaDB metadata filter dict, e.g.
                    {"page_title": {"$eq": "Decision Trees"}}.
                    When provided, BM25 results are post-filtered to only IDs
                    that ChromaDB also returned.

        Returns:
            RetrievalResult with top-k pairs sorted by RRF score.
        """
        candidate_count = k * 2  # fetch more candidates for RRF to blend

        # --- Dense retrieval ---
        chroma_kwargs: dict = {
            "query_texts": [query],
            "n_results": min(candidate_count, self._collection.count() or 1),
            "include": ["distances", "metadatas", "documents"],
        }
        if where:
            chroma_kwargs["where"] = where

        chroma_result = self._collection.query(**chroma_kwargs)

        chroma_ids: list[str] = chroma_result["ids"][0]
        chroma_distances: list[float] = chroma_result["distances"][0]

        # Map id → distance for later look-up
        id_to_distance: dict[str, float] = dict(zip(chroma_ids, chroma_distances))

        # --- Sparse retrieval (BM25) ---
        bm25_ids: list[str] = []
        bm25_scores: dict[str, float] = {}

        if self._bm25 is not None and self._bm25_corpus:
            tokens = _tokenize(query)
            scores = self._bm25.get_scores(tokens)

            # If a where filter is active, restrict BM25 to ChromaDB result IDs
            allowed_ids = set(chroma_ids) if where else None

            indexed_scores = sorted(
                enumerate(scores), key=lambda x: x[1], reverse=True
            )
            for corpus_idx, score in indexed_scores:
                if len(bm25_ids) >= candidate_count:
                    break
                pair_id = self._bm25_corpus[corpus_idx].id
                if allowed_ids is not None and pair_id not in allowed_ids:
                    continue
                bm25_ids.append(pair_id)
                bm25_scores[pair_id] = float(score)

        # --- RRF merge ---
        # Assign ranks (1-indexed)
        chroma_rank: dict[str, int] = {pid: r + 1 for r, pid in enumerate(chroma_ids)}
        bm25_rank: dict[str, int] = {pid: r + 1 for r, pid in enumerate(bm25_ids)}

        all_ids = set(chroma_ids) | set(bm25_ids)
        fallback_rank = candidate_count + 1  # rank assigned when absent from a list

        rrf_scores: dict[str, float] = {}
        for pid in all_ids:
            dense_r = chroma_rank.get(pid, fallback_rank)
            sparse_r = bm25_rank.get(pid, fallback_rank)
            rrf_scores[pid] = 1 / (_RRF_K + dense_r) + 1 / (_RRF_K + sparse_r)

        top_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)[:k]

        # --- Build QAPair list from BM25 corpus (O(n) but n≈150) ---
        id_to_pair: dict[str, QAPair] = {p.id: p for p in self._bm25_corpus}
        result_pairs: list[QAPair] = [
            id_to_pair[pid] for pid in top_ids if pid in id_to_pair
        ]

        # --- Compute score metadata ---
        top_cosine_distance: float = (
            min((id_to_distance.get(pid, 1.0) for pid in top_ids), default=1.0)
        )
        top_bm25_score: float = (
            max((bm25_scores.get(pid, 0.0) for pid in top_ids), default=0.0)
        )

        return RetrievalResult(
            pairs=result_pairs,
            top_cosine_distance=top_cosine_distance,
            top_bm25_score=top_bm25_score,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def rebuild_bm25(self, pairs: list[QAPair]) -> None:
        """Rebuild the in-memory BM25 index from the given pairs without upserting to ChromaDB."""
        if not pairs:
            return
        documents = [p.question + " " + p.answer for p in pairs]
        tokenized = [_tokenize(doc) for doc in documents]
        self._bm25 = BM25Okapi(tokenized)
        self._bm25_corpus = list(pairs)

    def is_populated(self) -> bool:
        """Return True if the ChromaDB collection contains at least one document."""
        return self._collection.count() > 0

    def delete_by_source(self, source: str) -> None:
        """Delete all documents whose metadata.source matches `source`."""
        self._collection.delete(where={"source": {"$eq": source}})

    def upsert_hitl_pair(self, pair: QAPair, corrected_answer: str = "") -> None:
        """
        Inject a human-corrected pair into ChromaDB for the current session.
        Uses the original answer if corrected_answer is empty.
        Marked with source="hitl" so it is cleaned up on restart.
        """
        answer_text = corrected_answer if corrected_answer.strip() else pair.answer
        document = pair.question + " " + answer_text
        hitl_id = pair.id + "_hitl"

        self._collection.upsert(
            ids=[hitl_id],
            documents=[document],
            metadatas=[
                {
                    "page_title": pair.page_title,
                    "section": pair.section,
                    "has_diagram": False,
                    "source": "hitl",
                }
            ],
        )

        # Rebuild BM25 to include the new pair
        hitl_pair = QAPair(
            id=hitl_id,
            question=pair.question,
            answer=answer_text,
            page_title=pair.page_title,
            section=pair.section,
            has_diagram=False,
        )
        self._bm25_corpus.append(hitl_pair)
        tokenized = [_tokenize(p.question + " " + p.answer) for p in self._bm25_corpus]
        self._bm25 = BM25Okapi(tokenized)
