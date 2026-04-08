# System Components Design Document

## Metadata
- **Status:** Draft
- **Author(s):** topu
- **Created:** 2026-04-06
- **Updated:** 2026-04-06

## Overview

This document catalogs every runtime component in the ML Study Bot system, defines their interfaces and responsibilities, and serves as the authoritative reference for engineers implementing or modifying the codebase. It covers the full stack: ingestion and parsing of Notion markdown exports, hybrid retrieval, guardrail layers, LLM integration, session logging, human-in-the-loop evaluation, and the Streamlit frontend.

The system is fully local. No external APIs are called at inference time. Ollama serves `llama3.2` for all LLM interactions, ChromaDB stores dense vector embeddings (via `all-MiniLM-L6-v2`), and `rank-bm25` handles sparse keyword retrieval. The two retrieval signals are fused via Reciprocal Rank Fusion. Source data is a Notion markdown export of approximately 148 Q&A pairs spread across 8 ML topic pages under `assets/notion/`.

This document does not describe the Streamlit UI in detail — that is covered in `005_streamlit_ui.md`. Cross-references are provided where the frontend integrates with backend components. All component paths are relative to the repository root.

## Goals

- Provide a single reference document engineers can consult when implementing any module without needing to re-read the full codebase.
- Define all public class and function signatures precisely so that modules can be developed independently against a stable contract.
- Document non-obvious behavior (state machines, guardrail thresholds, hash comparisons, RRF merge logic) in one place.
- Capture data type definitions (`QAPair`, `RetrievalResult`, `InsufficientContextError`) used across module boundaries.
- Record the dependency graph so that build and import order is unambiguous.

## Component Catalog

---

### MarkdownParser (`ml/parser.py`)

**Responsibility:** Walk all `.md` files under `assets/notion/` and parse Notion-style toggle Q&A blocks into a list of `QAPair` objects.

**Interface:**
```python
@dataclass
class QAPair:
    id: str               # deterministic: f"{page_slug}_{line_num}"
    question: str
    answer: str
    page_title: str
    section: str          # "main_ideas" | "exercises" | ""
    has_diagram: bool
    img_paths: list[str]

class MarkdownParser:
    def parse(self, directory: str = "assets/notion/") -> list[QAPair]: ...
```

**Dependencies:** `os`, `pathlib`, `dataclasses`, `re`

**Key logic:**
- Walks every `.md` file recursively under the given directory; file stem is used as `page_slug` (lowercased, spaces replaced with underscores).
- State machine with three states:
  1. **Scanning for section heading** — triggers on lines matching `^### `. Heading text is normalized: if it contains "main idea" (case-insensitive) the section becomes `"main_ideas"`; if it contains "exercise" the section becomes `"exercises"`; otherwise `""`.
  2. **Top-level bullet = question** — lines matching `^- ` (no leading whitespace) open a new `QAPair`. The dash and space are stripped; the remainder is the `question`. Any previously open pair is flushed to the output list.
  3. **Indented content = answer** — lines with leading whitespace (one or more spaces or a tab) that appear after a top-level bullet are accumulated into the `answer` field of the current open pair. Leading whitespace is stripped before accumulation.
- `has_diagram` is set `True` if any answer line contains a Markdown image reference (`![`).
- `img_paths` collects all image paths found in `![](<path>)` syntax within the answer block.
- `id` is set to `f"{page_slug}_{line_num}"` where `line_num` is the 1-indexed line number of the question bullet within its source file. This makes IDs deterministic across re-ingestion.
- Empty questions or empty answers (after stripping) are skipped.

---

### HybridRetriever (`ml/retriever.py`)

**Responsibility:** Accept a natural-language query and return the top-k most relevant `QAPair` objects by fusing dense (ChromaDB) and sparse (BM25) retrieval signals.

**Interface:**
```python
@dataclass
class RetrievalResult:
    pairs: list[QAPair]
    top_cosine_distance: float   # lowest cosine distance among returned pairs (best = 0.0)
    top_bm25_score: float        # highest BM25 score among returned pairs

class HybridRetriever:
    def __init__(
        self,
        collection_name: str = "ml_study_bot",
        chroma_persist_dir: str = "data/chroma",
    ) -> None: ...

    def index(self, pairs: list[QAPair]) -> None: ...

    def retrieve(
        self,
        query: str,
        k: int = 5,
        where: dict | None = None,
    ) -> RetrievalResult: ...

    def is_populated(self) -> bool: ...
```

**Dependencies:** `chromadb`, `rank_bm25.BM25Okapi`, `sentence_transformers` (via ChromaDB default embedder `all-MiniLM-L6-v2`), `ml/parser.py` (`QAPair`)

**Key logic:**
- `index()` upserts all pairs into ChromaDB using `pair.id` as the document ID. Metadata stored per document: `page_title`, `section`, `has_diagram`. The indexed text is `question + " " + answer`.
- BM25 index is built in-memory during `index()` over the same concatenated strings, tokenized by whitespace. The BM25 corpus order must match the order of `pairs` passed to `index()` to allow score-to-pair mapping.
- `retrieve()` runs both searches independently:
  - Dense: `collection.query(query_texts=[query], n_results=k*2, where=where)` — fetches `k*2` candidates to give RRF enough signal.
  - Sparse: tokenizes query by whitespace, calls `bm25.get_scores(query_tokens)`, selects top `k*2` indices.
- Reciprocal Rank Fusion merge: for each candidate pair, `rrf_score = 1/(k_rrf + dense_rank) + 1/(k_rrf + sparse_rank)` where `k_rrf=60`. Pairs only in one result set receive a rank of `k*2 + 1` for the missing signal. Top-k by RRF score are returned.
- `where` filter is passed directly to ChromaDB's metadata filter syntax (e.g., `{"page_title": {"$eq": "Neural Networks"}}`). BM25 does not support metadata filtering; when a `where` filter is active, BM25 candidate IDs are post-filtered to only those IDs returned by the ChromaDB query before RRF merge.
- `top_cosine_distance` is the minimum distance value from ChromaDB's returned distances for the final top-k set (lower is better).
- `top_bm25_score` is the maximum BM25 score among the final top-k set.
- `is_populated()` returns `True` if the ChromaDB collection contains at least one document.

---

### Guardrails (`ml/guardrails.py`)

**Responsibility:** Define the `InsufficientContextError` exception and the threshold constants that govern both retrieval-layer and LLM-layer refusals.

**Interface:**
```python
COSINE_DISTANCE_THRESHOLD: float  # default 0.8, overridable via .env
REFUSAL_PHRASE: str = "I don't have sufficient notes on that topic."

class InsufficientContextError(Exception):
    def __init__(self, layer: str, query: str) -> None: ...
    layer: str    # "retrieval" | "llm"
    query: str
```

**Dependencies:** `os` (reads `COSINE_DISTANCE_THRESHOLD` from environment), `dotenv` (loaded by caller)

**Key logic:**
- **Layer 1 — retrieval gate** (enforced in `StudyBot.retrieve()`): raises `InsufficientContextError(layer="retrieval", query=query)` if either:
  - `len(result.pairs) == 0`, OR
  - `result.top_bm25_score == 0.0` AND `result.top_cosine_distance > COSINE_DISTANCE_THRESHOLD`
- **Layer 2 — LLM refusal** (enforced in `OllamaClient`): raises `InsufficientContextError(layer="llm", query=query)` if the raw Ollama response string starts with the exact phrase `REFUSAL_PHRASE` (checked with `str.startswith`, case-sensitive).
- `COSINE_DISTANCE_THRESHOLD` is read from the environment at module import time via `float(os.getenv("COSINE_DISTANCE_THRESHOLD", "0.8"))`.
- `InsufficientContextError` is caught in `main.py` and in each Streamlit page; the handler prints or displays the list of available topics.
- This module does not import any other project module — it is a leaf dependency.

---

### OllamaClient (`ml/llm_client.py`)

**Responsibility:** Wrap the `ollama.chat` API and expose three task-specific prompt methods for RAG answering, quiz generation, and answer grading.

**Interface:**
```python
OLLAMA_MODEL: str          # read from env, e.g. "llama3.2"
REFUSAL_PHRASE: str        # imported from ml/guardrails.py

class OllamaClient:
    def __init__(self, model: str = OLLAMA_MODEL) -> None: ...

    def answer_from_snippets(
        self,
        query: str,
        snippets: list[QAPair],
    ) -> str: ...

    def quiz_from_snippets(
        self,
        snippets: list[QAPair],
    ) -> str: ...

    def grade_student_answer(
        self,
        question: str,
        student_answer: str,
        snippets: list[QAPair],
    ) -> str: ...
```

**Dependencies:** `ollama`, `ml/parser.py` (`QAPair`), `ml/guardrails.py` (`InsufficientContextError`, `REFUSAL_PHRASE`)

**Key logic:**
- All three methods build a `messages` list and call `ollama.chat(model=self.model, messages=messages)`, then return `response["message"]["content"]`.
- Every system prompt includes the Layer 2 refusal instruction: the model is told to respond with exactly `REFUSAL_PHRASE` (and nothing else) if it cannot answer strictly from the provided notes.
- After receiving the response, each method checks `response_text.startswith(REFUSAL_PHRASE)` and raises `InsufficientContextError(layer="llm", query=...)` if true.
- `answer_from_snippets`: system prompt enforces strict grounding ("answer only from the snippets below, do not add outside knowledge"). User message includes the query and formatted snippet text. The prompt instructs the model to cite the source question text for each claim.
- `quiz_from_snippets`: system prompt instructs the model to generate exactly one novel question that can be answered from the provided snippets only — not to reproduce a question verbatim. No `query` parameter; snippets alone drive generation.
- `grade_student_answer`: system prompt instructs the model to return a grade of exactly one of `Correct`, `Partial`, or `Incorrect` on the first line, followed by 2–3 sentences of feedback grounded in the snippets. Snippets are provided as reference material.
- Snippet formatting for all prompts: each `QAPair` is rendered as `"Q: {pair.question}\nA: {pair.answer}"`, separated by `"\n\n---\n\n"`.
- `OLLAMA_MODEL` is read from the environment at module import time via `os.getenv("OLLAMA_MODEL", "llama3.2")`.

---

### StudyBot (`backend/studybot.py`)

**Responsibility:** Orchestrate ingestion, indexing, retrieval, and session logging; serve as the single entry point for all backend operations called by the CLI and Streamlit app.

**Interface:**
```python
class StudyBot:
    def __init__(self) -> None: ...
    # Internally creates: MarkdownParser, HybridRetriever, StudyLogger

    def load_and_index(self, force_reingest: bool = False) -> None: ...

    def retrieve(
        self,
        query: str,
        k: int = 5,
        page_title_filter: str | None = None,
        section_filter: str | None = None,
    ) -> RetrievalResult: ...

    def full_corpus_text(self) -> str: ...
```

**Dependencies:** `ml/parser.py` (`MarkdownParser`, `QAPair`), `ml/retriever.py` (`HybridRetriever`, `RetrievalResult`), `ml/guardrails.py` (`InsufficientContextError`, `COSINE_DISTANCE_THRESHOLD`), `backend/logger.py` (`StudyLogger`)

**Key logic:**
- `load_and_index()`: if `force_reingest=False` and `retriever.is_populated()` returns `True`, skips parsing and embedding (fast startup). If `force_reingest=True` or ChromaDB is empty, calls `parser.parse()`, then `retriever.index(pairs)`. Stores the parsed pairs list as `self._pairs` for BM25 access and `full_corpus_text()`.
- `retrieve()`: constructs the ChromaDB `where` dict from `page_title_filter` and `section_filter` (uses `$eq` operator; omits the key if the filter value is `None`). Calls `retriever.retrieve(query, k, where)`. Applies Layer 1 guardrail check (see Guardrails section) and raises `InsufficientContextError` if triggered. Logs a `retrieval` event via `StudyLogger`.
- `full_corpus_text()`: concatenates all `QAPair` question+answer strings from `self._pairs` for use in naive (non-RAG) LLM mode where the full corpus is passed in context.
- `StudyLogger` is instantiated once inside `__init__` so that all method calls share the same session ID.
- `StudyBot` does not call `OllamaClient` directly — the caller (CLI `main.py` or Streamlit pages) is responsible for passing retrieval results to the appropriate `OllamaClient` method.

---

### StudyLogger (`backend/logger.py`)

**Responsibility:** Write structured JSONL event records for every retrieval, LLM response, quiz grade, HITL rating, and refusal during a session.

**Interface:**
```python
class StudyLogger:
    def __init__(self, log_dir: str = "data/logs") -> None: ...
    # Generates session_id = str(uuid.uuid4()) at construction time
    # Opens file: log_dir/session_YYYYMMDD_HHMMSS.jsonl

    def log_retrieval(
        self,
        query: str,
        result: RetrievalResult,
        mode: str,
    ) -> None: ...

    def log_llm_response(
        self,
        query: str,
        response: str,
        mode: str,
    ) -> None: ...

    def log_quiz_grade(
        self,
        question: str,
        student_answer: str,
        grade: str,
        feedback: str,
        mode: str,
    ) -> None: ...

    def log_hitl_rating(
        self,
        query: str,
        qa_pair_id: str,
        human_rating: str,
        mode: str,
    ) -> None: ...

    def log_refusal(
        self,
        query: str,
        layer: str,
        top_cosine_distance: float,
        top_bm25_score: float,
        mode: str,
    ) -> None: ...
```

**Dependencies:** `uuid`, `datetime`, `json`, `os`, `pathlib`, `backend/retriever.py` (`RetrievalResult`) — imported for type annotation only

**Key logic:**
- Each log method writes a single JSON object as one line to the session file, followed by `\n`. The file is opened in append mode.
- Every event record contains: `session_id` (uuid4 string, fixed for the session), `timestamp` (ISO 8601 UTC string), `mode` (caller-provided string, e.g. `"rag"`, `"quiz"`, `"naive"`), `event_type` (one of `"retrieval"`, `"llm_response"`, `"quiz_grade"`, `"hitl_rating"`, `"refusal"`).
- Type-specific fields beyond the common ones:
  - `retrieval`: `query`, `num_pairs_returned`, `top_cosine_distance`, `top_bm25_score`
  - `llm_response`: `query`, `response_length` (character count)
  - `quiz_grade`: `question`, `student_answer`, `grade`, `feedback`
  - `hitl_rating`: `query`, `qa_pair_id`, `human_rating`
  - `refusal`: `query`, `layer`, `top_cosine_distance`, `top_bm25_score`
- All log methods are wrapped in a `try/except Exception` block; on failure they call `warnings.warn(...)` and return silently — they never raise.
- `log_dir` is created with `os.makedirs(exist_ok=True)` on construction.

---

### HITLEvaluator (`backend/evaluation.py`)

**Responsibility:** Gate access to human-in-the-loop evaluation with a passkey, collect per-query human relevance ratings, persist results to CSV, and optionally inject approved pairs back into ChromaDB.

**Interface:**
```python
class HITLEvaluator:
    def __init__(
        self,
        csv_path: str = "data/human-in-the-loop-results.csv",
        retriever: HybridRetriever | None = None,
    ) -> None: ...

    def check_passkey(self, input_passkey: str) -> bool: ...

    def run_evaluation(
        self,
        query: str,
        result: RetrievalResult,
        session_id: str,
        mode: str,
    ) -> list[dict]: ...

    def inject_to_chroma(self, rows: list[dict]) -> None: ...

class AutoEvaluator:
    def __init__(self, retriever: HybridRetriever) -> None: ...

    def run_keyword_hit_rate(
        self,
        eval_pairs: list[tuple[str, list[str]]],
        k: int = 5,
    ) -> dict: ...
    # eval_pairs: list of (query, expected_keyword_list)
    # returns {"hit_rate": float, "num_queries": int, "hits": int}
```

**Dependencies:** `hashlib`, `os`, `csv`, `datetime`, `uuid`, `ml/parser.py` (`QAPair`), `ml/retriever.py` (`HybridRetriever`, `RetrievalResult`), `backend/logger.py` (`StudyLogger`)

**Key logic:**
- `check_passkey()`: computes `hashlib.sha256(input_passkey.encode()).hexdigest()` and compares against `hashlib.sha256(os.getenv("HITL_PASSKEY", "").encode()).hexdigest()`. Returns `True` only if the hashes match. Never compares plaintext strings.
- `run_evaluation()`: for each `QAPair` in `result.pairs`, prompts the human rater (via Streamlit UI in practice) to select a rating of `"relevant"`, `"partial"`, or `"not relevant"`, and optionally enter a corrected answer. Writes one CSV row per pair.
- CSV columns (in order): `timestamp`, `session_id`, `query`, `qa_pair_id`, `question`, `answer`, `page_title`, `section`, `human_rating`, `human_corrected_answer`, `added_to_chroma`
  - `added_to_chroma` defaults to `"False"` on write; updated to `"True"` after a successful `inject_to_chroma()` call.
- `inject_to_chroma()`: upserts each row in the provided list into the ChromaDB collection using `qa_pair_id` as the document ID. Uses `human_corrected_answer` as the answer text if non-empty, otherwise uses the original `answer`. After successful upsert, updates the CSV in-place to set `added_to_chroma=True` for those rows.
- HITL injections are not auto-persisted on server restart — ChromaDB is re-indexed from the original Notion data only. To make injections permanent they must be re-injected after each restart, or the CSV must be replayed manually.
- `AutoEvaluator.run_keyword_hit_rate()`: for each `(query, expected_keywords)` pair, calls `retriever.retrieve(query, k=k)`, checks whether any returned pair's question or answer contains all expected keywords (case-insensitive), counts hits, returns `{"hit_rate": float, "num_queries": int, "hits": int}`. Kept for regression testing.

---

### Streamlit App (`frontend/`)

**Responsibility:** Provide a four-tab web UI for RAG Q&A, quiz generation and grading, session metrics, and HITL evaluation.

**Interface:** See `005_streamlit_ui.md` for full page-by-page specification.

**Summary of tabs:**
- **RAG Q&A** — user types a query, optionally filters by page/section, calls `StudyBot.retrieve()` then `OllamaClient.answer_from_snippets()`
- **Quiz Me** — calls `StudyBot.retrieve()` then `OllamaClient.quiz_from_snippets()`, accepts student answer, calls `OllamaClient.grade_student_answer()`
- **Metrics & Logs** — reads JSONL log files from `data/logs/`, displays retrieval stats and refusal counts
- **Human-in-the-Loop** — passkey gate via `HITLEvaluator.check_passkey()`, then `HITLEvaluator.run_evaluation()` and optional `inject_to_chroma()`

**Shared sidebar fields:** model name (`OLLAMA_MODEL`), session ID, total queries this session, refusal count this session.

---

## Component Interaction Map

```
assets/notion/*.md
        |
        v
  MarkdownParser          (ml/parser.py)
        |
        | list[QAPair]
        v
   StudyBot               (backend/studybot.py)
   |       |
   |       | index(pairs)
   |       v
   |  HybridRetriever     (ml/retriever.py)
   |  |          |
   |  | ChromaDB | BM25 (in-memory)
   |  |          |
   |  +----------+
   |       | RetrievalResult
   |       v
   |  [Layer 1 guardrail check]  (ml/guardrails.py)
   |       |
   |       | RetrievalResult (or raises InsufficientContextError)
   v       v
StudyLogger            OllamaClient           (ml/llm_client.py)
(backend/logger.py)    |   |   |
                       |   |   +-- answer_from_snippets()
                       |   +------ quiz_from_snippets()
                       +---------- grade_student_answer()
                                   |
                              [Layer 2 guardrail check]
                                   |
                              str response (or raises InsufficientContextError)
                                   |
                                   v
                           Streamlit frontend / main.py
                                   |
                           HITLEvaluator (backend/evaluation.py)
                           |             |
                           CSV store     inject_to_chroma() --> HybridRetriever
```

---

## Data Types Reference

### QAPair

```python
from dataclasses import dataclass, field

@dataclass
class QAPair:
    id: str            # f"{page_slug}_{line_num}" — deterministic, stable across re-ingestion
    question: str      # stripped text of the top-level "- " bullet
    answer: str        # accumulated stripped text of all indented lines below the question
    page_title: str    # human-readable title of the source .md file
    section: str       # "main_ideas" | "exercises" | "" (empty if no ### heading precedes)
    has_diagram: bool  # True if any answer line contains "!["
    img_paths: list[str] = field(default_factory=list)  # paths extracted from ![](...) syntax
```

### RetrievalResult

```python
from dataclasses import dataclass

@dataclass
class RetrievalResult:
    pairs: list[QAPair]
    top_cosine_distance: float
    # Cosine distance of the nearest neighbor among returned pairs.
    # Range [0.0, 2.0]; lower is more similar. 0.0 = identical vector.
    top_bm25_score: float
    # Highest BM25Okapi score among returned pairs.
    # Range [0.0, inf); 0.0 means no term overlap with query.
```

### InsufficientContextError

```python
class InsufficientContextError(Exception):
    def __init__(self, layer: str, query: str) -> None:
        self.layer = layer   # "retrieval" | "llm"
        self.query = query
        super().__init__(f"[{layer}] Insufficient context for query: {query!r}")
```

### Guardrail Constants

```python
COSINE_DISTANCE_THRESHOLD: float = float(os.getenv("COSINE_DISTANCE_THRESHOLD", "0.8"))
REFUSAL_PHRASE: str = "I don't have sufficient notes on that topic."
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")
```

### HITL CSV Row (as dict)

```python
{
    "timestamp":               str,   # ISO 8601 UTC
    "session_id":              str,   # uuid4
    "query":                   str,
    "qa_pair_id":              str,   # QAPair.id
    "question":                str,   # QAPair.question
    "answer":                  str,   # QAPair.answer
    "page_title":              str,   # QAPair.page_title
    "section":                 str,   # QAPair.section
    "human_rating":            str,   # "relevant" | "partial" | "not relevant"
    "human_corrected_answer":  str,   # free text or "" if no correction provided
    "added_to_chroma":         str,   # "True" | "False"
}
```

---

## Dependency Graph

Build / import order (leaves first):

```
Level 0 (no project imports):
  ml/guardrails.py

Level 1 (imports only Level 0):
  ml/parser.py          -- no project deps
  ml/llm_client.py      -- imports ml/guardrails.py

Level 2 (imports Level 0–1):
  ml/retriever.py       -- imports ml/parser.py

Level 3 (imports Level 0–2):
  backend/logger.py     -- imports ml/retriever.py (type annotation only)
  backend/studybot.py   -- imports ml/parser.py, ml/retriever.py,
                           ml/guardrails.py, backend/logger.py

Level 4 (imports Level 0–3):
  backend/evaluation.py -- imports ml/parser.py, ml/retriever.py,
                           backend/logger.py

Level 5 (imports everything):
  frontend/*.py         -- imports backend/studybot.py, ml/llm_client.py,
                           backend/evaluation.py, backend/logger.py
  main.py               -- imports backend/studybot.py, ml/llm_client.py,
                           backend/evaluation.py, ml/guardrails.py
```

**Circular import check:** There are no circular dependencies. `ml/guardrails.py` and `ml/parser.py` are pure leaves. `backend/logger.py` imports `RetrievalResult` for type annotation only and must not import `StudyBot` or `HITLEvaluator`.
