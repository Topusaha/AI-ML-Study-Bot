# ML Study Bot Design Document

## Metadata
- **Status:** Draft
- **Author(s):** topu
- **Reviewers:**
- **Created:** 2026-04-06
- **Updated:** 2026-04-06
- **Implementation PR(s):**

---

## Overview

The existing DocuBot reads flat Markdown files and retrieves chunks via BM25 keyword matching against a generic doc corpus. We are replacing it with a **specialized ML Study Bot** that is:

- **Fully local** — no external API calls. Ollama serves both the LLM and the embedding model.
- **Q&A focused** — retrieves labeled question/answer pairs from your personal Notion ML notes.
- **Active learning** — students can ask questions (RAG), get quizzed (Quiz mode), and evaluate retrieval quality themselves (human-in-the-loop Evaluation mode).
- **Observable** — every interaction is logged as structured JSON so students and developers can track study progress, quiz accuracy, and retrieval quality over time.

The source data is a Notion export already downloaded to `assets/notion/` — **~148 Q&A pairs** across 8 ML topic pages. The Markdown export preserves toggle structure as indented bullet lists, so no Notion API is needed.

---

## Corpus Facts (from `assets/notion/`)

| Page | Q&A Pairs | Sections |
|------|-----------|----------|
| Training Models | 27 | Main Ideas, Exercises |
| Unsupervised Learning Techniques | 23 | Main Ideas, Exercises |
| Ensemble Learning and Random Forests | 20 | Main Ideas, Exercises |
| Dimensionality Reduction | 18 | Main Ideas, Exercises |
| Introduction to Artificial Neural Networks | 18 | Main Ideas, Exercises |
| Decision Trees | 17 | Main Ideas, Exercises |
| Support Vector Machines | 15 | Main Ideas, Exercises |
| Training Deep Neural Networks | 10 | Main Ideas, Exercises |
| **Total** | **~148** | |

**Toggle format in Markdown export (how toggles look after export):**
```markdown
- What is the CART algorithm?              ← question (0-indent bullet)
    - A greedy algorithm that...           ← answer line (4-space indent)
    - Splits training set recursively...
        - Sub-point with more detail       ← still answer (8-space indent)

    ![](Decision Trees .../Untitled.png)   ← diagram embedded in answer

- Next question                            ← new Q&A pair starts here
```

Images exist as local PNGs in per-page subfolders. **No Notion API required.**

---

## Goals

- Parse `assets/notion/` Markdown export into ~148 structured `QAPair` objects.
- Build a hybrid retriever: dense embeddings (Ollama `nomic-embed-text`) + BM25 + RRF merge.
- Expose three modes: **RAG Q&A**, **Quiz Me**, **Evaluation**.
- All inference runs locally through Ollama — zero external API calls, zero API keys.
- Log every interaction (query, retrieval, LLM response, quiz grade, HITL rating) to `logs/` as JSONL for study progress tracking and retrieval diagnostics.

---

## Technology Stack

| Layer | Technology | Replaces |
|-------|-----------|---------|
| LLM inference | Ollama (local) — `llama3.2` | Gemini API |
| Embeddings | ChromaDB default (`all-MiniLM-L6-v2`) | Gemini `text-embedding-004` |
| Vector store | ChromaDB (local persistent) | — (new) |
| Sparse retrieval | `rank-bm25` BM25Okapi | Same (keep) |
| Merge strategy | Reciprocal Rank Fusion | — (new) |
| Data parsing | Custom Markdown indentation parser | Heading-split chunker in `docubot.py` |
| **Logging** | **`logger.py` — JSONL to `logs/`** | **— (new)** |
| Config | `python-dotenv` `.env` for `OLLAMA_MODEL` | `GEMINI_API_KEY` |

**Updated `requirements.txt`:**
```
ollama>=0.2.0
chromadb>=0.5.0
rank-bm25>=0.2.2
python-dotenv>=1.0.0
```

**Ollama setup (one-time):**
```bash
ollama pull llama3.2          # or mistral, qwen2.5, etc.
ollama pull nomic-embed-text  # embedding model
```

---

## Proposed Solution

### High-Level Approach

**Ingest:** `MarkdownParser` walks each `.md` file in `assets/notion/`, reads the indentation structure, and emits `QAPair` objects. Each pair is embedded locally via `ollama.embeddings(model="nomic-embed-text", prompt=question + " " + answer)` and upserted into a ChromaDB collection. A BM25 index is built in-memory over the same text.

**Query (all three modes share the same retrieval path):** `HybridRetriever.retrieve(query)` runs dense cosine search (ChromaDB) and BM25 in parallel, merges with RRF, and returns top-k `QAPair` objects. The mode determines what happens next:

- **RAG Q&A** — pairs become context for an Ollama LLM prompt; answer is returned to user.
- **Quiz Me** — Ollama generates a novel question grounded in the pairs; student answers; Ollama grades against the source pairs.
- **Evaluation** — system retrieves pairs, shows them to the user, and asks "Did this answer your question? (y/n)". Tracks hit-rate per session.

### Key Components

- **MarkdownParser** (`ingester.py`) — indentation-based parser; emits `QAPair` objects with `page_title`, `section`, `has_diagram`, `img_paths`.
- **StudyBot** (`studybot.py`) — orchestrates parse → index → retrieve; replaces `DocuBot`; keeps same `retrieve()` / `build_index()` interface for `evaluation.py` compatibility.
- **HybridRetriever** (inside `StudyBot`) — ChromaDB dense search + BM25 + RRF merge.
- **OllamaClient** (`llm_client.py`) — replaces `GeminiClient`; wraps `ollama.chat()` with three prompt strategies: `answer_from_snippets()`, `quiz_from_snippets()`, `grade_student_answer()`.
- **Evaluator** (`evaluation.py`) — extends existing hit-rate harness with human-in-the-loop (HITL) mode: shows retrieved snippets, collects y/n feedback, reports per-topic hit-rate.
- **CLI Orchestrator** (`main.py`) — three modes replacing the existing five.

### Architecture Diagram

```
assets/notion/
  Hands On Machine Learning/
    Training Models ...md
    Decision Trees ...md
    ...6 more .md files
         │
         ▼
   MarkdownParser
   (indentation parser)
         │
         ▼
   list[QAPair]
   (question, answer,
    page_title, section,
    has_diagram)
         │
    ┌────┴──────┐
    ▼           ▼
ChromaDB     BM25Index
(nomic-      (rank-bm25,
 embed-text,  in-memory)
 persistent)
    │           │
    └────┬──────┘
         ▼
  HybridRetriever
  (RRF merge, top-k)
         │
   ┌─────┼───────┐
   ▼     ▼       ▼
Mode1  Mode2   Mode3
RAG    Quiz    Eval
Q&A    Me      (HITL)
   │     │       │
   └─────┴───────┘
         │
   OllamaClient
   (local LLM via
    ollama.chat())
         │
         ▼
   Terminal Output
```

---

## Design Considerations

### 1. Local LLM: Ollama Model Choice

**Context:** Ollama model quality varies significantly. We need a model that can follow structured prompts (grounding in snippets, refusal when unsure) and grade short free-text answers accurately.

**Options:**
- **Option A: `llama3.2` (3B)**
  - Pros: Fast; low RAM (~2GB); good instruction following for small tasks
  - Cons: Weaker at multi-step reasoning and nuanced grading
- **Option B: `mistral` (7B)**
  - Pros: Strong instruction following; good at structured output; widely tested
  - Cons: More RAM (~5GB); slightly slower
- **Option C: `qwen2.5` (7B)**
  - Pros: Strong at reasoning and Q&A tasks; competitive with Mistral
  - Cons: Same RAM footprint as Mistral

**Recommendation:** Make the model **configurable via `.env`** (`OLLAMA_MODEL=llama3.2`). Default to `llama3.2` for speed; switch to `mistral` or `qwen2.5` for better quality. No code change needed to swap.

---

### 2. Embedding Model

**Context:** We need a local embedding model for ChromaDB dense search. Ollama serves embedding models via `ollama.embeddings()`.

**Options:**
- **Option A: `nomic-embed-text` (via Ollama)**
  - Pros: Strong semantic embedding; 768-dim; runs locally; same Ollama client used for LLM
  - Cons: Requires `ollama pull nomic-embed-text`
- **Option B: `sentence-transformers` (e.g., `all-MiniLM-L6-v2`)**
  - Pros: Fast; well-benchmarked; no Ollama dependency for embeddings
  - Cons: Second library (`sentence-transformers`); adds torch dependency
- **Option C: ChromaDB default embedding (uses `all-MiniLM-L6-v2` internally)**
  - Pros: Zero extra code — ChromaDB handles it automatically
  - Cons: Less control; downloads model via `chromadb` dependency chain

**Recommendation:** Option C for MVP — ChromaDB's default embedding function just works with no extra code. Upgrade to Option A (`nomic-embed-text`) in Phase 2 for better semantic quality.

---

### 3. Chunking: Indentation Parser

**Context:** All 8 pages share a consistent structure (confirmed by reading actual files). The parser is a simple state machine.

**Parsing rules:**
```
Line starts with "# "  (single #)  → skip (page title already in filename)
Line starts with "## " or "### "   → update current page_title / section
Line starts with "^- "  (0 indent) → new QAPair; question = stripped line text
Line starts with "^    " (4+ indent) → append to current answer
Line starts with "^    ![](...)"   → set has_diagram=True; extract img path
Blank line / "---"                 → skip
```

Edge cases observed in actual files:
- Some answers have multi-level nesting (8 or 12 spaces) — treat all indented content as part of the answer.
- Images appear mid-answer — capture `img_path`; append `[diagram]` marker to answer text.
- Some questions have no answer body (rare) — skip the QAPair; log the question text.

---

### 4. Human-in-the-Loop Evaluation

**Context:** Traditional evaluation measures retrieval hit-rate against a keyword-to-file ground truth (existing `evaluation.py`). For a Q&A bot, the better signal is: *did the retrieved snippets actually answer the student's question?* Only a human can judge this.

**HITL Evaluation Flow:**
1. Student enters a query (or system uses sample queries from `dataset.py`).
2. `HybridRetriever.retrieve(query, k=3)` returns top-3 Q&A pairs.
3. System prints retrieved pairs.
4. System asks: `"Did these snippets answer your question? (y/n/partial)"`.
5. Student response is logged to `eval_log.jsonl`.
6. After N queries, system prints per-topic hit-rate summary.

**What gets tracked per entry in `eval_log.jsonl`:**
```json
{
  "query": "How does dropout work?",
  "retrieved_pages": ["Training Deep Neural Networks"],
  "retrieved_questions": ["Explain Dropout"],
  "human_rating": "y",
  "timestamp": "2026-04-06T..."
}
```

This gives a real signal on retrieval quality that can drive improvements (e.g., comparing BM25-only vs hybrid).

---

## Lifecycle of Code — RAG Q&A Mode

**User asks: "What is the difference between L1 and L2 regularization?"**

1. `main.py` → Mode 1 selected.
2. `StudyBot.retrieve("What is the difference between L1 and L2 regularization?", k=5)`
   - ChromaDB cosine search: surfaces "Explain Ridge Regression, Lasso Regression, and Elastic Net" from Training Models page.
   - BM25 search: surfaces same pair via keyword "L1", "L2", "regularization".
   - RRF merge: top-3 pairs returned.
3. `OllamaClient.answer_from_snippets(query, snippets)` — constructs grounded prompt; calls `ollama.chat(model="llama3.2", ...)`.
4. Answer printed to terminal with source questions cited.

## Lifecycle of Code — Quiz Me Mode

**User selects Quiz Me, optionally says "Decision Trees"**

1. `main.py` → Mode 2 selected.
2. `StudyBot.retrieve("Decision Trees", k=8)` — returns Decision Trees Q&A pairs.
3. `OllamaClient.quiz_from_snippets(snippets)` — Ollama generates one novel question grounded in snippets (e.g., "Given a node with 54 samples, calculate Gini impurity if class distribution is [0, 49, 5].").
4. Student types their answer.
5. `OllamaClient.grade_student_answer(question, student_answer, snippets)` — Ollama grades; prints feedback.
6. Loop: "Another question? (y/n)"

## Lifecycle of Code — Evaluation Mode

**Evaluation run with sample queries**

1. `main.py` → Mode 3 selected.
2. For each sample query in `dataset.py` (or user-provided):
   - Retrieve top-3 pairs.
   - Display retrieved Q&A pairs.
   - Prompt: `"Relevant? (y / n / partial)"`.
   - Log to `eval_log.jsonl`.
3. Print summary: per-page hit-rate, overall hit-rate, comparison vs prior run if log exists.

### Error Scenarios
- **Ollama not running:** Print clear message: `"Ollama is not running. Start it with: ollama serve"`. Retrieval-only mode still works.
- **Model not pulled:** Catch `ollama.ResponseError`; print `"Run: ollama pull llama3.2"`.
- **No retrieval results:** Print `"I don't have notes on that topic yet."` — no LLM call made.
- **Malformed indent in .md file:** Log warning with file + line number; skip the malformed pair; continue parsing.

---

## Detailed Design

### Data Model

```python
@dataclass
class QAPair:
    id: str              # deterministic: f"{page_slug}_{line_number}"
    question: str        # top-level bullet text (stripped of "- ")
    answer: str          # all indented content joined as text; "[diagram]" for images
    page_title: str      # e.g. "Decision Trees"
    section: str         # "main_ideas" | "exercises"
    has_diagram: bool
    img_paths: list[str] # relative paths to local PNG files
```

---

### `ingester.py` (new)

```python
class MarkdownParser:
    def parse_file(self, filepath: str) -> list[QAPair]:
        """State machine over lines. Tracks current section and open QAPair."""
        ...

    def parse_all(self, notion_dir: str) -> list[QAPair]:
        """Walk all .md files in notion_dir; skip the top-level index file."""
        ...
```

---

### `studybot.py` (new — replaces `docubot.py`)

```python
class StudyBot:
    def __init__(self, notion_dir: str, chroma_path: str = ".chroma"):
        ...

    def load_and_index(self, force_reingest: bool = False) -> int:
        """Parse Markdown, embed with ChromaDB default embedder, build BM25.
        Skips embedding if ChromaDB collection already populated (idempotent).
        Returns number of Q&A pairs indexed."""
        ...

    def retrieve(self, query: str, k: int = 5) -> RetrievalResult:
        """Hybrid BM25 + dense with RRF merge.
        Returns RetrievalResult with pairs + raw scores for guardrail gating."""
        ...

    def full_corpus_text(self) -> str:
        """For Mode 1 naive LLM — concatenate all Q&A pairs as plain text."""
        ...
```

---

### `llm_client.py` (replace `GeminiClient` with `OllamaClient`)

```python
class OllamaClient:
    def __init__(self, model: str = "llama3.2"):
        self.model = model

    def answer_from_snippets(self, query: str, snippets: list[QAPair]) -> str:
        """RAG answer. Instructs model to cite source questions and refuse if unsure."""
        ...

    def quiz_from_snippets(self, snippets: list[QAPair]) -> str:
        """Generate one novel grounded question. Model must not invent facts."""
        ...

    def grade_student_answer(self, question: str, student_answer: str,
                              snippets: list[QAPair]) -> str:
        """Grade free-text answer as Correct / Partial / Incorrect + feedback."""
        ...
```

---

### `evaluation.py` (extend existing)

```python
# Existing: keyword-based automated hit-rate
# New: HITL session that logs to eval_log.jsonl

class HITLEvaluator:
    def run_session(self, studybot: StudyBot, queries: list[str]):
        """Interactive loop: retrieve → display → collect y/n/partial → log."""
        ...

    def print_report(self, log_path: str = "eval_log.jsonl"):
        """Aggregate hit-rate by page_title from log file."""
        ...
```

---

### Updated CLI Modes

| Mode | Name | Description |
|------|------|-------------|
| 1 | RAG Q&A | `StudyBot.retrieve()` → `OllamaClient.answer_from_snippets()` |
| 2 | Quiz Me | `StudyBot.retrieve()` → `OllamaClient.quiz_from_snippets()` → student input → `grade_student_answer()` |
| 3 | Evaluation | HITL loop: retrieve → display → human rates relevance → log + report |

---

### File Changes Summary

| File | Action | What changes |
|------|--------|-------------|
| `ingester.py` | **Create** | `MarkdownParser` — indentation-based Q&A extraction |
| `studybot.py` | **Create** | Replaces `docubot.py`; hybrid retriever |
| `llm_client.py` | **Rewrite** | `GeminiClient` → `OllamaClient`; three prompt methods |
| `logger.py` | **Create** | `StudyLogger` — JSONL session logging + metrics report |
| `guardrails.py` | **Create** | `InsufficientContextError`, `RetrievalResult`, threshold constants |
| `main.py` | **Modify** | 3 modes; import `StudyBot`, `OllamaClient`, `StudyLogger` |
| `evaluation.py` | **Modify** | Add `HITLEvaluator` + `print_metrics_report()`; keep automated harness |
| `requirements.txt` | **Rewrite** | `ollama`, `chromadb`, `rank-bm25`, `python-dotenv` |
| `.env.example` | **Rewrite** | `OLLAMA_MODEL=llama3.2` |
| `docubot.py` | **Keep** | Don't delete until `evaluation.py` is fully migrated |
| `dataset.py` | **Keep** | Sample queries still used by Evaluation mode |

---

## Guardrails

The system refuses to answer in two distinct layers. Layer 1 fires before any Ollama call — pure retrieval signal. Layer 2 fires inside the LLM prompt — for cases where retrieval returns results but they don't actually answer the question.

---

### Layer 1 — Retrieval Gate (before LLM call)

`HybridRetriever.retrieve()` returns results **with confidence metadata**. Before calling Ollama, `StudyBot` checks two conditions:

**Condition A — Empty retrieval:**
BM25 returns zero results OR ChromaDB returns nothing above the minimum distance threshold.
```python
if len(results) == 0:
    raise InsufficientContextError("no_results", query)
```

**Condition B — Low confidence:**
RRF always returns up to k results (it re-ranks, doesn't filter). We need the underlying scores to gate. `retrieve()` returns a `RetrievalResult` that carries the top ChromaDB cosine distance alongside each pair.

```python
@dataclass
class RetrievalResult:
    pairs: list[QAPair]
    top_cosine_distance: float   # ChromaDB distance; lower = more similar
    top_bm25_score: float        # BM25 score of rank-1 result; 0.0 = no overlap

# Thresholds (tunable via .env)
COSINE_DISTANCE_THRESHOLD = 0.8   # above this = too dissimilar (0.0 = identical, 1.0 = orthogonal)
BM25_SCORE_THRESHOLD = 0.0        # exactly 0 means zero keyword overlap

if result.top_bm25_score == 0.0 and result.top_cosine_distance > COSINE_DISTANCE_THRESHOLD:
    raise InsufficientContextError("low_confidence", query)
```

Both signals must be weak simultaneously — if either has a good score, we trust retrieval and proceed.

**`InsufficientContextError`** is caught in `main.py` and printed as a user-friendly refusal:
```
I don't have sufficient notes to answer that question.
Topics I have notes on: Decision Trees, Training Models, Support Vector Machines,
Ensemble Learning, Dimensionality Reduction, Unsupervised Learning,
Neural Networks (Intro), Training Deep Neural Networks
```

The logger writes a `refusal` event (see Logging section) — no Ollama call is made.

---

### Layer 2 — LLM Refusal Prompt (inside Ollama call)

Even when Layer 1 passes, the retrieved snippets may not contain enough information to answer the specific question. The system prompt for every Ollama call includes a strict grounding instruction:

```
You are an ML study assistant. Answer ONLY using the study notes provided below.
If the notes do not contain enough information to answer the question,
respond with exactly this phrase and nothing else:
"I don't have sufficient notes on that topic."
Do not use outside knowledge. Do not guess. Do not partially answer if the notes are insufficient.
```

This handles cases like:
- Query: "Explain transformer self-attention" → notes don't cover transformers → LLM refuses even if retrieval returned loosely related neural network snippets.
- Query: "What is backpropagation?" → notes cover it under Neural Networks → LLM answers normally.

**Detection:** After calling Ollama, `OllamaClient` checks if the response starts with the refusal phrase:
```python
REFUSAL_PHRASE = "I don't have sufficient notes on that topic."

response = ollama.chat(...)
if response.strip().startswith(REFUSAL_PHRASE):
    raise InsufficientContextError("llm_refusal", query)
```

`main.py` catches this the same way as Layer 1 — prints the topic list and logs a `refusal` event.

---

### Refusal Event in Logger

A fifth event type added to `StudyLogger`:

```json
{
  "session_id": "a3f8...",
  "timestamp": "2026-04-06T14:35:12Z",
  "event_type": "refusal",
  "mode": "rag_qa",
  "query": "Explain transformer self-attention",
  "refusal_layer": "llm_refusal | low_confidence | no_results",
  "top_cosine_distance": 0.91,
  "top_bm25_score": 0.0
}
```

Refusals surface in the metrics report:
```
Refusals this session: 3
  - "Explain transformer self-attention"  [llm_refusal]
  - "What is XGBoost?"                    [low_confidence]
  - "How does BERT work?"                 [no_results]
→ These topics are gaps in your notes.
```

This turns refusals from failure signals into **actionable study gaps** — the student knows exactly what to add to their Notion notes.

---

### Threshold Configuration (`.env`)

```
OLLAMA_MODEL=llama3.2
COSINE_DISTANCE_THRESHOLD=0.8
BM25_SCORE_THRESHOLD=0.0
```

Thresholds are tunable without code changes. After running the Evaluation mode for a session, the student can inspect `refusal` events in the JSONL log and tighten or loosen the threshold.

---

## Logging Component

### What Gets Logged

Every user interaction writes one JSON object to `logs/session_<YYYYMMDD_HHMMSS>.jsonl` — one line per event. A new file is created each time the CLI starts. This keeps sessions isolated and makes it easy to compare runs.

**Common fields on every event:**
```json
{
  "session_id": "a3f8...",
  "timestamp": "2026-04-06T14:32:01.123Z",
  "mode": "rag_qa | quiz | evaluation",
  "event_type": "retrieval | llm_response | quiz_grade | hitl_rating"
}
```

**`retrieval` event** — emitted after every `HybridRetriever.retrieve()` call:
```json
{
  "event_type": "retrieval",
  "query": "What is the CART algorithm?",
  "retrieved": [
    {"id": "decision_trees_39", "page_title": "Decision Trees",
     "section": "main_ideas", "question": "Explain the CART Algorithm?"},
    {"id": "training_models_95", "page_title": "Training Models",
     "section": "exercises", "question": "..."}
  ],
  "retrieval_latency_ms": 42
}
```

**`llm_response` event** — emitted after every Ollama call (RAG Q&A and Quiz generation):
```json
{
  "event_type": "llm_response",
  "query": "What is the CART algorithm?",
  "answer": "The CART algorithm is a greedy algorithm...",
  "sources_cited": ["Explain the CART Algorithm?"],
  "llm_latency_ms": 1840,
  "model": "llama3.2"
}
```

**`quiz_grade` event** — emitted after each Quiz Me round:
```json
{
  "event_type": "quiz_grade",
  "generated_question": "Given a node with 54 samples...",
  "student_answer": "Gini = 1 - (49/54)^2 - (5/54)^2",
  "grade": "correct | partial | incorrect",
  "feedback": "Correct. You correctly applied the Gini formula...",
  "source_page": "Decision Trees",
  "source_section": "main_ideas",
  "grade_latency_ms": 920
}
```

**`hitl_rating` event** — emitted after each human rating in Evaluation mode:
```json
{
  "event_type": "hitl_rating",
  "query": "How does dropout prevent overfitting?",
  "retrieved_pages": ["Training Deep Neural Networks"],
  "human_rating": "y | n | partial"
}
```

---

### `logger.py` (new file)

```python
class StudyLogger:
    def __init__(self, log_dir: str = "logs"):
        # Creates logs/ dir if needed
        # Opens session file: logs/session_20260406_143201.jsonl
        # Generates session_id = uuid4()
        ...

    def log(self, event_type: str, mode: str, **fields):
        """Write one JSON line to the session log file."""
        ...

    # Convenience methods called by each component:
    def log_retrieval(self, query, retrieved, latency_ms, mode): ...
    def log_llm_response(self, query, answer, sources, latency_ms, model, mode): ...
    def log_quiz_grade(self, question, student_answer, grade, feedback,
                       source_page, source_section, latency_ms): ...
    def log_hitl_rating(self, query, retrieved_pages, rating): ...
    def log_refusal(self, query, refusal_layer, top_cosine_distance, 
                    top_bm25_score, mode): ...
```

`StudyLogger` is instantiated once in `main.py` and passed to `StudyBot`, `OllamaClient`, and `HITLEvaluator`. Each component calls the relevant convenience method — they never write JSON directly.

---

### Metrics Report (`evaluation.py` → `print_metrics_report()`)

At the end of any session, or when running in Evaluation mode, the system can compute metrics from the current session log or from all logs in `logs/`:

| Metric | Source events | Description |
|--------|--------------|-------------|
| **HITL hit-rate** | `hitl_rating` | `y` + `partial` / total, grouped by `retrieved_pages` |
| **Quiz accuracy** | `quiz_grade` | `correct` / total grades, grouped by `source_page` |
| **Avg retrieval latency** | `retrieval` | Mean `retrieval_latency_ms` across session |
| **Avg LLM latency** | `llm_response` | Mean `llm_latency_ms` across session |
| **Topics studied** | `retrieval` | Unique `page_title` values retrieved this session |
| **Weak topics** | `quiz_grade` | Pages where `incorrect` > 50% of quiz grades |
| **Zero-result queries** | `retrieval` | Queries where `retrieved` list is empty |

**Example end-of-session report:**
```
=== Session Summary ===
Mode: Quiz Me  |  Queries: 12  |  Duration: 14m

Quiz Accuracy by Topic:
  Decision Trees            ████████░░  80%  (8/10)
  Training Models           ██████░░░░  60%  (6/10)
  Support Vector Machines   ████░░░░░░  40%  (4/10)  ← needs work

Avg retrieval latency:  38ms
Avg LLM latency:       1.6s

Weak topics flagged: Support Vector Machines
```

---

### Log Storage & Rotation

```
logs/
  session_20260406_143201.jsonl   ← one file per CLI run
  session_20260406_160045.jsonl
  session_20260407_091122.jsonl
```

- Files are append-only JSONL — never overwritten.
- No automatic rotation for MVP; each session file is typically small (~2KB for a 10-query session).
- `logs/` is added to `.gitignore` — personal study data shouldn't be committed.

---

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Ollama not installed / model not pulled | High — app won't start | Med | Check `ollama.list()` on startup; print actionable error with exact `ollama pull` command |
| Local LLM quality lower than Gemini for grading | Med — imprecise feedback | Med | Configurable model via `.env`; user can switch to larger model |
| ChromaDB default embedder quality | Med — weaker semantic retrieval | Low | Phase 2: swap to `nomic-embed-text` via Ollama for better quality |
| Images lose context | Med — diagrams explain key concepts | High | Append `[diagram: filename.png]` in answer text; student can view image manually; Phase 2: Ollama vision model |
| Cosine threshold too tight — refuses valid questions | Med — frustrating UX | Med | Default threshold set conservatively (0.8); tunable via `.env`; refusals logged so threshold can be calibrated after a real session |
| LLM ignores refusal instruction and answers anyway | Med — hallucinated answer presented as fact | Low | Post-response `startswith(REFUSAL_PHRASE)` check is deterministic; doesn't rely on model compliance alone |
| Logger write failure (disk full, permissions) | Low — silent data loss | Low | Wrap all `log()` calls in try/except; print warning but never crash the main interaction |
| Log file exposes personal study data in commits | Low | Med | Add `logs/` to `.gitignore` on project creation |

---

## Rollout Plan

### Phase 1 — Parser + Ingest + ChromaDB
- [ ] `ingester.py`: `MarkdownParser.parse_all()` → verify ~148 pairs printed
- [ ] `studybot.py`: `load_and_index()` with ChromaDB default embedder
- [ ] Smoke test: `retrieve("gini impurity")` returns Decision Trees pairs

### Phase 2 — Hybrid Retrieval
- [ ] Add BM25 index + RRF merge to `StudyBot`
- [ ] Benchmark: BM25-only vs hybrid on 10 sample queries (manual inspection)

### Phase 3 — Ollama LLM + Modes + Guardrails
- [ ] `OllamaClient` with all three prompt methods + refusal phrase detection
- [ ] `guardrails.py`: `InsufficientContextError`, `RetrievalResult`, threshold constants
- [ ] Wire Layer 1 (retrieval gate) into `StudyBot.retrieve()`
- [ ] Wire Layer 2 (LLM refusal prompt) into `OllamaClient`
- [ ] `main.py` Modes 1, 2, 3 — catch `InsufficientContextError`, print topic list
- [ ] End-to-end test: out-of-domain query ("explain BERT") triggers refusal

### Phase 4 — Logging + Evaluation
- [ ] `logger.py`: `StudyLogger` with all four convenience methods
- [ ] Wire `StudyLogger` into `StudyBot`, `OllamaClient`, and `HITLEvaluator`
- [ ] `HITLEvaluator` HITL loop writes `hitl_rating` events
- [ ] `print_metrics_report()` reads session JSONL and prints summary table

---

## Open Questions

1. **Which Ollama model will you run locally?** Affects RAM requirements and prompt quality. `llama3.2` (3B) is fast; `mistral` or `qwen2.5` (7B) is more capable for grading.
2. **Quiz topic filter?** Should Mode 2 let you say "quiz me on Decision Trees only"? The `page_title` ChromaDB metadata filter makes this trivial — just needs a CLI prompt.
3. **Evaluation queries source?** Use the existing 8 queries in `dataset.py`, or let the user type queries interactively during an eval session?

---

## References
- [Existing system overview](../system/system_overview.md)
- [Corpus location](../../assets/notion/Hands%20On%20Machine%20Learning/)
- [ChromaDB docs](https://docs.trychroma.com)
- [Ollama Python client](https://github.com/ollama/ollama-python)
- [Reciprocal Rank Fusion paper](https://dl.acm.org/doi/10.1145/1571941.1572114)
