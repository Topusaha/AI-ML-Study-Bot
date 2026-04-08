# Repository Structure Design Document

## Metadata
- Status: Draft
- Author(s): topu
- Created: 2026-04-06
- Updated: 2026-04-06

---

## Overview

This document defines the canonical file layout for the ML Study Bot project as it transitions from a flat collection of Python scripts at the repo root into a structured, maintainable codebase. The redesign draws a hard boundary between pure ML logic, stateful orchestration, UI rendering, persisted data, and tests. Every file has exactly one home, determined by a small set of placement rules described below.

---

## Goals

- Eliminate ambiguity about where new code belongs.
- Enforce a strict one-way import chain so that ML components remain portable and testable in isolation.
- Keep all persisted data (source-of-truth content, generated artifacts, logs) under a single `data/` root so `.gitignore` rules are predictable.
- Provide a clear migration path from the existing flat layout with no regressions in functionality.
- Avoid over-engineering: no unnecessary packages, no `__init__.py` files unless a genuine namespace collision requires one.

---

## Proposed Structure

```
ml-study-bot/
├── backend/                    # Orchestration, session management, business logic
│   ├── studybot.py             # StudyBot orchestrator (load_and_index, retrieve)
│   ├── evaluation.py           # HITLEvaluator + AutoEvaluator + print_metrics_report()
│   └── logger.py               # StudyLogger (JSONL session logging)
│
├── ml/                         # Core ML components (stateless, pure logic)
│   ├── parser.py               # MarkdownParser → list[QAPair]
│   ├── retriever.py            # HybridRetriever (ChromaDB + BM25 + RRF)
│   ├── llm_client.py           # OllamaClient (answer, quiz, grade prompt methods)
│   ├── guardrails.py           # InsufficientContextError, RetrievalResult, thresholds
│   └── models.py               # QAPair dataclass (shared data model)
│
├── frontend/                   # Streamlit UI
│   ├── app.py                  # Entry point: st.set_page_config, tab routing, shared state
│   └── tabs/
│       ├── qa_tab.py           # RAG Q&A tab
│       ├── quiz_tab.py         # Quiz Me tab
│       ├── metrics_tab.py      # Metrics & Logs tab
│       └── hitl_tab.py         # Human-in-the-Loop tab
│
├── data/                       # All data assets (mostly gitignored)
│   ├── assets/
│   │   └── notion/             # Notion markdown export (committed source of truth)
│   │       ├── Hands On Machine Learning b470...md
│   │       └── Hands On Machine Learning/
│   │           ├── Decision Trees ...md
│   │           ├── Training Models ...md
│   │           └── ... (7 more .md files)
│   ├── logs/                   # Session JSONL files (gitignored)
│   ├── .chroma/                # ChromaDB persistence (gitignored)
│   └── human-in-the-loop-results.csv   # HITL ratings (gitignored from commits, kept locally)
│
├── tests/                      # Unit + integration tests
│   ├── test_parser.py          # MarkdownParser: verify ~148 pairs, section tags, diagram detection
│   ├── test_retriever.py       # HybridRetriever: smoke queries, RRF ordering
│   ├── test_guardrails.py      # InsufficientContextError raised correctly at each layer
│   └── test_hitl.py            # CSV write/read, passkey gate, inject_to_chroma + cleanup
│
├── main.py                     # CLI entry point (kept at root for simplicity)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Folder Rationale

### `ml/` — Stateless Pure Logic

`ml/` contains every component that takes inputs, applies ML or NLP logic, and returns outputs without touching the filesystem, a database, or session state. Code here must be side-effect-free: no logging calls, no file reads, no Streamlit imports.

The key benefit is testability. Because `ml/` components have no external dependencies beyond their function arguments, unit tests can instantiate them directly without mocking a session, a UI widget, or a file path. The `HybridRetriever`, `MarkdownParser`, `OllamaClient`, and data models all satisfy this criterion.

The `ml/` vs `backend/` split is the most important architectural boundary in the repo. A component belongs in `ml/` if and only if you could drop it into a completely different project (a Discord bot, a REST API, a Jupyter notebook) without any modification.

### `backend/` — Stateful Orchestration

`backend/` owns anything that coordinates multiple `ml/` components, manages application-level state (a loaded index, an active session), or writes to persistent storage (logs, CSV exports). It is the only layer that knows about file paths for data assets.

`StudyBot` is the canonical example: it calls `MarkdownParser` to load content, passes the results to `HybridRetriever` to build the index, and then exposes `retrieve()` and `answer()` methods that the frontend and CLI both call. It holds the index in memory across requests — that is state, so it belongs in `backend/`.

`StudyLogger` writes JSONL to `data/logs/`. File I/O is a side effect; it belongs in `backend/`, not `ml/`.

`HITLEvaluator` reads and writes `data/human-in-the-loop-results.csv` and coordinates ratings across sessions. Same reasoning.

### `frontend/` — Streamlit UI

`frontend/` contains every file that imports `streamlit`. Nothing outside `frontend/` should ever import `streamlit`. This isolation means the application logic can be tested and invoked from the CLI (`main.py`) without Streamlit being installed or running.

The `tabs/` sub-package splits the UI into one file per tab. Each tab file receives shared state (the `StudyBot` instance, session variables) as arguments from `app.py` rather than importing them as module-level globals. This makes the rendering logic easier to read and avoids Streamlit re-run side effects caused by top-level import execution.

`app.py` is the sole entry point: it calls `st.set_page_config`, initialises shared session state, and delegates to each tab module.

### `data/` — All Persisted Data

All data assets — whether committed source-of-truth content or generated runtime artifacts — live under `data/`. This single-root convention makes `.gitignore` rules trivial: one block of three lines covers everything that should not be committed. It also makes backup and volume-mount instructions in a future Docker setup straightforward.

`data/assets/notion/` is the only sub-directory that is committed to version control. Everything else (`logs/`, `.chroma/`, `human-in-the-loop-results.csv`) is local-only.

The old `assets/notion/` at the repo root is moved here during migration. The old `docs/` directory (AUTH.md, API_REFERENCE.md, DATABASE.md, SETUP.md) is deleted because it describes a generic web-app template, not this project.

### `tests/` — Unit and Integration Tests

All test files live in a flat `tests/` directory. Test files import from `ml/` and `backend/` directly; they never import from `frontend/`. Each test file maps one-to-one to a source module to make discovery predictable.

---

## Import Rules & Dependency Direction

Dependencies flow strictly downward. No layer may import from a layer above it.

```
┌─────────────────────────────────────────────┐
│                  main.py                    │  CLI entry point
│              (imports backend/ only)        │
└────────────────────┬────────────────────────┘
                     │
┌────────────────────▼────────────────────────┐
│               frontend/                     │  Streamlit UI
│           (imports backend/ only)           │
└────────────────────┬────────────────────────┘
                     │
┌────────────────────▼────────────────────────┐
│               backend/                      │  Orchestration + session
│             (imports ml/ only)              │
└────────────────────┬────────────────────────┘
                     │
┌────────────────────▼────────────────────────┐
│                  ml/                        │  Pure logic, no side effects
│      (imports only stdlib + third-party)    │
└─────────────────────────────────────────────┘
```

**Prohibited import patterns:**

| Importer | Must NOT import from |
|---|---|
| `ml/` | `backend/`, `frontend/`, `main.py` |
| `backend/` | `frontend/`, `main.py` |
| `frontend/` | `ml/` (must go through `backend/`) |
| `tests/` | `frontend/` |

The rule that `frontend/` must not import directly from `ml/` ensures that all business logic (guardrails, context thresholds, RRF fusion) passes through `backend/StudyBot`. This prevents the UI layer from bypassing application-level logic and makes it possible to swap out an `ml/` component without touching any tab file.

---

## Migration Plan

| Old file / directory | New location | Action |
|---|---|---|
| `main.py` | `main.py` (root) | Keep in place; update imports to use `backend.studybot`, `backend.evaluation` |
| `docubot.py` | — | Delete after `ml/parser.py` and `ml/retriever.py` are verified by tests |
| `llm_client.py` | `ml/llm_client.py` | Rewrite: replace `GeminiClient` with `OllamaClient`; keep same public interface (`.answer()`, `.quiz()`, `.grade()`) |
| `evaluation.py` | `backend/evaluation.py` | Migrate; add `HITLEvaluator` (CSV read/write, passkey gate, `inject_to_chroma`) alongside existing `AutoEvaluator` |
| `dataset.py` | `backend/dataset.py` | Move unchanged; 8 sample queries and fallback corpus still used by `AutoEvaluator` |
| `assets/notion/` | `data/assets/notion/` | Move directory; update all path references in `backend/studybot.py` |
| `docs/` | — | Delete entire directory (AUTH.md, API_REFERENCE.md, DATABASE.md, SETUP.md are not relevant to this project) |
| `model_card.md` | `memory/` or repo root | Keep at root or move to `memory/`; no functional impact |
| `memory/` | `memory/` (root) | Keep in place; design docs are not part of the application |
| `requirements.txt` | `requirements.txt` (root) | Keep in place; add `streamlit`, `chromadb`, `rank-bm25`, `ollama` as needed |
| `.env.example` | `.env.example` (root) | Keep in place; update keys to reflect OllamaClient (remove Gemini API key, add `OLLAMA_BASE_URL` if needed) |

**New files to create (no existing equivalent):**

| New file | Purpose |
|---|---|
| `backend/studybot.py` | StudyBot orchestrator — replaces the coordination logic currently spread across `main.py` and `docubot.py` |
| `backend/logger.py` | StudyLogger — JSONL session logging, currently absent |
| `ml/parser.py` | MarkdownParser — extracts QAPair list from Notion markdown; extracted from `docubot.py` |
| `ml/retriever.py` | HybridRetriever — ChromaDB + BM25 + RRF; replaces BM25-only logic in `docubot.py` |
| `ml/guardrails.py` | InsufficientContextError, RetrievalResult, score thresholds |
| `ml/models.py` | QAPair dataclass |
| `frontend/app.py` | Streamlit entry point |
| `frontend/tabs/qa_tab.py` | RAG Q&A tab |
| `frontend/tabs/quiz_tab.py` | Quiz Me tab |
| `frontend/tabs/metrics_tab.py` | Metrics & Logs tab |
| `frontend/tabs/hitl_tab.py` | Human-in-the-Loop tab |
| `tests/test_parser.py` | MarkdownParser unit tests |
| `tests/test_retriever.py` | HybridRetriever smoke tests |
| `tests/test_guardrails.py` | Guardrail threshold tests |
| `tests/test_hitl.py` | HITL CSV and ChromaDB injection tests |

---

## .gitignore Updates

Add the following block to `.gitignore`:

```gitignore
# Generated data artifacts
data/logs/
data/.chroma/
data/human-in-the-loop-results.csv
```

The `data/assets/notion/` directory is intentionally excluded from this block and remains committed as the source-of-truth corpus. All other subdirectories under `data/` are either runtime-generated or contain user-specific ratings that should not be shared.

If a future `data/` subdirectory is intended to be committed (e.g., a curated evaluation dataset), it must be explicitly documented here and added to an allowlist with `!data/path/to/committed/dir` in `.gitignore`.

---

## File Responsibilities Quick Reference

| File | One-line description |
|---|---|
| `main.py` | CLI entry point; parses mode flags and dispatches to `backend.studybot` |
| `backend/studybot.py` | Loads and indexes the Notion corpus; exposes `retrieve()` and `answer()` to callers |
| `backend/evaluation.py` | `AutoEvaluator` (keyword hit-rate) + `HITLEvaluator` (CSV ratings + ChromaDB injection) + `print_metrics_report()` |
| `backend/logger.py` | `StudyLogger`: appends structured session events to a per-session JSONL file under `data/logs/` |
| `ml/parser.py` | `MarkdownParser`: reads Notion `.md` files and returns a `list[QAPair]` |
| `ml/retriever.py` | `HybridRetriever`: combines ChromaDB dense retrieval and BM25 sparse retrieval via Reciprocal Rank Fusion |
| `ml/llm_client.py` | `OllamaClient`: wraps the Ollama HTTP API; exposes `.answer()`, `.quiz()`, `.grade()` prompt methods |
| `ml/guardrails.py` | Defines `InsufficientContextError`, `RetrievalResult`, and numeric thresholds for context sufficiency |
| `ml/models.py` | `QAPair` dataclass: the shared data model passed between parser, retriever, and evaluator |
| `frontend/app.py` | Streamlit entry point: `st.set_page_config`, shared session state initialisation, tab routing |
| `frontend/tabs/qa_tab.py` | Renders the RAG Q&A tab; calls `StudyBot.answer()` |
| `frontend/tabs/quiz_tab.py` | Renders the Quiz Me tab; calls `StudyBot` quiz methods and displays feedback |
| `frontend/tabs/metrics_tab.py` | Renders the Metrics & Logs tab; reads JSONL logs and displays evaluation scores |
| `frontend/tabs/hitl_tab.py` | Renders the Human-in-the-Loop tab; collects ratings and calls `HITLEvaluator.inject_to_chroma()` |
| `data/assets/notion/` | Committed Notion markdown export; the authoritative source corpus for the study bot |
| `data/logs/` | Runtime JSONL session logs written by `StudyLogger` (gitignored) |
| `data/.chroma/` | ChromaDB persistence directory (gitignored) |
| `data/human-in-the-loop-results.csv` | Accumulated HITL ratings from the frontend (gitignored, kept locally) |
| `tests/test_parser.py` | Verifies `MarkdownParser` produces ~148 `QAPair` objects with correct section tags and diagram flags |
| `tests/test_retriever.py` | Smoke-tests `HybridRetriever` with known queries; asserts RRF ordering is stable |
| `tests/test_guardrails.py` | Asserts `InsufficientContextError` is raised at the correct score thresholds in each layer |
| `tests/test_hitl.py` | Tests CSV write/read round-trip, passkey gate enforcement, and `inject_to_chroma` + cleanup |
| `requirements.txt` | Python dependency manifest |
| `.env.example` | Template for local environment variables (Ollama base URL, etc.) |
| `.gitignore` | Excludes `data/logs/`, `data/.chroma/`, `data/human-in-the-loop-results.csv`, and standard Python artifacts |
| `README.md` | Project overview, setup instructions, and usage guide |

---

## `__init__.py` Strategy

No `__init__.py` files are created in `ml/`, `backend/`, `frontend/`, or `tests/`.

All imports use direct module paths:

```python
# In backend/studybot.py
from ml.parser import MarkdownParser
from ml.retriever import HybridRetriever
from ml.llm_client import OllamaClient

# In frontend/tabs/qa_tab.py
from backend.studybot import StudyBot

# In main.py
from backend.studybot import StudyBot
from backend.evaluation import print_metrics_report
```

This works as long as the repo root is on `sys.path`, which is the default behavior when running `python main.py` from the root or when `pytest` is invoked from the root. If a future packaging step (e.g., a `pyproject.toml` with `src/` layout) is introduced, this section should be revisited and `__init__.py` files added at that point.

The one exception to watch for: if `frontend/tabs/` is treated as a Python package (i.e., if `app.py` does `from tabs.qa_tab import render`), a `frontend/tabs/__init__.py` may be needed depending on how Python resolves the import. The simplest fix is to import with the full path `from frontend.tabs.qa_tab import render` from within `app.py`, which requires no `__init__.py`.
