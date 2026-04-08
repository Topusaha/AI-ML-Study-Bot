# Streamlit UI Design Document

## Metadata
- Status: Draft
- Author(s): topu
- Created: 2026-04-06
- Updated: 2026-04-06

---

## Overview

This document specifies the full Streamlit UI for the ML Study Bot. The app is a single-page multi-tab interface (`app.py`) that surfaces four modes of interaction: RAG-based Q&A, quiz generation and grading, session metrics and log inspection, and a passkey-gated human-in-the-loop (HITL) evaluation panel. All backend calls go to a locally running Ollama instance (`llama3.2`) with ChromaDB + BM25 hybrid retrieval over ~148 Q&A pairs parsed from a Notion markdown export.

The UI is driven by `st.session_state` to avoid re-initializing the `StudyBot` and related services on every Streamlit rerun. A shared sidebar shows live session counters and exposes a force-reingest control.

---

## Goals

1. Provide a clean, low-friction interface for active ML study (ask questions, get quizzed, review weak areas).
2. Surface retrieval quality signals (latency, cosine distance, BM25 scores, refusals) without burying them.
3. Allow a developer/instructor to evaluate and correct retrieval results in-session via the HITL tab.
4. Keep all data local — no external API calls, no remote logging.
5. Make the session log downloadable so study progress can be reviewed offline.

---

## Tab Structure Overview

| Tab | Purpose |
|-----|---------|
| RAG Q&A | Ask a free-form ML question; get an LLM answer grounded in retrieved Q&A pairs |
| Quiz Me | Generate a quiz question from filtered topics/sections; submit and grade an answer |
| Metrics & Logs | Inspect session stats, quiz accuracy by topic, recent queries, refusals, and weak topics |
| Human-in-the-Loop | Passkey-gated panel to rate retrieved pairs, correct answers, and inject approved pairs into ChromaDB |

---

## Shared Sidebar

Rendered unconditionally on every page load using `st.sidebar`.

**Components:**
- `st.sidebar.title("ML Study Bot")` + `st.sidebar.caption("v1.0.0")`
- `st.sidebar.markdown(f"**Model:** llama3.2")`
- `st.sidebar.markdown(f"**Session:** {st.session_state.session_id[:8]}")`
- `st.sidebar.metric("Queries this session", st.session_state.query_count)`
- `st.sidebar.metric("Refusals this session", st.session_state.refusal_count)`
- `st.sidebar.divider()`
- "Reingest Notes" section:
  - `st.sidebar.button("Reingest Notes")` — on click, show a confirmation dialog via `st.sidebar.warning("Are you sure? This re-parses all Notion files and rebuilds the index.")` plus a `st.sidebar.button("Confirm Reingest")` that calls `st.session_state.studybot.load_and_index(force_reingest=True)` and then `st.sidebar.success("Reingest complete.")`.
  - Use a `st.session_state` flag (e.g., `reingest_confirm_pending: bool`) to hold the two-step confirmation state across reruns.

---

## Tab 1: RAG Q&A

### Layout

```
st.tabs(["RAG Q&A", "Quiz Me", "Metrics & Logs", "Human-in-the-Loop"])
  └── Tab 1
        st.text_input("Ask a question about ML", key="rag_query_input")
        st.button("Submit", key="rag_submit")
        --- (on submit) ---
        st.spinner("Retrieving and generating answer...")
        st.markdown answer box  (styled with st.success / st.info)
        st.expander("Sources")
          └── per retrieved pair: page_title, section badge, question text
        --- sidebar expander ---
        st.sidebar.expander("Recent Queries (last 5)")
          └── list of query strings
```

### User Flow

1. User types a question into `st.text_input("Ask a question about ML", key="rag_query_input")`.
2. User clicks `st.button("Submit", key="rag_submit")`.
3. A `st.spinner("Retrieving and generating answer...")` wraps the backend call sequence:
   - `st.session_state.studybot.retrieve(query)` — returns top-k `QAPair` objects.
   - `st.session_state.ollama_client.generate(query, context_pairs)` — returns answer string or raises `InsufficientContextError`.
   - `st.session_state.logger.log_retrieval(...)` — logs the event to the session JSONL.
4. On success:
   - Render the LLM answer inside a styled container: `st.success(answer)` or a custom `st.markdown` block with a light background using `st.container()` + CSS via `st.markdown("<style>...</style>", unsafe_allow_html=True)`.
   - Render `st.expander("Sources (N retrieved)")` containing a `st.markdown` list where each entry shows `**{pair.page_title}** — {pair.section} — _{pair.question}_`.
   - Increment `st.session_state.query_count += 1`.
   - Prepend query string to `st.session_state.rag_history` (capped at 5 entries).
5. On `InsufficientContextError`:
   - Render `st.warning("I don't have sufficient notes on that topic.")`.
   - Render a `st.markdown` bullet list of the 8 available topics (hard-coded constant or read from `StudyBot.topics`).
   - Increment `st.session_state.refusal_count += 1`.
   - Log the refusal event via `st.session_state.logger.log_refusal(...)`.

### Session State Interactions

- Reads: `studybot`, `ollama_client`, `logger`, `session_id`, `query_count`, `refusal_count`
- Writes: `query_count` (increment), `refusal_count` (increment), `rag_history` (prepend)

### Error States

| Condition | UI Response |
|-----------|-------------|
| `InsufficientContextError` | `st.warning` box with message and topic list |
| Ollama connection error / timeout | `st.error("Could not reach Ollama. Is it running on localhost:11434?")` |
| Empty query submitted | `st.warning("Please enter a question before submitting.")` — checked before any backend call |

---

## Tab 2: Quiz Me

### Layout

```
st.tabs(...)
  └── Tab 2
        st.selectbox("Topic", [...], key="quiz_topic_filter")
        st.radio("Section", ["All", "Main Ideas", "Exercises"], key="quiz_section_filter", horizontal=True)
        st.button("Generate Question", key="quiz_generate")
        --- (after generation) ---
        st.info(question_text)          # styled callout box
        st.text_area("Your Answer", key="quiz_answer_input")
        st.button("Submit Answer", key="quiz_submit")
        --- (after grading) ---
        grade badge (st.success / st.warning / st.error)
        st.markdown(feedback_text)
        st.button("Next Question", key="quiz_next")
```

### User Flow

1. User optionally selects a topic via `st.selectbox("Topic (optional)", ["All Topics", "Decision Trees", "Training Models", "Support Vector Machines", "Ensemble Learning and Random Forests", "Dimensionality Reduction", "Unsupervised Learning Techniques", "Introduction to Artificial Neural Networks", "Training Deep Neural Networks"], key="quiz_topic_filter")`.
2. User optionally selects a section via `st.radio("Section", ["All", "Main Ideas", "Exercises"], key="quiz_section_filter", horizontal=True)`.
3. User clicks `st.button("Generate Question", key="quiz_generate")`:
   - Backend: `st.session_state.studybot.sample_quiz_question(topic_filter, section_filter)` — returns a `QAPair` (or list used as context).
   - Store sampled pair(s) in `st.session_state.current_quiz_snippets`.
   - Generate the question text via `st.session_state.ollama_client.generate_quiz_question(snippets)` and store in `st.session_state.current_quiz_question`.
   - Render the question in `st.info(st.session_state.current_quiz_question)`.
4. User types an answer into `st.text_area("Your Answer", key="quiz_answer_input")`.
5. User clicks `st.button("Submit Answer", key="quiz_submit")`:
   - Backend: `st.session_state.ollama_client.grade_answer(question, user_answer, reference_snippets)` — returns `(grade: str, feedback: str)` where grade is one of `"Correct"`, `"Partial"`, `"Incorrect"`.
   - Render grade badge:
     - `"Correct"` → `st.success("Correct")`
     - `"Partial"` → `st.warning("Partial Credit")`
     - `"Incorrect"` → `st.error("Incorrect")`
   - Render `st.markdown(feedback_text)` below the badge.
   - Log the quiz event via `st.session_state.logger.log_quiz_grade(topic, section, grade)`.
6. User clicks `st.button("Next Question", key="quiz_next")`:
   - Clear `st.session_state.current_quiz_question`, `st.session_state.current_quiz_snippets`, and `st.session_state.quiz_answer_input` (via `st.session_state` reset or `st.rerun()`).
   - Return UI to step 1 (filter selectors + Generate button visible, no question shown).

### Session State Interactions

- Reads: `studybot`, `ollama_client`, `logger`, `session_id`
- Writes: `current_quiz_snippets`, `current_quiz_question`

### Error States

| Condition | UI Response |
|-----------|-------------|
| No Q&A pairs match the selected topic/section filter | `st.warning("No questions available for that combination. Try a different filter.")` |
| Empty answer submitted | `st.warning("Please write an answer before submitting.")` |
| Ollama grading error | `st.error("Grading failed. Please try again.")` |

---

## Tab 3: Metrics & Logs

### Layout

```
st.tabs(...)
  └── Tab 3
        st.subheader("Session Summary")
          st.columns(4): session_id[:8], total_queries, total_quiz_rounds, refusal_count, session_start
        st.divider()
        st.subheader("Quiz Accuracy by Topic")
          st.bar_chart(topic_accuracy_df)
        st.divider()
        st.subheader("Recent Queries")
          st.dataframe(recent_queries_df, use_container_width=True)
        st.divider()
        st.subheader("Refusals Log")
          st.dataframe(refusals_df, use_container_width=True)
        st.divider()
        st.subheader("Weak Topics")
          st.markdown list or st.dataframe
        st.divider()
        st.download_button("Download Session Log", ...)
```

### User Flow

On tab render, read the current session JSONL file from `st.session_state.logger.current_log_path` and parse it into event dicts. Derive each sub-section from the parsed events.

**Session Summary** — displayed using `st.metric` widgets in a 5-column `st.columns` row:
- Session ID: `st.session_state.session_id[:8]`
- Total Queries: count of `retrieval` events in JSONL
- Total Quiz Rounds: count of `quiz_grade` events in JSONL
- Refusals: count of `refusal` events in JSONL
- Session Start: `st.session_state.logger.session_start` formatted as `HH:MM:SS`

**Quiz Accuracy by Topic** — aggregate `quiz_grade` events grouped by `page_title`. Compute `accuracy = correct_count / total_count * 100`. Build a `pandas.DataFrame` with columns `["topic", "accuracy"]` and render with `st.bar_chart(df.set_index("topic"))`. Show `st.caption("No quiz data yet.")` if no `quiz_grade` events exist.

**Recent Queries** — filter last 10 `retrieval` events. Build a `pandas.DataFrame` with columns `["timestamp", "query", "retrieved_pages", "latency_ms"]`. Render with `st.dataframe(df, use_container_width=True)`.

**Refusals Log** — filter all `refusal` events. Build a `pandas.DataFrame` with columns `["timestamp", "query", "layer", "cosine_distance", "bm25_score"]`. Render with `st.dataframe(df, use_container_width=True)`. Show `st.caption("No refusals this session.")` if empty.

**Weak Topics** — from the quiz accuracy dataframe, filter rows where `accuracy < 50` and `attempt_count >= 3`. Render as a `st.warning` block listing flagged topics, or `st.success("No weak topics detected yet.")` if none.

**Download Log** — `st.download_button("Download Session Log (.jsonl)", data=raw_jsonl_bytes, file_name=f"session_{session_id[:8]}.jsonl", mime="application/jsonl")`.

### Session State Interactions

- Reads: `logger`, `session_id`, `query_count`, `refusal_count`
- Writes: none (read-only tab)

### Error States

| Condition | UI Response |
|-----------|-------------|
| Log file not yet created (zero events) | All sub-sections show `st.caption("No data yet.")` gracefully |
| JSONL parse error on a line | Skip malformed line; show `st.warning("Some log entries could not be parsed.")` |

---

## Tab 4: Human-in-the-Loop

### Layout

```
st.tabs(...)
  └── Tab 4
        st.text_input("Enter HITL passkey", type="password", key="hitl_passkey_input")
        st.button("Verify", key="hitl_verify")
        st.error(...)   # shown on mismatch
        --- (if hitl_unlocked) ---
        st.subheader("Retrieve & Rate")
        st.text_input("Query", key="hitl_query_input")
        st.button("Retrieve", key="hitl_retrieve")
        --- per retrieved pair ---
        st.expander(f"Result {i+1}: {pair.page_title}")
          st.markdown(page_title, section_badge, bold question, answer)
          st.radio("Relevance", ["Relevant", "Partial", "Not Relevant"], key=f"hitl_rating_{i}")
          st.text_area("Corrected answer (optional)", key=f"hitl_correction_{i}")
        st.button("Save Ratings", key="hitl_save")
        st.divider()
        st.subheader("Pending Injections")
        st.dataframe(pending_df)
        st.multiselect / checkboxes for row selection
        st.button("Add Selected to ChromaDB", key="hitl_inject")
        st.download_button("Download CSV", ...)
```

### User Flow

**Authentication:**
1. User enters passkey in `st.text_input("Enter HITL passkey", type="password", key="hitl_passkey_input")`.
2. User clicks `st.button("Verify", key="hitl_verify")`.
3. Compute `hashlib.sha256(entered_passkey.encode()).hexdigest()` and compare against `hashlib.sha256(os.getenv("HITL_PASSKEY", "").encode()).hexdigest()`.
4. On mismatch: `st.error("Invalid passkey")`. Leave `st.session_state.hitl_unlocked = False`.
5. On match: `st.session_state.hitl_unlocked = True`. Rerun to reveal unlocked UI.

**Retrieve & Rate (unlocked only):**
1. User enters a query in `st.text_input("Query", key="hitl_query_input")`.
2. User clicks `st.button("Retrieve", key="hitl_retrieve")`.
3. Backend: `st.session_state.studybot.retrieve(query, top_k=5)` — returns 5 `QAPair` objects stored in `st.session_state.hitl_results`.
4. Render one `st.expander(f"Result {i+1}: {pair.page_title} — {pair.section}")` per pair containing:
   - `st.markdown(f"**Page:** {pair.page_title}")` + section badge via colored `st.markdown` with inline HTML.
   - `st.markdown(f"**Question:** {pair.question}")`.
   - `st.markdown(pair.answer)`.
   - `st.radio("Relevance", ["Relevant", "Partial", "Not Relevant"], key=f"hitl_rating_{i}", horizontal=True)`.
   - `st.text_area("Corrected answer (optional)", key=f"hitl_correction_{i}", height=80)`.

**Save Ratings:**
1. User clicks `st.button("Save Ratings", key="hitl_save")`.
2. Collect ratings and corrections from `st.session_state` using keys `hitl_rating_{i}` and `hitl_correction_{i}`.
3. Call `HITLEvaluator.save_ratings(query, results, ratings, corrections)` which appends rows to `data/human-in-the-loop-results.csv`. Each row includes: `session_id`, `timestamp`, `query`, `page_title`, `section`, `question`, `answer`, `corrected_answer`, `rating`, `added_to_chroma` (default `False`).
4. Show `st.toast(f"Saved {N} ratings to CSV.")`.

**Pending Injections:**
1. Read `data/human-in-the-loop-results.csv` and filter rows where `added_to_chroma == False`.
2. Render with `st.dataframe(pending_df, use_container_width=True)`.
3. Provide row selection via `st.multiselect("Select rows to inject (by index)", options=pending_df.index.tolist(), key="hitl_selected_ids")`.
4. User clicks `st.button("Add Selected to ChromaDB", key="hitl_inject")`:
   - Call `HITLEvaluator.inject_to_chroma(selected_ids)` — embeds and upserts selected pairs into the running ChromaDB collection.
   - Update CSV rows: set `added_to_chroma = True` for injected IDs.
   - Show `st.success(f"{X} pairs injected into ChromaDB for this session.")`.
   - Show `st.warning("Note: ChromaDB HITL data is cleared on server restart. CSV is permanent.")`.

**Download CSV:**
- `st.download_button("Download CSV", data=csv_bytes, file_name="human-in-the-loop-results.csv", mime="text/csv")`.

### Session State Interactions

- Reads: `studybot`, `session_id`, `hitl_unlocked`
- Writes: `hitl_unlocked` (on verify), `hitl_results` (on retrieve)
- Widget keys (ephemeral per rerun): `hitl_passkey_input`, `hitl_query_input`, `hitl_rating_{i}`, `hitl_correction_{i}`, `hitl_selected_ids`

### Error States

| Condition | UI Response |
|-----------|-------------|
| Invalid passkey | `st.error("Invalid passkey")` |
| `HITL_PASSKEY` env var not set | `st.error("HITL_PASSKEY environment variable is not configured.")` — checked before verify logic |
| Empty query on Retrieve | `st.warning("Please enter a query.")` |
| CSV file missing on Pending Injections load | `st.caption("No ratings saved yet.")` |
| ChromaDB injection failure | `st.error("Injection failed: {error message}")` |

---

## Session State Reference Table

| Key | Type | Set Where | Read Where |
|-----|------|-----------|------------|
| `studybot` | `StudyBot` | App startup (`app.py` init block) | All tabs, sidebar reingest |
| `ollama_client` | `OllamaClient` | App startup | Tab 1 (generate), Tab 2 (generate + grade) |
| `logger` | `StudyLogger` | App startup | Tab 1 (log retrieval/refusal), Tab 2 (log grade), Tab 3 (read log) |
| `session_id` | `str` | App startup (`str(uuid4())`) | Sidebar display, Tab 3 summary, Tab 4 CSV rows |
| `hitl_unlocked` | `bool` | App startup (`False`); Tab 4 verify | Tab 4 (gate unlocked UI) |
| `query_count` | `int` | App startup (`0`); Tab 1 on submit | Sidebar metric |
| `refusal_count` | `int` | App startup (`0`); Tab 1 on refusal | Sidebar metric |
| `current_quiz_snippets` | `list[QAPair]` | Tab 2 on generate | Tab 2 on grade (as reference context) |
| `current_quiz_question` | `str` | Tab 2 on generate | Tab 2 display, Tab 2 on grade |
| `rag_history` | `list[str]` | Tab 1 on submit (prepend, cap at 5) | Sidebar recent queries expander |
| `hitl_results` | `list[QAPair]` | Tab 4 on retrieve | Tab 4 rating expanders |
| `reingest_confirm_pending` | `bool` | Sidebar reingest button click | Sidebar confirm button visibility |

---

## Startup Init Sequence

Placed at the top of `app.py`, before `st.tabs(...)` is called. Runs exactly once per browser session because it is gated on `"studybot" not in st.session_state`.

```python
import os
from uuid import uuid4
import streamlit as st
from src.studybot import StudyBot
from src.ollama_client import OllamaClient
from src.logger import StudyLogger

if "studybot" not in st.session_state:
    st.session_state.studybot = StudyBot(
        notion_dir="data/assets/notion",
        chroma_path="data/.chroma"
    )
    st.session_state.studybot.load_and_index()   # idempotent — skips if already indexed
    st.session_state.ollama_client = OllamaClient(
        model=os.getenv("OLLAMA_MODEL", "llama3.2")
    )
    st.session_state.logger = StudyLogger(log_dir="data/logs")
    st.session_state.session_id = str(uuid4())
    st.session_state.hitl_unlocked = False
    st.session_state.query_count = 0
    st.session_state.refusal_count = 0
    st.session_state.rag_history = []
    st.session_state.current_quiz_snippets = []
    st.session_state.current_quiz_question = ""
    st.session_state.hitl_results = []
    st.session_state.reingest_confirm_pending = False
```

`load_and_index()` checks for an existing ChromaDB collection and skips re-embedding if the collection is already populated and `force_reingest=False` (default). This keeps cold-start time low on repeated `streamlit run` calls.

---

## Key UI Decisions

**Why `st.tabs` instead of `st.sidebar` navigation or multi-page (`pages/`) layout:**
Tabs keep all four modes in a single module, which simplifies shared session state. Multi-page apps in Streamlit re-initialize `st.session_state` across pages unless explicitly persisted; tabs do not. Since `StudyBot` construction (embedding model load + ChromaDB connect) is expensive, a single-page tab layout ensures the init block runs once and the instance is reused across all modes.

**Why `st.session_state` for `StudyBot`, not a module-level global:**
Streamlit reruns the entire script on every user interaction. A module-level global would survive within a single worker process but is not safe across multiple workers and is harder to test. `st.session_state` is Streamlit's canonical mechanism for per-session persistent objects; it is reset cleanly when the browser tab is closed.

**Why a two-step confirmation for "Reingest Notes":**
Force reingestion wipes and rebuilds the ChromaDB collection, which takes several seconds and discards any HITL-injected pairs for the current session. A single accidental click would silently destroy in-session data. The two-step `st.button` + confirm pattern uses `reingest_confirm_pending` in session state to hold the pending state across the rerun triggered by the first click.

**Why SHA-256 for HITL passkey comparison (not plain equality):**
Comparing against a hashed env var prevents the plaintext passkey from appearing in logs, tracebacks, or `st.session_state` inspector output. Both sides are hashed before comparison so the env var itself can be stored as a hash, keeping the plaintext out of `.env` files if desired.

**Why `st.expander` for Sources (Tab 1) and HITL pairs (Tab 4):**
Source citations and retrieved pair details are secondary information in both contexts. Defaulting them to collapsed keeps the primary content (LLM answer; rating controls) visually dominant without hiding the detail for users who want it.

**Why `st.bar_chart` (not `st.plotly_chart`) for quiz accuracy:**
`st.bar_chart` requires no additional dependency and renders natively. The chart data (topic vs. accuracy%) has no requirement for interactivity beyond hover; a Plotly chart would add complexity without benefit here.

**Why grade is rendered with `st.success` / `st.warning` / `st.error` instead of a custom component:**
These built-in callouts provide immediate semantic color coding (green/yellow/red) that matches the Correct/Partial/Incorrect grading scale without any custom CSS, keeping the implementation simple and consistent with the rest of the app's visual language.
