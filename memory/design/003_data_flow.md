# Data Flow Design Document

## Metadata
- Status: Draft
- Author(s): topu
- Created: 2026-04-06
- Updated: 2026-04-06

---

## Overview

This document traces every data flow in the ML Study Bot end-to-end: from raw Notion markdown files on disk
through parsing, indexing, retrieval, LLM inference, evaluation, and human-in-the-loop correction. Each
flow is described as an ordered sequence of steps with explicit input/output types at every system boundary.

The bot runs fully locally. The LLM is `llama3.2` served by Ollama. Vector storage is ChromaDB (collection
name `ml_notes`). Sparse retrieval uses BM25Okapi (rank-bm25). The UI is Streamlit. Source material is
~148 Q&A pairs parsed from Notion markdown exports at `assets/notion/`.

---

## Goals

1. Provide a single reference that lets an engineer trace any piece of data from its origin (a `.md` file or
   a user keystroke) to its final resting place (ChromaDB, a CSV row, a JSONL log entry, or the Streamlit UI).
2. Document every type boundary explicitly so that callers and callees can be refactored independently.
3. Capture all error / refusal paths in one consolidated section to avoid hunting across four separate flows.
4. Identify exactly what state persists between flows and where it lives, so restart / recovery behaviour is
   unambiguous.

---

## Flow 1: Ingest Pipeline

### Step-by-step trace

**Step 1 — Read source files from disk**

- Trigger: `StudyBot.load_and_index()` is called on application startup (or manually).
- Guard: before doing any work, check `chroma_collection.count()`. If `count > 0` and
  `force_reingest=False`, skip steps 2–5 and return immediately. (HITL cleanup described in
  "State That Persists Between Flows" runs before this check.)
- Input: file-system glob over `assets/notion/*.md` → list of `pathlib.Path` objects.

**Step 2 — Parse markdown into QAPair objects**

- Caller: `StudyBot.load_and_index()` calls `MarkdownParser.parse_all(paths)`.
- Internal state the parser maintains while scanning a single file line-by-line:
  - `current_section: str` — updated whenever a `## Heading` line is encountered.
  - `open_qa_pair: QAPair | None` — the QAPair currently being built.
- Parsing rules applied per line:
  - Line matches `^- ` (top-level bullet) → close any `open_qa_pair`, push it to results, open a new
    `QAPair` with `question = line[2:].strip()`.
  - Line matches `^  ` or `^\t` (indented continuation) → append to `open_qa_pair.answer`.
  - Line contains `![]()` markdown image syntax → set `open_qa_pair.has_diagram = True` and append
    the extracted image path to `open_qa_pair.img_paths`.
- Output: `list[QAPair]` (~148 objects across all files).

**Step 3 — Upsert into ChromaDB**

- Caller: `StudyBot.load_and_index()` iterates the list and calls
  `chroma_collection.upsert(ids=[qa.id], documents=[qa.question + " " + qa.answer], metadatas=[{...}])`.
- The `document` string is the concatenation of question and answer separated by a single space; this is
  the text that ChromaDB embeds and stores.
- Output: ChromaDB collection `ml_notes` populated with N documents.

**Step 4 — Build BM25 index in memory**

- Caller: `StudyBot.load_and_index()` constructs
  `BM25Okapi([tokenize(qa.question + " " + qa.answer) for qa in qa_pairs])`.
- The `tokenize()` helper lowercases and splits on whitespace/punctuation.
- Output: `BM25Okapi` instance stored as `StudyBot._bm25` and a parallel list
  `StudyBot._bm25_corpus: list[QAPair]` that maps BM25 corpus indices back to `QAPair` objects.

**Step 5 — Return**

- `StudyBot.load_and_index()` returns `None`; side-effects are the populated ChromaDB collection and the
  in-memory BM25 index.

### Input / Output types at each boundary

| Boundary | Input type | Output type |
|---|---|---|
| Disk → `MarkdownParser.parse_all()` | `list[pathlib.Path]` | `list[QAPair]` |
| `QAPair` → `chroma_collection.upsert()` | `QAPair.id: str`, `document: str`, `metadata: dict` | ChromaDB write (no return value used) |
| `QAPair` list → `BM25Okapi(...)` | `list[list[str]]` (tokenized corpus) | `BM25Okapi` instance |

### QAPair fields (referenced throughout all flows)

```
QAPair:
  id:           str          # stable hash of (page_title, question)
  question:     str
  answer:       str
  page_title:   str          # e.g. "Decision Trees"
  section:      str          # e.g. "Overfitting"
  has_diagram:  bool
  img_paths:    list[str]
```

### ChromaDB metadata schema per document

```
{
  "page_title":  str,   # e.g. "SVMs"
  "section":     str,
  "has_diagram": bool,
  "source":      str    # "notion" for original data; "hitl" for injected corrections
}
```

### ASCII diagram

```
assets/notion/*.md (8 files, ~148 Q&A pairs)
         |
         | list[pathlib.Path]
         v
  MarkdownParser.parse_all()
    [line-by-line scan]
    [state: current_section, open_qa_pair]
    [top-level "- " -> QAPair.question]
    [indented lines -> QAPair.answer]
    [![]() -> has_diagram=True, img_paths.append()]
         |
         | list[QAPair]  (~148)
         v
  StudyBot.load_and_index()
    |                           |
    | QAPair.id,                | list[list[str]]
    | question+" "+answer,      | (tokenized docs)
    | metadata                  v
    v                     BM25Okapi(corpus)
  chroma_collection             |
  .upsert(...)                  | BM25Okapi instance
    |                           | stored as StudyBot._bm25
    v                           v
  ChromaDB collection     in-memory BM25 index
  "ml_notes"              (StudyBot._bm25_corpus)
```

---

## Flow 2: RAG Q&A Mode

### Step-by-step trace

**Step 1 — User input**

- User types a natural-language query string in the Streamlit Q&A tab and submits.
- Input: `query: str`.

**Step 2 — Hybrid retrieval (`StudyBot.retrieve`)**

- `StudyBot.retrieve(query, k=5)` is called.

  **2a — Dense retrieval (ChromaDB)**
  - `chroma_collection.query(query_texts=[query], n_results=5)`
  - Returns: `ids: list[list[str]]`, `distances: list[list[float]]`, `documents: list[list[str]]`,
    `metadatas: list[list[dict]]`.
  - The distance metric is L2 (lower = more similar). The top result has the smallest distance, stored as
    `top_cosine_distance` (named for conceptual clarity even though the underlying metric is L2).

  **2b — Sparse retrieval (BM25)**
  - `StudyBot._bm25.get_scores(tokenize(query))` returns `np.ndarray` of float scores, one per corpus
    position. Higher = more relevant. `top_bm25_score = scores.max()`.

  **2c — Reciprocal Rank Fusion (RRF)**
  - Dense ranking: sort ChromaDB `ids` by ascending `distance`.
  - Sparse ranking: sort `StudyBot._bm25_corpus` indices by descending BM25 score.
  - RRF score per candidate: `sum(1 / (rank + k_rrf))` across both ranked lists (k_rrf = 60 typical).
  - Take top-k=5 by RRF score → `merged_ids: list[str]`.

**Step 3 — Guardrail Layer 1 (retrieval quality)**

- `if len(merged_ids) == 0` → raise `InsufficientContextError(reason="no_results", query=query)`.
- `if top_bm25_score == 0.0 AND top_cosine_distance > 0.8` → raise
  `InsufficientContextError(reason="low_confidence", query=query)`.

**Step 4 — Return `RetrievalResult`**

- Resolve `merged_ids` to `list[QAPair]` by looking up `StudyBot._bm25_corpus`.
- Return `RetrievalResult(pairs=list[QAPair], top_cosine_distance=float, top_bm25_score=float)`.

**Step 5 — Log retrieval**

- `StudyLogger.log_retrieval(query, retrieved=RetrievalResult, latency_ms=int, mode="rag_qa")`
- Appends one JSON line to the JSONL log file (path: `data/study_log.jsonl`).

**Step 6 — LLM answer generation (`OllamaClient.answer_from_snippets`)**

- Builds prompt:
  - System message: grounding instruction + literal refusal phrase constant `REFUSAL_PHRASE`
    (e.g. `"I don't have sufficient notes on that topic."`).
  - User message: formatted snippets (question + answer text from each `QAPair`) + original `query`.
- Calls `ollama.chat(model="llama3.2", messages=[system_msg, user_msg])`.
- Receives `response_text: str`.

**Step 7 — Guardrail Layer 2 (LLM refusal detection)**

- `if response_text.startswith(REFUSAL_PHRASE)` → raise
  `InsufficientContextError(reason="llm_refusal", query=query)`.
- Otherwise → `answer: str = response_text`.

**Step 8 — Log LLM response**

- `StudyLogger.log_llm_response(query, answer, sources=list[QAPair], latency_ms=int, model="llama3.2", mode="rag_qa")`
- Appends one JSON line to `data/study_log.jsonl`.

**Step 9 — Display in Streamlit**

- Render `answer` string in the chat panel.
- Render source citations below the answer: for each `QAPair` in `RetrievalResult.pairs`, show
  `page_title`, `section`, and `question`.

**Step 10 — Error path (any `InsufficientContextError`)**

- `StudyLogger.log_refusal(query, layer=reason, cosine_dist=float, bm25_score=float, mode="rag_qa")`
- Streamlit displays: `"I don't have sufficient notes on that topic."` and a list of available topics.

### Input / Output types at each boundary

| Boundary | Input type | Output type |
|---|---|---|
| Streamlit → `StudyBot.retrieve()` | `query: str`, `k: int` | `RetrievalResult` |
| `StudyBot.retrieve()` → `chroma_collection.query()` | `query_texts: list[str]`, `n_results: int` | `ids`, `distances`, `documents`, `metadatas` (ChromaDB response dicts) |
| `StudyBot.retrieve()` → `BM25Okapi.get_scores()` | `list[str]` (tokens) | `np.ndarray[float]` |
| RRF → Guardrail Layer 1 | `merged_ids: list[str]`, `top_cosine_distance: float`, `top_bm25_score: float` | `RetrievalResult` or raises `InsufficientContextError` |
| `StudyBot.retrieve()` → `OllamaClient.answer_from_snippets()` | `query: str`, `snippets: list[QAPair]` | `answer: str` or raises `InsufficientContextError` |
| `OllamaClient` → `ollama.chat()` | `model: str`, `messages: list[dict]` | `response: ollama.ChatResponse` |
| Guardrail Layer 2 → caller | `response_text: str` | `answer: str` or raises `InsufficientContextError` |

### ASCII diagram

```
User: query (str)
         |
         v
  StudyBot.retrieve(query, k=5)
    |                        |
    | query_texts=[query]    | tokenize(query) -> list[str]
    v                        v
  chroma_collection        BM25Okapi
  .query(n_results=5)      .get_scores(tokens)
    |                        |
    | ids, distances,        | np.ndarray[float]
    | documents, metadatas   | top_bm25_score=scores.max()
    v                        v
         RRF merge
         (dense rank + sparse rank -> RRF scores)
              |
              | merged_ids: list[str]
              v
         Guardrail Layer 1
         (no_results / low_confidence checks)
              |
              | RetrievalResult(pairs, top_cosine_distance, top_bm25_score)
              v
  StudyLogger.log_retrieval(...)  -->  data/study_log.jsonl
              |
              | list[QAPair] as snippets
              v
  OllamaClient.answer_from_snippets(query, snippets)
    |
    | messages=[system_msg, user_msg]
    v
  ollama.chat(model="llama3.2")
    |
    | response_text: str
    v
  Guardrail Layer 2
  (startswith REFUSAL_PHRASE?)
    |
    | answer: str
    v
  StudyLogger.log_llm_response(...)  -->  data/study_log.jsonl
    |
    v
  Streamlit: render answer + source citations

  [on InsufficientContextError at any layer]
    -> StudyLogger.log_refusal(...)  -->  data/study_log.jsonl
    -> Streamlit: display refusal message + topic list
```

---

## Flow 3: Quiz Me Mode

### Step-by-step trace

**Step 1 — User initiates quiz**

- User navigates to the Quiz Me tab in Streamlit.
- Optionally selects a topic filter from a dropdown (e.g. `"Decision Trees"`). If no filter, topic is
  `"general ML"`.
- Input: `topic: str | None`.

**Step 2 — Retrieve relevant snippets**

- `StudyBot.retrieve(query=topic or "general ML", k=8, page_title_filter=topic)`
- If `page_title_filter` is set, ChromaDB query includes a `where={"page_title": topic}` metadata filter.
- Same hybrid retrieval and Guardrail Layer 1 logic as Flow 2. Returns `RetrievalResult`.

**Step 3 — Generate quiz question (`OllamaClient.quiz_from_snippets`)**

- Input: `snippets: list[QAPair]` from `RetrievalResult.pairs`.
- Prompt instructs the model: generate ONE novel question grounded ONLY in the provided notes.
- Calls `ollama.chat(model="llama3.2", messages=[...])`.
- Output: `generated_question: str`.

**Step 4 — Display question to user**

- Streamlit renders `generated_question`.
- User types their answer into a text area.
- Input: `student_answer: str`.

**Step 5 — Grade student answer (`OllamaClient.grade_student_answer`)**

- Input: `generated_question: str`, `student_answer: str`, `snippets: list[QAPair]`.
- Prompt provides: the generated question, the student's answer, and the reference snippets.
- Calls `ollama.chat(model="llama3.2", messages=[...])`.
- Output: `GradeResult` dict:
  ```
  {
    "grade":    str,   # one of: "correct" | "partial" | "incorrect"
    "feedback": str
  }
  ```

**Step 6 — Log quiz grade**

- `StudyLogger.log_quiz_grade(generated_question, student_answer, grade, feedback, source_page, source_section, latency_ms)`
- `source_page` and `source_section` are taken from the top-ranked `QAPair` in `RetrievalResult.pairs`.
- Appends one JSON line to `data/study_log.jsonl`.

**Step 7 — Display results**

- Streamlit renders `GradeResult.grade` and `GradeResult.feedback`.

**Step 8 — Loop prompt**

- Streamlit shows a "Another question?" button.
  - Yes → go to Step 2.
  - No → return to idle state.

### Input / Output types at each boundary

| Boundary | Input type | Output type |
|---|---|---|
| Streamlit → `StudyBot.retrieve()` | `query: str`, `k: int`, `page_title_filter: str \| None` | `RetrievalResult` |
| `StudyBot.retrieve()` → ChromaDB | `query_texts: list[str]`, `n_results: int`, `where: dict \| None` | ChromaDB response dicts |
| `RetrievalResult` → `OllamaClient.quiz_from_snippets()` | `snippets: list[QAPair]` | `generated_question: str` |
| Streamlit → `OllamaClient.grade_student_answer()` | `generated_question: str`, `student_answer: str`, `snippets: list[QAPair]` | `GradeResult: dict` |
| `GradeResult` → `StudyLogger.log_quiz_grade()` | `grade: str`, `feedback: str`, plus context fields | JSONL append |

### ASCII diagram

```
User: topic filter (str | None)
         |
         v
  StudyBot.retrieve(query=topic or "general ML", k=8,
                    page_title_filter=topic)
    [same hybrid retrieval + Guardrail Layer 1 as Flow 2]
         |
         | RetrievalResult(pairs: list[QAPair], ...)
         v
  OllamaClient.quiz_from_snippets(snippets)
    |
    | messages: "generate ONE novel question grounded ONLY in notes"
    v
  ollama.chat(model="llama3.2")
    |
    | generated_question: str
    v
  Streamlit: display generated_question
         |
         | student_answer: str (user input)
         v
  OllamaClient.grade_student_answer(
      generated_question, student_answer, snippets)
    |
    | messages: question + student_answer + reference snippets
    v
  ollama.chat(model="llama3.2")
    |
    | GradeResult: {grade: str, feedback: str}
    v
  StudyLogger.log_quiz_grade(...)  -->  data/study_log.jsonl
    |
    v
  Streamlit: display grade + feedback
    |
    v
  "Another question?" button  --> loop back to retrieve()
```

---

## Flow 4: HITL Flow

### Step-by-step trace

**Step 1 — Authentication**

- User navigates to the HITL tab in Streamlit and enters a passkey.
- Guard: `hashlib.sha256(passkey.encode()).hexdigest() == hashlib.sha256(os.environ["HITL_PASSKEY"].encode()).hexdigest()`
  - Mismatch → Streamlit shows error message; HITL interface remains locked.
  - Match → HITL interface is unlocked for the session.

**Step 2 — Query input**

- User types a query or selects from a dropdown of sample queries.
- Input: `query: str`.

**Step 3 — Retrieval**

- `StudyBot.retrieve(query, k=5)` — identical hybrid retrieval + Guardrail Layer 1 as Flow 2.
- Output: `RetrievalResult(pairs: list[QAPair], ...)`.

**Step 4 — Display retrieved pairs with rating UI**

- For each `QAPair` in `RetrievalResult.pairs`, Streamlit renders:
  - `qa.question` and `qa.answer` text.
  - Three rating buttons: `[Relevant]`, `[Partial]`, `[Not Relevant]`.
  - Optional text area: `"Provide corrected/better answer"`.

**Step 5 — User rates pairs and clicks "Save Ratings"**

- Input per pair:
  ```
  {
    qa_pair_id:               str,
    question:                 str,
    answer:                   str,
    page_title:               str,
    section:                  str,
    human_rating:             str,   # "relevant" | "partial" | "not_relevant"
    human_corrected_answer:   str    # empty string if no correction provided
  }
  ```

**Step 6 — `HITLEvaluator.save_ratings(rows)`**

- Appends one row per rated pair to `data/human-in-the-loop-results.csv`.
- CSV columns:
  ```
  timestamp, session_id, query, qa_pair_id, question, answer,
  page_title, section, human_rating, human_corrected_answer, added_to_chroma
  ```
- `added_to_chroma` is initialized to `False` on append.
- Also calls `StudyLogger.log_hitl_rating(query, retrieved_pages=list[str], ratings=list[dict])`,
  which appends to `data/study_log.jsonl`.

**Step 7 — Streamlit confirms save**

- Displays: `"Saved to CSV. Use 'Add to ChromaDB' to inject into retrieval."`.
- Shows a table of rows from the CSV where `added_to_chroma == False`.

**Step 8 — User selects rows and clicks "Add to ChromaDB"**

- Input: `selected_row_ids: list[str]` (the `qa_pair_id` values of rows to inject).

**Step 9 — `HITLEvaluator.inject_to_chroma(selected_row_ids)`**

- For each selected row:
  - Resolves `document = question + " " + (human_corrected_answer if non-empty else answer)`.
  - Calls `chroma_collection.upsert(id=qa_pair_id + "_hitl", document=document, metadata={page_title, section, has_diagram=False, source="hitl"})`.
- Updates `data/human-in-the-loop-results.csv`: sets `added_to_chroma = True` for injected rows.
- Rebuilds the BM25 index: fetches the full updated corpus from ChromaDB (or re-runs ingest), reconstructs
  `StudyBot._bm25` and `StudyBot._bm25_corpus` to include the newly added HITL pairs.

**Step 10 — Confirmation**

- Streamlit displays: `"X pairs added to ChromaDB for this session."`.

**Step 11 — Restart behaviour (HITL cleanup)**

- On server restart, `StudyBot.load_and_index()` runs before any collection count check.
- First action: `chroma_collection.delete(where={"source": "hitl"})`.
- This removes all HITL-injected documents from ChromaDB; only `source="notion"` documents remain.
- Then the normal idempotent guard (`if count > 0: skip re-embed`) applies to the cleaned collection.
- `data/human-in-the-loop-results.csv` is never touched by this cleanup; it is the permanent audit trail.

### Input / Output types at each boundary

| Boundary | Input type | Output type |
|---|---|---|
| Streamlit → auth check | `passkey: str` | `bool` (unlocked/locked) |
| Streamlit → `StudyBot.retrieve()` | `query: str`, `k: int` | `RetrievalResult` |
| Rating UI → `HITLEvaluator.save_ratings()` | `rows: list[dict]` (see schema above) | CSV append + JSONL append |
| `HITLEvaluator.inject_to_chroma()` → `chroma_collection.upsert()` | `id: str`, `document: str`, `metadata: dict` | ChromaDB write |
| `inject_to_chroma()` → BM25 rebuild | full corpus (re-fetched or re-constructed) | new `BM25Okapi` instance |
| `load_and_index()` → `chroma_collection.delete()` | `where: {"source": "hitl"}` | ChromaDB deletes |

### ASCII diagram

```
User: passkey (str)
         |
         | SHA-256(passkey) == SHA-256(HITL_PASSKEY)?
         v
  [locked]  [unlocked: HITL interface shown]
                  |
                  | query: str
                  v
         StudyBot.retrieve(query, k=5)
           [same hybrid retrieval + Guardrail Layer 1]
                  |
                  | RetrievalResult(pairs: list[QAPair])
                  v
         Streamlit: display each QAPair
           + [Relevant] [Partial] [Not Relevant] buttons
           + optional corrected-answer text area
                  |
                  | user submits ratings
                  v
         HITLEvaluator.save_ratings(rows: list[dict])
           |                           |
           | append rows               | log event
           v                           v
  data/human-in-the-loop-    data/study_log.jsonl
  results.csv
  (added_to_chroma=False)
                  |
                  | user selects rows, clicks "Add to ChromaDB"
                  | selected_row_ids: list[str]
                  v
         HITLEvaluator.inject_to_chroma(selected_row_ids)
           |                              |
           | upsert(id=qa_pair_id+"_hitl",| update added_to_chroma=True
           |   document=q+" "+corrected,  | in CSV
           |   metadata={source:"hitl"})  |
           v                              v
  ChromaDB "ml_notes"            data/human-in-the-loop-
  (HITL docs present             results.csv
   until next restart)
           |
           | rebuild BM25 index
           v
  StudyBot._bm25  (updated in-memory)
  StudyBot._bm25_corpus (updated in-memory)

  --- ON SERVER RESTART ---

  StudyBot.load_and_index(force_reingest=False)
    |
    | chroma_collection.delete(where={"source": "hitl"})
    v
  ChromaDB "ml_notes"
  (HITL docs removed; only source="notion" remains)
    |
    | collection.count() > 0 -> skip re-embed
    v
  Normal startup (BM25 rebuilt from notion-only corpus)
```

---

## Error / Refusal Paths

All three active flows (Flow 2, 3, 4) share the same guardrail architecture. `InsufficientContextError` is
raised at up to three distinct layers; only Flow 2 and Flow 3 involve an LLM response, so Guardrail Layer 2
does not apply to Flow 4.

### `InsufficientContextError` fields

```
InsufficientContextError:
  reason:  str    # "no_results" | "low_confidence" | "llm_refusal"
  query:   str
```

### Guardrail Layer 1 — retrieval quality (all flows)

| Condition | `reason` | Raised in |
|---|---|---|
| `len(merged_ids) == 0` | `"no_results"` | `StudyBot.retrieve()` |
| `top_bm25_score == 0.0 AND top_cosine_distance > 0.8` | `"low_confidence"` | `StudyBot.retrieve()` |

### Guardrail Layer 2 — LLM refusal detection (Flow 2 and Flow 3 only)

| Condition | `reason` | Raised in |
|---|---|---|
| `response_text.startswith(REFUSAL_PHRASE)` | `"llm_refusal"` | `OllamaClient.answer_from_snippets()` |

### Catch and handling (all flows)

```
except InsufficientContextError as e:
    StudyLogger.log_refusal(
        query=e.query,
        layer=e.reason,
        cosine_dist=top_cosine_distance,  # 1.0 sentinel if no results
        bm25_score=top_bm25_score,        # 0.0 sentinel if no results
        mode="rag_qa" | "quiz" | "hitl"
    )
    # append to data/study_log.jsonl

    # Streamlit display:
    st.warning("I don't have sufficient notes on that topic.")
    st.write("Available topics: Decision Trees, Training Models, SVMs, ...")
```

### `StudyLogger.log_refusal` JSONL schema

```json
{
  "event":        "refusal",
  "timestamp":    "ISO-8601",
  "mode":         "rag_qa | quiz | hitl",
  "query":        "...",
  "layer":        "no_results | low_confidence | llm_refusal",
  "cosine_dist":  0.92,
  "bm25_score":   0.0
}
```

---

## State That Persists Between Flows

### ChromaDB collection `ml_notes` — `data/chroma/` (on disk)

- Written by: `StudyBot.load_and_index()` (Flow 1), `HITLEvaluator.inject_to_chroma()` (Flow 4).
- Read by: `StudyBot.retrieve()` (Flows 2, 3, 4).
- Deleted from by: `StudyBot.load_and_index()` at startup (`delete(where={"source": "hitl"})`).
- Documents with `metadata.source == "notion"` persist indefinitely across restarts.
- Documents with `metadata.source == "hitl"` are ephemeral: present only until the next server restart.

### In-memory BM25 index — `StudyBot._bm25`, `StudyBot._bm25_corpus`

- Written by: `StudyBot.load_and_index()` (Flow 1), `HITLEvaluator.inject_to_chroma()` (Flow 4 post-inject rebuild).
- Read by: `StudyBot.retrieve()` (Flows 2, 3, 4).
- Does NOT persist across restarts. Rebuilt from ChromaDB corpus on every startup (after HITL cleanup).
- After a HITL inject within a session, the BM25 index temporarily includes HITL documents until restart.

### `data/human-in-the-loop-results.csv` — CSV on disk

- Written by: `HITLEvaluator.save_ratings()` (append rows), `HITLEvaluator.inject_to_chroma()` (update `added_to_chroma` column).
- Never deleted or truncated by the application.
- Serves as the permanent audit trail for all human ratings and corrections.
- Not read by any retrieval path; read only by the HITL tab UI to show uninjected rows.

### `data/study_log.jsonl` — append-only JSONL on disk

- Written by: `StudyLogger.log_retrieval()`, `StudyLogger.log_llm_response()`, `StudyLogger.log_refusal()`,
  `StudyLogger.log_quiz_grade()`, `StudyLogger.log_hitl_rating()` — across all flows.
- Never read by the application at runtime (read by external analysis tools only).
- Columns / event types present: `retrieval`, `llm_response`, `refusal`, `quiz_grade`, `hitl_rating`.

### Source files — `assets/notion/*.md`

- Read by: `MarkdownParser.parse_all()` (Flow 1, on first startup or forced reingest).
- Never written by the application.
- Changes take effect only on next `load_and_index(force_reingest=True)`.

### `.env` — `HITL_PASSKEY` and other secrets

- Read by: HITL authentication check (Flow 4).
- Never written by the application.
- Not committed to version control (see `.env.example` for template).
