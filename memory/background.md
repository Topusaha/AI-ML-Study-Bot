# Background: RAG, Evaluation, and Logging Concepts

This document explains the conceptual foundations behind every major design decision in the ML Study Bot. It is a reference for understanding *why* the system is built the way it is, not *how* to implement it (see the design docs for that).

---

## Part 1: Retrieval-Augmented Generation (RAG)

### What RAG Is

A standard LLM answers questions from its training data — knowledge baked into its weights at training time. It cannot answer questions about your personal notes, and it hallucinates when asked about things it doesn't know well.

**Retrieval-Augmented Generation (RAG)** fixes this by giving the LLM access to a document corpus at *inference time*. Instead of asking "what do you know about gini impurity?", the system:

1. Searches the corpus for relevant passages
2. Injects those passages into the LLM prompt as context
3. Instructs the LLM to answer *only* from that context

The LLM becomes a reasoning layer over retrieved evidence, not a knowledge store. This separates two concerns: *what information is available* (retrieval) and *how to synthesize an answer* (generation).

### Why RAG Is Right for This System

The corpus here is your personal Notion ML notes — 148 Q&A pairs not present in any LLM's training data. A pure LLM could answer general ML questions, but it cannot answer questions the way *you* understand them, using *your* examples and *your* phrasing. RAG anchors answers to your specific notes.

The labeled structure (toggle = question/answer) makes this even stronger: we know exactly what the "right" answer looks like for each question. This is a labeled retrieval corpus, which is rare and valuable.

---

### Chunking Strategy

**What chunking is:** Before retrieval, documents are split into smaller units called *chunks*. The chunk is the unit of retrieval — when a query comes in, the system finds the most relevant chunks, not the most relevant documents.

**Why chunk size matters:**
- Too large: a chunk returns an entire page on "Gradient Descent" when the student asked about one specific step. The LLM gets flooded with irrelevant context.
- Too small: a chunk is one sentence, missing the surrounding context needed to understand it.

**Our chunking decision — one chunk = one Q&A pair:**
The toggle structure gives us natural, semantically complete chunks. Each toggle is self-contained: the question is a complete thought, the answer explains it fully. This is a better boundary than anything we could compute algorithmically.

Contrast with the existing DocuBot's chunking: it splits on `## ` Markdown headings, producing chunks of ~20 Q&A pairs each. Retrieval returns an entire section (e.g., all of "Regularization") when the student asked about one concept within it. That's too coarse.

**Topic injection:** Because each chunk is isolated from its surrounding context, we add the parent heading (`page_title`, `section`) as metadata and prepend `[Page: X | Section: Y]` to each snippet in the LLM prompt. The LLM gets section context without the chunk size being inflated.

---

### Vector Embeddings and Dense Retrieval

**What an embedding is:** A function that maps text to a point in high-dimensional space (e.g., 768 dimensions) such that semantically similar texts are geometrically close. "What causes overfitting?" and "high variance in a model" land near each other even though they share no words.

**Cosine similarity / cosine distance:**
- Cosine similarity = 1 means vectors point in the same direction (identical meaning).
- Cosine distance = 1 - cosine_similarity. ChromaDB returns distance, so lower = more relevant.
- Our guardrail threshold `COSINE_DISTANCE_THRESHOLD = 0.8` says: if the closest chunk is still 0.8 away (nearly orthogonal), we don't trust the retrieval result.

**Why dense retrieval is necessary for ML concept questions:**
BM25 (keyword matching) fails on paraphrase. "What is the bias-variance tradeoff?" shares no words with an answer that says "increasing model complexity increases variance and decreases bias." Dense embeddings capture the conceptual overlap that keyword search misses.

**ChromaDB:** A local persistent vector database. It stores embeddings and metadata for each chunk, and runs approximate nearest-neighbor search at query time. Persisted to `data/.chroma/` so embeddings are not recomputed on every run.

**Embedding input decision — `question + " " + answer`:**
We embed the full Q&A pair, not just the question or just the answer. This maximizes retrieval surface area: queries that paraphrase the question and queries that describe the answer concept both find the pair.

---

### BM25 and Sparse Retrieval

**What BM25 is:** BM25 (Best Match 25) is a probabilistic keyword ranking function. It scores documents by:
- Term frequency (TF): how often the query term appears in the document
- Inverse document frequency (IDF): how rare the term is across all documents (rare terms are more informative)
- Document length normalization: longer documents don't get unfair advantage

BM25 is exact — it scores zero for any document that contains none of the query terms. This is a weakness for semantic queries, but a strength for exact-match queries like "CART algorithm" or "L1 norm".

**BM25Okapi** (used via `rank-bm25`): The standard modern variant of BM25, with two tuning parameters (k1 and b) controlling term frequency saturation and length normalization.

**Why keep BM25 alongside dense retrieval:**
Students sometimes type exact ML terms: "elastic net", "CART", "softmax". Dense retrieval can miss these if the embedding space doesn't strongly differentiate jargon. BM25 catches them reliably. The combination covers both semantic and lexical overlap.

---

### Hybrid Retrieval and Reciprocal Rank Fusion (RRF)

**The problem with running two retrievers:** Dense search returns a ranked list. BM25 returns a different ranked list. How do you combine them into one list without needing to know absolute scores (which are on different scales)?

**Reciprocal Rank Fusion (RRF):**
For each document, RRF computes a combined score based on its *rank* in each list, not its raw score:

```
RRF_score(doc) = Σ  1 / (k + rank_in_list_i)
```

Where `k` is a constant (typically 60) that dampens the influence of very high-ranked documents. A document ranked #1 in both lists gets `1/(60+1) + 1/(60+1) ≈ 0.033`. A document ranked #1 in one and #20 in the other gets `1/61 + 1/80 ≈ 0.029`.

**Why RRF works:**
- No need to normalize scores across retrievers (they're on incompatible scales)
- No learned weights to tune — works well out of the box
- Robust to one retriever having a bad run on a particular query
- Simple: ~5 lines of Python

**Design decision — both signals must be weak to refuse:**
If BM25 score > 0 (even one keyword matched) OR dense distance < threshold (semantic match found), we trust retrieval and call the LLM. Only when *both* signals are weak do we refuse at the retrieval layer. This reduces false refusals.

---

### Guardrails and Refusal

**The hallucination problem:** Without constraints, an LLM will synthesize a plausible-sounding answer even when its context doesn't support one. For a study tool, a confident wrong answer is worse than no answer — it teaches incorrect information.

**Two-layer guardrail design:**

**Layer 1 — Retrieval gate (before LLM call):**
Fires on objective signal from the retrieval system: empty results, or both BM25 and cosine scores below threshold. This is cheap (~40ms) and deterministic. No LLM call is made.

**Layer 2 — LLM refusal prompt (inside Ollama call):**
Even when retrieval finds something, the snippets may not answer the specific question. The system prompt instructs the LLM to respond with an exact phrase (`"I don't have sufficient notes on that topic."`) if the context is insufficient. After the call, the response is checked for this phrase deterministically — we don't rely on the model's judgement about whether it followed the instruction.

**Why the exact-phrase check matters:**
If we just asked the model "are you sure?", it might say "yes" and still be wrong. By looking for a specific, unlikely-in-normal-text phrase with `startswith()`, we get a binary signal regardless of model quality.

**Refusals as study gap diagnostics:**
Every refusal is logged with the query text and refusal layer. The metrics report surfaces these as gaps in the notes — topics the student needs to add to their Notion pages.

---

## Part 2: Evaluation

### What Evaluation Means for a RAG System

Evaluation for a RAG system has two distinct concerns:

1. **Retrieval quality:** Did the retriever return the right chunks for the query?
2. **Answer quality:** Did the LLM generate a correct, grounded answer from those chunks?

Most evaluation work focuses on retrieval, because it's easier to measure (ground truth exists) and because retrieval quality is the primary bottleneck — if the right chunks aren't retrieved, the best LLM in the world can't generate a good answer.

---

### Automated Evaluation: Hit-Rate

**Hit-rate (retrieval precision@k):** For a given query, is the expected source document present in the top-k retrieved results? If yes, that's a hit.

```
hit_rate = hits / total_queries
```

**Ground truth:** For the automated evaluator (existing `evaluation.py` pattern), ground truth is a mapping from sample queries to expected source pages:
```python
{"What is gini impurity?": "Decision Trees",
 "Explain dropout": "Training Deep Neural Networks", ...}
```

This is a *weak* form of ground truth — it checks which page was retrieved, not which specific Q&A pair. But it's useful as a regression test to catch retrieval regressions when changing chunking or retrieval strategy.

**Limitation:** Automated hit-rate doesn't measure answer quality. A retriever can surface the right page but the wrong chunk, and still pass the automated test.

---

### Human-in-the-Loop (HITL) Evaluation

**Why HITL is necessary:**
Automated metrics measure retrieval against pre-written ground truth. But:
- Ground truth is hard to write completely (you can't enumerate all valid queries)
- Answer quality requires human judgement
- The real question is: "did this actually help the student learn?"

HITL evaluation captures the ground truth signal that automated metrics cannot.

**Our HITL flow:**
1. Student retrieves snippets for a query
2. Student rates each retrieved pair: Relevant / Partial / Not Relevant
3. Optionally provides a corrected/better answer
4. Ratings are saved to `human-in-the-loop-results.csv`
5. Selected high-quality pairs can be injected into ChromaDB to improve future retrieval

**Three-point rating scale (Relevant / Partial / Not Relevant):**
- Binary (yes/no) loses the signal from "sort of relevant" — which is the most informative case for diagnosing retrieval problems
- Five-point scales add cognitive load with little additional signal for this use case
- Three-point scale is the standard in information retrieval evaluation (TREC guidelines)

**HITL as corpus expansion:**
When a student provides a corrected answer, that's a high-quality labeled example that didn't exist in the original notes. Injecting it into ChromaDB improves retrieval for future similar queries. This is a lightweight form of *active learning* — human feedback improves the system over time.

**Why HITL data doesn't persist across server restarts:**
HITL injections are experimental — they haven't been validated at scale. Keeping the base corpus clean (only verified Notion notes) and the HITL data ephemeral in ChromaDB means a restart returns to a known good state. The CSV is the permanent audit trail; ChromaDB is the live operational index.

**Passkey gate:**
HITL actions modify the retrieval corpus, which affects every subsequent query. This is a privileged operation. A passkey (SHA-256 hashed, never logged as plaintext) ensures only the note author — the person who understands the ground truth — can perform HITL ratings.

---

### Quiz Accuracy as an Evaluation Signal

Quiz mode produces a different kind of evaluation signal: not "did the retriever find the right thing?" but "did the student learn the material?"

**Quiz grades as proxy for content quality:**
If a student consistently answers questions on "Decision Trees" correctly but fails on "Support Vector Machines", two interpretations are possible:
1. The student hasn't studied SVMs enough
2. The SVM notes are unclear or incomplete

The metrics report surfaces per-topic accuracy with a "weak topic" flag (< 50% accuracy with ≥ 3 attempts). This makes both possibilities visible — the student can decide whether to study more or improve their notes.

**Grade reliability:**
LLM grading of free-text answers has known failure modes (overconfident "correct" on shallow answers, "partial" for borderline cases). The design mitigates this by grounding the grading prompt strictly in the retrieved snippets: the LLM can only cite evidence from the provided context, not from general knowledge. This reduces the chance of grading an answer as correct for the wrong reasons.

---

## Part 3: Logging

### Why Structured Logging

**Structured logging** means writing log events as machine-readable records (JSON) rather than human-readable strings. Compare:

```
# Unstructured
2026-04-06 14:32 - Retrieved 3 results for query "gini impurity" in 42ms

# Structured
{"timestamp": "2026-04-06T14:32:01Z", "event_type": "retrieval",
 "query": "gini impurity", "retrieved_count": 3, "latency_ms": 42,
 "retrieved_pages": ["Decision Trees"]}
```

Both record the same event. But the structured version can be aggregated, filtered, and charted programmatically. The unstructured version requires regex parsing to extract any metric.

**JSONL (JSON Lines):** One JSON object per line. Each line is a valid, self-contained JSON record. This format is:
- Appendable — no need to parse the whole file to add a record
- Streamable — can process line-by-line without loading the whole file into memory
- Recoverable — a truncated file is still valid up to the last complete line
- Compatible with standard tools (`jq`, pandas `read_json(lines=True)`)

---

### Session-Based Log Files

Each CLI/Streamlit run creates a new log file: `data/logs/session_YYYYMMDD_HHMMSS.jsonl`. This design choice:

- **Isolates sessions:** Comparing a run before and after adding HITL data to ChromaDB is trivial — two files, two metrics reports
- **No rotation complexity:** Small corpus, short sessions → files are tiny (~2KB for 10 queries)
- **Preserves history:** Every session is recoverable; nothing is overwritten

The tradeoff is that cross-session aggregation (e.g., "how has my quiz accuracy on Decision Trees improved over the past week?") requires reading multiple files. For MVP this is acceptable — the metrics report reads the current session file only.

---

### The Five Event Types and What They Measure

**`retrieval`** — measures retrieval system performance:
- `latency_ms`: Is retrieval fast enough for interactive use? Target < 100ms.
- `retrieved_pages`: Which topics are being queried? Topic distribution analysis.
- Empty `retrieved` list: Queries with no results → gaps in corpus.

**`llm_response`** — measures LLM performance:
- `llm_latency_ms`: How long does Ollama take? Useful for model comparison.
- `sources_cited`: Did the LLM actually cite the retrieved sources? (Prompt adherence)
- `model`: Records which model was in use when a session ran.

**`quiz_grade`** — measures student learning:
- `grade` by `source_page`: Per-topic accuracy over time
- `grade_latency_ms`: Quiz grading latency (second Ollama call per round)

**`hitl_rating`** — measures retrieval quality from human signal:
- `human_rating` by `retrieved_pages`: Per-topic hit-rate from human perspective
- Complement to automated hit-rate — captures what keyword ground truth misses

**`refusal`** — measures corpus coverage gaps:
- `refusal_layer`: Distinguishes retrieval failures (`no_results`, `low_confidence`) from LLM failures (`llm_refusal`)
- `query` text: The specific topic the notes don't cover → actionable note-taking guidance

---

### Metrics as Feedback Loops

The logging system creates three feedback loops:

**Loop 1 — Retrieval quality:**
`retrieval` + `hitl_rating` events → HITL hit-rate per topic → identify topics where retrieval is weak → investigate chunk quality or add more notes

**Loop 2 — Student learning:**
`quiz_grade` events → accuracy per topic → weak topic flags → student studies more or improves notes

**Loop 3 — Corpus coverage:**
`refusal` events → list of uncovered topics → student adds notes → HITL injection expands corpus

Each loop closes by feeding information back to the student or system designer to improve the next session. This is the core value proposition of instrumented observability in an educational tool.

---

## Concept Summary Table

| Concept | What it is | Why it matters here |
|---|---|---|
| RAG | Inject retrieved docs into LLM prompt at inference time | Anchors answers to personal notes, prevents hallucination |
| Chunking | Split corpus into retrieval units | One Q&A pair per chunk = precise, self-contained retrieval unit |
| Dense embeddings | Map text to vector space; similar meaning = close vectors | Handles semantic/paraphrase queries BM25 misses |
| BM25 | Keyword ranking with TF-IDF weighting | Handles exact ML jargon queries; free, fast |
| Hybrid retrieval | Run both retrievers; merge result lists | Best of both — neither alone is sufficient |
| RRF | Rank-based score fusion for merging lists | No score normalization needed; robust; 5 lines of code |
| Retrieval guardrail (L1) | Gate before LLM call based on retrieval scores | Cheap refusal; no wasted Ollama calls for off-topic queries |
| LLM refusal prompt (L2) | Instruct model to use exact phrase if context insufficient | Deterministic hallucination prevention inside the LLM call |
| Hit-rate | % of queries where expected source is in top-k results | Fast automated retrieval regression test |
| HITL evaluation | Human rates retrieved results as relevant/partial/not | Ground truth signal automated metrics can't capture |
| HITL corpus expansion | Inject human-validated pairs into ChromaDB | Active learning — human feedback improves future retrieval |
| Quiz accuracy | % of quiz rounds graded correct per topic | Proxy for both student learning and note quality |
| Structured logging | Machine-readable JSON events per interaction | Enables aggregation, charting, cross-session comparison |
| JSONL | One JSON object per line | Appendable, streamable, recoverable |
| Session log files | New file per run | Session isolation; easy before/after comparison |
| Refusal logging | Log every refused query with reason and scores | Converts failures into actionable corpus gap diagnostics |
