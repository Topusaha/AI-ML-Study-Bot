# AI/ML Study Bot

An AI-powered study assistant that helps you learn machine learning concepts from your own notes. It uses hybrid retrieval (BM25 + ChromaDB dense embeddings) with a local Ollama LLM to answer questions, generate quizzes, and evaluate retrieval quality.

---

## Features

- **RAG Q&A** — Ask free-form questions about ML topics; the bot retrieves relevant note snippets and generates a grounded answer with source citations.
- **Quiz Me** — The bot generates a question from your notes, you answer, and it grades you with detailed feedback.
- **HITL Evaluation** — Human-in-the-loop loop: retrieve snippets for sample queries, rate relevance, and view a session metrics report.
- **Streamlit UI** — Web interface for all three modes with session metrics.
- **Guardrails** — Refuses to answer when retrieval confidence is too low (BM25 score + cosine distance thresholds).

---

## Topics

Notes are sourced from *Hands On Machine Learning* (Notion export) and cover:

- Decision Trees
- Training Models
- Support Vector Machines
- Ensemble Learning and Random Forests
- Dimensionality Reduction
- Unsupervised Learning Techniques
- Introduction to Artificial Neural Networks
- Training Deep Neural Networks

---

## Setup

### 1. Install Python dependencies

    pip install -r requirements.txt

### 2. Start Ollama and pull models

    ollama serve
    ollama pull llama3.2
    ollama pull nomic-embed-text   # optional, for upgraded embeddings

### 3. (Optional) Configure environment variables

    cp .env.example .env

Edit `.env` if you need to override any defaults.

---

## Running

### CLI

    python main.py

Choose a mode at the prompt:

- **1** — RAG Q&A
- **2** — Quiz Me
- **3** — Evaluation (HITL)
- **q** — Quit

### Streamlit UI

    streamlit run main.py

---

## Project Structure

```
backend/
  studybot.py      # Orchestrates ingestion, indexing, retrieval, and logging
  evaluation.py    # HITL evaluator and metrics reporting
  dataset.py       # Sample queries for evaluation mode
  logger.py        # Session logging (retrieval, quiz grades, HITL ratings, refusals)

ml/
  retriever.py     # HybridRetriever: ChromaDB dense + BM25 sparse via RRF fusion
  models.py        # Data models: QAPair, RetrievalResult
  parser.py        # Parses Notion markdown exports into Q&A pairs
  guardrails.py    # Thresholds and InsufficientContextError
  llm_client.py    # OllamaClient: answer, quiz generation, and grading prompts

frontend/
  tabs/
    qa_tab.py      # RAG Q&A Streamlit tab
    hitl_tab.py    # HITL evaluation Streamlit tab
    metrics_tab.py # Session metrics Streamlit tab

data/
  assets/notion/   # Notion markdown exports (source notes)
  .chroma/         # Persisted ChromaDB vector store
  logs/            # Session log files (JSONL)

tests/             # pytest test suite
main.py            # CLI and Streamlit entry point
```

---

## How It Works

1. **Ingestion** — `MarkdownParser` parses Notion exports into `QAPair` objects (question + answer + metadata).
2. **Indexing** — `HybridRetriever` upserts pairs into a persistent ChromaDB collection and builds an in-memory BM25 index.
3. **Retrieval** — On a query, both ChromaDB (cosine similarity) and BM25 (keyword) retrieve candidates; scores are fused via Reciprocal Rank Fusion (RRF).
4. **Guardrail** — If top BM25 score and cosine distance both fall below thresholds, an `InsufficientContextError` is raised and the bot refuses to answer.
5. **Generation** — Retrieved snippets are passed to `OllamaClient`, which calls a local `llama3.2` model to generate an answer, quiz question, or grade.

---

## Running Tests

    pytest

---

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com) running locally with `llama3.2` pulled
- No external database or cloud services required
