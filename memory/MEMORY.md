# Memory Index

## System
- [System Overview](system/system_overview.md) — Architecture of the existing DocuBot (BM25, Gemini, 3 modes)
- [Background](background.md) — Concepts behind all design decisions: RAG, chunking, BM25, RRF, guardrails, HITL eval, structured logging

## Design
- [001 ML Study Bot](design/001_ml_study_bot.md) — Master design: Ollama+ChromaDB+BM25 RAG, Quiz, HITL, Guardrails, Logging
- [002 System Components](design/002_system_components.md) — Component catalog: signatures, dependencies, data types (QAPair, RetrievalResult)
- [003 Data Flow](design/003_data_flow.md) — End-to-end traces: Ingest, RAG Q&A, Quiz Me, HITL; error/refusal paths; state persistence
- [004 Repo Structure](design/004_repo_structure.md) — backend/ ml/ frontend/ data/ tests/ layout; import rules; migration plan
- [005 Streamlit UI](design/005_streamlit_ui.md) — Four tabs (Q&A, Quiz, Metrics, HITL); session state keys; startup sequence
