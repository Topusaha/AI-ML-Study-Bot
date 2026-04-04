---
name: DocuBot Chunking Strategy
description: How and why docs/ files are split into sub-document chunks for BM25 retrieval
type: project
---

## What the docs look like

All four files in `docs/` are structured markdown with `##` section headings:

- **AUTH.md** — 6 sections: Overview, Token Generation, Validating Requests, Environment Variables, Client Workflow, Common Failure Cases
- **DATABASE.md** — 6 sections: Overview, Connection Configuration, Tables, Query Helpers, Common Failure Cases, Notes for Development
- **API_REFERENCE.md** — 5 sections: Base URL, Authentication Endpoints, User Data Endpoints, Project Data Endpoints, Error Formats
- **SETUP.md** — 7 sections: Requirements, Install Dependencies, Environment Variables, Initialize the Database, Running the Application, Using the Docs Assistant, Troubleshooting, Resetting the Environment

Each `##` section is a self-contained topic (100–300 words). `###` headings exist inside some sections (individual API endpoints) but are too granular to use as chunk boundaries.

## Chosen strategy: split at `##` headings

`_chunk_by_section(text, filename)` uses `re.split(r'(?=\n## )', text)` to split each file into sections. Each chunk gets a `chunk_id`:

- `"FILENAME#Section Heading"` for `##` sections
- `"FILENAME"` for any preamble before the first `##` (title + intro paragraph)

`load_documents` now calls `_chunk_by_section` and `self.documents` holds section-level `(chunk_id, text)` pairs instead of whole-file pairs. `build_index`, `score_document`, and `retrieve` are unchanged — BM25 operates over sections automatically.

## Why `##` over alternatives

| Option | Problem |
|---|---|
| Paragraphs | Code blocks get split from their explanatory text |
| `###` headings | Too small — one chunk per API endpoint, BM25 scores degrade on tiny texts |
| Fixed-size slices | Arbitrary cuts mid-sentence or mid-code-block |
| Whole files | BM25 scores a 3 KB file as one unit; a query about "token expiry" scores all of AUTH.md equally even though only one section is relevant |

`##` sections are the natural semantic unit in these docs: each covers one concept, includes its own code examples, and is large enough for BM25 to score meaningfully.

**Why:** Retrieval precision improves because the LLM receives the exact section that answers the query rather than an entire file padded with unrelated content.
**How to apply:** When adding new docs to `docs/`, structure them with `##` headings so `_chunk_by_section` picks them up correctly. Avoid `##` headings that are only one or two lines — merge them into their parent section.
