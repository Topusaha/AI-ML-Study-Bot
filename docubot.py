"""
Core DocuBot class responsible for:
- Loading documents from the docs/ folder
- Building a simple retrieval index (Phase 1)
- Retrieving relevant snippets (Phase 1)
- Supporting retrieval only answers
- Supporting RAG answers when paired with Gemini (Phase 2)
"""

import os
import re
import glob
from rank_bm25 import BM25Okapi

class DocuBot:
    def __init__(self, docs_folder="docs", llm_client=None):
        """
        docs_folder: directory containing project documentation files
        llm_client: optional Gemini client for LLM based answers
        """
        self.docs_folder = docs_folder
        self.llm_client = llm_client

        # Load documents into memory
        self.documents = self.load_documents()  # List of (filename, text)

        # Build a retrieval index (implemented in Phase 1)
        self.index = self.build_index(self.documents)

    # -----------------------------------------------------------
    # Document Loading
    # -----------------------------------------------------------

    def load_documents(self):
        """
        Loads all .md and .txt files inside docs_folder and splits each file
        into section-level chunks at ## headings.
        Returns a list of tuples: (chunk_id, text)
        where chunk_id = "FILENAME#Section Heading" (or just "FILENAME" for
        the preamble before the first ## heading).
        """
        docs = []
        pattern = os.path.join(self.docs_folder, "*.*")
        for path in glob.glob(pattern):
            if path.endswith(".md") or path.endswith(".txt"):
                with open(path, "r", encoding="utf8") as f:
                    text = f.read()
                filename = os.path.basename(path)
                docs.extend(self._chunk_by_section(text, filename))
        return docs

    def _chunk_by_section(self, text, filename):
        """
        Splits a markdown document into chunks at ## headings (not ###).
        Each chunk is (chunk_id, section_text) where
        chunk_id = "FILENAME#Heading" for ## sections,
        or just "FILENAME" for any preamble before the first ## heading.
        """
        # Split just before lines that start with exactly '## '
        parts = re.split(r'(?=\n## )', text)
        chunks = []
        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
            heading_match = re.match(r'^## (.+)', part)
            if heading_match:
                chunk_id = f"{filename}#{heading_match.group(1).strip()}"
            else:
                chunk_id = filename
            chunks.append((chunk_id, part))
        return chunks

    # -----------------------------------------------------------
    # Index Construction (Phase 1)
    # -----------------------------------------------------------

    def build_index(self, documents):
        """
        Builds a BM25Okapi index over the loaded documents.
        Stores tokenized corpus and filenames so score_document and retrieve
        can look up scores by index position.
        """
        self._bm25_filenames = [filename for filename, _ in documents]
        tokenized_corpus = [
            text.lower().split() for _, text in documents
        ]
        return BM25Okapi(tokenized_corpus)

    # -----------------------------------------------------------
    # Scoring and Retrieval (Phase 1)
    # -----------------------------------------------------------

    def score_document(self, query, text):
        """
        Returns the BM25 score for a single document against the query.
        Tokenizes both query and text, then uses get_scores() and picks
        the score for the matching document position.
        """
        tokenized_query = query.lower().split()
        scores = self.index.get_scores(tokenized_query)
        tokenized_text = text.lower().split()
        for i, (_, doc_text) in enumerate(self.documents):
            if doc_text.lower().split() == tokenized_text:
                return scores[i]
        return 0.0

    def retrieve(self, query, top_k=3):
        """
        Uses BM25 to score all documents and return the top_k most relevant
        as a list of (filename, text) sorted by score descending.
        """
        tokenized_query = query.lower().split()
        scores = self.index.get_scores(tokenized_query)
        ranked = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )
        results = []
        for i in ranked[:top_k]:
            if scores[i] > 0:
                filename, text = self.documents[i]
                results.append((filename, text))
        return results

    # -----------------------------------------------------------
    # Answering Modes
    # -----------------------------------------------------------

    def answer_retrieval_only(self, query, top_k=3):
        """
        Phase 1 retrieval only mode.
        Returns raw snippets and filenames with no LLM involved.
        """
        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        formatted = []
        for filename, text in snippets:
            formatted.append(f"[{filename}]\n{text}\n")

        return "\n---\n".join(formatted)

    def answer_rag(self, query, top_k=3):
        """
        Phase 2 RAG mode.
        Uses student retrieval to select snippets, then asks Gemini
        to generate an answer using only those snippets.
        """
        if self.llm_client is None:
            raise RuntimeError(
                "RAG mode requires an LLM client. Provide a GeminiClient instance."
            )

        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        return self.llm_client.answer_from_snippets(query, snippets)

    # -----------------------------------------------------------
    # Bonus Helper: concatenated docs for naive generation mode
    # -----------------------------------------------------------

    def full_corpus_text(self):
        """
        Returns all documents concatenated into a single string.
        This is used in Phase 0 for naive 'generation only' baselines.
        """
        return "\n\n".join(text for _, text in self.documents)
