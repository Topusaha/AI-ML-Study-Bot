"""
Shared data models for the ML Study Bot.
This module has no project-level dependencies — it is a leaf import.
"""

from dataclasses import dataclass, field


@dataclass
class QAPair:
    """A single question/answer pair parsed from a Notion markdown export."""

    id: str
    # Deterministic: f"{page_slug}_{line_num}" (1-indexed line of the question bullet)

    question: str
    # Stripped text of the top-level "- " bullet

    answer: str
    # Accumulated stripped text of all indented lines below the question.
    # Image references are replaced with "[diagram: <filename>]" markers.

    page_title: str
    # Human-readable title derived from the source .md filename
    # e.g. "Decision Trees"

    section: str
    # "main_ideas" | "exercises" | "" (empty if no ### heading precedes the pair)

    has_diagram: bool
    # True if any answer line contains a Markdown image reference ("![")

    img_paths: list[str] = field(default_factory=list)
    # Relative paths extracted from ![](<path>) syntax within the answer block


@dataclass
class RetrievalResult:
    """Output of HybridRetriever.retrieve()."""

    pairs: list[QAPair]

    top_cosine_distance: float
    # Cosine distance of the nearest neighbour among returned pairs.
    # Range [0.0, 2.0]; lower is more similar. 0.0 == identical vector.
    # Named "cosine" for conceptual clarity; ChromaDB internally uses L2.

    top_bm25_score: float
    # Highest BM25Okapi score among returned pairs.
    # Range [0.0, inf); 0.0 means zero term overlap with the query.
