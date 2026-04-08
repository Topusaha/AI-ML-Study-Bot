"""
Unit tests for MarkdownParser (ml/parser.py).

Run with: pytest tests/test_parser.py -v
"""

import pytest
from pathlib import Path
from ml.parser import MarkdownParser
from ml.models import QAPair


NOTION_DIR = str(Path(__file__).parent.parent / "data" / "assets" / "notion")


@pytest.fixture(scope="module")
def pairs() -> list[QAPair]:
    parser = MarkdownParser()
    return parser.parse(NOTION_DIR)


def test_total_pair_count(pairs):
    """Should parse approximately 148 Q&A pairs across all 8 pages."""
    assert 120 <= len(pairs) <= 180, (
        f"Expected ~148 pairs, got {len(pairs)}"
    )


def test_all_pairs_have_nonempty_question(pairs):
    for p in pairs:
        assert p.question.strip(), f"Empty question in pair {p.id}"


def test_all_pairs_have_nonempty_answer(pairs):
    for p in pairs:
        assert p.answer.strip(), f"Empty answer in pair {p.id}"


def test_section_tags_are_valid(pairs):
    valid_sections = {"main_ideas", "exercises", ""}
    for p in pairs:
        assert p.section in valid_sections, (
            f"Unexpected section {p.section!r} in pair {p.id}"
        )


def test_diagram_pairs_have_img_paths(pairs):
    diagram_pairs = [p for p in pairs if p.has_diagram]
    assert len(diagram_pairs) > 0, "Expected at least some pairs with diagrams"
    for p in diagram_pairs:
        assert len(p.img_paths) > 0, (
            f"has_diagram=True but img_paths is empty for pair {p.id}"
        )


def test_ids_are_unique(pairs):
    ids = [p.id for p in pairs]
    assert len(ids) == len(set(ids)), "Duplicate QAPair IDs found"


def test_page_titles_cover_expected_topics(pairs):
    titles = {p.page_title for p in pairs}
    expected_substrings = [
        "Decision Trees",
        "Training Models",
        "Support Vector Machines",
    ]
    for substr in expected_substrings:
        assert any(substr in t for t in titles), (
            f"Expected a page with '{substr}' in title, got titles: {titles}"
        )
