"""
Smoke tests for HybridRetriever (ml/retriever.py).

These tests build a small in-memory fixture corpus rather than hitting the
real Notion data, so they run without the full dataset present.

Run with: pytest tests/test_retriever.py -v
"""

import pytest
from ml.retriever import HybridRetriever
from ml.models import QAPair, RetrievalResult

_TMPDIR = "/tmp/test_chroma_ml_study_bot"


@pytest.fixture(scope="module")
def retriever_with_data() -> HybridRetriever:
    r = HybridRetriever(
        collection_name="test_collection",
        chroma_persist_dir=_TMPDIR,
    )
    pairs = [
        QAPair(
            id="decision_trees_1",
            question="What is Gini impurity?",
            answer="Gini impurity measures the probability of misclassification at a node.",
            page_title="Decision Trees",
            section="main_ideas",
            has_diagram=False,
        ),
        QAPair(
            id="training_models_1",
            question="What is gradient descent?",
            answer="An optimisation algorithm that iteratively adjusts parameters to minimise loss.",
            page_title="Training Models",
            section="main_ideas",
            has_diagram=False,
        ),
        QAPair(
            id="svms_1",
            question="What is the margin in an SVM?",
            answer="The distance between the decision boundary and the nearest support vectors.",
            page_title="Support Vector Machines",
            section="main_ideas",
            has_diagram=False,
        ),
    ]
    r.index(pairs)
    return r


def test_is_populated(retriever_with_data):
    assert retriever_with_data.is_populated()


def test_retrieve_returns_result(retriever_with_data):
    result = retriever_with_data.retrieve("gini impurity decision tree", k=2)
    assert isinstance(result, RetrievalResult)
    assert len(result.pairs) > 0


def test_retrieve_top_result_is_relevant(retriever_with_data):
    result = retriever_with_data.retrieve("gini impurity", k=1)
    assert len(result.pairs) > 0, "Expected at least 1 result"
    assert result.pairs[0].page_title == "Decision Trees", (
        f"Expected top result from Decision Trees, got {result.pairs[0].page_title}"
    )


def test_retrieve_scores_populated(retriever_with_data):
    result = retriever_with_data.retrieve("gradient descent loss minimisation", k=2)
    assert 0.0 <= result.top_cosine_distance <= 2.0
    assert result.top_bm25_score >= 0.0


def test_retrieve_with_page_title_filter(retriever_with_data):
    result = retriever_with_data.retrieve(
        "algorithm",
        k=3,
        where={"page_title": {"$eq": "Training Models"}},
    )
    for pair in result.pairs:
        assert pair.page_title == "Training Models"


def test_retrieve_empty_query_returns_results(retriever_with_data):
    """Empty or nonsense query should not crash — may return low-score results."""
    result = retriever_with_data.retrieve("xyzzy_nonsense_query", k=3)
    assert isinstance(result, RetrievalResult)
