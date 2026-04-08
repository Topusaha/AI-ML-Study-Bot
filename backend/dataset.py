"""
Sample queries for the ML Study Bot Evaluation mode.

These queries cover the 8 topic pages in the Notion corpus and are used by
AutoEvaluator.run_keyword_hit_rate() and the HITL evaluation session in main.py.
"""

SAMPLE_QUERIES: list[str] = [
    "What is the CART algorithm and how does it split nodes?",
    "Explain the bias-variance tradeoff in model training.",
    "How does a Support Vector Machine maximize the margin?",
    "What is Gini impurity and how is it calculated?",
    "Describe the difference between bagging and boosting.",
    "How does PCA reduce dimensionality?",
    "What is k-means clustering and how does it converge?",
    "Explain backpropagation in neural networks.",
    "What is dropout regularization and why does it work?",
    "How does gradient descent update model parameters?",
]

# Expected keywords per query — used by AutoEvaluator for keyword hit-rate testing.
# Each entry is (query, list_of_keywords_expected_in_retrieved_text).
EVAL_PAIRS: list[tuple[str, list[str]]] = [
    (SAMPLE_QUERIES[0], ["cart", "split"]),
    (SAMPLE_QUERIES[1], ["bias", "variance"]),
    (SAMPLE_QUERIES[2], ["margin", "support vector"]),
    (SAMPLE_QUERIES[3], ["gini"]),
    (SAMPLE_QUERIES[4], ["bagging", "boosting"]),
    (SAMPLE_QUERIES[5], ["pca", "principal"]),
    (SAMPLE_QUERIES[6], ["k-means", "cluster"]),
    (SAMPLE_QUERIES[7], ["backprop"]),
    (SAMPLE_QUERIES[8], ["dropout"]),
    (SAMPLE_QUERIES[9], ["gradient", "descent"]),
]
