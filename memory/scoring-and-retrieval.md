# Scoring and Retrieval — Implementation Plan

## Basic Python Implementation (Phase 1 Goal)

### `build_index`

```python
def build_index(self, documents):
    index = {}
    for filename, text in documents:
        words = text.lower().split()
        for word in words:
            word = word.strip(".,!?;:\"'()[]{}")  # strip punctuation
            if word:
                if word not in index:
                    index[word] = []
                if filename not in index[word]:
                    index[word].append(filename)
    return index
```

- Splits every document's text on whitespace
- Lowercases and strips punctuation from each token
- Maps each token → list of filenames that contain it

### `score_document`

```python
def score_document(self, query, text):
    query_words = query.lower().split()
    text_lower = text.lower()
    score = 0
    for word in query_words:
        if word in text_lower:
            score += 1
    return score
```

- Splits query into words
- Counts how many query words appear anywhere in the document text
- Returns that count as the numeric score
- Different queries produce different scores because different words match different documents

### `retrieve`

```python
def retrieve(self, query, top_k=3):
    scored = []
    for filename, text in self.documents:
        score = self.score_document(query, text)
        if score > 0:
            scored.append((score, filename, text))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [(filename, text) for _, filename, text in scored[:top_k]]
```

- Scores every document
- Filters out zero-score documents
- Sorts descending by score
- Returns top_k as (filename, text) pairs

---

## Why Results Differ Per Query

The score is purely a word overlap count. A query about "authentication" will hit words in AUTH.md more
than DATABASE.md. A query about "schema" will do the reverse. The numeric score is the signal that
separates relevant from irrelevant.

---

## Advanced Techniques and the Science Behind Them

### 1. TF-IDF (Term Frequency – Inverse Document Frequency)

**What it does:** Weights words by how *rare* they are across the whole corpus.

**The science:**
- **TF (Term Frequency):** how often a word appears in *this* document. Common words within a doc signal relevance.
- **IDF (Inverse Document Frequency):** `log(total_docs / docs_containing_word)`. Words that appear in
  *every* document (like "the", "is") get a near-zero IDF weight, so they don't pollute scores. Rare
  words get high IDF.
- Final score = TF × IDF for each query word, summed up.

**Why it's better:** The basic approach counts "the" the same as "authentication". TF-IDF automatically
downweights common words and upweights rare, meaningful ones.

**Library:** `sklearn.feature_extraction.text.TfidfVectorizer`

---

### 2. BM25 (Best Match 25)

**What it does:** An evolution of TF-IDF used in production search engines (Elasticsearch, Lucene).

**The science:**
- Adds **document length normalization** — a long doc naturally contains more words, so raw TF inflates
  its score unfairly. BM25 penalizes longer documents.
- Uses a **saturation function** so repeating a word 100 times doesn't give 100x the score of repeating
  it once — relevance saturates.
- Two tunable parameters: `k1` (term frequency saturation) and `b` (length normalization strength).

**Why it's better than TF-IDF:** More robust scoring in practice. The standard baseline for keyword
search in academia and industry.

**Library:** `rank_bm25`

---

### 3. Dense Embeddings / Semantic Search

**What it does:** Converts query and documents into high-dimensional vectors using a neural model, then
retrieves by **cosine similarity** in vector space.

**The science:**
- A language model (e.g. `sentence-transformers`) encodes text into a vector where semantic meaning is
  captured — "car" and "automobile" will have nearby vectors even though they share no words.
- Retrieval becomes a nearest-neighbor search: find the document vectors closest to the query vector.
- Addresses the **vocabulary mismatch problem** that kills keyword methods: a query "how do I log in"
  can match a doc about "authentication flow" even with zero word overlap.

**Why it's better:** Understands *meaning* not just surface words. Used in modern RAG systems.

**Libraries:** `sentence-transformers`, `faiss`, `chromadb`

---

## Technique Comparison

| Technique        | Handles Synonyms | Speed     | Complexity | Best For               |
|------------------|------------------|-----------|------------|------------------------|
| Word Overlap     | No               | Very fast | Trivial    | Learning / tiny docs   |
| TF-IDF           | No               | Fast      | Low        | Small corpora          |
| BM25             | No               | Fast      | Low-Medium | Production keyword search |
| Dense Embeddings | Yes              | Slower    | High       | Semantic / RAG systems |

---

## Recommendation for DocuBot

- **Phase 1:** Word overlap (basic Python, no dependencies) — good enough for small doc sets
- **Upgrade path:** Swap `score_document` for BM25 with `rank_bm25` for noticeably better results
- **Phase 2 / RAG:** Dense embeddings pair naturally with an LLM since retrieval is semantic, not lexical
