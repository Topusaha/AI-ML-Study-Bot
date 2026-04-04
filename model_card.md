# DocuBot Model Card

This model card is a short reflection on your DocuBot system. Fill it out after you have implemented retrieval and experimented with all three modes:

1. Naive LLM over full docs  
2. Retrieval only  
3. RAG (retrieval plus LLM)


---

## 1. System Overview

**What is DocuBot trying to do?**  
Describe the overall goal in 2 to 3 sentences.

> _Your answer here._
Docubot is trying to take an internal document and be able to answer questions on it. It has the ability to do so in 3 different ways. The first is to scan the entire document and load it into its context window, the second is to rank them using the algorithm BM25, and the third is using RAG. 

**What inputs does DocuBot take?**  
For example: user question, docs in folder, environment variables.

> _Your answer here._
DocuBot takes in an api key from an enviroment variable to be able to connect to gemini. It also takes in the document, a loading/ranking system which is pre-defined, and the user and system prompts. User prompt is given while system promptd are pre-defined. 

**What outputs does DocuBot produce?**

> _Your answer here._
The outputs of the model is the answer to the question as well as some confidence around it. If the model is unsure based on a confidence crieteria it will let the user know that it can not answer the question. 
---

## 2. Retrieval Design

**How does your retrieval system work?**  
Describe your choices for indexing and scoring.

- How do you turn documents into an index?
- How do you score relevance for a query?
- How do you choose top snippets?

> _Your answer here._
We turn documents into an index by chuncking with lower() and split() and passing it o the BM25 algorithm to rank. The documents are loaded form the docs and split via headings at section levels.

We score them by taking in the user prompt and ranking them again using BM25 scoring to put a score on each section we have in our index.

We then retrieve the top k chuncks and the default is 3 so we return the top 3 chuncks that matches what the user is asking through the BM25 scoring system. 

**What tradeoffs did you make?**  
For example: speed vs precision, simplicity vs accuracy.

> _Your answer here._
We chuncked via section level and not sentence level because BM25 can be diluted when it scores a long section. 

We reduce hullucinations by having a strict refusal to answer based on a confidence score. 

BM25 is fast and we don't use embeddings or a vector search which would slow things down. 

---

## 3. Use of the LLM (Gemini)

**When does DocuBot call the LLM and when does it not?**  
Briefly describe how each mode behaves.

- Naive LLM mode: 
- Retrieval only mode:
- RAG mode:

> _Your answer here._

The Naive LLM mode calls the model everytime and it sees the entire doc to determine how to handle the users question. 

The retrieval only model never calls an LLM but uses BM25 scores to compute the top-k results to give back to the user. 

Rag Mode only calls the LLM after retrival of the top k results which is sent to the LLM to reason and answer.

**What instructions do you give the LLM to keep it grounded?**  
Summarize the rules from your prompt. For example: only use snippets, say "I do not know" when needed, cite files.

> _Your answer here._
If the snnipets aren't able to provide enough context the model has the ability to refuse to answer so it does not make things up. 

---

## 4. Experiments and Comparisons

Run the **same set of queries** in all three modes. Fill in the table with short notes.

You can reuse or adapt the queries from `dataset.py`.

| Query | Naive LLM: helpful or harmful? | Retrieval only: helpful or harmful? | RAG: helpful or harmful? | Notes |
|------|---------------------------------|--------------------------------------|---------------------------|-------|
| Where is the auth token generated? | **Harmful** — hallucinated SQLite/PostgreSQL details, never mentioned `generate_access_token` or `auth_utils.py` | **Helpful** — correctly retrieved AUTH.md#Token Generation with the right function name, env var, and payload fields | **Harmful** — fabricated code snippet and "expires in 30 min" detail not in the docs; mixed in irrelevant content | Retrieval only was the clear winner here |
| How do I connect to the database? | **Harmful** — invented Python code using `sqlite3`/`psycopg2` that does not exist in the docs | **Partially helpful** — retrieved SETUP.md which mentions `DATABASE_URL` but the actual connection steps aren't in the docs | **Harmful** — leaked its own system prompt in the response; gave no useful answer | None of the modes answered this well because the docs don't contain connection code |
| Which endpoint lists all users? | **Partially helpful** — mentioned `/api/users` (correct) but buried in irrelevant database comparison text | **Helpful** — retrieved API_REFERENCE.md (the right document); top-level chunk retrieved but specific endpoint detail is in a sub-section | **Harmful** — generated an email/template format instead of an answer; completely off-topic | Naive LLM got the right answer by accident; retrieval found the right doc |
| How does a client refresh an access token? | **Harmful** — completely wrong; discussed DATABASE_URL mismatches and database migration instead | **Helpful** — AUTH.md#Client Workflow clearly showed the 4-step flow including `/api/refresh`; also retrieved the POST /api/refresh endpoint spec | **Harmful** — generated meta-instructions about how to answer instead of actually answering | Retrieval only was the clear winner; tinyllama struggled badly with RAG prompts |

**What patterns did you notice?**  

- When does naive LLM look impressive but untrustworthy?  
- When is retrieval only clearly better?  
- When is RAG clearly better than both?

> **Naive LLM looks impressive but untrustworthy** when the question is about something common like database connections the LLM produces confident and good code but the code is not from the docs. 
>
> **Retrieval only is clearly better** for factual lookup questions like "where is X defined?" or "what endpoint does Y?"  BM25 finds the exact section and returns the raw text with no hallucinations.
>
> **RAG should be better** when the retrieved snippets need synthesis or when the answer spans multiple chunks, but with a very small model like tinyllama, the RAG prompt was too complex and the model confused itself. It sometimes echoed the prompt or generating nonsense.

---

## 5. Failure Cases and Guardrails

**Describe at least two concrete failure cases you observed.**  
For each one, say:

- What was the question?  
- What did the system do?  
- What should have happened instead?

> **Failure case 1: Naive LLM hallucinating a database connection**
> - **Question:** “How do I connect to the database?”
> - **What the system did:** The naive LLM generated confident, well-formatted Python code using `sqlite3` and `psycopg2` with specific connection strings — none of which exist anywhere in the docs.
> - **What should have happened:** The model should have said “I do not know based on the docs I have” because the documentation only mentions the `DATABASE_URL` environment variable, not how to write connection code.

> **Failure case 2: RAG with tinyllama echoing its own prompt**
> - **Question:** “How does a client refresh an access token?”
> - **What the system did:** The RAG mode (using tinyllama) generated a list of meta-instructions about how to answer questions — essentially paraphrasing the rules from the system prompt — instead of actually answering.
> - **What should have happened:** The model should have used the retrieved AUTH.md#Client Workflow snippet (which clearly shows step 4: “Refresh the token when it expires by calling `/api/refresh`”) to give a direct answer. Retrieval-only mode answered this correctly; the LLM made it worse.

**When should DocuBot say “I do not know based on the docs I have”?**  
Give at least two specific situations.

> 1. **When the question asks about something not covered in the docs** — for example, “How do I connect to the database?” The docs only mention `DATABASE_URL` as an environment variable; there is no connection code. Rather than inventing code, the system should refuse.
>
> 2. **When BM25 retrieval returns no results with a score above zero** — if no document section matches the query at all, there is no basis for an answer and the system should refuse rather than guess.

**What guardrails did you implement?**  
Examples: refusal rules, thresholds, limits on snippets, safe defaults.

> - **Explicit refusal instruction in the RAG prompt:** The LLM is told to reply exactly “I do not know based on the docs I have.” if the snippets are insufficient, rather than guessing.
> - **BM25 score threshold in retrieval:** `retrieve()` only returns snippets with a score greater than zero — documents with no keyword overlap are excluded entirely, preventing irrelevant context from being passed to the LLM.
> - **Short-circuit on empty snippets:** Both `answer_from_snippets` and `answer_rag` return the refusal string immediately if the retriever returns nothing, without ever calling the LLM.
> - **Citation requirement:** The prompt instructs the LLM to mention which files it relied on, making it easier to spot when an answer is not actually grounded in the docs.

---

## 6. Limitations and Future Improvements

**Current limitations**  
List at least three limitations of your DocuBot system.

1. Makes up code 
2. Small open source model not able to have complex reasonings
3. Bm25 only uses keywords so if you use a word that isn't in its vocabulary it won't be able to match on the proper context

**Future improvements**  
List two or three changes that would most improve reliability or usefulness.

1. Migrate away from BM25 and use something like vector search with cosine similiarity
2. Use a stronger model so it can have more complex usecases and understanding
3. Improve the quality of the docs so the data is more rich for processing

---

## 7. Responsible Use

**Where could this system cause real world harm if used carelessly?**  
Think about wrong answers, missing information, or over trusting the LLM.

> _Your answer here._
This system can be used well in docs for libaries like docker, python packages, and popular frameworks. It will be helpful for useers to interact with a model for simple questions and fixes that relates to it.

**What instructions would you give real developers who want to use DocuBot safely?**  
Write 2 to 4 short bullet points.

- Don't fully trust without verifying sources yourself
- If the model responds with "I don't know" try adding more context to improve the confidence score


---
