"""
Ollama client wrapper used by DocuBot.

Handles:
- Sending prompts to a locally running Ollama instance
- Naive "generation only" answers over the full docs corpus (Phase 0)
- RAG style answers that use only retrieved snippets (Phase 2)

Experiment with:
- Prompt wording
- Refusal conditions
- How strictly the model is instructed to use only the provided context
- Swapping OLLAMA_MODEL for a different local model (e.g. "phi3:mini")
"""

import requests

# Central place to update the model name if needed.
# Run `ollama pull tinyllama` (or another model) before using.
OLLAMA_BASE_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "tinyllama"


class GeminiClient:
    """
    Ollama-backed client with the same interface as the original GeminiClient,
    so DocuBot needs no changes.

    Usage:
        client = GeminiClient()
        answer = client.naive_answer_over_full_docs(query, all_text)
        # or
        answer = client.answer_from_snippets(query, snippets)
    """

    def __init__(self):
        pass  # No API key needed for local Ollama

    def _generate(self, prompt):
        payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
        response = requests.post(OLLAMA_BASE_URL, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "").strip()

    # -----------------------------------------------------------
    # Phase 0: naive generation over full docs
    # -----------------------------------------------------------

    def naive_answer_over_full_docs(self, query, all_text):
        prompt = f"""You are a documentation assistant. Use the documentation below to answer the developer question.

Documentation:
{all_text}

Developer question: {query}
"""
        return self._generate(prompt)

    # -----------------------------------------------------------
    # Phase 2: RAG style generation over retrieved snippets
    # -----------------------------------------------------------

    def answer_from_snippets(self, query, snippets):
        """
        Phase 2:
        Generate an answer using only the retrieved snippets.

        snippets: list of (filename, text) tuples selected by DocuBot.retrieve

        The prompt:
        - Shows each snippet with its filename
        - Instructs the model to rely only on these snippets
        - Requires an explicit "I do not know" refusal when needed
        """

        if not snippets:
            return "I do not know based on the docs I have."

        context_blocks = []
        for filename, text in snippets:
            block = f"File: {filename}\n{text}\n"
            context_blocks.append(block)

        context = "\n\n".join(context_blocks)

        prompt = f"""
You are a cautious documentation assistant helping developers understand a codebase.

You will receive:
- A developer question
- A small set of snippets from project files

Your job:
- Answer the question using only the information in the snippets.
- If the snippets do not provide enough evidence, refuse to guess.

Snippets:
{context}

Developer question:
{query}

Rules:
- Use only the information in the snippets. Do not invent new functions,
  endpoints, or configuration values.
- If the snippets are not enough to answer confidently, reply exactly:
  "I do not know based on the docs I have."
- When you do answer, briefly mention which files you relied on.
"""

        return self._generate(prompt)
