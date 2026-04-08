"""
CLI entry point for the ML Study Bot.

Modes:
  1) RAG Q&A   — hybrid retrieval + Ollama LLM answer
  2) Quiz Me   — Ollama generates a question; student answers; Ollama grades
  3) Evaluation — HITL loop: retrieve → display → human rates → log + report

Usage:
  python main.py

Requires Ollama to be running:
  ollama serve
  ollama pull llama3.2
  ollama pull nomic-embed-text   # optional, Phase 2 upgrade
"""

from dotenv import load_dotenv
load_dotenv()

import sys

from backend.studybot import StudyBot
from backend.evaluation import HITLEvaluator, print_metrics_report
from backend.dataset import SAMPLE_QUERIES
from ml.llm_client import OllamaClient
from ml.guardrails import InsufficientContextError


def _print_topic_list() -> None:
    print("Topics I have notes on:")
    for t in StudyBot.AVAILABLE_TOPICS:
        print(f"  - {t}")


# ---------------------------------------------------------------------------
# Mode 1: RAG Q&A
# ---------------------------------------------------------------------------

def run_rag_mode(bot: StudyBot, ollama: OllamaClient) -> None:
    print("\n--- RAG Q&A Mode ---")
    print("Type your question (or 'back' to return to menu).\n")

    while True:
        query = input("Question: ").strip()
        if query.lower() in ("back", "q", ""):
            break

        try:
            result = bot.retrieve(query, k=5, mode="rag")
            answer = ollama.answer_from_snippets(query, result.pairs)
            bot.logger.log_llm_response(query=query, response=answer, mode="rag")
            print(f"\nAnswer:\n{answer}")
            print(f"\nSources: {', '.join(p.page_title for p in result.pairs)}\n")
        except InsufficientContextError as e:
            print(f"\n{e}\n")
            _print_topic_list()
            print()
        except Exception as exc:
            if "connection" in str(exc).lower() or "refused" in str(exc).lower():
                print("\nOllama is not running. Start it with: ollama serve\n")
            else:
                print(f"\nError: {exc}\n")


# ---------------------------------------------------------------------------
# Mode 2: Quiz Me
# ---------------------------------------------------------------------------

def run_quiz_mode(bot: StudyBot, ollama: OllamaClient) -> None:
    print("\n--- Quiz Me Mode ---")
    print("Optionally enter a topic to focus on (or press Enter for any topic).")
    topic = input("Topic: ").strip() or None

    while True:
        try:
            query = topic or "general ML concepts"
            result = bot.retrieve(query, k=8, page_title_filter=topic, mode="quiz")
            question = ollama.quiz_from_snippets(result.pairs)
        except InsufficientContextError:
            print("No notes found for that topic.")
            _print_topic_list()
            break
        except Exception as exc:
            print(f"Could not generate question: {exc}")
            break

        print(f"\nQuestion: {question}")
        student_answer = input("Your answer: ").strip()

        if not student_answer:
            print("Skipping grade (no answer provided).")
        else:
            try:
                feedback = ollama.grade_student_answer(question, student_answer, result.pairs)
                print(f"\nFeedback:\n{feedback}\n")
                grade = feedback.split("\n")[0].strip().lower()
                bot.logger.log_quiz_grade(
                    question=question,
                    student_answer=student_answer,
                    grade=grade,
                    feedback=feedback,
                    mode="quiz",
                    page_title=result.pairs[0].page_title if result.pairs else "",
                )
            except Exception as exc:
                print(f"Grading error: {exc}")

        again = input("Another question? (y/n): ").strip().lower()
        if again != "y":
            break


# ---------------------------------------------------------------------------
# Mode 3: Evaluation (HITL)
# ---------------------------------------------------------------------------

def run_evaluation_mode(bot: StudyBot) -> None:
    print("\n--- Evaluation Mode ---")
    print("Using sample queries. Answer y / n / partial for each retrieval.\n")

    evaluator = HITLEvaluator(
        csv_path="data/human-in-the-loop-results.csv",
        logger=bot.logger,
    )

    queries_input = input(
        "Press Enter to use built-in sample queries, or type a single query: "
    ).strip()
    queries = [queries_input] if queries_input else SAMPLE_QUERIES

    for query in queries:
        print(f"\nQuery: {query}")
        try:
            result = bot.retrieve(query, k=3, mode="hitl")
        except InsufficientContextError:
            print("  → No relevant notes found. Skipping.\n")
            continue

        for i, pair in enumerate(result.pairs, start=1):
            print(f"  [{i}] ({pair.page_title}) Q: {pair.question}")

        rating = input("Relevant? (y / n / partial): ").strip().lower()
        if rating not in ("y", "n", "partial"):
            rating = "n"

        for pair in result.pairs:
            bot.logger.log_hitl_rating(
                query=query,
                qa_pair_id=pair.id,
                human_rating=rating,
                mode="hitl",
            )

    print("\n--- Session Report ---")
    print_metrics_report(str(bot.logger.current_log_path))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("ML Study Bot")
    print("============\n")

    bot = StudyBot()
    print("Loading and indexing notes...")
    bot.load_and_index()
    print()

    try:
        ollama = OllamaClient()
        has_llm = True
    except Exception as exc:
        print(f"Warning: Ollama unavailable — LLM modes disabled. ({exc})\n")
        ollama = None
        has_llm = False

    while True:
        print("Choose a mode:")
        print("  1) RAG Q&A")
        print("  2) Quiz Me")
        print("  3) Evaluation (HITL)")
        print("  q) Quit")
        choice = input("Choice: ").strip().lower()

        if choice == "q":
            print("\nGoodbye.")
            break
        elif choice == "1":
            if not has_llm:
                print("LLM unavailable. Run: ollama serve && ollama pull llama3.2\n")
            else:
                run_rag_mode(bot, ollama)
        elif choice == "2":
            if not has_llm:
                print("LLM unavailable. Run: ollama serve && ollama pull llama3.2\n")
            else:
                run_quiz_mode(bot, ollama)
        elif choice == "3":
            run_evaluation_mode(bot)
        else:
            print("Unknown choice.\n")


if __name__ == "__main__":
    main()
