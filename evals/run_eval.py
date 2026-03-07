#!/usr/bin/env python
"""Automated evaluation runner for devdocs-rag collections.

Parses an eval XML file, runs each question against the live ChromaDB
collection, checks must_contain terms, and prints a scored results table.

Usage:
    uv run python evals/run_eval.py
    uv run python evals/run_eval.py --eval evals/samsung_tv_eval.xml
    uv run python evals/run_eval.py --collection samsung_tv --n-results 5
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from devdocs_rag.embedding import EmbeddingModel  # noqa: E402
from devdocs_rag.store import DocStore  # noqa: E402


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Question:
    id: str
    topic: str
    prompt: str
    query: str
    terms: list[str]


@dataclass
class QuestionResult:
    question: Question
    hits: list[str]           # content returned by search_docs
    missing_terms: list[str]
    passed: bool


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------

def load_questions(eval_file: Path) -> tuple[str, list[Question]]:
    """Parse eval XML and return (collection_name, questions)."""
    tree = ET.parse(eval_file)
    root = tree.getroot()
    collection = root.get("collection", "samsung_tv")

    questions: list[Question] = []
    for q in root.findall("question"):
        tool = q.find("tools/tool")
        query = tool.findtext("suggested_query", "").strip() if tool is not None else ""
        terms = [t.text.strip() for t in q.findall("must_contain/term") if t.text]
        questions.append(Question(
            id=q.get("id", "?"),
            topic=q.get("topic", ""),
            prompt=(q.findtext("prompt") or "").strip(),
            query=query,
            terms=terms,
        ))
    return collection, questions


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_question(
    question: Question,
    collection: str,
    store: DocStore,
    n_results: int,
) -> QuestionResult:
    results = store.search(collection, question.query, n_results=n_results)
    combined = "\n".join(r.content for r in results)
    missing = [t for t in question.terms if t.lower() not in combined.lower()]
    return QuestionResult(
        question=question,
        hits=[r.content[:120] for r in results],
        missing_terms=missing,
        passed=len(missing) == 0,
    )


def run_eval(
    eval_file: Path,
    collection_override: str | None,
    n_results: int,
) -> list[QuestionResult]:
    collection, questions = load_questions(eval_file)
    if collection_override:
        collection = collection_override

    print(f"Loading embedding model (first run will take ~30s)...")
    model = EmbeddingModel()
    store = DocStore(embedding_model=model)

    print(f"Collection : {collection}")
    print(f"Questions  : {len(questions)}")
    print(f"n_results  : {n_results}")
    print()

    results: list[QuestionResult] = []
    for q in questions:
        r = run_question(q, collection, store, n_results)
        status = "PASS" if r.passed else "FAIL"
        print(f"  Q{q.id:>2} [{status}] {q.topic:<12}  {q.query[:55]}")
        if r.missing_terms:
            print(f"            missing: {', '.join(r.missing_terms)}")
        results.append(r)

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(results: list[QuestionResult]) -> None:
    passed = sum(1 for r in results if r.passed)
    total = len(results)

    print()
    print("=" * 65)
    print(f"  Score: {passed}/{total}  ({100 * passed // total}%)")
    print("=" * 65)
    print()
    print(f"| {'#':>2} | {'Topic':<14} | {'Result':<6} | Missing terms")
    print(f"|{'---':->4}|{'':->16}|{'':->8}|{'':->30}")
    for r in results:
        q = r.question
        status = "PASS" if r.passed else "FAIL"
        missing = ", ".join(r.missing_terms) if r.missing_terms else "-"
        print(f"| {q.id:>2} | {q.topic:<14} | {status:<6} | {missing}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RAG accuracy evaluation against a ChromaDB collection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--eval",
        default=str(_ROOT / "evals" / "samsung_tv_eval.xml"),
        help="Path to eval XML file",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Override the collection name from the XML",
    )
    parser.add_argument(
        "--n-results",
        type=int,
        default=5,
        help="Number of search results to retrieve per question",
    )
    args = parser.parse_args()

    results = run_eval(
        eval_file=Path(args.eval),
        collection_override=args.collection,
        n_results=args.n_results,
    )
    print_report(results)

    passed = sum(1 for r in results if r.passed)
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
