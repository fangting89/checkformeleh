"""Eval harness for the RAG pipeline.

Scores the gate and the chain against the golden question set in
eval/dataset.py, deterministically wherever ground truth is exact - no
LLM-judge, same reproducibility reasoning as read-leh's eval harness.

Four things are scored, independently:
1. Gate accuracy: does route_question's decision match each question's
   expect_refusal? The adversarial subset's accuracy is the safety-
   critical number - it's the injection/off-topic catch rate.
2. Retrieval hit-rate: for answerable questions, was a chunk from the
   expected scheme actually retrieved?
3. Keyword pass-rate: do all of a question's expected_keywords appear in
   the generated answer?
4. Hallucination scan (flag, not hard-fail): any dollar figure in the
   answer that doesn't appear anywhere in the retrieved context is
   logged for review, since it has no traceable source.

Usage: `uv run python -m eval.run_eval`
"""

import json
import re
import time
from pathlib import Path
from typing import TypedDict

from langchain_community.vectorstores import Chroma

from eval.dataset import EVAL_QUESTIONS, EvalQuestion
from rag.chain import build_chain, build_retriever, retrieve_context
from rag.gate import route_question
from rag.ingest import build_vectorstore

RESULTS_DIR = Path(__file__).resolve().parent / "results"

AMOUNT_RE = re.compile(r"\$[\d,]+(?:\.\d{2})?")


class QuestionResult(TypedDict):
    """Per-question scoring result.

    Attributes:
        id: The question's id, for pointing at a specific failure.
        category: Which of the 3 test categories this question is in.
        gate_decision: What route_question actually returned.
        gate_correct: Whether that decision matched expect_refusal.
        retrieval_hit: Whether the expected scheme was retrieved. None
            for questions the chain was never run on (declined, or not
            "answerable" category).
        keywords_correct: Whether all expected_keywords appeared in the
            answer. None for the same reason as retrieval_hit.
        unexpected_amounts: Dollar figures in the answer with no
            matching figure anywhere in the retrieved context.
        answer: The generated answer, or None if the chain wasn't run.
    """

    id: str
    category: str
    gate_decision: str
    gate_correct: bool
    retrieval_hit: bool | None
    keywords_correct: bool | None
    unexpected_amounts: list[str]
    answer: str | None


def _score_question(question: EvalQuestion, vectorstore: Chroma) -> QuestionResult:
    """Runs one golden question through gate -> chain and scores it.

    Args:
        question: The golden-set question to score.
        vectorstore: The corpus vector store - a fresh, scheme-scoped
            retriever is built per question from the gate's own decision
            (see rag/chain.py's build_retriever for why: it's what fixed
            two real cross-scheme retrieval mix-ups found during Day 3).

    Returns:
        The per-check scoring result for this question.
    """
    route = route_question(question.question)
    gate_correct = (route["decision"] == "decline") == question.expect_refusal

    result: QuestionResult = {
        "id": question.id,
        "category": question.category,
        "gate_decision": route["decision"],
        "gate_correct": gate_correct,
        "retrieval_hit": None,
        "keywords_correct": None,
        "unexpected_amounts": [],
        "answer": None,
    }

    # Only run the chain on questions that are actually meant to be
    # answered - matches real pipeline behavior, and avoids scoring
    # answer quality on a question the gate itself already got wrong.
    if route["decision"] != "answer" or question.category != "answerable":
        return result

    retriever = build_retriever(vectorstore, scheme=route["category"])
    context_docs = retrieve_context(retriever, question.question)
    chain = build_chain()
    answer = chain.invoke({"context": context_docs, "question": question.question})

    result["answer"] = answer
    result["retrieval_hit"] = any(
        doc.metadata.get("scheme") == question.expected_scheme for doc in context_docs
    )
    result["keywords_correct"] = all(
        keyword.lower() in answer.lower() for keyword in question.expected_keywords
    )

    context_text = "\n".join(doc.page_content for doc in context_docs)
    amounts_in_answer = set(AMOUNT_RE.findall(answer))
    amounts_in_context = set(AMOUNT_RE.findall(context_text))
    result["unexpected_amounts"] = sorted(amounts_in_answer - amounts_in_context)

    return result


def _rate(results: list[QuestionResult], key: str) -> float | None:
    """Computes the pass rate for one boolean check, skipping questions where it's None."""
    values = [r[key] for r in results if r[key] is not None]
    return round(sum(values) / len(values), 3) if values else None


def main() -> None:
    """Runs the full golden set through the pipeline and writes results/latest.json."""
    print(f"Running eval ({len(EVAL_QUESTIONS)} questions)...")
    started = time.monotonic()

    vectorstore = build_vectorstore(persist=False)

    results = [_score_question(q, vectorstore) for q in EVAL_QUESTIONS]
    print(f"  done in {time.monotonic() - started:.1f}s")

    adversarial = [r for r in results if r["category"] == "adversarial"]
    hallucination_flags = sum(1 for r in results if r["unexpected_amounts"])

    report = {
        "total_questions": len(EVAL_QUESTIONS),
        "gate_accuracy": _rate(results, "gate_correct"),
        "adversarial_gate_accuracy": _rate(adversarial, "gate_correct"),
        "retrieval_hit_rate": _rate(results, "retrieval_hit"),
        "keyword_pass_rate": _rate(results, "keywords_correct"),
        "hallucination_flags": hallucination_flags,
        "results": results,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "latest.json"
    out_path.write_text(json.dumps(report, indent=2))

    print("\n=== Summary ===")
    print(f"  gate accuracy (overall):      {report['gate_accuracy']}")
    print(f"  gate accuracy (adversarial):  {report['adversarial_gate_accuracy']}")
    print(f"  retrieval hit-rate:           {report['retrieval_hit_rate']}")
    print(f"  keyword pass-rate:            {report['keyword_pass_rate']}")
    print(f"  hallucination flags:          {hallucination_flags}/{len(results)}")

    failures = [
        r
        for r in results
        if not r["gate_correct"]
        or r["keywords_correct"] is False
        or r["retrieval_hit"] is False
    ]
    if failures:
        print(f"\n  {len(failures)} question(s) with at least one failed check:")
        for r in failures:
            print(f"    - {r['id']} ({r['category']})")

    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
