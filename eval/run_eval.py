"""Tiny eval harness.

Reports two required numbers (deliberately simple and free to compute —
no LLM-judge call needed, since the "tiny eval" brief doesn't call for
one):

  1. Retrieval metric: Recall@k — did a chunk from the expected source
     document appear anywhere in the top-k retrieved chunks for that
     question? (k = TOP_K below)
  2. Answer-quality check: keyword containment — does the agent's final
     answer contain the expected keyword(s) (e.g. the correct number/rate)?
     This catches the common RAG failure mode where retrieval succeeds but
     the model still answers from its own (wrong) prior knowledge.

Plus three RAGAS-style metrics, computed via direct embedding cosine
similarity rather than an LLM judge (same reasoning as above — proportional
to a "tiny eval," no extra API spend, no judge-model variance):

  - context_relevance = cosine(question, context) — did retrieval fetch
    relevant context? Eval-only: doesn't depend on the answer, so it's a
    retrieval-quality signal, not a live per-call guardrail signal.
  - answer_relevance  = cosine(question, answer)  — did the answer
    actually address the question? Also computed live as a runtime
    guardrail in src/agent.py; reused here from the same Agent.ask() call.
  - faithfulness      = cosine(context, answer)   — is the answer
    consistent with the retrieved context? Same dual-use as above.

Reported separately per question, not blended into one composite score:
a single number can't show which of the three failed.

Also reports a reranker on/off comparison — retrieval-only (no extra LLM
calls, since reranking only affects retrieval ordering, not generation):
Recall@k with/without, and whether the expected chunk's RANK POSITION
moved, which is a more informative signal than Recall@k alone at this
corpus size (~4-5 docs/tenant), where the expected chunk is almost always
present in top-k regardless — the real question is whether reranking
changes ORDER, not presence. This is the empirical evidence for "implement
reranking and justify your choice," rather than an assertion either way.

Usage:
    python -m eval.run_eval
"""
from __future__ import annotations

import json
from pathlib import Path

from src import vector_store
from src.agent import Agent
from src.citations import cosine_similarity
from src.config import Settings
from src.retriever import close_store, retrieve
from src.trace import configure_console_encoding

TOP_K = 5
REPO_ROOT = Path(__file__).resolve().parent.parent


def _rank_of_expected(chunks, expected_source: str) -> int | None:
    """1-indexed rank position of the first chunk from expected_source, or
    None if it isn't present at all in this result set."""
    for i, c in enumerate(chunks, start=1):
        if c.source == expected_source:
            return i
    return None


def run_rerank_comparison(qa_pairs: list[dict], settings: Settings) -> None:
    print("\n=== Reranker comparison (retrieval-only, no extra LLM calls) ===")
    recall_off = recall_on = 0
    rank_moved = 0
    n = len(qa_pairs)

    for qa in qa_pairs:
        chunks_off = retrieve(qa["tenant_id"], qa["question"], settings, top_k=TOP_K, rerank=False)
        chunks_on = retrieve(qa["tenant_id"], qa["question"], settings, top_k=TOP_K, rerank=True)

        rank_off = _rank_of_expected(chunks_off, qa["expected_source"])
        rank_on = _rank_of_expected(chunks_on, qa["expected_source"])
        recall_off += rank_off is not None
        recall_on += rank_on is not None
        if rank_off != rank_on:
            rank_moved += 1

        print(f"[{qa['id']}] expected={qa['expected_source']!r}  rank_without_rerank={rank_off}  rank_with_rerank={rank_on}")

    print(f"\nRecall@{TOP_K} without rerank: {recall_off}/{n} = {recall_off / n:.2f}")
    print(f"Recall@{TOP_K} with rerank:    {recall_on}/{n} = {recall_on / n:.2f}")
    print(f"Questions where the expected chunk's rank position changed: {rank_moved}/{n}")
    if rank_moved == 0:
        print(
            "No rank changes observed - consistent with our own reasoning that a "
            "~4-5 doc/tenant corpus is too small for reranking to matter here; "
            "kept off by default (RERANK_ENABLED=0) but pluggable rather than "
            "removed entirely, since a larger real corpus is exactly where this "
            "would start to help."
        )


def main() -> None:
    configure_console_encoding()
    settings = Settings()
    qa_pairs = json.loads((REPO_ROOT / "eval" / "qa_pairs.json").read_text())

    recall_hits = 0
    quality_hits = 0
    context_relevances, answer_relevances, faithfulnesses = [], [], []
    rows = []

    for qa in qa_pairs:
        chunks = retrieve(qa["tenant_id"], qa["question"], settings, top_k=TOP_K)
        retrieved_sources = {c.source for c in chunks}
        recall_hit = qa["expected_source"] in retrieved_sources
        recall_hits += recall_hit

        agent = Agent(tenant_id=qa["tenant_id"], settings=settings)
        result = agent.ask(qa["question"])
        answer = result["answer"] or ""
        quality_hit = all(kw.lower() in answer.lower() for kw in qa["expected_keywords"])
        quality_hits += quality_hit

        context_relevance = None
        if chunks and answer:
            context_text = "\n\n".join(c.text for c in chunks)
            vecs = list(vector_store.embed_dense([qa["question"], context_text], settings))
            context_relevance = cosine_similarity(vecs[0], vecs[1])
            context_relevances.append(context_relevance)

        answer_relevance, faithfulness = result["answer_relevance"], result["faithfulness"]
        if answer_relevance is not None:
            answer_relevances.append(answer_relevance)
        if faithfulness is not None:
            faithfulnesses.append(faithfulness)

        rows.append(
            {
                "id": qa["id"],
                "tenant_id": qa["tenant_id"],
                "recall_hit": recall_hit,
                "quality_hit": quality_hit,
                "context_relevance": context_relevance,
                "answer_relevance": answer_relevance,
                "faithfulness": faithfulness,
                "answer": answer,
            }
        )
        print(f"[{qa['id']}] recall_hit={recall_hit} quality_hit={quality_hit}")
        print(
            f"    context_relevance={context_relevance}  answer_relevance={answer_relevance}  "
            f"faithfulness={faithfulness}"
        )
        print(f"    answer: {answer}\n")

    n = len(qa_pairs)
    print(f"Recall@{TOP_K}: {recall_hits}/{n} = {recall_hits / n:.2f}")
    print(f"Answer-quality (keyword containment): {quality_hits}/{n} = {quality_hits / n:.2f}")

    def _avg(values: list[float]) -> str:
        return f"{sum(values) / len(values):.2f}" if values else "n/a"

    print(f"context_relevance (avg, Q<->C): {_avg(context_relevances)}")
    print(f"answer_relevance  (avg, Q<->A): {_avg(answer_relevances)}")
    print(f"faithfulness      (avg, A<->C): {_avg(faithfulnesses)}")

    (REPO_ROOT / "eval" / "results.json").write_text(json.dumps(rows, indent=2))

    run_rerank_comparison(qa_pairs, settings)
    # See retriever.close_store(): closes the cached Qdrant client on this
    # still-live interpreter rather than relying on __del__ during shutdown.
    close_store(settings)


if __name__ == "__main__":
    main()
