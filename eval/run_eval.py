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

PLACEHOLDER NOTE: eval/qa_pairs.json currently targets the placeholder docs
in data/. Replace both once the real Westpac docs + Q/A pairs arrive.

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
from src.retriever import retrieve

TOP_K = 5
REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
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


if __name__ == "__main__":
    main()
