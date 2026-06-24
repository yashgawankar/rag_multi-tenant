"""Tiny eval harness.

Reports two numbers, deliberately simple and free to compute (no LLM-judge
call needed, since the "tiny eval" brief doesn't call for one):

  1. Retrieval metric: Recall@k — did a chunk from the expected source
     document appear anywhere in the top-k retrieved chunks for that
     question? (k = TOP_K below)
  2. Answer-quality check: keyword containment — does the agent's final
     answer contain the expected keyword(s) (e.g. the correct number/rate)?
     This catches the common RAG failure mode where retrieval succeeds but
     the model still answers from its own (wrong) prior knowledge.

PLACEHOLDER NOTE: eval/qa_pairs.json currently targets the placeholder docs
in data/. Replace both once the real Westpac docs + Q/A pairs arrive.

Usage:
    python -m eval.run_eval
"""
from __future__ import annotations

import json
from pathlib import Path

from src.agent import Agent
from src.config import Settings
from src.retriever import retrieve

TOP_K = 5
REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    settings = Settings()
    qa_pairs = json.loads((REPO_ROOT / "eval" / "qa_pairs.json").read_text())

    recall_hits = 0
    quality_hits = 0
    rows = []

    for qa in qa_pairs:
        chunks = retrieve(qa["tenant_id"], qa["question"], settings, top_k=TOP_K)
        retrieved_sources = {c.source for c in chunks}
        recall_hit = qa["expected_source"] in retrieved_sources
        recall_hits += recall_hit

        agent = Agent(tenant_id=qa["tenant_id"], settings=settings)
        answer = agent.ask(qa["question"])["answer"] or ""
        quality_hit = all(kw.lower() in answer.lower() for kw in qa["expected_keywords"])
        quality_hits += quality_hit

        rows.append(
            {
                "id": qa["id"],
                "tenant_id": qa["tenant_id"],
                "recall_hit": recall_hit,
                "quality_hit": quality_hit,
                "answer": answer,
            }
        )
        print(f"[{qa['id']}] recall_hit={recall_hit} quality_hit={quality_hit}")
        print(f"    answer: {answer}\n")

    n = len(qa_pairs)
    print(f"Recall@{TOP_K}: {recall_hits}/{n} = {recall_hits / n:.2f}")
    print(f"Answer-quality (keyword containment): {quality_hits}/{n} = {quality_hits / n:.2f}")

    (REPO_ROOT / "eval" / "results.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
