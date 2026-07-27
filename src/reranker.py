"""Pluggable cross-encoder reranker (FlashRank, ms-marco-TinyBERT-L-2-v2).

Confirmed lightweight before adopting: `pip show flashrank` lists numpy,
onnxruntime, requests, tokenizers, tqdm as its only dependencies — every
one of those is already pulled in by fastembed, which this project already
depends on for embeddings. No torch, no new heavy ML framework, ~3MB model.

Pluggable rather than always-on specifically so eval/run_eval.py can run
the same 5 questions with reranking on vs. off and report a real,
measured before/after comparison — rather than asserting whether it helps
at this corpus size (~4-5 docs/tenant), the same empirical-over-assumed
approach used for score_threshold and the citation-grounding thresholds
elsewhere in this codebase.

Reranking operates ONLY on an already-isolation-verified candidate list —
it never touches which tenant's data enters the pipeline, only the order
(and truncation) of chunks already confirmed safe by retriever.py's
per-hit tenant_id assertion. It cannot become an isolation bypass.
"""
from __future__ import annotations

from functools import lru_cache

from flashrank import Ranker, RerankRequest

from src.retriever import RetrievedChunk

RERANK_CANDIDATE_MULTIPLIER = 4  # fetch this many x top_k before reranking down


@lru_cache(maxsize=1)
def _ranker() -> Ranker:
    return Ranker(cache_dir="./storage/flashrank_cache")


def rerank(query: str, chunks: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
    if not chunks:
        return chunks

    passages = [{"id": i, "text": c.text} for i, c in enumerate(chunks)]
    request = RerankRequest(query=query, passages=passages)
    results = _ranker().rerank(request)  # already sorted best-first by FlashRank

    reordered = []
    for r in results[:top_n]:
        chunk = chunks[r["id"]]
        # Replace the RRF fusion score with the reranker's own relevance
        # score (cast from numpy.float32 to a native float — same
        # JSON-serialization lesson as citations.cosine_similarity).
        reordered.append(
            RetrievedChunk(
                text=chunk.text,
                source=chunk.source,
                chunk_index=chunk.chunk_index,
                score=float(r["score"]),
                tenant_id=chunk.tenant_id,
            )
        )
    return reordered
