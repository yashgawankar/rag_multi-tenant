"""Retrieval entry point used by the agent — the ONLY code in this repo
allowed to query the shared vector store.

Isolation now rests on two things, both inside this one audited path:
  1. vector_store.py applies a mandatory `must` filter on tenant_id to
     every stage of the hybrid search (both prefetches and the fusion).
  2. This module re-checks every single hit's tenant_id payload against
     the requested tenant before it's allowed anywhere near the LLM
     context, and raises TenantIsolationViolation if they ever disagree.

Since the data physically lives in one shared collection now (see
vector_store.py for why — mirrors a real production multitenancy pattern
rather than a toy per-tenant-store design), the filter in (1) is what
actually keeps tenants apart, not physical separation. That makes the
assertion in (2) more than a defensive afterthought: it is the backstop
for the one thing that could realistically go wrong — a future code path
that calls hybrid_search without going through this function, or a typo'd
filter. By making this the *only* function that calls hybrid_search at
all (nothing else in the codebase imports vector_store directly), "did
every call site remember the filter" stops being a fleet-wide question
and becomes a property of one function, which this module's own tests
(tests/test_isolation.py) verify directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from src.config import Settings
from src.trace import trace
from src.vector_store import SharedVectorStore

# reranker imported lazily inside retrieve() instead of at module level —
# reranker.py imports flashrank, and retriever.py should stay importable
# (and its own tests runnable) even in an environment where flashrank
# isn't installed, since reranking is an optional, off-by-default
# pluggable feature, not a hard dependency of retrieval itself.


class TenantIsolationViolation(RuntimeError):
    """Raised if a retrieved hit's tenant_id does not match the requested
    tenant. Carries structured attributes (not just a message string) so
    callers like src/audit.py can record violation detail without
    parsing exception text."""

    def __init__(self, message: str, requested_tenant: str, hit_tenant: str | None, source: str | None):
        super().__init__(message)
        self.requested_tenant = requested_tenant
        self.hit_tenant = hit_tenant
        self.source = source


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source: str
    chunk_index: int
    score: float
    tenant_id: str


@lru_cache(maxsize=1)
def _store(settings: Settings) -> SharedVectorStore:
    return SharedVectorStore(settings=settings)


def close_store(settings: Settings) -> None:
    """Explicitly close the cached SharedVectorStore's Qdrant client, if
    one was ever created for this settings (a no-op otherwise — a chat
    session that only ever called get_account_balance never touches the
    vector store at all).

    Call this at normal process exit (see scripts/chat.py, eval/run_eval.py)
    instead of letting the process fall through to interpreter shutdown.
    QdrantClient itself defines a __del__ that calls close() as a
    last-resort safety net, but @lru_cache(maxsize=1) on _store keeps this
    object alive for the whole process, so __del__ only ever fires during
    interpreter teardown — by which point some stdlib internals (e.g.
    sys.meta_path) are already gone, and close() raises an ImportError
    from inside __del__. Python can't propagate an exception out of a
    finalizer, so it prints "Exception ignored in: ..." to stderr instead
    and moves on; harmless (no corrupted state, no nonzero exit code), but
    alarming to see after pressing Ctrl+C. Closing here, on a live
    interpreter, avoids relying on __del__ at all.
    """
    if _store.cache_info().currsize == 0:
        return
    _store(settings).close()
    _store.cache_clear()


def retrieve(
    tenant_id: str,
    query: str,
    settings: Settings,
    top_k: int = 5,
    score_threshold: float = 0.5,
    rerank: bool | None = None,
) -> list[RetrievedChunk]:
    """rerank: None (default) reads Settings.rerank_enabled; True/False
    overrides it explicitly — used by eval/run_eval.py to run the same
    questions with reranking on vs. off for a measured comparison, rather
    than asserting whether it helps at this corpus size."""
    use_rerank = settings.rerank_enabled if rerank is None else rerank
    fetch_k = top_k
    if use_rerank:
        from src.reranker import RERANK_CANDIDATE_MULTIPLIER

        fetch_k = top_k * RERANK_CANDIDATE_MULTIPLIER
    trace(
        f"[RETRIEVER] tenant={tenant_id!r} query={query!r} top_k={top_k} "
        f"score_threshold={score_threshold} rerank={use_rerank}"
    )

    store = _store(settings)
    hits = store.hybrid_search(query, tenant_id=tenant_id, top_k=fetch_k, score_threshold=score_threshold)
    trace(f"[RETRIEVER] hybrid_search (filtered to tenant_id={tenant_id!r}) returned {len(hits)} hit(s)")

    chunks: list[RetrievedChunk] = []
    for hit in hits:
        payload = hit.payload or {}
        hit_tenant = payload.get("tenant_id")
        if hit_tenant != tenant_id:
            trace(f"[RETRIEVER] !! ISOLATION VIOLATION !! hit tenant_id={hit_tenant!r} != requested {tenant_id!r}")
            raise TenantIsolationViolation(
                f"Retrieval for tenant_id={tenant_id!r} returned a chunk tagged "
                f"tenant_id={hit_tenant!r} (source={payload.get('source')!r}). "
                "The vector store's tenant_id filter should have excluded this — "
                "aborting rather than passing cross-tenant content to the LLM.",
                requested_tenant=tenant_id,
                hit_tenant=hit_tenant,
                source=payload.get("source"),
            )
        trace(
            f"[RETRIEVER]   ok: source={payload.get('source')!r} chunk={payload.get('chunk_index')} "
            f"score={hit.score:.3f} tenant_id_check=PASS"
        )
        chunks.append(
            RetrievedChunk(
                text=payload.get("text", ""),
                source=payload.get("source", "unknown"),
                chunk_index=payload.get("chunk_index", -1),
                score=hit.score,
                tenant_id=hit_tenant,
            )
        )

    if use_rerank and chunks:
        # fetch_k > top_k only in this branch (see above), so this is the
        # step that actually truncates back down to top_k; when reranking
        # is off, hybrid_search already returned at most top_k directly.
        from src.reranker import rerank as rerank_fn

        chunks = rerank_fn(query, chunks, top_n=top_k)
        trace(
            f"[RETRIEVER] reranked down to top {len(chunks)}: "
            + ", ".join(f"{c.source}#{c.chunk_index}={c.score:.3f}" for c in chunks)
        )

    return chunks
