"""Retrieval entry point used by the agent.

This module is the single choke point through which all document retrieval
flows, and it is where Layer 3 of the isolation design lives: an explicit
runtime assertion that every hit returned actually belongs to the requested
tenant. Layers 1-2 (separate on-disk stores + payload tenant_id, see
vector_store.py) should make a cross-tenant hit structurally impossible —
this assertion exists so that *if* that ever stops being true (a bug, a
future refactor to shared infra, a copy-paste error), retrieval fails loudly
instead of silently leaking a tenant's content into another tenant's answer.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from src.config import Settings
from src.vector_store import TenantStore


class TenantIsolationViolation(RuntimeError):
    """Raised if a retrieved hit's tenant_id does not match the requested tenant."""


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source: str
    chunk_index: int
    score: float
    tenant_id: str


@lru_cache(maxsize=None)
def _store_for(tenant_id: str, settings: Settings) -> TenantStore:
    return TenantStore(tenant_id=tenant_id, settings=settings)


def retrieve(tenant_id: str, query: str, settings: Settings, top_k: int = 5) -> list[RetrievedChunk]:
    store = _store_for(tenant_id, settings)
    hits = store.hybrid_search(query, top_k=top_k)

    chunks: list[RetrievedChunk] = []
    for hit in hits:
        payload = hit.payload or {}
        hit_tenant = payload.get("tenant_id")
        if hit_tenant != tenant_id:
            raise TenantIsolationViolation(
                f"Retrieval for tenant_id={tenant_id!r} returned a chunk tagged "
                f"tenant_id={hit_tenant!r} (source={payload.get('source')!r}). "
                "Aborting rather than passing cross-tenant content to the LLM."
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
    return chunks
