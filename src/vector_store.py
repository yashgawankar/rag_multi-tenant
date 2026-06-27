"""Shared vector store for all tenants.

Isolation design (see README for full rationale):
  One Qdrant collection, shared by every tenant. Every point's payload
  carries tenant_id, which is the ONLY thing distinguishing one tenant's
  data from another's at the storage layer. Isolation is enforced by:
    - a payload index on tenant_id with is_tenant=True (Qdrant's documented
      multitenancy optimization — note: a no-op in embedded/local mode,
      see _ensure_collection),
    - a mandatory `must` filter on tenant_id applied to every search, at
      every stage of the hybrid query (both prefetches and the final fusion),
    - retriever.py's post-fetch assertion as the last-resort backstop.
  This mirrors how Qdrant is actually used for multi-tenant RAG in
  production (one collection, indexed tenant filter), rather than the
  per-tenant-physical-store design this replaced. The trade-off: isolation
  now depends on the filter being correctly applied on every query path,
  not on physical impossibility. That's why retriever.py is the *only*
  code in this repo allowed to call hybrid_search — one audited choke
  point instead of trusting every call site.

Qdrant's embedded/local mode (QdrantClient(path=...)) needs no server,
no Docker, and no API key — it persists straight to disk. Known limitation
of this mode specifically: payload indexes (including is_tenant) have no
effect locally — confirmed empirically, Qdrant prints a UserWarning and
filtering still works correctly, just via unindexed scan instead of an
optimized lookup. At this corpus size that's not observable; on a real
Qdrant server the index would actually activate.
"""
from __future__ import annotations

from functools import lru_cache

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models

from src.config import Settings, shared_storage_path, validate_tenant
from src.trace import trace

COLLECTION_NAME = "docs"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
TENANT_FIELD = "tenant_id"


@lru_cache(maxsize=1)
def _dense_embedder(model_name: str) -> TextEmbedding:
    return TextEmbedding(model_name=model_name)


@lru_cache(maxsize=1)
def _sparse_embedder(model_name: str) -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name=model_name)


def _tenant_filter(tenant_id: str) -> models.Filter:
    return models.Filter(
        must=[models.FieldCondition(key=TENANT_FIELD, match=models.MatchValue(value=tenant_id))]
    )


class SharedVectorStore:
    """A single Qdrant client + collection shared by every tenant.
    tenant_id is required on every write and every search — there is no
    method on this class that can read or write without one."""

    def __init__(self, settings: Settings):
        path = shared_storage_path(settings)
        trace(f"[VECTOR_STORE] opening shared embedded Qdrant store at path={path!r}")
        self._client = QdrantClient(path=path)
        self._dense = _dense_embedder(settings.embedding_model)
        self._sparse = _sparse_embedder(settings.sparse_embedding_model)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if self._client.collection_exists(COLLECTION_NAME):
            return
        dense_dim = len(next(self._dense.embed(["dimension probe"])))
        self._client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=dense_dim, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={SPARSE_VECTOR_NAME: models.SparseVectorParams()},
        )
        # is_tenant=True is Qdrant's documented multitenancy optimization —
        # clusters one tenant's vectors together on disk for better cache
        # locality. No-op in embedded/local mode (see module docstring);
        # included anyway because this is the real production call.
        self._client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=TENANT_FIELD,
            field_schema=models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD, is_tenant=True),
        )
        trace(f"[VECTOR_STORE] created collection {COLLECTION_NAME!r} with tenant_id payload index (is_tenant=True)")

    def upsert_chunks(self, points: list[dict]) -> None:
        """points: [{"id": int, "text": str, "payload": {...}}]; payload MUST
        include a valid tenant_id — checked here as a last-resort guard
        against untagged data ever entering the shared collection."""
        texts = [p["text"] for p in points]
        dense_vecs = list(self._dense.embed(texts))
        sparse_vecs = list(self._sparse.embed(texts))

        qdrant_points = []
        for point, dense, sparse in zip(points, dense_vecs, sparse_vecs):
            payload = point["payload"]
            validate_tenant(payload.get(TENANT_FIELD))
            qdrant_points.append(
                models.PointStruct(
                    id=point["id"],
                    vector={
                        DENSE_VECTOR_NAME: dense.tolist(),
                        SPARSE_VECTOR_NAME: models.SparseVector(
                            indices=sparse.indices.tolist(),
                            values=sparse.values.tolist(),
                        ),
                    },
                    payload=payload,
                )
            )
        trace(f"[VECTOR_STORE] upserting {len(qdrant_points)} point(s) into shared collection")
        self._client.upsert(collection_name=COLLECTION_NAME, points=qdrant_points)

    def hybrid_search(
        self, query: str, tenant_id: str, top_k: int = 5, score_threshold: float = 0.5
    ) -> list[models.ScoredPoint]:
        """score_threshold is applied to the fused RRF score, not the raw
        dense/sparse scores (those are on incompatible scales — see
        qdrant_client/hybrid/fusion.py). With our settings (no custom
        weights or ranking_constant_k), RRF score per candidate = sum of
        1/(rank+2) across whichever of the dense/sparse lists it appears
        in, zero-indexed rank. Max per-list contribution is 1/(0+2)=0.5,
        so a 0.5 threshold means: only keep candidates that are at least
        as good as ranking #1 in one of the two retrieval methods alone."""
        validate_tenant(tenant_id)
        tenant_filter = _tenant_filter(tenant_id)

        dense_vec = next(self._dense.embed([query]))
        sparse_vec = next(self._sparse.embed([query]))
        trace(
            f"[VECTOR_STORE] embedded query into dense({len(dense_vec)}-dim) + "
            f"sparse({len(sparse_vec.indices)} nonzero terms) vectors; prefetching top {top_k * 4} "
            f"from each (filtered to tenant_id={tenant_id!r}), RRF-fusing, then keeping fused "
            f"score >= {score_threshold} down to top {top_k}"
        )

        result = self._client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=dense_vec.tolist(),
                    using=DENSE_VECTOR_NAME,
                    limit=top_k * 4,
                    filter=tenant_filter,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_vec.indices.tolist(),
                        values=sparse_vec.values.tolist(),
                    ),
                    using=SPARSE_VECTOR_NAME,
                    limit=top_k * 4,
                    filter=tenant_filter,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=tenant_filter,
            score_threshold=score_threshold,
            limit=top_k,
        )
        return result.points

    def count(self, tenant_id: str | None = None) -> int:
        query_filter = _tenant_filter(tenant_id) if tenant_id else None
        return self._client.count(COLLECTION_NAME, count_filter=query_filter).count

    def count_tokens(self, text: str) -> int:
        """Real token count from the actual dense embedding model's
        tokenizer (bge-small truncates silently past 512 tokens — this is
        used as chunking.py's length_fn so chunks never exceed that
        unnoticed, instead of approximating with character count)."""
        return self._dense.token_count([text])
