"""Per-tenant vector store.

Isolation design (see README for full rationale):
  Layer 1 - PHYSICAL: each tenant gets its own embedded Qdrant storage
            directory (a separate on-disk database, not a shared one).
            A TenantStore for tenant_a literally holds no client handle
            that could ever reach tenant_b's files.
  Layer 2 - LOGICAL (redundant, defence-in-depth): every point payload
            still carries tenant_id, so the data is self-describing even
            if it were ever migrated into a shared collection later.

Qdrant's embedded/local mode (QdrantClient(path=...)) needs no server,
no Docker, and no API key — it persists straight to disk, which is why it
was chosen over a hosted/server deployment for this exercise.
"""
from __future__ import annotations

from functools import lru_cache

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models

from src.config import Settings, tenant_storage_path

COLLECTION_NAME = "docs"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


@lru_cache(maxsize=1)
def _dense_embedder(model_name: str) -> TextEmbedding:
    return TextEmbedding(model_name=model_name)


@lru_cache(maxsize=1)
def _sparse_embedder(model_name: str) -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name=model_name)


class TenantStore:
    """A Qdrant client scoped to exactly one tenant's on-disk storage path."""

    def __init__(self, tenant_id: str, settings: Settings):
        self.tenant_id = tenant_id
        self._settings = settings
        self._client = QdrantClient(path=tenant_storage_path(tenant_id, settings))
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

    def upsert_chunks(self, points: list[dict]) -> None:
        """points: [{"id": int, "text": str, "payload": {...}}]; payload MUST
        include tenant_id (asserted here as a last-resort guard against a
        caller accidentally mixing tenants during ingestion)."""
        texts = [p["text"] for p in points]
        dense_vecs = list(self._dense.embed(texts))
        sparse_vecs = list(self._sparse.embed(texts))

        qdrant_points = []
        for point, dense, sparse in zip(points, dense_vecs, sparse_vecs):
            payload = point["payload"]
            if payload.get("tenant_id") != self.tenant_id:
                raise ValueError(
                    f"Refusing to upsert: payload tenant_id={payload.get('tenant_id')!r} "
                    f"does not match store tenant {self.tenant_id!r}"
                )
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
        self._client.upsert(collection_name=COLLECTION_NAME, points=qdrant_points)

    def hybrid_search(self, query: str, top_k: int = 5) -> list[models.ScoredPoint]:
        dense_vec = next(self._dense.embed([query]))
        sparse_vec = next(self._sparse.embed([query]))

        result = self._client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=dense_vec.tolist(),
                    using=DENSE_VECTOR_NAME,
                    limit=top_k * 4,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_vec.indices.tolist(),
                        values=sparse_vec.values.tolist(),
                    ),
                    using=SPARSE_VECTOR_NAME,
                    limit=top_k * 4,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
        )
        return result.points

    def count(self) -> int:
        return self._client.count(COLLECTION_NAME).count
