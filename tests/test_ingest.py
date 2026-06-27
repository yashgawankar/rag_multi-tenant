"""Regression test for a real bug found during development: re-running
ingestion after a source file was edited or deleted left its old chunks
behind forever, since upsert only adds/overwrites — it never removes a
point whose ID no longer corresponds to anything on disk. Reproduced by
hand (deleted a test doc, re-ingested, the old chunk was still
retrievable), fixed via SharedVectorStore.delete_tenant_data, called at
the start of ingest_tenant.

This test temporarily replaces tenant_a's real docs with a throwaway
directory to prove the point, then restores the real corpus in a
`finally` block — TENANTS is hardcoded to exactly tenant_a/tenant_b
(see src/config.py), so there's no spare fake tenant to use without
risking leaving other tests' fixture data corrupted if this test fails
mid-way.

Note: never hold a SharedVectorStore reference alive across an
ingest_tenant() call — it opens its own client internally, and embedded
Qdrant only allows one open handle per path at a time (the same lock
this codebase has hit before). Each verification below opens its own
short-lived store, used and discarded before the next ingest_tenant call.
"""
import tempfile
from pathlib import Path

from src.config import Settings
from src.ingest import ingest_tenant
from src.vector_store import SharedVectorStore

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_TENANT_A_DOCS = REPO_ROOT / "data" / "tenant_a"

settings = Settings()


def _tenant_a_sources_and_count() -> tuple[set, int]:
    store = SharedVectorStore(settings)
    points = store._client.scroll(collection_name="docs", limit=100, with_payload=True)[0]
    sources = {p.payload["source"] for p in points if p.payload["tenant_id"] == "tenant_a"}
    return sources, store.count(tenant_id="tenant_a")


def test_reingesting_removes_chunks_for_deleted_or_edited_files():
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)

            (tmp_dir / "doc_one.md").write_text("First throwaway document for this test.")
            (tmp_dir / "doc_two.md").write_text("Second throwaway document for this test.")
            ingest_tenant("tenant_a", tmp_dir, settings)

            sources, count = _tenant_a_sources_and_count()
            assert sources == {"doc_one.md", "doc_two.md"}
            assert count == 2

            # Simulate "doc_two.md was deleted" by re-ingesting a directory
            # that no longer contains it.
            (tmp_dir / "doc_two.md").unlink()
            ingest_tenant("tenant_a", tmp_dir, settings)

            sources, count = _tenant_a_sources_and_count()
            assert sources == {"doc_one.md"}, (
                f"Expected only doc_one.md to remain for tenant_a after doc_two.md "
                f"was deleted from disk, but found: {sources}"
            )
            assert count == 1
    finally:
        # Restore the real corpus regardless of pass/fail, so other tests
        # (and a human re-running the suite) see the actual documents again.
        ingest_tenant("tenant_a", REAL_TENANT_A_DOCS, settings)
