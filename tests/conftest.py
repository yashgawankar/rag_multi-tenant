"""Shared test fixtures.

src.retriever._store is an lru_cache(maxsize=1) singleton holding an open
embedded-Qdrant client. Embedded mode only allows one open client per
storage path at a time (a real constraint this test suite has hit
repeatedly: test_isolation.py, the model bake-off, and now cross-test-file
ordering between test_audit.py and test_ingest.py — whichever test runs
first populates the cache and keeps it alive for the rest of the pytest
session, colliding with any other test that opens its own fresh
SharedVectorStore against the same path, e.g. via ingest_tenant).

Clearing the cache after every test guarantees no test leaves a stale
open client behind for the next one to collide with — without this,
test pass/fail becomes dependent on alphabetical file ordering, which is
exactly the kind of fragility a real test suite shouldn't have.
"""
import pytest


@pytest.fixture(autouse=True)
def _clear_shared_store_cache():
    yield
    from src.retriever import _store

    _store.cache_clear()
