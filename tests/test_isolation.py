"""Cross-tenant leakage tests.

Both tenants' data lives in ONE shared Qdrant collection (see
src/vector_store.py) — there is no physical separation to fall back on
here, so these tests are actually exercising the tenant_id filter, not a
folder boundary. Run after `python -m scripts.ingest_all` so the shared
store is populated.
"""
import pytest

from src.config import Settings
from src.retriever import _store, retrieve

settings = Settings()


@pytest.mark.parametrize(
    "asking_tenant,forbidden_term,other_only_query",
    [
        ("tenant_a", "horizon", "Horizon Saver Horizon Plus"),
        ("tenant_b", "skyline", "Skyline Saver Skyline Rewards"),
    ],
)
def test_no_cross_tenant_hits(asking_tenant, forbidden_term, other_only_query):
    chunks = retrieve(asking_tenant, other_only_query, settings, top_k=5)
    for chunk in chunks:
        # every hit's tenant_id is already asserted == asking_tenant inside
        # retrieve() (it would raise TenantIsolationViolation otherwise);
        # this also sanity-checks the content itself never mentions the
        # other tenant's product names.
        assert chunk.tenant_id == asking_tenant
        assert forbidden_term not in chunk.text.lower()


def test_both_tenants_are_genuinely_colocated_in_one_collection():
    """Proves this is testing a filter, not a folder boundary: both
    tenants' points really do live in the same collection. Reuses
    retriever.py's cached store (rather than opening a second
    SharedVectorStore) — embedded Qdrant only allows one open handle
    per path at a time, see the [VECTOR_STORE] lock errors this codebase
    has hit twice already when that rule wasn't respected."""
    store = _store(settings)
    assert store.count() == store.count(tenant_id="tenant_a") + store.count(tenant_id="tenant_b")
    assert store.count(tenant_id="tenant_a") > 0
    assert store.count(tenant_id="tenant_b") > 0


def test_unknown_tenant_is_rejected():
    from src.config import validate_tenant

    with pytest.raises(ValueError):
        validate_tenant("tenant_does_not_exist")
