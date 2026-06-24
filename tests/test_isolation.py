"""Cross-tenant leakage tests.

These ask each tenant questions whose answers only exist in the OTHER
tenant's docs, and assert that nothing from the other tenant comes back.
Run after `python -m scripts.ingest_all` so both stores are populated.
"""
import pytest

from src.config import Settings
from src.retriever import retrieve

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


def test_each_tenant_has_isolated_collection_path():
    from src.config import tenant_storage_path

    path_a = tenant_storage_path("tenant_a", settings)
    path_b = tenant_storage_path("tenant_b", settings)
    assert path_a != path_b
