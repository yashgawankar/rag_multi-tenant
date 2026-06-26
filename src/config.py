import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

TENANTS = ("tenant_a", "tenant_b")


@dataclass(frozen=True)
class Settings:
    llm_base_url: str = os.environ["LLM_BASE_URL"]
    llm_api_key: str = os.environ["LLM_API_KEY"]
    llm_model: str = os.environ["LLM_MODEL"]

    qdrant_storage_root: str = os.environ.get("QDRANT_STORAGE_ROOT", "./storage/qdrant")
    embedding_model: str = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    sparse_embedding_model: str = os.environ.get("SPARSE_EMBEDDING_MODEL", "Qdrant/bm25")


def validate_tenant(tenant_id: str) -> None:
    """Stand-in for a real tenant registry lookup (see README). Every
    code path that accepts a tenant_id should route through this check
    before it touches the vector store."""
    if tenant_id not in TENANTS:
        raise ValueError(f"Unknown tenant_id: {tenant_id!r}. Known tenants: {TENANTS}")


def shared_storage_path(settings: Settings) -> str:
    """One Qdrant store shared by all tenants — see README 'Isolation
    approach' for why this replaced the earlier per-tenant-path design.
    Isolation is enforced by a mandatory payload filter, not by physical
    separation."""
    return os.path.join(settings.qdrant_storage_root, "shared")
