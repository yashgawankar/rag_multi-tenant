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


def tenant_storage_path(tenant_id: str, settings: Settings) -> str:
    """Each tenant gets a separate on-disk Qdrant store — physical isolation,
    not a shared collection with a filter. See README 'Isolation' section."""
    if tenant_id not in TENANTS:
        raise ValueError(f"Unknown tenant_id: {tenant_id!r}. Known tenants: {TENANTS}")
    return os.path.join(settings.qdrant_storage_root, tenant_id)
