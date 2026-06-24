"""Ingest every tenant's docs/ directory into its own isolated vector store.

Usage:
    python -m scripts.ingest_all
"""
from pathlib import Path

from src.config import TENANTS, Settings
from src.ingest import ingest_tenant

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    settings = Settings()
    for tenant_id in TENANTS:
        docs_dir = REPO_ROOT / "data" / tenant_id
        count = ingest_tenant(tenant_id, docs_dir, settings)
        print(f"[{tenant_id}] ingested {count} chunks from {docs_dir}")


if __name__ == "__main__":
    main()
