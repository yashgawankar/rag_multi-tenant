"""Load a tenant's docs directory, chunk them, and upsert into the shared
vector store, tagged with that tenant's id."""
from __future__ import annotations

import hashlib
from pathlib import Path

from pypdf import PdfReader

from src.chunking import chunk_text
from src.config import Settings
from src.trace import trace
from src.vector_store import SharedVectorStore

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


def _read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8")


def _stable_id(tenant_id: str, source: str, chunk_index: int) -> int:
    digest = hashlib.sha256(f"{tenant_id}:{source}:{chunk_index}".encode()).hexdigest()
    return int(digest[:16], 16)


def ingest_tenant(tenant_id: str, docs_dir: Path, settings: Settings) -> int:
    store = SharedVectorStore(settings=settings)

    points = []
    for path in sorted(docs_dir.iterdir()):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        text = _read_file(path)
        chunks = chunk_text(text)
        trace(f"[INGEST] tenant={tenant_id!r} file={path.name!r} -> {len(chunks)} chunk(s)")
        for chunk in chunks:
            points.append(
                {
                    "id": _stable_id(tenant_id, path.name, chunk.chunk_index),
                    "text": chunk.text,
                    "payload": {
                        "tenant_id": tenant_id,
                        "source": path.name,
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                    },
                }
            )

    if points:
        store.upsert_chunks(points)
    return len(points)
