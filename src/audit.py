"""Structured JSON audit log — one JSONL record per Agent.ask() call.

Separate from src/trace.py: trace() is human-readable, off-by-default,
stdout-only, for development. This module is the durable, parseable,
forensic record — controlled independently via AUDIT=1 (default on), and
designed to answer "what did this tenant's session see, and did
isolation hold" without needing trace() output at all.

Granularity is deliberately one record per Agent.ask() call, not one
record per internal trace() site — a banking reviewer reconstructs a
session's data exposure as a whole, not a sequence of disconnected
low-level events.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.citations import CitationCheck
from src.retriever import RetrievedChunk

_write_lock = threading.Lock()


def _audit_enabled() -> bool:
    return os.environ.get("AUDIT", "1") == "1"


def _audit_dir() -> Path:
    # Read lazily, not at import time, so tests can override AUDIT_DIR
    # via env var without needing to reimport this module.
    return Path(os.environ.get("AUDIT_DIR", "./audit"))


@dataclass
class AuditRecord:
    """Accumulates everything observed during one Agent.ask() call.
    Built incrementally by Agent — it's the only thing with the full
    picture: question, tool calls, retrieved chunks, final answer."""

    tenant_id: str
    question: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    tool_calls: list[dict] = field(default_factory=list)
    retrieved: list[dict] = field(default_factory=list)
    isolation_violations: list[dict] = field(default_factory=list)
    cross_tenant_access_attempts: list[dict] = field(default_factory=list)
    answer: str | None = None
    submit_answer_status: str | None = None  # "ok" | "malformed" | "not_called" | "llm_error"
    submit_answer_error: str | None = None
    citation_checks: list[dict] = field(default_factory=list)
    answer_relevance: float | None = None  # Q<->A
    faithfulness: float | None = None  # A<->C

    def record_tool_call(self, name: str, arguments: dict) -> None:
        self.tool_calls.append({"name": name, "arguments": arguments})

    def record_submit_answer_status(self, status: str, error: str | None = None) -> None:
        self.submit_answer_status = status
        self.submit_answer_error = error

    def record_citation_checks(self, checks: list[CitationCheck]) -> None:
        for c in checks:
            self.citation_checks.append(
                {
                    "source": c.source,
                    "chunk_index": c.chunk_index,
                    "exists": c.exists,
                    "source_mechanism": c.source_mechanism,
                    "claim": c.claim,
                    "token_overlap_score": c.token_overlap_score,
                    "cosine_score": c.cosine_score,
                    "weakly_grounded": c.weakly_grounded,
                }
            )

    def record_guardrail(self, answer_relevance: float | None, faithfulness: float | None) -> None:
        self.answer_relevance = answer_relevance
        self.faithfulness = faithfulness

    def record_retrieved_chunks(self, chunks: list[RetrievedChunk]) -> None:
        for c in chunks:
            self.retrieved.append(
                {
                    "source": c.source,
                    "chunk_index": c.chunk_index,
                    "score": c.score,
                    "tenant_id": c.tenant_id,
                    # if a violation had occurred, retrieve() would have
                    # raised before this chunk ever reached here.
                    "isolation_check": "pass",
                }
            )

    def record_isolation_violation(self, requested_tenant: str, hit_tenant: str, source: str | None) -> None:
        self.isolation_violations.append(
            {"requested_tenant": requested_tenant, "hit_tenant": hit_tenant, "source": source}
        )

    def record_cross_tenant_access_attempt(self, requested_tenant: str, account_id: str) -> None:
        """Distinct from record_isolation_violation: isolation_violations
        means our own retrieval filter failed (should never happen — a
        five-alarm bug). This means mock_tool's own guardrail CORRECTLY
        refused a request for an account belonging to a different tenant —
        no data was leaked, the tool worked exactly as designed. Logged
        separately so the two are never conflated in the audit trail."""
        self.cross_tenant_access_attempts.append(
            {"requested_tenant": requested_tenant, "account_id": account_id}
        )

    def finalize(self, answer: str | None) -> None:
        self.answer = answer

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "tenant_id": self.tenant_id,
            "question": self.question,
            "tool_calls": self.tool_calls,
            "retrieved": self.retrieved,
            "isolation_ok": len(self.isolation_violations) == 0,
            "isolation_violations": self.isolation_violations,
            "cross_tenant_access_attempts": self.cross_tenant_access_attempts,
            "answer": self.answer,
            "submit_answer_status": self.submit_answer_status,
            "submit_answer_error": self.submit_answer_error,
            "citation_checks": self.citation_checks,
            "citations_ok": all(c["exists"] for c in self.citation_checks),
            "answer_relevance": self.answer_relevance,
            "faithfulness": self.faithfulness,
        }


def write_audit_record(record: AuditRecord) -> None:
    """Append one JSON line to today's audit file. No-ops if AUDIT=0.

    Concurrency note: append-mode + an in-process lock is correct for a
    single process (or multiple threads within one). True multi-process
    write-safety (e.g. via portalocker/flock) is intentionally out of
    scope for this assignment's scale — noted here rather than silently
    assumed away.
    """
    if not _audit_enabled():
        return
    audit_dir = _audit_dir()
    audit_dir.mkdir(parents=True, exist_ok=True)
    path = audit_dir / f"{datetime.now(timezone.utc):%Y-%m-%d}.jsonl"
    line = json.dumps(record.to_dict(), ensure_ascii=False)
    with _write_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
