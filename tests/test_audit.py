"""Tests for the structured JSON audit log (src/audit.py).

AUDIT_DIR is read lazily inside write_audit_record, not at import time,
specifically so these tests can redirect writes to a tmp directory via
monkeypatch.setenv without needing to reimport the module.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from src.agent import Agent
from src.audit import AuditRecord


def _read_last_record(audit_dir: Path) -> dict:
    path = audit_dir / f"{datetime.now(timezone.utc):%Y-%m-%d}.jsonl"
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return json.loads(lines[-1])


def test_ask_writes_a_complete_audit_record(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIT", "1")
    monkeypatch.setenv("AUDIT_DIR", str(tmp_path))

    agent = Agent(tenant_id="tenant_a")
    result = agent.ask("What is the bonus interest rate on the Skyline Saver account?")

    record = _read_last_record(tmp_path)
    assert record["tenant_id"] == "tenant_a"
    assert record["question"] == "What is the bonus interest rate on the Skyline Saver account?"
    assert record["isolation_ok"] is True
    assert record["isolation_violations"] == []
    assert record["answer"] == result["answer"]
    assert len(record["retrieved"]) > 0
    for chunk in record["retrieved"]:
        assert chunk["isolation_check"] == "pass"
        assert chunk["tenant_id"] == "tenant_a"
        assert {"source", "chunk_index", "score"} <= chunk.keys()
    assert any(call["name"] == "search_docs" for call in record["tool_calls"])
    assert record["submit_answer_status"] == "ok"
    assert record["submit_answer_error"] is None
    assert record["citations_ok"] is True
    assert isinstance(record["citation_checks"], list)
    assert record["answer_relevance"] is not None
    assert record["faithfulness"] is not None
    assert isinstance(record["answer_relevance"], float)
    assert isinstance(record["faithfulness"], float)


def test_audit_disabled_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIT", "0")
    monkeypatch.setenv("AUDIT_DIR", str(tmp_path))

    Agent(tenant_id="tenant_a").ask("What is the annual fee on the Skyline Rewards credit card?")

    assert list(tmp_path.iterdir()) == []


def test_isolation_violation_serializes_correctly():
    """Cheaper and more deterministic than provoking a real
    TenantIsolationViolation through the live filter (which the rest of
    this codebase is specifically designed to make very hard to trigger)
    — directly verifies the worst-case path's data shape instead."""
    record = AuditRecord(tenant_id="tenant_a", question="some question")
    record.record_isolation_violation(
        requested_tenant="tenant_a", hit_tenant="tenant_b", source="credit_card_rewards.md"
    )
    record.finalize(None)

    as_dict = record.to_dict()
    assert as_dict["isolation_ok"] is False
    assert as_dict["isolation_violations"] == [
        {"requested_tenant": "tenant_a", "hit_tenant": "tenant_b", "source": "credit_card_rewards.md"}
    ]


def test_malformed_submit_answer_round_trips_correctly():
    """Same rationale as the isolation-violation test above: verify the
    unhappy-path data shape directly rather than relying on a real model
    to misbehave on demand."""
    record = AuditRecord(tenant_id="tenant_a", question="some question")
    record.record_submit_answer_status("malformed", "KeyError: 'answer'")
    record.finalize("I wasn't able to format a response correctly. Please try rephrasing your question.")

    as_dict = record.to_dict()
    assert as_dict["submit_answer_status"] == "malformed"
    assert as_dict["submit_answer_error"] == "KeyError: 'answer'"
    assert as_dict["citation_checks"] == []
    assert as_dict["citations_ok"] is True  # all() on an empty list — no citations means nothing failed
    assert as_dict["answer_relevance"] is None
    assert as_dict["faithfulness"] is None
