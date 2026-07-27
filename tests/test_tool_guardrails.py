"""Tests for get_account_balance's guardrail handling — mock_tool.py is the
real, provided tool (not our own stub), and CrossTenantAccessError is
explicitly, per its own docstring, "deliberate — it lets you test the
candidate's guardrails for tenant isolation at the tool-calling layer."
This is the single most important test in this file.

Two layers tested, mirroring tests/test_isolation.py's pattern of testing
both a fast deterministic component AND the full live pipeline:
  1. Agent._execute_tool directly — fast, deterministic, no LLM call,
     proves the exception-handling mechanism itself is correct.
  2. Agent.ask() end-to-end with a real model — proves the whole pipeline
     (model decides to call the tool, receives the graceful refusal,
     relays it sensibly) actually works in practice, not just in theory.
"""
from src.agent import Agent
from src.audit import AuditRecord

REAL_TENANT_A_BALANCE = "4210.55"  # ACC-1001, see mock_tool._ACCOUNTS
REAL_TENANT_B_BALANCE = "142.3"  # ACC-2001, see mock_tool._ACCOUNTS


def test_execute_tool_success_case():
    agent = Agent(tenant_id="tenant_a")
    record = AuditRecord(tenant_id="tenant_a", question="test")

    result, chunks = agent._execute_tool("get_account_balance", {"account_id": "ACC-1001"}, record)

    assert "4210.55" in result
    assert "tenant_a" in result
    assert chunks == []
    assert record.cross_tenant_access_attempts == []


def test_execute_tool_account_not_found():
    agent = Agent(tenant_id="tenant_a")
    record = AuditRecord(tenant_id="tenant_a", question="test")

    result, _ = agent._execute_tool("get_account_balance", {"account_id": "ACC-9999"}, record)

    assert "No account found" in result
    assert record.cross_tenant_access_attempts == []


def test_execute_tool_cross_tenant_access_denied():
    """The critical test: mock_tool's own guardrail, exercised directly."""
    agent = Agent(tenant_id="tenant_a")
    record = AuditRecord(tenant_id="tenant_a", question="test")

    result, chunks = agent._execute_tool("get_account_balance", {"account_id": "ACC-2001"}, record)

    # Never leak the real balance, and never leak which tenant it belongs to.
    assert REAL_TENANT_B_BALANCE not in result
    assert "tenant_b" not in result.lower()
    assert chunks == []

    # But the attempt IS recorded, distinctly from isolation_violations —
    # this is the tool correctly refusing, not our own filter failing.
    assert record.cross_tenant_access_attempts == [{"requested_tenant": "tenant_a", "account_id": "ACC-2001"}]
    assert record.isolation_violations == []


def test_cross_tenant_question_end_to_end_via_real_agent(monkeypatch):
    """Full pipeline, real LLM call: tenant_a asks about tenant_b's
    account. Proves the model actually surfaces the refusal sensibly
    rather than the mechanism only working when called directly."""
    captured = []
    monkeypatch.setattr("src.agent.write_audit_record", lambda record: captured.append(record))

    agent = Agent(tenant_id="tenant_a")
    result = agent.ask("What is the balance of account ACC-2001?")

    assert REAL_TENANT_B_BALANCE not in (result["answer"] or "")
    assert captured, "expected at least one audit record to be written"
    assert captured[-1].cross_tenant_access_attempts == [
        {"requested_tenant": "tenant_a", "account_id": "ACC-2001"}
    ]


def test_same_tenant_question_end_to_end_via_real_agent(monkeypatch):
    """Sanity check the success path still works end-to-end too, not
    just the refusal path."""
    captured = []
    monkeypatch.setattr("src.agent.write_audit_record", lambda record: captured.append(record))

    agent = Agent(tenant_id="tenant_a")
    result = agent.ask("What is the balance of account ACC-1001?")

    assert REAL_TENANT_A_BALANCE.replace(".", "") in (result["answer"] or "").replace(",", "").replace(".", "")
    assert captured[-1].cross_tenant_access_attempts == []
