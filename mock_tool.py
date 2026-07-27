"""
Mock structured-data tool for the RAG take-home assignment.

The agent should call `get_account_balance(tenant_id, account_id)` when a question
requires account-level structured data rather than document retrieval.

This is intentionally a simple, dependency-free, deterministic stub. Do NOT change
the function signature or the error contract — the live follow-up session relies on it.

Behaviour summary
-----------------
- Returns a balance dict for a known (tenant_id, account_id) pair.
- Raises AccountNotFoundError for an unknown account_id within a valid tenant.
- Raises CrossTenantAccessError if the account_id belongs to a DIFFERENT tenant
  (this is deliberate — it lets you test the candidate's guardrails for tenant
  isolation at the tool-calling layer).
- Raises UnknownTenantError for an unrecognised tenant_id.
"""

from __future__ import annotations


class ToolError(Exception):
    """Base class for all mock-tool errors."""


class UnknownTenantError(ToolError):
    pass


class AccountNotFoundError(ToolError):
    pass


class CrossTenantAccessError(ToolError):
    pass


# Deterministic fake data. account_id -> record.
_ACCOUNTS = {
    # Acme Bank (tenant_a)
    "ACC-1001": {"tenant_id": "tenant_a", "balance": 4210.55, "currency": "AUD", "type": "Acme Everyday"},
    "ACC-1002": {"tenant_id": "tenant_a", "balance": 18999.00, "currency": "AUD", "type": "Acme Saver"},
    "ACC-1003": {"tenant_id": "tenant_a", "balance": -320.10, "currency": "AUD", "type": "Acme Everyday"},
    # Globex Insurance (tenant_b) — "balance" here represents premium owing
    "ACC-2001": {"tenant_id": "tenant_b", "balance": 142.30, "currency": "AUD", "type": "Globex Motor premium"},
    "ACC-2002": {"tenant_id": "tenant_b", "balance": 0.00, "currency": "AUD", "type": "Globex Home premium"},
}

_VALID_TENANTS = {"tenant_a", "tenant_b"}


def get_account_balance(tenant_id: str, account_id: str) -> dict:
    """Return balance info for an account, enforcing tenant ownership.

    Args:
        tenant_id: The authenticated tenant making the request.
        account_id: The account to look up.

    Returns:
        dict with keys: account_id, tenant_id, balance, currency, type.

    Raises:
        UnknownTenantError: tenant_id is not recognised.
        AccountNotFoundError: account_id does not exist.
        CrossTenantAccessError: account_id exists but belongs to another tenant.
    """
    if tenant_id not in _VALID_TENANTS:
        raise UnknownTenantError(f"Unknown tenant_id: {tenant_id!r}")

    record = _ACCOUNTS.get(account_id)
    if record is None:
        raise AccountNotFoundError(f"No account {account_id!r} for tenant {tenant_id!r}")

    if record["tenant_id"] != tenant_id:
        raise CrossTenantAccessError(
            f"Account {account_id!r} does not belong to tenant {tenant_id!r}"
        )

    return {"account_id": account_id, **record}


# JSON schema for the tool, if the candidate wants to register it for function calling.
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_account_balance",
        "description": "Get the current balance/premium-owing for an account belonging to the tenant.",
        "parameters": {
            "type": "object",
            "properties": {
                "tenant_id": {"type": "string", "description": "Authenticated tenant id"},
                "account_id": {"type": "string", "description": "Account id, e.g. ACC-1001"},
            },
            "required": ["tenant_id", "account_id"],
        },
    },
}


if __name__ == "__main__":
    # Quick self-check.
    print(get_account_balance("tenant_a", "ACC-1001"))
    for tid, aid in [("tenant_a", "ACC-9999"), ("tenant_a", "ACC-2001"), ("tenant_x", "ACC-1001")]:
        try:
            get_account_balance(tid, aid)
        except ToolError as e:
            print(f"{type(e).__name__}: {e}")
 