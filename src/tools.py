"""Mock tool stub.

PLACEHOLDER — Westpac said they'll provide the real `get_account_balance`
stub; this one exists so the agent loop is runnable/testable now. Swap the
function body for the real one when it arrives; the schema and the
tenant_id-pinning behaviour in agent.py should not need to change.

Isolation note: tenant_id is a required argument here because the brief
specifies it, but the agent (src/agent.py) never lets the LLM choose it —
it is always overwritten server-side with the tenant_id of the current
session before the tool executes. An LLM cannot be prompt-injected into
fetching another tenant's balance through this tool.
"""
from __future__ import annotations

GET_ACCOUNT_BALANCE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_account_balance",
        "description": "Look up the current balance for one of the current tenant's accounts.",
        "parameters": {
            "type": "object",
            "properties": {
                "tenant_id": {
                    "type": "string",
                    "description": "Tenant identifier. Ignored if it does not match the active session's tenant.",
                },
                "account_id": {
                    "type": "string",
                    "description": "The account identifier to look up, e.g. 'ACC-1001'.",
                },
            },
            "required": ["tenant_id", "account_id"],
        },
    },
}


def get_account_balance(tenant_id: str, account_id: str) -> dict:
    """Fake deterministic balance, scoped by tenant so two tenants never
    collide on the same account_id by coincidence."""
    seed = sum(map(ord, f"{tenant_id}:{account_id}"))
    balance = round(100 + (seed % 5000) + (seed % 97) / 100, 2)
    return {
        "tenant_id": tenant_id,
        "account_id": account_id,
        "balance": balance,
        "currency": "AUD",
    }
