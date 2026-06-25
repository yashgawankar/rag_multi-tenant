"""Mock tool stub.

PLACEHOLDER — Westpac said they'll provide the real `get_account_balance`
stub; this one exists so the agent loop is runnable/testable now. Swap the
function body for the real one when it arrives.

Isolation note: the real Python function takes `tenant_id`, because the
brief's stub signature requires it. But GET_ACCOUNT_BALANCE_SCHEMA below —
the JSON description handed to the LLM — deliberately does NOT include
tenant_id as a parameter. The schema we show the model and the function we
actually call don't have to match 1:1. The model is never asked for a
tenant_id, never generates one, and has no way to think about other
tenants through this tool at all — it's not "the model proposes one and we
override it," there's simply nothing for it to propose. `agent.py` supplies
the real tenant_id itself, from the session, when it calls the function.
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
                "account_id": {
                    "type": "string",
                    "description": "The account identifier to look up, e.g. 'ACC-1001'.",
                },
            },
            "required": ["account_id"],
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
