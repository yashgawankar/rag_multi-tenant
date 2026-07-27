"""Wraps the real, provided `mock_tool.get_account_balance` — its function
signature and error contract (UnknownTenantError / AccountNotFoundError /
CrossTenantAccessError) are used exactly as given, per "do not change its
signature or error behaviour."

Isolation note: `mock_tool.TOOL_SCHEMA` includes `tenant_id` as a parameter
the model would supply — but GET_ACCOUNT_BALANCE_SCHEMA below deliberately
does NOT. The schema we show the model and the function we actually call
don't have to match 1:1. The model is never asked for a tenant_id, never
generates one, and has no way to think about other tenants through this
tool at all. `agent.py` supplies the real tenant_id itself, from the
session, when it calls the function — this is a schema-design decision on
our side, not a change to the provided tool's signature or behaviour.

CrossTenantAccessError specifically is the real tool's deliberate guardrail
test (per its own docstring): it fires when a correctly-tenant-pinned
request asks for an account that belongs to a different tenant. See
agent.py's exception handling for how this is caught and logged.
"""
from __future__ import annotations

from mock_tool import (  # noqa: F401 - re-exported for callers
    AccountNotFoundError,
    CrossTenantAccessError,
    ToolError,
    UnknownTenantError,
    get_account_balance,
)

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
