"""Toggleable flow tracing.

Set TRACE=1 in .env (or the environment) to print [TAG] lines showing every
point where control flow branches: LLM decides tool-vs-answer, which tool
gets dispatched, the tenant-pin override firing, and each retrieval hit's
isolation check. Off by default so normal runs (and the eval harness) stay
quiet; flip it on when you want to see the mechanics live.
"""
import os

TRACE_ENABLED = os.environ.get("TRACE", "0") == "1"


def trace(message: str) -> None:
    if TRACE_ENABLED:
        print(message)
