"""Toggleable flow tracing.

Set TRACE=1 in .env (or the environment) to print [TAG] lines showing every
point where control flow branches: LLM decides tool-vs-answer, which tool
gets dispatched, the tenant-pin override firing, and each retrieval hit's
isolation check. Off by default so normal runs (and the eval harness) stay
quiet; flip it on when you want to see the mechanics live.
"""
import os
import sys

TRACE_ENABLED = os.environ.get("TRACE", "0") == "1"


def configure_console_encoding() -> None:
    """Call once, at the top of an entry point (scripts/chat.py,
    eval/run_eval.py) — makes ALL print() calls in that process tolerant
    of Unicode characters a Windows console's codepage can't render (smart
    quotes, em-dashes, non-breaking hyphens — real model output routinely
    contains these). trace()'s own try/except below only protects trace()
    calls; this protects everything, including run_eval.py's own
    print(answer) on a live model's actual generated text, which is what
    actually crashed a real eval run before this fix existed."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")


def trace(message: str) -> None:
    if not TRACE_ENABLED:
        return
    try:
        print(message)
    except UnicodeEncodeError:
        # Windows consoles (cp1252 etc.) can't render arbitrary Unicode —
        # and message often contains raw model output (repr'd), which can
        # legitimately include em-dashes, smart quotes, non-breaking
        # hyphens, etc. Hit three separate variants of this in one
        # session (an arrow character, an em-dash, a non-breaking hyphen)
        # before fixing it here once instead of patching each new
        # character as it turns up. Trace output is diagnostic, not the
        # eval's actual result — never let it crash a real run.
        print(message.encode(sys.stdout.encoding or "ascii", errors="replace").decode(sys.stdout.encoding or "ascii"))
