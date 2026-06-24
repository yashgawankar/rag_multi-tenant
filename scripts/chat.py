"""Simple CLI to chat with the agent for a chosen tenant.

Usage:
    python -m scripts.chat tenant_a
"""
import sys

from src.agent import Agent
from src.config import TENANTS


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in TENANTS:
        print(f"Usage: python -m scripts.chat <tenant_id>  (one of {TENANTS})")
        sys.exit(1)

    tenant_id = sys.argv[1]
    agent = Agent(tenant_id=tenant_id)
    print(f"Chatting as {tenant_id}. Ctrl+C to exit.\n")

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        result = agent.ask(question)
        print(result["answer"])
        if result["retrieved"]:
            sources = ", ".join(f"{c.source}#{c.chunk_index}" for c in result["retrieved"])
            print(f"  (sources: {sources})")
        print()


if __name__ == "__main__":
    main()
