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
            print(f"  (sources retrieved: {sources})")
        for c in result["citations"]:
            flag = "MISSING" if not c.exists else ("weak" if c.weakly_grounded else "ok")
            print(f"  (cited [{c.source}#{c.chunk_index}] via {c.source_mechanism}: {flag})")
        ar, fa = result["answer_relevance"], result["faithfulness"]
        if ar is not None or fa is not None:
            ar_s = f"{ar:.2f}" if ar is not None else "n/a"
            fa_s = f"{fa:.2f}" if fa is not None else "n/a"
            low = (ar is not None and ar < 0.5) or (fa is not None and fa < 0.5)
            print(f"  (answer_relevance(Q<->A)={ar_s}  faithfulness(A<->C)={fa_s}{' — LOW, see audit log' if low else ''})")
        print()


if __name__ == "__main__":
    main()
