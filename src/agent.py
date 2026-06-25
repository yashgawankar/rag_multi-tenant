"""Custom, hand-rolled tool-calling agent loop (no LangChain/LlamaIndex —
see README for why transparency was preferred over a framework here).

The agent is instantiated per-session with a fixed tenant_id. The LLM is
never given tenant_id as something it can supply — see GET_ACCOUNT_BALANCE_SCHEMA
in src/tools.py, which omits it entirely from the tool's parameters. The
model has no way to think about, generate, or be tricked into requesting a
different tenant's data through this tool, because tenant_id was never part
of its vocabulary for this tool in the first place. agent.py supplies the
real tenant_id itself, from the session, every time the tool is called.
"""
from __future__ import annotations

import json

from src.config import Settings
from src.llm import LLMClient
from src.retriever import RetrievedChunk, retrieve
from src.tools import GET_ACCOUNT_BALANCE_SCHEMA, get_account_balance
from src.trace import trace

SEARCH_DOCS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_docs",
        "description": "Search the tenant's own document corpus for policy/product information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
            },
            "required": ["query"],
        },
    },
}

SYSTEM_PROMPT = """You are a banking assistant for a single tenant.
Answer using ONLY information returned by the search_docs tool for
document/policy questions, or get_account_balance for balance questions.
If search_docs returns no relevant content, say you don't have that
information rather than guessing. Never claim to know another tenant's
information — you only have access to this tenant's data."""


class Agent:
    def __init__(self, tenant_id: str, settings: Settings | None = None):
        self.tenant_id = tenant_id
        self.settings = settings or Settings()
        self.llm = LLMClient(self.settings)

    def _execute_tool(self, name: str, arguments: dict) -> tuple[str, list[RetrievedChunk]]:
        trace(f"[AGENT] >> executing tool: {name}({arguments})")

        if name == "search_docs":
            chunks = retrieve(self.tenant_id, arguments["query"], self.settings)
            if not chunks:
                trace("[AGENT] << search_docs found nothing")
                return "No relevant documents found.", []
            rendered = "\n\n".join(f"[{c.source}#{c.chunk_index}] {c.text}" for c in chunks)
            trace(f"[AGENT] << search_docs returning {len(chunks)} chunk(s) to the model")
            return rendered, chunks

        if name == "get_account_balance":
            # arguments only ever contains "account_id" — tenant_id isn't in
            # the tool's schema (src/tools.py), so the model never supplies
            # one. We pass the session's real tenant_id here ourselves.
            result = get_account_balance(tenant_id=self.tenant_id, account_id=arguments["account_id"])
            trace(f"[AGENT] << get_account_balance returning {result}")
            return json.dumps(result), []

        raise ValueError(f"Unknown tool: {name}")

    def ask(self, question: str) -> dict:
        trace(f"[AGENT] session tenant={self.tenant_id!r}  question={question!r}")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        tools = [SEARCH_DOCS_SCHEMA, GET_ACCOUNT_BALANCE_SCHEMA]
        retrieved_chunks: list[RetrievedChunk] = []

        for iteration in range(4):  # bounded loop, avoids runaway tool-calling
            trace(f"[AGENT] --- loop iteration {iteration}: calling LLM ---")
            response = self.llm.chat(messages, tools=tools)
            message = response.choices[0].message

            if not message.tool_calls:
                trace("[AGENT] model returned a final answer (no tool_calls) -> exiting loop")
                deduped = list({(c.source, c.chunk_index): c for c in retrieved_chunks}.values())
                return {"answer": message.content, "retrieved": deduped}

            trace(f"[AGENT] model requested {len(message.tool_calls)} tool call(s) -> continuing loop")
            messages.append(message.model_dump(exclude_none=True))
            for call in message.tool_calls:
                arguments = json.loads(call.function.arguments)
                tool_result, chunks = self._execute_tool(call.function.name, arguments)
                retrieved_chunks.extend(chunks)
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": tool_result}
                )

        trace("[AGENT] !! exhausted tool-call budget (4 iterations) without a final answer")
        return {"answer": "I wasn't able to resolve this within the tool-call budget.", "retrieved": retrieved_chunks}
