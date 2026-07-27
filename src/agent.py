"""Custom, hand-rolled tool-calling agent loop (no LangChain/LlamaIndex —
see README for why transparency was preferred over a framework here).

The agent is instantiated per-session with a fixed tenant_id. The LLM is
never given tenant_id as something it can supply — see GET_ACCOUNT_BALANCE_SCHEMA
in src/tools.py, which omits it entirely from the tool's parameters. The
model has no way to think about, generate, or be tricked into requesting a
different tenant's data through this tool, because tenant_id was never part
of its vocabulary for this tool in the first place. agent.py supplies the
real tenant_id itself, from the session, every time the tool is called.

Final answers are finalized via a forced `submit_answer` tool call rather
than plain text, so citations arrive as structured, schema-validated data
(source/chunk_index/claim) instead of being regex-parsed out of free
prose. Two independent things can go wrong with this, logged separately
(see _finalize* methods and src/citations.py): the model might not call
submit_answer at all (submit_answer_status="not_called"), or it might call
it with malformed arguments (submit_answer_status="malformed") — neither
is assumed away, both degrade gracefully rather than crashing the request.
"""
from __future__ import annotations

import json

from src import vector_store
from src.audit import AuditRecord, write_audit_record
from src.citations import COSINE_THRESHOLD, CitationCheck, cosine_similarity, verify_citations
from src.config import Settings
from src.llm import LLMClient
from src.retriever import RetrievedChunk, TenantIsolationViolation, retrieve
from src.tools import (
    AccountNotFoundError,
    CrossTenantAccessError,
    GET_ACCOUNT_BALANCE_SCHEMA,
    UnknownTenantError,
    get_account_balance,
)
from src.trace import trace

# Same provisional threshold as citations.COSINE_THRESHOLD, shared until
# calibration (see plan) determines whether the coarse runtime guardrail
# and the fine-grained per-citation check should use different cutoffs.
GUARDRAIL_THRESHOLD = COSINE_THRESHOLD

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

SUBMIT_ANSWER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_answer",
        "description": "Call this to give your final answer once you have everything you need.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "description": "Your final answer to the user."},
                "citations": {
                    "type": "array",
                    "description": (
                        "One entry per factual claim drawn from search_docs. "
                        "Leave empty if your answer only used get_account_balance, or is a refusal."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {
                                "type": "string",
                                "description": "The source filename from the [source#chunk_index] tag.",
                            },
                            "chunk_index": {
                                "type": "integer",
                                "description": "The chunk index from the [source#chunk_index] tag.",
                            },
                            "claim": {
                                "type": "string",
                                "description": "The specific fact this citation supports.",
                            },
                        },
                        "required": ["source", "chunk_index", "claim"],
                    },
                },
            },
            "required": ["answer", "citations"],
        },
    },
}

SYSTEM_PROMPT = """You are a banking assistant for a single tenant.
Answer using ONLY information returned by the search_docs tool for
document/policy questions, or get_account_balance for balance questions.
If search_docs returns no relevant content, say you don't have that
information rather than guessing. Never claim to know another tenant's
information — you only have access to this tenant's data.

You MUST call submit_answer to give your final answer — never answer
directly as plain text. For every claim drawn from search_docs, include
a citation in submit_answer's citations array referencing the exact
[source#chunk_index] tag shown with that chunk, plus the specific claim
it supports. If your answer only used get_account_balance, or is a
refusal/"I don't know", leave citations empty — do not invent a citation
tag for those cases."""

# A worked example, prepended as real prior turns (not described in
# prose) so the model has an actual demonstration in the exact format
# it has to reproduce — chosen over zero-shot specifically because the
# model-compliance gap (will it call submit_answer at all, correctly) is
# the single biggest weakness of this whole mechanism. Real, accepted
# cost: this persists in `messages` for the whole ask() call, so it's
# resent on every API call within that invocation, not a one-time cost.
FEW_SHOT_EXAMPLE = [
    {"role": "user", "content": "What's the fee on the rewards card?"},
    {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "example_call_1",
                "type": "function",
                "function": {"name": "search_docs", "arguments": '{"query": "rewards card annual fee"}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "example_call_1", "content": "[example.md#0] Annual fee is $99."},
    {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "example_call_2",
                "type": "function",
                "function": {
                    "name": "submit_answer",
                    "arguments": (
                        '{"answer": "The annual fee is $99.", "citations": '
                        '[{"source": "example.md", "chunk_index": 0, "claim": "Annual fee is $99."}]}'
                    ),
                },
            }
        ],
    },
]


def _fmt(x: float | None) -> str:
    return f"{x:.2f}" if x is not None else "n/a"


class Agent:
    def __init__(self, tenant_id: str, settings: Settings | None = None):
        self.tenant_id = tenant_id
        self.settings = settings or Settings()
        self.llm = LLMClient(self.settings)

    def _embed(self, texts: list[str]):
        return vector_store.embed_dense(texts, self.settings)

    def _execute_tool(
        self, name: str, arguments: dict, record: AuditRecord
    ) -> tuple[str, list[RetrievedChunk]]:
        trace(f"[AGENT] >> executing tool: {name}({arguments})")
        record.record_tool_call(name, arguments)

        if name == "search_docs":
            try:
                chunks = retrieve(self.tenant_id, arguments["query"], self.settings)
            except TenantIsolationViolation as e:
                # The audit write must never soften this failure — record
                # the violation for the forensic trail, then re-raise
                # unchanged so the request still aborts exactly as before.
                record.record_isolation_violation(e.requested_tenant, e.hit_tenant, e.source)
                write_audit_record(record)
                raise
            if not chunks:
                trace("[AGENT] << search_docs found nothing")
                return "No relevant documents found.", []
            rendered = "\n\n".join(f"[{c.source}#{c.chunk_index}] {c.text}" for c in chunks)
            trace(f"[AGENT] << search_docs returning {len(chunks)} chunk(s) to the model")
            record.record_retrieved_chunks(chunks)
            return rendered, chunks

        if name == "get_account_balance":
            # arguments only ever contains "account_id" — tenant_id isn't in
            # the tool's schema (src/tools.py), so the model never supplies
            # one. We pass the session's real tenant_id here ourselves.
            account_id = arguments["account_id"]
            try:
                result = get_account_balance(tenant_id=self.tenant_id, account_id=account_id)
                trace(f"[AGENT] << get_account_balance returning {result}")
                return json.dumps(result), []
            except CrossTenantAccessError:
                # mock_tool's own guardrail correctly refusing a request for
                # an account belonging to a different tenant — no data was
                # leaked, this is the tool working exactly as designed. Not
                # the same severity as TenantIsolationViolation (which means
                # OUR filter failed); logged to a distinct audit field and
                # the loop continues normally rather than aborting.
                trace(f"[AGENT] !! cross-tenant access attempt: tenant={self.tenant_id!r} account_id={account_id!r} !!")
                record.record_cross_tenant_access_attempt(self.tenant_id, account_id)
                # Deliberately generic — never reveals which tenant the
                # account actually belongs to.
                return "That account is not associated with your tenant.", []
            except AccountNotFoundError:
                trace(f"[AGENT] << get_account_balance: no such account {account_id!r}")
                return "No account found with that ID.", []
            except UnknownTenantError:
                # Defensive only — should be unreachable since self.tenant_id
                # is always one of the two known tenants (see config.TENANTS).
                trace(f"[AGENT] !! unexpected UnknownTenantError for tenant={self.tenant_id!r} !!")
                return "Unable to look up account information right now.", []

        raise ValueError(f"Unknown tool: {name}")

    def _finalize_via_submit_answer(self, call, question: str, retrieved_chunks, record: AuditRecord) -> dict:
        structured_citations: list[dict] = []
        try:
            args = json.loads(call.function.arguments)
            answer_text = args["answer"]
            structured_citations = args.get("citations") or []
            record.record_submit_answer_status("ok")
            trace(f"[AGENT] submit_answer_status=ok, {len(structured_citations)} citation(s)")
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            trace(f"[AGENT] !! submit_answer arguments malformed: {e} !!")
            # Best-effort partial recovery: salvage "answer" even if
            # citations is broken/missing, rather than losing it too.
            answer_text = None
            try:
                partial = json.loads(call.function.arguments)
                answer_text = partial.get("answer") if isinstance(partial, dict) else None
            except (json.JSONDecodeError, TypeError):
                pass
            if not answer_text:
                answer_text = "I wasn't able to format a response correctly. Please try rephrasing your question."
            record.record_submit_answer_status("malformed", str(e))

        return self._finalize(answer_text, structured_citations, question, retrieved_chunks, record)

    def _finalize_not_called(self, answer_text: str | None, question: str, retrieved_chunks, record: AuditRecord) -> dict:
        record.record_submit_answer_status("not_called")
        trace("[AGENT] submit_answer_status=not_called, using plain content")
        return self._finalize(answer_text or "", [], question, retrieved_chunks, record)

    def _finalize(
        self,
        answer_text: str,
        structured_citations: list[dict],
        question: str,
        retrieved_chunks: list[RetrievedChunk],
        record: AuditRecord,
    ) -> dict:
        deduped = list({(c.source, c.chunk_index): c for c in retrieved_chunks}.values())

        checks = verify_citations(structured_citations, answer_text, deduped, embed_fn=self._embed)
        record.record_citation_checks(checks)
        for c in checks:
            trace(
                f"[AGENT] citation check: source={c.source!r} chunk={c.chunk_index} "
                f"mechanism={c.source_mechanism} exists={c.exists} weakly_grounded={c.weakly_grounded}"
            )
        regex_only = [c for c in checks if c.source_mechanism == "regex"]
        if regex_only:
            trace(f"[AGENT] regex safety net recovered {len(regex_only)} citation(s) not in structured array")

        answer_relevance = faithfulness = None
        if deduped and answer_text:
            context_text = "\n\n".join(c.text for c in deduped)
            vecs = list(self._embed([question, answer_text, context_text]))
            answer_relevance = cosine_similarity(vecs[0], vecs[1])  # Q<->A
            faithfulness = cosine_similarity(vecs[2], vecs[1])  # A<->C
        record.record_guardrail(answer_relevance, faithfulness)

        is_low = (answer_relevance is not None and answer_relevance < GUARDRAIL_THRESHOLD) or (
            faithfulness is not None and faithfulness < GUARDRAIL_THRESHOLD
        )
        trace(
            f"[AGENT] guardrail: answer_relevance(Q<->A)={_fmt(answer_relevance)}  "
            f"faithfulness(A<->C)={_fmt(faithfulness)}" + (" LOW" if is_low else "")
        )

        record.finalize(answer_text)
        write_audit_record(record)
        return {
            "answer": answer_text,
            "retrieved": deduped,
            "citations": checks,
            "answer_relevance": answer_relevance,
            "faithfulness": faithfulness,
        }

    def ask(self, question: str) -> dict:
        trace(f"[AGENT] session tenant={self.tenant_id!r}  question={question!r}")
        record = AuditRecord(tenant_id=self.tenant_id, question=question)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *FEW_SHOT_EXAMPLE,
            {"role": "user", "content": question},
        ]
        tools = [SEARCH_DOCS_SCHEMA, GET_ACCOUNT_BALANCE_SCHEMA, SUBMIT_ANSWER_SCHEMA]
        retrieved_chunks: list[RetrievedChunk] = []

        for iteration in range(4):  # bounded loop, avoids runaway tool-calling
            trace(f"[AGENT] --- loop iteration {iteration}: calling LLM ---")
            response = self.llm.chat(messages, tools=tools)
            message = response.choices[0].message
            trace(
                f"[AGENT] raw message from API: role={message.role!r} "
                f"content={message.content!r} tool_calls={message.tool_calls!r}"
            )

            # submit_answer is a TERMINATING tool call — if present among
            # this turn's tool_calls, finalize via it and ignore any other
            # tool calls requested alongside it in the same turn (acting on
            # them would be wasted work once the model has decided to finish).
            submit_call = None
            if message.tool_calls:
                for call in message.tool_calls:
                    if call.function.name == "submit_answer":
                        submit_call = call
                        break

            if submit_call is not None:
                trace("[AGENT] model called submit_answer -> finalizing")
                return self._finalize_via_submit_answer(submit_call, question, retrieved_chunks, record)

            if not message.tool_calls:
                trace("[AGENT] !! model returned plain content without calling submit_answer -> not_called fallback")
                return self._finalize_not_called(message.content, question, retrieved_chunks, record)

            trace(f"[AGENT] model requested {len(message.tool_calls)} tool call(s) -> continuing loop")
            messages.append(message.model_dump(exclude_none=True))
            for call in message.tool_calls:
                arguments = json.loads(call.function.arguments)
                tool_result, chunks = self._execute_tool(call.function.name, arguments, record)
                retrieved_chunks.extend(chunks)
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": tool_result}
                )

        trace("[AGENT] !! exhausted tool-call budget (4 iterations) without a final answer")
        return self._finalize_not_called(
            "I wasn't able to resolve this within the tool-call budget.", question, retrieved_chunks, record
        )
