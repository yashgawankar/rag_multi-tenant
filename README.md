# Mini Tenant-Aware RAG Agent

A minimal but defensible tenant-isolated RAG pipeline + agent for two
tenants (`tenant_a`, `tenant_b`), built for the Westpac take-home.

> **Status:** scaffolded ahead of receiving the real docs/Q&A pairs from
> Westpac. `data/tenant_a/` and `data/tenant_b/` currently hold placeholder
> banking docs so the whole pipeline is runnable end-to-end today. Swap in
> the real files and re-run ingestion — no code changes should be needed.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env   # fill in your LLM_API_KEY (free Groq key by default)

python -m scripts.ingest_all        # ingests both tenants into the shared vector store
python -m scripts.chat tenant_a     # chat as tenant_a
python -m eval.run_eval             # retrieval + answer-quality metrics
pytest tests/                       # isolation tests
```

No Docker, no external server — Qdrant runs embedded, persisting to
`./storage/qdrant/shared/` (git-ignored). All tenants share this one store;
isolation is enforced by a mandatory filter, not by separate paths — see
"Isolation approach" below for why.

## Isolation approach

This is the part I expect to be scrutinized most, so it's layered
deliberately rather than relying on a single mechanism. The design mirrors
a pattern I've implemented in production before — one shared collection,
an indexed tenant filter — rather than a toy per-tenant-store setup that
wouldn't survive contact with a real tenant count.

**Layer 1 — one shared Qdrant collection, with tenant_id as an indexed,
mandatory filter on every query.** All tenants' chunks live in the same
collection (`src/vector_store.py`). At collection-creation time, a payload
index is created on `tenant_id` with `is_tenant=True` — Qdrant's documented
multitenancy optimization, which clusters one tenant's vectors together on
disk for better cache locality. Every search — both the dense and sparse
prefetch stages, and the final RRF fusion — carries a `must` filter on
`tenant_id` (`SharedVectorStore.hybrid_search`). I tested this combination
directly: a hybrid query with the filter applied to both prefetches and the
top-level query correctly returns only the requested tenant's points even
when both tenants' vectors score identically against the query (see
`tests/test_isolation.py::test_both_tenants_are_genuinely_colocated_in_one_collection`,
which proves the data really is colocated — this isn't a folder boundary
doing the work, the filter is). I rejected an earlier version of this
codebase that gave each tenant a fully separate on-disk Qdrant store
(no shared collection at all) — that's simpler to *demonstrate* in a 2-tenant
demo, but doesn't scale: Qdrant's own guidance is against per-tenant
collections for "many small tenants" (fixed per-collection overhead,
operational burden multiplied by tenant count), and embedded mode only
allows one open client per path at a time, which I hit firsthand running
concurrent processes against it during development.

**Known limitation, found by testing rather than assumed:** Qdrant's
embedded/local mode (used here — no server, no Docker, no API key, see
below) silently no-ops payload indexes; I confirmed this empirically
(`UserWarning: Payload indexes have no effect in the local Qdrant`).
Filtering itself still works correctly without the index — I verified this
directly too — it just falls back to an unindexed scan instead of an
optimized lookup, which is unobservable at 10 documents per tenant. On a
real Qdrant server the `is_tenant` index would actually activate. I'm
calling the real production API anyway (rather than skipping it) so the
code matches what I'd actually ship, with the local-mode caveat documented
rather than hidden.

**Layer 2 — runtime assertion in the retriever.** `src/retriever.py` checks
every single hit's `tenant_id` against the requested tenant before it's
allowed anywhere near the LLM context, and raises `TenantIsolationViolation`
if they ever disagree. This is more than a defensive afterthought here:
since isolation now genuinely depends on the filter being applied correctly
(there's no physical separation backstopping it), this assertion is the
last line of defense against a future code path that calls
`hybrid_search` directly instead of going through this module.
`src/retriever.py` is the *only* file in this codebase that imports
`vector_store.py` — making "did every call site remember the filter" a
property of one audited function instead of a fleet-wide assumption.

**Layer 3 — the LLM is never given `tenant_id` to begin with.** The real
`get_account_balance` function takes `tenant_id` (the brief's stub signature
requires it), but the *tool schema* shown to the model — `GET_ACCOUNT_BALANCE_SCHEMA`
in `src/tools.py` — only exposes `account_id` as a parameter. The schema we
hand the LLM and the function we actually call don't have to match 1:1.
The model never sees, generates, or reasons about a `tenant_id` for this
tool at all; `src/agent.py` supplies the real one from the session every
time. This is stronger than "let the model propose a value and override it
server-side" (an earlier version of this code did exactly that) — there's
no proposed value to override, so there's no path through which a
confused or adversarial prompt could even attempt to influence which
tenant's balance gets fetched.

**Layer 4 — isolation eval.** `tests/test_isolation.py` and the eval set
explicitly probe each tenant with queries about the *other* tenant's
products and assert zero leakage, plus a direct proof that both tenants'
data is genuinely colocated in one collection (not just "assumed" from the
absence of separate folders).

**Trade-off, stated plainly:** a shared collection with a mandatory filter
is the right call for real scale, but it does mean isolation is a property
you have to prove (tests, the retriever choke-point, the assertion) rather
than something structurally guaranteed by separate storage. I'm accepting
that trade-off deliberately because it's the pattern that actually holds up
past a handful of tenants — the alternative (physical per-tenant stores)
would have been the *easier* thing to build for a 2-tenant take-home, not
the more correct one.

## Chunking choice

Hand-rolled paragraph-aware splitter (`src/chunking.py`), not a fixed-size
token splitter. These are short, structured policy/product documents where
each paragraph is usually a self-contained clause — splitting mid-sentence
to hit an exact token count loses more context than it gains at this
document length. The splitter packs paragraphs into ~800-character windows
with a ~120-character overlap carried into the next chunk, and falls back to
sentence-level splitting only if a single paragraph is unusually long. Will
revisit chunk size once I see the real docs — if they're longer/denser
(e.g. multi-page PDFs), a token-aware splitter would be worth the extra
dependency.

## Retrieval

Hybrid dense + sparse search fused with Qdrant's native RRF (`models.Fusion.RRF`),
no reranker. Dense embeddings via `fastembed` (`BAAI/bge-small-en-v1.5`,
local, no API key); sparse via `fastembed`'s BM25 implementation
(`Qdrant/bm25`). Hybrid was worth the (small) extra complexity because
banking docs mix natural-language questions with exact terms that matter
(account names, dollar figures, day counts) where lexical match outperforms
pure embedding similarity. Skipped a cross-encoder reranker — with ~10 docs
per tenant the candidate pool is too small for reranking to meaningfully
change top-k ordering; it would be over-engineering relative to the corpus
size, even though it's cheap to add later (`src/vector_store.py:hybrid_search`
already returns scored candidates a reranker could re-sort).

## Agent design

Custom tool-calling loop (`src/agent.py`), not LangChain/LlamaIndex. Two
tools: `search_docs` (wraps the isolated retriever) and
`get_account_balance` (the provided mock stub). Deliberately hand-rolled
rather than framework-based so every part of the loop — including the
tenant-pinning behaviour above — is something I can walk through line by
line in the follow-up call, rather than something a framework did for me.

LLM access is provider-agnostic (`src/llm.py`): Groq, Gemini, and OpenRouter
all expose OpenAI-compatible chat-completions endpoints, so one thin
wrapper around the `openai` SDK covers all three — switching providers is a
`.env` edit (see `.env.example`), not a code change. Default is Groq,
`llama-4-scout-17b-16e-instruct` — see the model-choice note in the Eval
section below for why this specific model and not the largest one available.

## Eval

`eval/qa_pairs.json` — 5 Q/A pairs (currently against the placeholder docs;
will be replaced with real ones once provided). `eval/run_eval.py` reports:

1. **Retrieval metric — Recall@5:** did a chunk from the expected source
   document appear anywhere in the top-5 retrieved chunks for that question?
2. **Answer-quality check — keyword containment:** does the agent's final
   answer contain the expected fact (e.g. the correct rate/day-count)? This
   catches the common RAG failure mode where retrieval succeeds but the
   model answers from its own (wrong) prior instead of the retrieved text.

Both are deliberately cheap/deterministic rather than an LLM-as-judge, given
the brief's "tiny eval" framing — no extra API spend, no judge-model
variance to explain.

**Latest run** (placeholder docs, Groq `llama-4-scout-17b-16e-instruct`):

```
Recall@5: 5/5 = 1.00
Answer-quality (keyword containment): 5/5 = 1.00
```

Earlier runs on `llama-3.1-8b-instant` scored 4/5 on answer-quality — retrieval
correctly surfaced the right chunk every time (`recall_hit=True`), but the
small 8B model occasionally answered "I don't have that information" anyway
despite the fact being present in context, a genuine, reproducible failure
mode (retrieval success ≠ answer correctness — exactly the gap this check
exists to catch). A 4-model bake-off against the same 5 questions
(`llama-3.1-8b-instant`, `llama-4-scout-17b-16e-instruct`, `openai/gpt-oss-20b`,
`qwen/qwen3-32b`) showed all three larger models scoring 5/5 with reliable
tool-calling; `llama-4-scout-17b-16e-instruct` was picked as the new default
for giving the most direct answers without extra markdown/commentary
clutter. Full per-question output is in `eval/results.json`.

**Note on model choice:** `llama-3.3-70b-versatile` (Groq's largest general
model) has a reproducible tool-calling bug — it emits its function call as
literal text (`<function=search_docs{...}</function>`) instead of a
structured `tool_calls` response, which the OpenAI SDK then rejects as a
400. That ruled it out regardless of quality. Among the models that call
tools correctly, `llama-3.1-8b-instant` works but is the smallest/cheapest
and showed real answer-quality flakiness above. `llama-4-scout-17b-16e-instruct`,
`openai/gpt-oss-20b`, and `qwen/qwen3-32b` all called tools reliably and
scored 5/5 in the bake-off; `llama-4-scout` is the default for the cleanest
output formatting, but any of the three would be a reasonable choice —
this is documented as a deliberate, tested decision rather than "whichever
model happened to be set first."

## What I'd improve with more time

- **Real Qdrant server instead of embedded mode**: would make the
  `is_tenant` payload index actually activate (it's a documented no-op in
  embedded/local mode — see Isolation approach above), and would remove the
  one-client-per-path limitation that caused lock contention during
  development whenever two processes touched the same store concurrently.
- **Token-aware, fully recursive chunking**: swap the hand-rolled splitter
  for `langchain-text-splitters`' `RecursiveCharacterTextSplitter` with a
  `tiktoken`-based length function and percentage-based overlap (10-20%).
  The current splitter only falls back from paragraphs to sentences, not
  to line breaks or raw characters, so a long bullet-point list without
  blank lines between items could end up under-split — a real gap I'd
  rather fix with a battle-tested splitter than patch by hand.
- **LLM retry-with-backoff**: a transient Groq 503 currently crashes the
  whole `ask()` call (hit this live during development);
  `LLMClient.chat()` should retry transient 5xx/rate-limit errors with
  exponential backoff before giving up.
- **Anti-hallucination measures**: `temperature=0` for deterministic
  output, explicit citation requirements in the system prompt, and
  possibly a chain-of-verification pass (extract claims from the draft
  answer, verify each against retrieved context before returning it).
- **Context ordering for the "lost in the middle" effect**: LLMs attend
  most reliably to the start and end of a long context window; the most
  relevant retrieved chunk should be placed first (and possibly repeated
  last) rather than left in whatever order retrieval returned it.
- **Richer eval**: LLM-as-judge for faithfulness/answer-relevance once
  there's a real budget for it, plus MRR alongside Recall@k.
- **Reranking**: add a cross-encoder rerank step (bi-encoder retrieval for
  recall, cross-encoder reranking for precision) once the corpus is large
  enough per tenant for it to matter — at ~10 docs/tenant the candidate
  pool is too small for reranking to change top-k ordering meaningfully.
- **Citations in answers**: surface `[source#chunk_index]` inline in the
  agent's final answer, not just alongside it, for auditability.
- **Streaming + multi-turn memory** in `scripts/chat.py` (currently
  single-shot per question, no conversation history carried across turns).
- **Structured logging** of every retrieval (tenant_id, query, hit sources,
  isolation-assertion outcome) for an auditable trail — important for a
  banking context specifically. The `TRACE=1` flag's print-based tracing
  was built for interactively understanding the system during development,
  not as production observability.
