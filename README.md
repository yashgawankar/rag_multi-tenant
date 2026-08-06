# Mini Tenant-Aware RAG Agent

A tenant-isolated RAG pipeline and agent over banking/insurance product
documents for two tenants — `tenant_a` (Acme Bank) and `tenant_b` (Globex
Insurance). The agent answers questions from each tenant's own documents,
or by calling a mock `get_account_balance` tool, and never lets one
tenant's data reach the other's session.

## Requirements

- Python 3.11+
- An API key for an OpenAI-compatible LLM provider (Groq's free tier works
  out of the box — see `.env.example` for Gemini/OpenRouter alternatives)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # fill in LLM_API_KEY
```

No external database server is required — vector storage runs embedded
(Qdrant's local/on-disk mode) and persists to `./storage/` (git-ignored).

## Ingest the documents

```bash
python -m scripts.ingest_all
```

Run this once before chatting or evaluating, and again any time the files
under `data/tenant_a/` or `data/tenant_b/` change. Ingestion is safe to
re-run: each tenant's existing chunks are wiped and rebuilt from what's on
disk, so edited or removed source files never leave stale chunks behind.

## Chat with the agent

```bash
python -m scripts.chat tenant_a
python -m scripts.chat tenant_b
```

Each session is scoped to one tenant for its entire duration. Every
answer prints its citations and two lightweight quality scores
(`answer_relevance`, `faithfulness`) alongside the response.

## Run the eval

```bash
python -m eval.run_eval
```

Runs the 5 Q/A pairs in `eval/qa_pairs.json` end-to-end through the real
agent, reporting Recall@5, keyword-containment quality, and RAGAS-style
context relevance / answer relevance / faithfulness (all computed by
direct embedding cosine similarity, no LLM judge). Also runs a rerank
on/off comparison over the same questions. Per-question output is written
to `eval/results.json`.

## Run the tests

```bash
pytest tests/
```

35 tests covering chunking, ingestion, isolation (including a live
cross-tenant probe), citation grounding, audit logging, and the tool
guardrails.

## Project structure

```
data/<tenant>/         Source documents per tenant
src/config.py          Settings (env vars) and the tenant registry
src/chunking.py        Recursive, token-aware document splitter
src/ingest.py          Loads, chunks, and indexes documents
src/vector_store.py    Embedded Qdrant client, embeddings, hybrid search
src/retriever.py       The only code path allowed to query the store
src/reranker.py        Optional cross-encoder reranking (off by default)
src/agent.py           Tool-calling agent loop
src/tools.py           get_account_balance tool wrapper (tenant-pinned schema)
src/citations.py       Two-tier citation grounding checks
src/audit.py           Structured JSON audit log, one record per turn
src/trace.py           Optional human-readable flow tracing
src/llm.py             LLM client (any OpenAI-compatible provider)
scripts/                CLI entry points (ingest, chat)
eval/                   Q/A pairs and eval harness
tests/                  Unit, integration, and isolation tests
mock_tool.py            Provided balance-lookup tool (unmodified)
```

## Configuration

All configuration is via environment variables, loaded from `.env`
(see `.env.example` for the full list and alternative LLM providers):

- `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` — LLM provider settings
- `QDRANT_STORAGE_ROOT` — where the vector store persists on disk
- `EMBEDDING_MODEL` / `SPARSE_EMBEDDING_MODEL` — local embedding models
- `RERANK_ENABLED` — set to `1` to enable the cross-encoder reranker
- `AUDIT` / `AUDIT_DIR` — structured JSON audit log (on by default)
- `TRACE` — set to `1` to print step-by-step flow tracing during a run

## Data

`data/tenant_a/` and `data/tenant_b/` contain the provided Acme Bank and
Globex Insurance documents (FAQ, fees, product terms, refund/claims
policy, support). Replace these with different documents and re-run
`python -m scripts.ingest_all` — no code changes required.

---

## Design notes

### Isolation approach

Every tenant's documents live in **one shared Qdrant collection**, not
one store per tenant. Each point's payload carries a `tenant_id`, and
that's the only thing distinguishing one tenant's data from another's at
the storage layer. This mirrors how multi-tenant RAG is actually run in
production against a real vector database — one collection, indexed
tenant filter — rather than a toy design that isolates tenants by giving
each one a separate physical store.

Isolation is enforced in three independent layers, not one:

1. A mandatory `must` filter on `tenant_id`, applied to every stage of
   every query — both the dense and sparse prefetches and the final RRF
   fusion (`src/vector_store.py`).
2. A payload index on `tenant_id` with `is_tenant=True`, Qdrant's
   documented multi-tenancy optimization (a no-op in the embedded/local
   mode this project runs in, but the real call a production deployment
   against a Qdrant server would use).
3. A post-fetch assertion in `src/retriever.py`: every single hit's
   `tenant_id` payload is re-checked against the requested tenant before
   it's allowed anywhere near the LLM context. Any mismatch raises and
   aborts the request rather than degrading silently.

`src/retriever.py` is deliberately the *only* code in the repo allowed to
call the store's search method — nothing else imports `vector_store` for
querying. That turns "did every call site remember the filter" from a
fleet-wide question into a property of one function, and that function's
own test (`tests/test_isolation.py`) verifies it directly, including a
live end-to-end cross-tenant probe.

The same pattern extends to the tool-calling side: `get_account_balance`'s
schema (`src/tools.py`) never exposes `tenant_id` as something the model
can supply — the model has no vocabulary to even attempt a cross-tenant
request through this tool. The real `tenant_id` is injected server-side
from the session on every call. The provided `mock_tool.py` also runs its
own guardrail (`CrossTenantAccessError`) if an account belongs to a
different tenant; the agent treats that as expected tool behavior, not a
system failure, and logs it to its own audit field, distinct from an
actual isolation violation.

### Chunking

Documents are split by a recursive, tiered splitter
(`src/chunking.py`): paragraph → line → sentence → word → character. Each
piece is checked against a target size and only descended to a finer
separator if it's still too large — most paragraphs need no more than the
first tier. Chunks are then greedily packed back up toward the target
size so retrieval isn't left with one chunk per tiny paragraph, and each
new chunk is seeded with a word-boundary-aware overlap of the previous
chunk's tail so a fact sitting near a chunk boundary keeps context on
both sides.

Size is measured through a pluggable `length_fn` rather than a hardcoded
metric. Ingestion passes the real embedding model's own tokenizer
(`store.count_tokens`), not a character-count approximation — `bge-small`
truncates silently past 512 tokens, so sizing chunks in the same units
the model actually consumes avoids chunks that get silently cut off
during embedding.

### Retrieval

Retrieval is hybrid: dense (`bge-small-en-v1.5`) and sparse (`Qdrant/bm25`)
search fused with Qdrant's native Reciprocal Rank Fusion, filtered to the
requesting tenant at every stage. A `score_threshold` on the fused score
trims low-confidence results before they ever reach the LLM. An optional
FlashRank cross-encoder reranker sits behind a feature flag
(`RERANK_ENABLED`) — pluggable rather than always-on specifically so
`eval/run_eval.py` can report a measured before/after comparison instead
of assuming reranking helps. At this corpus size (~4-5 documents per
tenant) it measurably doesn't move the top-k ordering; it's kept as an
opt-in path for when a real deployment's corpus grows.

### Grounded answers and guardrails

The agent never answers as free text. It's required to call a
`submit_answer` tool with a structured `citations` array
(source/chunk_index/claim per claim), so citations arrive as
schema-validated data instead of being regex-parsed out of prose (a regex
scan still runs as a safety net for anything that lands in the text
anyway). Each citation is checked in two tiers: existence (does the cited
chunk actually appear in what was retrieved this turn — free,
deterministic) and grounding (does the claimed text actually relate to
that chunk's real content, via token overlap and cosine similarity,
thresholds calibrated against real eval output rather than guessed).
Separately, a whole-answer guardrail computes answer relevance
(question↔answer) and faithfulness (answer↔retrieved context) via direct
embedding similarity and logs when either is low. None of this blocks a
response — the assignment scope is "flag," not "self-correct" — but every
number is written to the structured JSON audit log
(`src/audit.py`, one record per turn) alongside tool calls, retrieved
chunks, and any isolation or cross-tenant events, independent of the
human-readable trace output used for local debugging.

### Framework choice

The agent loop is hand-rolled rather than built on LangChain/LangGraph.
For a two-tool, bounded-iteration agent like this one, a framework's
abstractions (chains, graph state, callback wiring) buy little and cost
transparency — the tenant-isolation logic, the forced-`submit_answer`
citation mechanism, and the audit/guardrail hooks all live in plain,
readable Python where every step is traceable. That trade-off would
flip for a larger multi-agent or branching workflow, where LangGraph's
explicit state graph and checkpointing genuinely earn their weight.

## Scaling considerations

**To ~100 tenants / ~10M documents:** the shared-collection-plus-filter
design is exactly the pattern Qdrant recommends at this scale, and the
`is_tenant=True` payload index (already in the code, currently a no-op
locally) becomes a real optimization on an actual Qdrant server —
clustering each tenant's vectors together on disk instead of an
unindexed full-collection scan. Ingestion would move from a synchronous
CLI script to a queue-backed background job so a large tenant's re-index
doesn't block others, and the `@lru_cache(maxsize=1)` singletons on the
embedder/store objects (fine for a single-process demo) would need to
become shared, properly-sized caches or a dedicated embedding service.

**Cost, latency, observability in production:** embedding is local and
free today (fastembed, CPU-only, no API cost); the real per-request cost
is the LLM call, so caching identical questions and capping the
tool-call loop (already bounded at 4 iterations) both matter directly at
scale. The structured JSON audit log is the seed of real observability —
in production it would ship to a log pipeline rather than local files, with
the isolation-violation and cross-tenant-attempt fields wired to alerts
rather than just being queryable after the fact.

## What I'd do differently with more time

- Fetch-size math currently compounds when reranking is enabled
  (`retriever.py`'s already-multiplied candidate count is passed into
  `vector_store.py`'s own independent multiplier), over-fetching more
  than intended — a known, not-yet-fixed inefficiency, harmless at this
  corpus size but worth flattening into one place before it scales.
- The reranker currently has no score floor of its own — it always
  returns exactly `top_n` results regardless of how low the best
  candidate scores. Worth adding once there's a real reranked-score
  distribution to calibrate a threshold against.
- Citation-grounding thresholds are calibrated against 6 real data points
  from one eval run, not a proper labeled set — good enough to catch the
  one bad case seen so far, but not something I'd trust unchanged at
  real scale without more examples.

## A case where the system fails

Early in development, a smaller/weaker model correctly cited the right
source document for a fee question but under-claimed the fee — its
`submit_answer` call cited chunk text that stated the fee's full
conditions (e.g. "waived if X, otherwise $10") while asserting only the
unconditional part. Tier 1 citation checks passed (the chunk was real
and retrieved), and Tier 2 grounding passed too (the claim and the chunk
text score highly similar, since most of the sentence really does
overlap) — this is a genuine blind spot in embedding-similarity-based
grounding: it verifies *relatedness*, not *entailment* or *completeness*.
A model can partially misstate a fact drawn from a real, correctly-cited
chunk, and cosine similarity alone won't catch it. Closing this gap
properly would need an actual entailment check (small NLI model or an
LLM-judge pass), which this project deliberately avoided for cost/latency
reasons — a trade-off worth surfacing rather than hiding.
