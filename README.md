# Mini Tenant-Aware RAG Agent

A tenant-isolated RAG pipeline and agent over banking product documents for
two tenants (`tenant_a`, `tenant_b`). The agent answers questions from each
tenant's own documents, or by calling a mock `get_account_balance` tool.

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
and persists to `./storage/` (git-ignored).

## Ingest the documents

```bash
python -m scripts.ingest_all
```

Run this once before chatting or evaluating, and again any time the files
under `data/tenant_a/` or `data/tenant_b/` change.

## Chat with the agent

```bash
python -m scripts.chat tenant_a
python -m scripts.chat tenant_b
```

Each session is scoped to one tenant for its entire duration.

## Run the eval

```bash
python -m eval.run_eval
```

Reports a retrieval metric (Recall@5) and an answer-quality check
(keyword containment) against the 5 Q/A pairs in `eval/qa_pairs.json`.
Per-question output is written to `eval/results.json`.

## Run the tests

```bash
pytest tests/
```

## Project structure

```
data/<tenant>/        Source documents per tenant
src/config.py         Settings (env vars) and tenant registry
src/chunking.py       Document splitter
src/vector_store.py   Embedded Qdrant client, embeddings, hybrid search
src/ingest.py         Loads, chunks, and indexes documents
src/retriever.py      Tenant-scoped retrieval used by the agent
src/llm.py            LLM client (OpenAI-compatible)
src/tools.py          Mock get_account_balance tool
src/agent.py          Tool-calling agent loop
src/trace.py          Optional flow tracing (see Configuration)
scripts/              CLI entry points (ingest, chat)
eval/                 Q/A pairs and eval harness
tests/                Isolation tests
```

## Configuration

All configuration is via environment variables, loaded from `.env`
(see `.env.example` for the full list and alternative LLM providers):

- `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` — LLM provider settings
- `QDRANT_STORAGE_ROOT` — where the vector store persists on disk
- `EMBEDDING_MODEL` / `SPARSE_EMBEDDING_MODEL` — local embedding models
- `TRACE` — set to `1` to print step-by-step flow tracing during a run
  (tool calls, retrieval hits, etc.); `0` (default) is silent

## Data

`data/tenant_a/` and `data/tenant_b/` contain sample banking documents
(savings accounts, credit cards, home loans, transfers, fees, privacy
policy, FAQ). Replace these with real documents and re-run
`python -m scripts.ingest_all` — no code changes required.
