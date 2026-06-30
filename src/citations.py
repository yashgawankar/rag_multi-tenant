"""Citation grounding checks — two cheap tiers, no LLM call, no NLI model.

Tier 1 (existence): does a cited (source, chunk_index) actually appear in
what was retrieved this turn? Deterministic, free.

Tier 2 (hybrid grounding): for citations that pass Tier 1, does the
model's claimed supporting text actually relate to the cited chunk's
real content? Computes BOTH a lexical signal (token overlap) and a
semantic signal (cosine similarity, reusing the already-loaded dense
embedder) because they have different, complementary blind spots —
token overlap misses paraphrase, cosine similarity is tolerant of
"topically related but not actually entailing" content (the same
character of false-positive that let an irrelevant chunk survive
retrieval's score_threshold elsewhere in this codebase). Flags with OR
logic, not a weighted blend: this is a safety flag meant to catch
problems, and a blend would let one strong signal mask the other's red
flag — contrast with the whole-answer Faithfulness/Answer Relevance
guardrail in agent.py, which IS a quality score where averaging legs
would make sense; different goal, different combination rule.

Citations can arrive two ways: from the model's structured submit_answer
tool call (primary, gives an explicit `claim` string per citation), or
recovered via the regex safety net over the final answer text (secondary,
catches a citation embedded in prose that never made it into the
structured array). Every CitationCheck records which mechanism produced
it via `source_mechanism`.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from src.retriever import RetrievedChunk

#  Calibrated against real data (5 genuinely-grounded citations from the
#  eval set, plus the known-bad home-loan-cited-for-savings-claim case),
#  not guessed — see the session notes for the actual numbers. Findings:
#
#  token_overlap: the 5 GOOD citations scored 0.078-0.128 — i.e. an
#  0.15 threshold flagged 100% of genuinely well-grounded citations as
#  false positives, because the model paraphrases rather than quoting
#  verbatim, so word-level overlap is naturally low even when grounding
#  is fine. The 1 BAD citation scored 0.069 — actually *lower* than two
#  of the good ones (0.078, 0.084), meaning token overlap barely
#  discriminates good from bad in this regime at all. Lowered to 0.05,
#  which means token overlap effectively stops being the discriminating
#  signal here and cosine carries the real weight — an honest finding,
#  not the original design assumption.
TOKEN_OVERLAP_THRESHOLD = 0.05
#
#  cosine_score: the 5 GOOD citations scored 0.616-0.778. The 1 BAD
#  citation scored 0.597 — correctly the lowest of all 6, but only 0.02
#  below the closest good case. This embedding space compresses
#  similarity toward a high floor (a deliberately unrelated control
#  scored 0.50 earlier this session), so the margin is thin. Raised from
#  an initial guess of 0.5 (which would have caught NONE of this,
#  including the bad case) to 0.60 — just above the one bad data point,
#  just below the closest good one. This is a best-effort threshold from
#  6 real data points, not a robust one — flagged explicitly as needing
#  more flagged/unflagged examples before being trusted at real scale.
COSINE_THRESHOLD = 0.60

CITATION_TAG_RE = re.compile(r"\[([\w.\-]+)#(\d+)\]")


@dataclass(frozen=True)
class CitationCheck:
    source: str
    chunk_index: int
    exists: bool
    source_mechanism: str  # "structured" | "regex"
    claim: str | None = None
    token_overlap_score: float | None = None
    cosine_score: float | None = None
    weakly_grounded: bool | None = None


def cosine_similarity(vec_a, vec_b) -> float:
    dot = sum(x * y for x, y in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(x * x for x in vec_a))
    norm_b = math.sqrt(sum(y * y for y in vec_b))
    # fastembed returns numpy arrays, so dot/norm_a/norm_b are numpy
    # scalars (e.g. float32) — cast to a native Python float so this is
    # always JSON-serializable downstream (audit log, eval results.json)
    # without every caller needing to remember to do it.
    return float(dot / (norm_a * norm_b)) if norm_a and norm_b else 0.0


def _token_overlap(a: str, b: str) -> float:
    tokens_a = set(re.findall(r"\w+", a.lower()))
    tokens_b = set(re.findall(r"\w+", b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def extract_regex_citations(answer_text: str) -> list[tuple[str, int]]:
    """Supplementary safety net: every [source#chunk_index]-shaped tag
    found in free text, regardless of whether submit_answer's structured
    `citations` array was used, malformed, or never called at all."""
    return [(m.group(1), int(m.group(2))) for m in CITATION_TAG_RE.finditer(answer_text)]


def verify_citations(
    structured_citations: list[dict],
    answer_text: str,
    retrieved_chunks: list[RetrievedChunk],
    embed_fn=None,
) -> list[CitationCheck]:
    """structured_citations: [{"source", "chunk_index", "claim"}, ...]
    from submit_answer's parsed arguments (empty list if not_called or
    malformed). embed_fn(texts) -> list[vector]; if None, Tier 2 reports
    only token_overlap_score (cosine_score stays None) — lets this module
    be unit-tested without loading any model."""
    retrieved_by_key = {(c.source, c.chunk_index): c for c in retrieved_chunks}
    seen: set[tuple[str, int]] = set()
    checks: list[CitationCheck] = []

    for entry in structured_citations:
        source, chunk_index = entry.get("source"), entry.get("chunk_index")
        seen.add((source, chunk_index))
        checks.append(
            _check_one(source, chunk_index, entry.get("claim"), "structured", retrieved_by_key, embed_fn)
        )

    for source, chunk_index in extract_regex_citations(answer_text):
        if (source, chunk_index) in seen:
            continue  # already covered by the structured array
        seen.add((source, chunk_index))
        checks.append(_check_one(source, chunk_index, None, "regex", retrieved_by_key, embed_fn))

    return checks


def _check_one(source, chunk_index, claim, mechanism, retrieved_by_key, embed_fn) -> CitationCheck:
    chunk = retrieved_by_key.get((source, chunk_index))
    if chunk is None:
        return CitationCheck(source=source, chunk_index=chunk_index, exists=False, source_mechanism=mechanism)

    # Regex-recovered citations have no model-stated claim text to compare
    # against — Tier 2 can't run without something to compare, so it's
    # skipped (exists=True, grounding fields left None) rather than guessing.
    if claim is None:
        return CitationCheck(
            source=source, chunk_index=chunk_index, exists=True, source_mechanism=mechanism
        )

    token_score = _token_overlap(claim, chunk.text)
    cosine_score = None
    if embed_fn is not None:
        vecs = list(embed_fn([claim, chunk.text]))
        cosine_score = cosine_similarity(vecs[0], vecs[1])

    weakly_grounded = token_score < TOKEN_OVERLAP_THRESHOLD or (
        cosine_score is not None and cosine_score < COSINE_THRESHOLD
    )

    return CitationCheck(
        source=source,
        chunk_index=chunk_index,
        exists=True,
        source_mechanism=mechanism,
        claim=claim,
        token_overlap_score=token_score,
        cosine_score=cosine_score,
        weakly_grounded=weakly_grounded,
    )
