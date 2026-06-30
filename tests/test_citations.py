"""Unit tests for citation grounding (src/citations.py) — no model
loading required; embed_fn=None exercises Tier 1 + the token-overlap half
of Tier 2 only, which is enough to prove the logic without needing
fastembed loaded for every test."""
from src.citations import (
    CitationCheck,
    cosine_similarity,
    extract_regex_citations,
    verify_citations,
)
from src.retriever import RetrievedChunk


def _chunk(source, chunk_index, text, tenant_id="tenant_a"):
    return RetrievedChunk(text=text, source=source, chunk_index=chunk_index, score=1.0, tenant_id=tenant_id)


def test_tier1_existence_failure_citation_not_retrieved():
    checks = verify_citations(
        structured_citations=[{"source": "fake_doc.md", "chunk_index": 99, "claim": "made up"}],
        answer_text="The rate is 5%.",
        retrieved_chunks=[],
    )
    assert len(checks) == 1
    assert checks[0].exists is False
    assert checks[0].source_mechanism == "structured"


def test_tier1_existence_pass():
    chunk = _chunk("savings.md", 0, "The bonus rate is 4.25% p.a.")
    checks = verify_citations(
        structured_citations=[{"source": "savings.md", "chunk_index": 0, "claim": "The bonus rate is 4.25% p.a."}],
        answer_text="The rate is 4.25%.",
        retrieved_chunks=[chunk],
    )
    assert checks[0].exists is True


def test_tier2_flags_mismatched_claim_real_scenario():
    """Mirrors the real home-loan/savings mismatch found earlier this
    session: a chunk about LVR/comparison-rate definitions cited for a
    savings-interest-rate claim it has nothing to do with.

    Calibrating against real eval data showed token_overlap barely
    discriminates good from bad in this regime (the real bad case
    scored 0.069 — *lower* than two genuinely good citations) — cosine
    is the signal that actually carries this, so this test uses a fake
    embed_fn returning near-orthogonal vectors rather than relying on
    token overlap alone, to test the logic that real evidence says
    matters, without needing a real model loaded for a unit test."""
    home_loan_chunk = _chunk(
        "home_loan_product_disclosure.md",
        1,
        "LVR means Loan to Value Ratio, the loan amount expressed as a percentage "
        "of the lender's valuation of the security property.",
    )
    checks = verify_citations(
        structured_citations=[
            {
                "source": "home_loan_product_disclosure.md",
                "chunk_index": 1,
                "claim": "The bonus interest rate on the Skyline Saver account is 4.25% p.a.",
            }
        ],
        answer_text="The bonus rate is 4.25%.",
        retrieved_chunks=[home_loan_chunk],
        embed_fn=lambda texts: [[1.0, 0.0], [0.0, 1.0]],  # orthogonal: claim vs. chunk
    )
    assert checks[0].exists is True
    assert checks[0].cosine_score == 0.0
    assert checks[0].weakly_grounded is True


def test_tier2_passes_matching_claim():
    """Identical text: token_overlap alone (1.0) is enough to pass here,
    confirming the non-cosine path works when overlap genuinely is high
    (which, per the calibration finding above, real paraphrased claims
    usually don't have — this is the easy case, not the typical one)."""
    chunk = _chunk("savings.md", 0, "The bonus interest rate on the Skyline Saver account is 4.25% p.a.")
    checks = verify_citations(
        structured_citations=[
            {
                "source": "savings.md",
                "chunk_index": 0,
                "claim": "The bonus interest rate on the Skyline Saver account is 4.25% p.a.",
            }
        ],
        answer_text="The bonus rate is 4.25%.",
        retrieved_chunks=[chunk],
    )
    assert checks[0].weakly_grounded is False


def test_tier2_passes_via_cosine_when_token_overlap_is_low():
    """The realistic case per calibration: a paraphrased claim with low
    token overlap but high cosine similarity should NOT be flagged —
    this is exactly the scenario all 5 real eval citations hit."""
    chunk = _chunk("savings.md", 0, "Interest rate: 4.25% p.a., calculated daily, paid monthly.")
    checks = verify_citations(
        structured_citations=[
            {"source": "savings.md", "chunk_index": 0, "claim": "The bonus rate is 4.25 percent annually."}
        ],
        answer_text="The bonus rate is 4.25%.",
        retrieved_chunks=[chunk],
        embed_fn=lambda texts: [[1.0, 0.0], [0.95, 0.31]],  # high cosine (~0.95)
    )
    assert checks[0].weakly_grounded is False


def test_cosine_similarity_sanity():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert abs(cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-9
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0  # zero vector, no crash


def test_extract_regex_citations():
    tags = extract_regex_citations("The rate is 4.25% [savings.md#0], fee is $0 [fees.md#2].")
    assert tags == [("savings.md", 0), ("fees.md", 2)]


def test_regex_safety_net_recovers_citation_missing_from_structured_array():
    chunk = _chunk("savings.md", 0, "The bonus rate is 4.25% p.a.")
    checks = verify_citations(
        structured_citations=[],  # model put nothing in the structured array...
        answer_text="The bonus rate is 4.25% p.a. [savings.md#0]",  # ...but did cite it in prose
        retrieved_chunks=[chunk],
    )
    assert len(checks) == 1
    assert checks[0].source_mechanism == "regex"
    assert checks[0].exists is True
    # No model-stated claim text available for a regex-recovered citation,
    # so Tier 2 grounding fields are skipped rather than guessed.
    assert checks[0].claim is None
    assert checks[0].weakly_grounded is None


def test_no_duplicate_when_citation_present_in_both_structured_and_prose():
    chunk = _chunk("savings.md", 0, "The bonus rate is 4.25% p.a.")
    checks = verify_citations(
        structured_citations=[{"source": "savings.md", "chunk_index": 0, "claim": "The bonus rate is 4.25% p.a."}],
        answer_text="The bonus rate is 4.25% p.a. [savings.md#0]",
        retrieved_chunks=[chunk],
    )
    assert len(checks) == 1  # not double-counted
    assert checks[0].source_mechanism == "structured"  # structured array wins precedence
