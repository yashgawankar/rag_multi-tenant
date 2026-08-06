"""Direct unit tests for the chunker — previously validated only
indirectly through eval/isolation results."""
from src.chunking import chunk_text


def test_bullet_list_without_blank_lines_still_splits():
    """The original chunker only split on blank lines; a bullet list
    written as one line per item (no blank line between items) would
    never be divided there. This must be caught by the line-break tier."""
    bullets = "\n".join(f"- Item {i}: some descriptive text without a period" for i in range(20))
    chunks = chunk_text(bullets, target_size=80, overlap_chars=10)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 80 + 10  # generous bound; overlap can push slightly over


def test_short_text_is_a_single_chunk():
    chunks = chunk_text("A short paragraph that fits easily within the target size.", target_size=200)
    assert len(chunks) == 1


def test_respects_custom_length_fn():
    """length_fn lets the real call site measure in tokens instead of
    characters; a trivial custom length_fn should be honored exactly."""
    text = "one two three four five six seven eight nine ten"

    def word_count(s: str) -> int:
        return len(s.split())

    chunks = chunk_text(text, target_size=3, overlap_chars=1, length_fn=word_count)
    for chunk in chunks:
        assert word_count(chunk.text) <= 3 + 1  # + small overlap allowance


def test_pathological_single_long_word_terminates():
    """No whitespace, no punctuation, no blank lines at all — only the
    character-level last-resort tier can split this. Must not hang or
    raise, and must still respect target_size."""
    long_token = "x" * 500
    chunks = chunk_text(long_token, target_size=50, overlap_chars=5)
    assert len(chunks) > 1
    assert all(len(c.text) <= 50 + 5 for c in chunks)


def test_overlap_carries_context_into_next_chunk():
    para1 = "First paragraph with some identifying content ZZZMARKERZZZ here."
    para2 = "Second paragraph that is unrelated to the first one entirely."
    text = f"{para1}\n\n{para2}"
    chunks = chunk_text(text, target_size=len(para1) + 1, overlap_chars=20)
    assert len(chunks) == 2
    assert "ZZZMARKERZZZ" in chunks[0].text
    # raw para1[-20:] is "t ZZZMARKERZZZ here." - a mid-word cut of
    # "content". _overlap_tail trims it forward to the next real word,
    # so chunk 2 must start with "ZZZMARKERZZZ", never a stray "t".
    assert chunks[1].text.startswith("ZZZMARKERZZZ here.")
    assert para2 in chunks[1].text


def test_overlap_tail_never_starts_mid_word():
    """Direct regression test for the mid-word-cut bug found by tracing
    real overlap output on data/tenant_b/claims-policy.md: a raw
    text[-overlap_chars:] slice cut "Overview" into "w" + "Overvie",
    producing a chunk starting with the single stray character 'w'."""
    from src.chunking import _overlap_tail

    text = "...this document describes how claims are lodged and assessed."
    # raw text[-11:] is "d assessed." - a genuine mid-word cut of "lodged".
    raw = text[-11:]
    assert raw == "d assessed."

    tail = _overlap_tail(text, overlap_chars=11)
    assert tail == "assessed."  # trimmed forward to the next real word
    assert text.endswith(tail)  # still a genuine substring, not fabricated


def test_rejects_overlap_not_smaller_than_target():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("text", target_size=100, overlap_chars=100)
