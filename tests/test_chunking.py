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
    # chunk 2 should start with the literal last-20-chars tail of chunk 1,
    # which is how _pack constructs the overlap — not just "contains the
    # marker somewhere," but specifically carries that exact tail forward.
    assert chunks[1].text.startswith(para1[-20:])
    assert para2 in chunks[1].text


def test_rejects_overlap_not_smaller_than_target():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("text", target_size=100, overlap_chars=100)
