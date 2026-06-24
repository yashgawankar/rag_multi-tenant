"""Hand-rolled, dependency-light chunker.

Splits on paragraph boundaries first (these docs are short markdown/text
files with meaningful paragraph structure), then greedily packs paragraphs
into ~target_chars windows with a sentence-aware overlap carried into the
next chunk. Falls back to a hard character split for any single paragraph
that exceeds target_chars on its own (e.g. a dense table-like block).

Chosen over a fixed-size token splitter because the source docs are short,
structured policy/product documents where a clause rarely needs splitting
mid-sentence — paragraph-aware splitting keeps each chunk self-contained,
which matters more than token-count precision at this corpus size.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    text: str
    chunk_index: int


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _hard_split(paragraph: str, target_chars: int) -> list[str]:
    sentences = _SENTENCE_END.split(paragraph)
    pieces, current = [], ""
    for sentence in sentences:
        if current and len(current) + 1 + len(sentence) > target_chars:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def chunk_text(text: str, target_chars: int = 800, overlap_chars: int = 120) -> list[Chunk]:
    if overlap_chars >= target_chars:
        raise ValueError("overlap_chars must be smaller than target_chars")

    paragraphs: list[str] = []
    for para in _split_paragraphs(text):
        if len(para) > target_chars:
            paragraphs.extend(_hard_split(para, target_chars))
        else:
            paragraphs.append(para)

    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= target_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)
        overlap_tail = current[-overlap_chars:] if current else ""
        current = f"{overlap_tail}\n\n{para}".strip() if overlap_tail else para

    if current:
        chunks.append(current)

    return [Chunk(text=c, chunk_index=i) for i, c in enumerate(chunks)]
