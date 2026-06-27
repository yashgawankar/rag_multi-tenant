"""Hand-rolled, dependency-light chunker — recursive-separator splitting
with a pluggable length function, not a fixed-size splitter.

Splitting descends through separator tiers, only when a piece is too big
to stand as its own chunk: blank-line paragraphs -> single line breaks ->
sentence boundaries -> whitespace-separated words -> raw character window
(absolute last resort, guarantees termination even on a single long token
like a URL with no whitespace at all). Earlier versions of this module
only had the first and third tiers, which meant a bullet list written as
one line per item with no blank lines between them (very common in real
documents) would never get split at the line tier, and would only be
caught by the sentence tier if items happened to end in punctuation —
neither is guaranteed. The line-break tier closes that gap.

`length_fn` defaults to `len` (character count) so this module stays a
pure, dependency-free, fast-to-test function on its own. The real
ingestion path (src/ingest.py) passes in the actual embedding model's
token counter instead — char count is only a rough proxy for token count,
and bge-small truncates silently past 512 tokens with no error, so the
real call site measures in real tokens, not characters.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

LengthFn = Callable[[str], int]

_BLANK_LINE = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE = re.compile(r"\s+")

# Tried in order; "" (empty string) means "split into individual characters"
# and is always last — it can always make a piece smaller, guaranteeing
# the recursion terminates no matter how pathological the input.
_SEPARATORS: list[str] = ["\n\n", "\n", "SENTENCE", "WORD", ""]


@dataclass(frozen=True)
class Chunk:
    text: str
    chunk_index: int


def _split_once(text: str, separator: str) -> list[str]:
    if separator == "\n\n":
        parts = _BLANK_LINE.split(text)
    elif separator == "\n":
        parts = text.split("\n")
    elif separator == "SENTENCE":
        parts = _SENTENCE_END.split(text)
    elif separator == "WORD":
        parts = _WHITESPACE.split(text)
    elif separator == "":
        parts = list(text)
    else:
        raise ValueError(f"Unknown separator: {separator!r}")
    return [p.strip() for p in parts if p.strip()]


def _atomize(text: str, target_size: int, length_fn: LengthFn, separators: list[str]) -> list[str]:
    """Recursively split text so that every returned piece individually
    satisfies length_fn(piece) <= target_size, descending through
    separator tiers only as needed."""
    if length_fn(text) <= target_size:
        return [text]
    if not separators:
        return [text]  # nothing left to try; caller accepts the overflow

    separator, *rest = separators
    pieces = _split_once(text, separator)
    if len(pieces) <= 1:
        # This separator didn't actually divide the text — try the next tier.
        return _atomize(text, target_size, length_fn, rest)

    atoms: list[str] = []
    for piece in pieces:
        atoms.extend(_atomize(piece, target_size, length_fn, rest))
    return atoms


def _pack(atoms: list[str], target_size: int, overlap_chars: int, length_fn: LengthFn) -> list[str]:
    """Greedily glue adjacent small atoms back together up to target_size,
    carrying a character-based overlap tail into the next chunk."""
    chunks: list[str] = []
    current = ""
    for atom in atoms:
        candidate = f"{current}\n\n{atom}".strip() if current else atom
        if length_fn(candidate) <= target_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
        overlap_tail = current[-overlap_chars:] if current else ""
        current = f"{overlap_tail}\n\n{atom}".strip() if overlap_tail else atom

    if current:
        chunks.append(current)
    return chunks


def chunk_text(
    text: str,
    target_size: int = 800,
    overlap_chars: int = 120,
    length_fn: LengthFn = len,
) -> list[Chunk]:
    if overlap_chars >= target_size:
        raise ValueError("overlap_chars must be smaller than target_size")

    atoms = _atomize(text, target_size, length_fn, _SEPARATORS)
    chunks = _pack(atoms, target_size, overlap_chars, length_fn)
    return [Chunk(text=c, chunk_index=i) for i, c in enumerate(chunks)]
