"""Splits a document into overlapping, size-bounded chunks for embedding
and retrieval.

The algorithm runs in two phases:

1. Split — break the document down into pieces small enough to embed,
   trying progressively finer separators (paragraph, line, sentence,
   word, character) and only descending to a finer one where a piece is
   still too large. Most paragraphs need no more than the first tier.
2. Pack — greedily glue those small pieces back together up to a target
   size, so retrieval isn't left with one chunk per tiny paragraph. Each
   new chunk is seeded with a short overlap of the previous chunk's tail,
   so a fact sitting near a chunk boundary keeps some surrounding context
   on both sides.

Size is measured via a pluggable `length_fn` (default: `len`, i.e. raw
character count) rather than a hardcoded metric, so callers that care
about token limits — e.g. an embedding model with a fixed max sequence
length — can pass in a real tokenizer instead.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

LengthFn = Callable[[str], int]

_BLANK_LINE = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE = re.compile(r"\s+")

# Separator tiers, coarsest first. CHARACTER is the last resort: splitting
# into individual characters always succeeds on any text longer than one
# character, which guarantees the split phase terminates regardless of
# input (e.g. one long URL with no whitespace at all).
PARAGRAPH, LINE, SENTENCE, WORD, CHARACTER = range(5)


@dataclass(frozen=True)
class Chunk:
    """One piece of a chunked document.

    Attributes:
        text: The chunk's text content.
        chunk_index: Position of this chunk within the document, starting
            at 0, in the order chunk_text() produced them.
    """

    text: str
    chunk_index: int


def _split_by_tier(text: str, tier: int) -> list[str]:
    """Split text once, using the separator for the given tier.

    Returns the resulting pieces with blank/whitespace-only ones dropped.
    A single-item result means this tier's separator does not occur in
    the text (e.g. no blank lines to split on).
    """
    if tier == PARAGRAPH:
        pieces = _BLANK_LINE.split(text)
    elif tier == LINE:
        pieces = text.split("\n")
    elif tier == SENTENCE:
        pieces = _SENTENCE_END.split(text)
    elif tier == WORD:
        pieces = _WHITESPACE.split(text)
    else:  # CHARACTER
        pieces = list(text)
    return [p.strip() for p in pieces if p.strip()]


def _split_into_atoms(text: str, target_size: int, length_fn: LengthFn) -> list[str]:
    """Break text into pieces that each satisfy length_fn(piece) <= target_size.

    Makes one pass per separator tier, from coarsest to finest. On each
    pass, only pieces still over target_size are split further; anything
    already small enough is carried through unchanged.
    """
    atoms = [text]

    for tier in (PARAGRAPH, LINE, SENTENCE, WORD, CHARACTER):
        next_atoms = []
        for atom in atoms:
            if length_fn(atom) <= target_size:
                next_atoms.append(atom)
                continue

            pieces = _split_by_tier(atom, tier)
            if len(pieces) > 1:
                next_atoms.extend(pieces)
            else:
                next_atoms.append(atom)  # this tier made no progress; try the next one
        atoms = next_atoms

    return atoms


def _overlap_tail(text: str, overlap_chars: int) -> str:
    """Return the last overlap_chars of text, trimmed forward to the next
    word boundary.

    A raw text[-overlap_chars:] slice can land inside a word rather than
    between two words. Trimming to the following space keeps the overlap
    a sequence of whole words, which is both more readable and more
    useful as a lexical/semantic signal than a partial word fragment. If
    no space is found in the slice (e.g. it lands inside one very long
    token), the raw slice is returned unchanged.
    """
    raw = text[-overlap_chars:]
    space = raw.find(" ")
    if space == -1:
        return raw
    return raw[space + 1:]


def _pack_into_chunks(atoms: list[str], target_size: int, overlap_chars: int, length_fn: LengthFn) -> list[str]:
    """Greedily combine small atoms into chunks of up to target_size.

    Atoms are joined in order, accumulating into the current chunk until
    the next one would push it over target_size. At that point the
    current chunk is closed off, and the next chunk is seeded with the
    closed chunk's overlap tail (see _overlap_tail) so context carries
    across the boundary.
    """
    chunks: list[str] = []
    current = ""

    for atom in atoms:
        if current:
            candidate = f"{current}\n\n{atom}".strip()
        else:
            candidate = atom

        if length_fn(candidate) <= target_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
            overlap_tail = _overlap_tail(current, overlap_chars)
            current = f"{overlap_tail}\n\n{atom}".strip()
        else:
            current = atom

    if current:
        chunks.append(current)

    return chunks


def chunk_text(
    text: str,
    target_size: int = 800,
    overlap_chars: int = 120,
    length_fn: LengthFn = len,
) -> list[Chunk]:
    """Split text into overlapping, size-bounded chunks.

    Args:
        text: The document text to chunk.
        target_size: Maximum size of each chunk, as measured by length_fn.
        overlap_chars: Number of characters of trailing context carried
            from one chunk into the next. Must be smaller than target_size.
        length_fn: Function used to measure size; defaults to character
            count. Pass a tokenizer's count function to size chunks by
            token count instead.

    Returns:
        A list of Chunk objects in document order.

    Raises:
        ValueError: If overlap_chars is not smaller than target_size.
    """
    if overlap_chars >= target_size:
        raise ValueError("overlap_chars must be smaller than target_size")

    atoms = _split_into_atoms(text, target_size, length_fn)
    chunks = _pack_into_chunks(atoms, target_size, overlap_chars, length_fn)
    return [Chunk(text=c, chunk_index=i) for i, c in enumerate(chunks)]
