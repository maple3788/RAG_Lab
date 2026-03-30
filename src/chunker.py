from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence


_WHITESPACE_RE = re.compile(r"\s+")


def _simple_tokenize(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    text = _WHITESPACE_RE.sub(" ", text)
    return text.split(" ")


@dataclass(frozen=True)
class Chunk:
    doc_id: int
    chunk_id: int
    text: str


def chunk_text(
    text: str,
    *,
    doc_id: int,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> List[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")

    tokens = _simple_tokenize(text)
    if not tokens:
        return []

    chunks: List[Chunk] = []
    start = 0
    cid = 0
    step = chunk_size - chunk_overlap
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(Chunk(doc_id=doc_id, chunk_id=cid, text=" ".join(chunk_tokens)))
        cid += 1
        if end == len(tokens):
            break
        start += step
    return chunks


def chunk_documents(
    documents: Sequence[str],
    *,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    for i, doc in enumerate(documents):
        all_chunks.extend(
            chunk_text(
                doc,
                doc_id=i,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )
    return all_chunks


def chunks_to_texts(chunks: Iterable[Chunk]) -> List[str]:
    return [c.text for c in chunks]

