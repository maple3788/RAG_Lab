from __future__ import annotations

from typing import Iterable, Sequence


def recall_at_k(retrieved_texts: Sequence[str], ground_truth: str) -> float:
    gt = (ground_truth or "").strip().lower()
    if not gt:
        return 0.0
    for t in retrieved_texts:
        if gt in (t or "").lower():
            return 1.0
    return 0.0


def mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    if not xs:
        return 0.0
    return sum(xs) / len(xs)

