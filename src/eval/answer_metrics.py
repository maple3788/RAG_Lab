from __future__ import annotations

import re
import string
from typing import Iterable, List


def _normalize_answer(text: str) -> str:
    """Lowercase, strip, collapse whitespace — lightweight SQuAD-style norm."""
    text = (text or "").lower().replace("-", " ")
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _tokens(text: str) -> List[str]:
    return _normalize_answer(text).split()


def exact_match(prediction: str, ground_truth: str) -> float:
    return (
        1.0 if _normalize_answer(prediction) == _normalize_answer(ground_truth) else 0.0
    )


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_toks = _tokens(prediction)
    gold_toks = _tokens(ground_truth)
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0
    pred_counts: dict[str, int] = {}
    for t in pred_toks:
        pred_counts[t] = pred_counts.get(t, 0) + 1
    gold_counts: dict[str, int] = {}
    for t in gold_toks:
        gold_counts[t] = gold_counts.get(t, 0) + 1
    overlap = 0
    for t, c in pred_counts.items():
        overlap += min(c, gold_counts.get(t, 0))
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_toks)
    recall = overlap / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def gold_answer_hit(prediction: str, ground_truth: str) -> float:
    """1.0 if the gold answer string appears in the prediction (case-insensitive)."""
    gt = (ground_truth or "").strip()
    if not gt:
        return 0.0
    pred_l = (prediction or "").lower()
    return 1.0 if gt.lower() in pred_l else 0.0


def mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    if not xs:
        return 0.0
    return sum(xs) / len(xs)
