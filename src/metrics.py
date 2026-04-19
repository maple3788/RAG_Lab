from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Optional, Sequence

from src.answer_metrics import exact_match, gold_answer_hit, token_f1


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


def approx_token_count(*texts: str) -> int:
    """Heuristic token estimate when tiktoken is unavailable (≈4 chars/token)."""
    n = sum(len(t or "") for t in texts)
    return max(1, n // 4)


def compute_answer_metrics(prediction: str, reference: str) -> dict[str, float]:
    """Token F1, gold substring hit, and exact match for live UI evaluation."""
    ref = (reference or "").strip()
    pred = prediction or ""
    if not ref:
        return {"token_f1": 0.0, "gold_hit": 0.0, "exact_match": 0.0}
    return {
        "token_f1": float(token_f1(pred, ref)),
        "gold_hit": float(gold_answer_hit(pred, ref)),
        "exact_match": float(exact_match(pred, ref)),
    }


def retrieval_recall_proxy(retrieved_texts: Sequence[str], ground_truth: str) -> float:
    """Alias for dashboards that want an explicit retrieval-quality signal."""
    return recall_at_k(retrieved_texts, ground_truth)


def composite_ragas_score(scores: Mapping[str, Any]) -> Optional[float]:
    """
    Single Y-axis score from RAGAS-style dict (faithfulness, context precision, answer accuracy).
    Uses only finite numeric values present.
    """
    keys = (
        "response_groundedness",
        "context_relevance",
        "answer_accuracy",
    )
    vals: list[float] = []
    for k in keys:
        v = scores.get(k)
        if v is None:
            continue
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x):
            vals.append(x)
    if not vals:
        return None
    return mean(vals)
