from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from sentence_transformers import CrossEncoder


@dataclass(frozen=True)
class Reranker:
    name: str
    model: CrossEncoder

    def rerank(
        self, query: str, candidates: Sequence[str], *, top_k: int
    ) -> Tuple[List[str], np.ndarray]:
        if not candidates:
            return [], np.asarray([], dtype=np.float32)
        pairs = [(query, c) for c in candidates]
        scores = self.model.predict(pairs)
        scores = np.asarray(scores, dtype=np.float32)
        order = np.argsort(-scores)
        top = order[:top_k]
        reranked = [candidates[int(i)] for i in top]
        return reranked, scores[top]


def load_reranker(model_name: str, *, device: str | None = None) -> Reranker:
    kwargs = {}
    if device is not None:
        kwargs["device"] = device
    model = CrossEncoder(model_name, **kwargs)
    return Reranker(name=model_name, model=model)
