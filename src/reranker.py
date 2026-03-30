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


def load_reranker(model_name: str) -> Reranker:
    model = CrossEncoder(model_name)
    return Reranker(name=model_name, model=model)

