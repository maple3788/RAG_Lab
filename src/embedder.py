from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class EmbeddingModel:
    name: str
    model: SentenceTransformer
    normalize: bool = True

    def encode(self, texts: Sequence[str], *, batch_size: int = 64) -> np.ndarray:
        vecs = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        if not isinstance(vecs, np.ndarray):
            vecs = np.asarray(vecs)
        return vecs.astype(np.float32, copy=False)


def load_embedding_model(model_name: str, *, normalize: bool = True) -> EmbeddingModel:
    model = SentenceTransformer(model_name)
    return EmbeddingModel(name=model_name, model=model, normalize=normalize)


def is_e5(model_name: str) -> bool:
    mn = model_name.lower()
    return "e5" in mn


def prepare_query(model_name: str, query: str) -> str:
    # E5 models commonly expect "query: ..." prefix.
    if is_e5(model_name):
        return f"query: {query}"
    return query


def prepare_passage(model_name: str, passage: str) -> str:
    if is_e5(model_name):
        return f"passage: {passage}"
    return passage


def prepare_passages(model_name: str, passages: Iterable[str]) -> List[str]:
    return [prepare_passage(model_name, p) for p in passages]

