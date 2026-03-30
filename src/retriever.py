from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import faiss
import numpy as np


@dataclass(frozen=True)
class FaissIndex:
    index: faiss.Index
    dim: int


def build_faiss_index(embeddings: np.ndarray) -> FaissIndex:
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be 2D array (n, d)")
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32, copy=False)

    n, d = embeddings.shape
    if n == 0:
        raise ValueError("embeddings is empty")

    # Use inner product. With normalized vectors this is cosine similarity.
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return FaissIndex(index=index, dim=d)


def search(
    faiss_index: FaissIndex,
    query_embeddings: np.ndarray,
    *,
    top_k: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    if query_embeddings.ndim == 1:
        query_embeddings = query_embeddings.reshape(1, -1)
    if query_embeddings.dtype != np.float32:
        query_embeddings = query_embeddings.astype(np.float32, copy=False)
    if query_embeddings.shape[1] != faiss_index.dim:
        raise ValueError(
            f"query dim {query_embeddings.shape[1]} != index dim {faiss_index.dim}"
        )
    scores, indices = faiss_index.index.search(query_embeddings, top_k)
    return scores, indices


def gather_texts_by_indices(texts: Sequence[str], indices: Sequence[int]) -> List[str]:
    out: List[str] = []
    for i in indices:
        if i < 0:
            continue
        out.append(texts[int(i)])
    return out

