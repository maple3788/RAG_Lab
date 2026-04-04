"""
Hybrid retrieval: BM25 (sparse) + dense (FAISS), merged via Reciprocal Rank Fusion (RRF).

Useful when dense embeddings miss lexical/table signals (e.g. "accuracy" vs "F1 85.99").
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from src.embedder import EmbeddingModel, prepare_query
from src.retriever import FaissIndex, gather_texts_by_indices, search


def tokenize_for_bm25(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass(frozen=True)
class BM25Resources:
    bm25: BM25Okapi
    tokenized_corpus: List[List[str]]


def build_bm25_resources(corpus_chunks: Sequence[str]) -> BM25Resources:
    tokenized = [tokenize_for_bm25(c) for c in corpus_chunks]
    # rank_bm25 tolerates empty docs poorly; ensure at least one token
    tokenized = [t if t else ["empty"] for t in tokenized]
    return BM25Resources(bm25=BM25Okapi(tokenized), tokenized_corpus=tokenized)


def _dense_ranked_indices(
    question: str,
    embedder: EmbeddingModel,
    faiss_index: FaissIndex,
    corpus_chunks: Sequence[str],
    *,
    top_k: int,
) -> List[int]:
    k = min(top_k, len(corpus_chunks))
    if k <= 0:
        return []
    q = prepare_query(embedder.name, question)
    q_emb = embedder.encode([q])
    _, idx = search(faiss_index, q_emb, top_k=k)
    return [int(i) for i in idx[0].tolist() if i >= 0]


def _bm25_ranked_indices(
    question: str,
    resources: BM25Resources,
    corpus_len: int,
    *,
    top_k: int,
) -> List[int]:
    k = min(top_k, corpus_len)
    if k <= 0:
        return []
    q_tokens = tokenize_for_bm25(question)
    if not q_tokens:
        q_tokens = ["query"]
    scores = resources.bm25.get_scores(q_tokens)
    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    out: List[int] = []
    for i in order:
        if len(out) >= k:
            break
        out.append(int(i))
    return out


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[int]],
    *,
    rrf_k: int = 60,
    max_results: int,
) -> List[int]:
    """
    RRF: score(doc) = sum_i 1 / (rrf_k + rank_i(doc)).
    """
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            if doc_id < 0:
                continue
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)

    merged = sorted(scores.keys(), key=lambda d: (-scores[d], d))
    return merged[:max_results]


def fused_top_indices(
    question: str,
    *,
    embedder: EmbeddingModel,
    corpus_chunks: Sequence[str],
    faiss_index: FaissIndex,
    bm25_resources: BM25Resources,
    retrieve_k: int,
    rrf_k: int = 60,
    fusion_list_k: int | None = None,
) -> List[int]:
    """
    Top-``retrieve_k`` chunk indices after merging dense and BM25 rankings via RRF.

    ``fusion_list_k``: how many candidates to take from *each* retriever before RRF (default:
    same as ``retrieve_k``). Use a value **larger** than ``retrieve_k`` (up to corpus size)
    so a chunk that is mid-ranked on one list can still win after fusion (common hybrid pattern).
    """
    n = len(corpus_chunks)
    k_out = min(retrieve_k, n)
    if k_out <= 0:
        return []
    k_lists = min(fusion_list_k if fusion_list_k is not None else k_out, n)
    dense_ranks = _dense_ranked_indices(
        question, embedder, faiss_index, corpus_chunks, top_k=k_lists
    )
    bm25_ranks = _bm25_ranked_indices(
        question, bm25_resources, n, top_k=k_lists
    )
    return reciprocal_rank_fusion(
        [dense_ranks, bm25_ranks],
        rrf_k=rrf_k,
        max_results=k_out,
    )


def retrieve_hybrid_pool(
    question: str,
    *,
    embedder: EmbeddingModel,
    corpus_chunks: Sequence[str],
    faiss_index: FaissIndex,
    bm25_resources: BM25Resources,
    retrieve_k: int,
    rrf_k: int = 60,
    fusion_list_k: int | None = None,
) -> List[str]:
    indices = fused_top_indices(
        question,
        embedder=embedder,
        corpus_chunks=corpus_chunks,
        faiss_index=faiss_index,
        bm25_resources=bm25_resources,
        retrieve_k=retrieve_k,
        rrf_k=rrf_k,
        fusion_list_k=fusion_list_k,
    )
    return gather_texts_by_indices(corpus_chunks, indices)
