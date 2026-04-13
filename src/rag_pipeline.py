from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List, Optional, Sequence, Tuple

from prometheus_client import Histogram

from src.chunker import chunk_documents, chunks_to_texts
from src.embedder import (
    EmbeddingModel,
    prepare_passages,
    prepare_query,
)
from src.loader import QAExample
from src.metrics import mean, recall_at_k
from src.hybrid_retrieval import BM25Resources, retrieve_hybrid_pool
from src.retriever import FaissIndex, build_faiss_index, gather_texts_by_indices, search
from src.reranker import Reranker

# Latency histograms (seconds, Prometheus convention). Used by ``retrieve_passages_pool_and_final``
# and optionally by ``record_rag_*`` for code paths outside this module (e.g. Streamlit ``run_pipeline``).
RAG_RETRIEVAL_SECONDS = Histogram(
    "rag_retrieval_seconds",
    "Wall time for retrieval in rag_pipeline (dense or hybrid + optional rerank to final pool)",
    buckets=(
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        30.0,
        60.0,
    ),
)
RAG_GENERATION_SECONDS = Histogram(
    "rag_generation_seconds",
    "Wall time for LLM answer generation in the RAG stack",
    buckets=(
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        30.0,
        60.0,
        120.0,
        300.0,
    ),
)


def record_rag_retrieval_latency(seconds: float) -> None:
    """Observe retrieval latency for callers that do not use ``retrieve_passages_pool_and_final``."""
    if seconds >= 0.0:
        RAG_RETRIEVAL_SECONDS.observe(seconds)


def record_rag_generation_latency(seconds: float) -> None:
    """Observe generation latency (e.g. from Streamlit or custom RAG loops)."""
    if seconds >= 0.0:
        RAG_GENERATION_SECONDS.observe(seconds)


@dataclass(frozen=True)
class RAGConfig:
    top_k: int = 3
    chunk_size: int = 512
    chunk_overlap: int = 64
    use_rerank: bool = False
    rerank_top_k: int = 3


@dataclass
class RAGArtifacts:
    chunk_texts: List[str]
    index_dim: int


def build_retrieval_corpus(
    examples: Sequence[QAExample],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    docs: List[str] = []
    for ex in examples:
        docs.extend(list(ex.contexts))
    chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunks_to_texts(chunks)


def build_corpus_chunks_from_documents(
    documents: Sequence[str],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    """Chunk arbitrary document strings (e.g. full BEIR corpus bodies) into passage strings."""
    chunks = chunk_documents(
        list(documents), chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return chunks_to_texts(chunks)


def build_retrieval_index(
    embedder: EmbeddingModel,
    corpus_chunks: Sequence[str],
) -> FaissIndex:
    passages = prepare_passages(embedder.name, corpus_chunks)
    corpus_emb = embedder.encode(passages)
    return build_faiss_index(corpus_emb)


def retrieve_passages_pool_and_final(
    question: str,
    embedder: EmbeddingModel,
    corpus_chunks: Sequence[str],
    faiss_index: FaissIndex,
    *,
    retrieve_k: int,
    reranker: Optional[Reranker] = None,
    final_k: int = 3,
    bm25_resources: Optional[BM25Resources] = None,
    rrf_k: int = 60,
    fusion_list_k: int | None = None,
) -> Tuple[List[str], List[str]]:
    """
    ``pool``: FAISS top-``retrieve_k`` passages (before rerank), or hybrid BM25+dense+RRF if ``bm25_resources`` is set.
    ``final``: passages passed to the LLM (after rerank or slice to ``final_k``).
    """
    t0 = perf_counter()
    try:
        k = min(retrieve_k, len(corpus_chunks))
        if k <= 0:
            return [], []
        if bm25_resources is not None:
            pool = retrieve_hybrid_pool(
                question,
                embedder=embedder,
                corpus_chunks=corpus_chunks,
                faiss_index=faiss_index,
                bm25_resources=bm25_resources,
                retrieve_k=k,
                rrf_k=rrf_k,
                fusion_list_k=fusion_list_k,
            )
        else:
            q = prepare_query(embedder.name, question)
            q_emb = embedder.encode([q])
            _, idx = search(faiss_index, q_emb, top_k=k)
            pool = gather_texts_by_indices(corpus_chunks, idx[0].tolist())

        if reranker is not None:
            final_texts, _ = reranker.rerank(
                question, pool, top_k=min(final_k, len(pool))
            )
        else:
            final_texts = pool[: min(final_k, len(pool))]
        return pool, final_texts
    finally:
        RAG_RETRIEVAL_SECONDS.observe(perf_counter() - t0)


def retrieve_passages_for_query(
    question: str,
    embedder: EmbeddingModel,
    corpus_chunks: Sequence[str],
    faiss_index: FaissIndex,
    *,
    retrieve_k: int,
    reranker: Optional[Reranker] = None,
    final_k: int = 3,
    bm25_resources: Optional[BM25Resources] = None,
    rrf_k: int = 60,
    fusion_list_k: int | None = None,
) -> List[str]:
    """Retrieve up to ``retrieve_k`` (dense or hybrid+RRF); optionally rerank down to ``final_k`` passages."""
    _, final_texts = retrieve_passages_pool_and_final(
        question,
        embedder,
        corpus_chunks,
        faiss_index,
        retrieve_k=retrieve_k,
        reranker=reranker,
        final_k=final_k,
        bm25_resources=bm25_resources,
        rrf_k=rrf_k,
        fusion_list_k=fusion_list_k,
    )
    return final_texts


def evaluate_retrieval(
    examples: Sequence[QAExample],
    *,
    embedder: EmbeddingModel,
    corpus_chunks: Sequence[str],
    top_k: int,
    reranker: Optional[Reranker] = None,
    rerank_top_k: int = 3,
) -> Dict[str, float]:
    faiss_index = build_retrieval_index(embedder, corpus_chunks)

    scores: List[float] = []
    for ex in examples:
        candidate_texts = retrieve_passages_for_query(
            ex.question,
            embedder,
            corpus_chunks,
            faiss_index,
            retrieve_k=top_k,
            reranker=reranker,
            final_k=rerank_top_k if reranker is not None else top_k,
        )

        scores.append(recall_at_k(candidate_texts, ex.answer))

    # Metric name matches how many chunks we score: initial top_k from FAISS, or
    # rerank_top_k after reranking (not the larger retrieve_k when reranking).
    recall_k = rerank_top_k if reranker is not None else top_k
    return {
        f"recall@{recall_k}": float(mean(scores)),
        "n_questions": float(len(examples)),
        "n_chunks": float(len(corpus_chunks)),
        "index_dim": float(faiss_index.dim),
    }

