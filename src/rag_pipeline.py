from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from src.chunker import chunk_documents, chunks_to_texts
from src.embedder import (
    EmbeddingModel,
    prepare_passages,
    prepare_query,
)
from src.loader import QAExample
from src.metrics import mean, recall_at_k
from src.retriever import FaissIndex, build_faiss_index, gather_texts_by_indices, search
from src.reranker import Reranker


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
) -> Tuple[List[str], List[str]]:
    """
    ``pool``: FAISS top-``retrieve_k`` passages (before rerank).
    ``final``: passages passed to the LLM (after rerank or slice to ``final_k``).
    """
    k = min(retrieve_k, len(corpus_chunks))
    if k <= 0:
        return [], []
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


def retrieve_passages_for_query(
    question: str,
    embedder: EmbeddingModel,
    corpus_chunks: Sequence[str],
    faiss_index: FaissIndex,
    *,
    retrieve_k: int,
    reranker: Optional[Reranker] = None,
    final_k: int = 3,
) -> List[str]:
    """Retrieve up to `retrieve_k` from FAISS; optionally rerank down to `final_k` passages."""
    _, final_texts = retrieve_passages_pool_and_final(
        question,
        embedder,
        corpus_chunks,
        faiss_index,
        retrieve_k=retrieve_k,
        reranker=reranker,
        final_k=final_k,
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

