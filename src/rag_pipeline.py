from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

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


def build_retrieval_index(
    embedder: EmbeddingModel,
    corpus_chunks: Sequence[str],
) -> FaissIndex:
    passages = prepare_passages(embedder.name, corpus_chunks)
    corpus_emb = embedder.encode(passages)
    return build_faiss_index(corpus_emb)


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
    k = min(retrieve_k, len(corpus_chunks))
    if k <= 0:
        return []
    q = prepare_query(embedder.name, question)
    q_emb = embedder.encode([q])
    _, idx = search(faiss_index, q_emb, top_k=k)
    candidate_texts = gather_texts_by_indices(corpus_chunks, idx[0].tolist())

    if reranker is not None:
        candidate_texts, _ = reranker.rerank(
            question, candidate_texts, top_k=min(final_k, len(candidate_texts))
        )
    else:
        candidate_texts = candidate_texts[: min(final_k, len(candidate_texts))]
    return candidate_texts


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

