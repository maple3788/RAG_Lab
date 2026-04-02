from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from src.answer_metrics import (
    exact_match,
    gold_answer_hit,
    mean,
    token_f1,
)
from src.context_truncation import TruncationStrategy, truncate_context
from src.embedder import EmbeddingModel
from src.generator import TextGenerator
from src.loader import QAExample
from src.prompts import format_rag_prompt
from src.rag_pipeline import (
    build_corpus_chunks_from_documents,
    build_retrieval_index,
    retrieve_passages_for_query,
)
from src.reranker import Reranker
from src.retriever import FaissIndex


def passages_to_context(passages: Sequence[str], *, separator: str = "\n\n") -> str:
    parts = [f"[{i + 1}] {p}" for i, p in enumerate(passages)]
    return separator.join(parts)


@dataclass(frozen=True)
class RAGGenerationConfig:
    retrieve_k: int = 10
    final_k: int = 3
    use_rerank: bool = False
    prompt_template: str = ""
    max_context_chars: int = 6000
    truncation: TruncationStrategy = "head"
    #: True: index only ``QAExample.contexts`` per question (TriviaQA RC). Ignores shared ``faiss_index``.
    per_example_retrieval: bool = False
    chunk_size: int = 512
    chunk_overlap: int = 64


def evaluate_rag_answer_quality(
    examples: Sequence[QAExample],
    *,
    embedder: EmbeddingModel,
    corpus_chunks: Sequence[str],
    generator: TextGenerator,
    config: RAGGenerationConfig,
    reranker: Optional[Reranker] = None,
    faiss_index: Optional[FaissIndex] = None,
) -> Dict[str, float]:
    """
    End-to-end: retrieve → build prompt → generate → EM / F1 / gold hit (mean over examples).
    Pass ``faiss_index`` to reuse the same index across runs (large corpora).
    """
    if not config.per_example_retrieval and faiss_index is None:
        faiss_index = build_retrieval_index(embedder, corpus_chunks)

    ems: List[float] = []
    f1s: List[float] = []
    hits: List[float] = []

    def _gold_strings(ex: QAExample) -> List[str]:
        if ex.answer_aliases:
            return list(ex.answer_aliases)
        return [ex.answer]

    for ex in examples:
        if config.per_example_retrieval:
            chunks = build_corpus_chunks_from_documents(
                list(ex.contexts),
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )
            if not chunks:
                ems.append(0.0)
                f1s.append(0.0)
                hits.append(0.0)
                continue
            fi = build_retrieval_index(embedder, chunks)
            passages = retrieve_passages_for_query(
                ex.question,
                embedder,
                chunks,
                fi,
                retrieve_k=config.retrieve_k,
                reranker=reranker if config.use_rerank else None,
                final_k=config.final_k,
            )
        else:
            passages = retrieve_passages_for_query(
                ex.question,
                embedder,
                corpus_chunks,
                faiss_index,  # type: ignore[arg-type]
                retrieve_k=config.retrieve_k,
                reranker=reranker if config.use_rerank else None,
                final_k=config.final_k,
            )
        raw_context = passages_to_context(passages)
        context = truncate_context(
            raw_context, config.max_context_chars, config.truncation
        )
        prompt = format_rag_prompt(
            config.prompt_template, context=context, question=ex.question
        )
        prediction = generator.generate(prompt)
        golds = _gold_strings(ex)
        ems.append(max(exact_match(prediction, g) for g in golds))
        f1s.append(max(token_f1(prediction, g) for g in golds))
        hits.append(max(gold_answer_hit(prediction, g) for g in golds))

    return {
        "exact_match": mean(ems),
        "token_f1": mean(f1s),
        "gold_hit": mean(hits),
        "n_questions": float(len(examples)),
    }
