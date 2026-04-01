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
from src.rag_pipeline import build_retrieval_index, retrieve_passages_for_query
from src.reranker import Reranker


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


def evaluate_rag_answer_quality(
    examples: Sequence[QAExample],
    *,
    embedder: EmbeddingModel,
    corpus_chunks: Sequence[str],
    generator: TextGenerator,
    config: RAGGenerationConfig,
    reranker: Optional[Reranker] = None,
) -> Dict[str, float]:
    """
    End-to-end: retrieve → build prompt → generate → EM / F1 / gold hit (mean over examples).
    """
    faiss_index = build_retrieval_index(embedder, corpus_chunks)

    ems: List[float] = []
    f1s: List[float] = []
    hits: List[float] = []

    for ex in examples:
        passages = retrieve_passages_for_query(
            ex.question,
            embedder,
            corpus_chunks,
            faiss_index,
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
        ems.append(exact_match(prediction, ex.answer))
        f1s.append(token_f1(prediction, ex.answer))
        hits.append(gold_answer_hit(prediction, ex.answer))

    return {
        "exact_match": mean(ems),
        "token_f1": mean(f1s),
        "gold_hit": mean(hits),
        "n_questions": float(len(examples)),
    }
