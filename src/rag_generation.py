from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from traceback import format_exc
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from src.answer_metrics import (
    exact_match,
    gold_answer_hit,
    mean,
    token_f1,
)
from src.context_truncation import TruncationStrategy, truncate_context
from src.embedder import EmbeddingModel, prepare_query
from src.generator import TextGenerator
from src.loader import QAExample
from src.prompts import format_rag_prompt
from src.hybrid_retrieval import BM25Resources, build_bm25_resources
from src.rag_pipeline import (
    build_corpus_chunks_from_documents,
    build_retrieval_index,
    retrieve_passages_for_hyde_document,
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
    #: BM25 + dense RRF pool (per-example corpus only); ignored unless ``per_example_retrieval``.
    use_hybrid: bool = False
    #: Candidates per retriever before RRF; ``None`` = use full corpus (capped by chunk count).
    fusion_list_k: Optional[int] = None
    rrf_k: int = 60
    #: If true, use embedding-similarity cache to reuse answers for semantically similar queries.
    use_semantic_cache: bool = False
    #: Similarity threshold (cosine on query embeddings) for semantic cache hit.
    semantic_cache_threshold: float = 0.93
    #: Upper bound on in-memory semantic cache entries.
    semantic_cache_max_entries: int = 512
    #: Optional fallback: rewrite query once if no passages were retrieved.
    rewrite_on_empty_retrieval: bool = False
    #: ``multi_query``: LLM paraphrases / sub-queries, merge + dedupe then rerank on original
    #: question. ``hyde``: hypothetical passage, dense retrieval via passage embedding only
    #: (BM25 hybrid is not fused with HyDE). ``none``: single query string.
    query_expansion: Literal["none", "multi_query", "hyde"] = "none"
    #: Max queries including the original (``multi_query`` only).
    expansion_max_queries: int = 4
    #: If true, record per-example failures and continue evaluation instead of raising.
    continue_on_error: bool = True


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _rewrite_query(generator: TextGenerator, question: str) -> str:
    prompt = (
        "Rewrite the user question for retrieval only. Keep intent unchanged, "
        "expand abbreviations if obvious, include key entities, output one line.\n\n"
        f"Question: {question}\nRewritten query:"
    )
    rewritten = generator.generate(prompt).strip()
    return rewritten if rewritten else question


def _multi_query_variants(
    generator: TextGenerator,
    question: str,
    *,
    max_queries: int,
) -> List[str]:
    max_queries = max(1, max_queries)
    prompt = (
        "For passage retrieval, list short search queries that could find relevant text. "
        "Use paraphrases, synonyms, and alternative keyword phrasings. "
        "One query per line; no numbering, bullets, or explanation.\n\n"
        f"Question:\n{question}\n"
    )
    raw = generator.generate(prompt).strip()
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    out: List[str] = [question]
    lower_seen = {question.lower()}
    for ln in lines:
        if len(out) >= max_queries:
            break
        low = ln.lower()
        if low not in lower_seen:
            lower_seen.add(low)
            out.append(ln)
    return out


def _hyde_hypothetical_passage(generator: TextGenerator, question: str) -> str:
    prompt = (
        "Write a short hypothetical passage (2-4 sentences) that could appear in a document "
        "and would help answer the question. Use concrete terms and entities when possible. "
        "Do not explain the task.\n\n"
        f"Question: {question}\n\nPassage:"
    )
    text = generator.generate(prompt).strip()
    return text if text else question


def _retrieve_merged_multi_query(
    queries: Sequence[str],
    original_question: str,
    embedder: EmbeddingModel,
    corpus_chunks: Sequence[str],
    faiss_index: FaissIndex,
    *,
    retrieve_k: int,
    reranker: Optional[Reranker],
    final_k: int,
    bm25_resources: Optional[BM25Resources],
    rrf_k: int,
    fusion_list_k: Optional[int],
) -> List[str]:
    """Per-variant retrieval without rerank; dedupe by full passage text; one rerank on merged pool."""
    seen: set[str] = set()
    pool: List[str] = []
    cap = max(retrieve_k * 4, final_k * 4, 32)
    for q in queries:
        if len(pool) >= cap:
            break
        part = retrieve_passages_for_query(
            q,
            embedder,
            corpus_chunks,
            faiss_index,
            retrieve_k=retrieve_k,
            reranker=None,
            final_k=retrieve_k,
            bm25_resources=bm25_resources,
            rrf_k=rrf_k,
            fusion_list_k=fusion_list_k,
        )
        for t in part:
            if t not in seen:
                seen.add(t)
                pool.append(t)
    pool = pool[:cap]
    if reranker is not None:
        texts, _ = reranker.rerank(
            original_question, pool, top_k=min(final_k, len(pool))
        )
        return list(texts)
    return pool[: min(final_k, len(pool))]


def _run_retrieval_for_question(
    question: str,
    *,
    force_plain: bool,
    generator: TextGenerator,
    config: RAGGenerationConfig,
    embedder: EmbeddingModel,
    corpus_chunks: Sequence[str],
    faiss_index: FaissIndex,
    reranker: Optional[Reranker],
    bm25_resources: Optional[BM25Resources],
    fusion_list_k: Optional[int],
) -> List[str]:
    rr = reranker if config.use_rerank else None
    if force_plain or config.query_expansion == "none":
        return retrieve_passages_for_query(
            question,
            embedder,
            corpus_chunks,
            faiss_index,
            retrieve_k=config.retrieve_k,
            reranker=rr,
            final_k=config.final_k,
            bm25_resources=bm25_resources,
            rrf_k=config.rrf_k,
            fusion_list_k=fusion_list_k,
        )
    if config.query_expansion == "multi_query":
        variants = _multi_query_variants(
            generator,
            question,
            max_queries=config.expansion_max_queries,
        )
        return _retrieve_merged_multi_query(
            variants,
            question,
            embedder,
            corpus_chunks,
            faiss_index,
            retrieve_k=config.retrieve_k,
            reranker=rr,
            final_k=config.final_k,
            bm25_resources=bm25_resources,
            rrf_k=config.rrf_k,
            fusion_list_k=fusion_list_k,
        )
    hyde = _hyde_hypothetical_passage(generator, question)
    return retrieve_passages_for_hyde_document(
        hyde,
        question,
        embedder,
        corpus_chunks,
        faiss_index,
        retrieve_k=config.retrieve_k,
        reranker=rr,
        final_k=config.final_k,
    )


def evaluate_rag_answer_quality(
    examples: Sequence[QAExample],
    *,
    embedder: EmbeddingModel,
    corpus_chunks: Sequence[str],
    generator: TextGenerator,
    config: RAGGenerationConfig,
    reranker: Optional[Reranker] = None,
    faiss_index: Optional[FaissIndex] = None,
    return_per_example: bool = False,
) -> Dict[str, Any]:
    """
    End-to-end: retrieve → build prompt → generate → EM / F1 / gold hit (mean over examples).
    Pass ``faiss_index`` to reuse the same index across runs (large corpora).

    Optional ``config.query_expansion``: ``multi_query`` (LLM paraphrases, merged retrieval),
    ``hyde`` (hypothetical document embedding; dense only), or ``none``. If
    ``rewrite_on_empty_retrieval`` fires, the retry uses a single rewritten query without
    expansion to limit extra LLM calls.
    """
    if not config.per_example_retrieval and faiss_index is None:
        faiss_index = build_retrieval_index(embedder, corpus_chunks)

    ems: List[float] = []
    f1s: List[float] = []
    hits: List[float] = []
    lat_retrieve_ms: List[float] = []
    lat_generate_ms: List[float] = []
    lat_total_ms: List[float] = []
    lat_rewrite_ms: List[float] = []
    cache_hits = 0.0
    cache_lookups = 0.0
    failures = 0.0
    semantic_cache: List[Tuple[np.ndarray, str]] = []
    per_example_rows: List[Dict[str, Any]] = []

    def _gold_strings(ex: QAExample) -> List[str]:
        if ex.answer_aliases:
            return list(ex.answer_aliases)
        return [ex.answer]

    def _record_failure(
        ex: QAExample,
        *,
        prediction: str = "",
        passages: Optional[Sequence[str]] = None,
        retrieve_ms: float = 0.0,
        rewrite_ms: float = 0.0,
        generate_ms: float = 0.0,
        error: str,
        error_traceback: Optional[str] = None,
        token_count: int = 1,
    ) -> None:
        nonlocal failures
        failures += 1.0
        ems.append(0.0)
        f1s.append(0.0)
        hits.append(0.0)
        lat_retrieve_ms.append(retrieve_ms)
        lat_rewrite_ms.append(rewrite_ms)
        lat_generate_ms.append(generate_ms)
        lat_total_ms.append((perf_counter() - t0_total) * 1000.0)
        if return_per_example:
            row: Dict[str, Any] = {
                "question": ex.question,
                "reference_answer": ex.answer,
                "prediction": prediction,
                "retrieved_passages": list(passages or []),
                "token_f1": 0.0,
                "gold_hit": 0.0,
                "exact_match": 0.0,
                "latency_total_ms": lat_total_ms[-1],
                "token_count": token_count,
                "error": error,
            }
            if error_traceback is not None:
                row["error_traceback"] = error_traceback
            per_example_rows.append(row)

    for ex in examples:
        t0_total = perf_counter()
        cache_lookups += 1.0
        cached_prediction: Optional[str] = None
        passages: List[str] = []
        rewrite_ms = 0.0
        retrieve_ms = 0.0
        generate_ms = 0.0
        q_emb: Optional[np.ndarray] = None

        try:
            if config.use_semantic_cache:
                q_text = prepare_query(embedder.name, ex.question)
                q_emb = embedder.encode([q_text])[0]
                best_sim = -1.0
                best_answer = None
                for emb, ans in semantic_cache:
                    sim = _cosine_similarity(q_emb, emb)
                    if sim > best_sim:
                        best_sim = sim
                        best_answer = ans
                if best_answer is not None and best_sim >= config.semantic_cache_threshold:
                    cache_hits += 1.0
                    cached_prediction = best_answer

            if config.per_example_retrieval:
                chunks = build_corpus_chunks_from_documents(
                    list(ex.contexts),
                    chunk_size=config.chunk_size,
                    chunk_overlap=config.chunk_overlap,
                )
                if not chunks:
                    _record_failure(
                        ex,
                        passages=[],
                        error="No chunks available for per-example retrieval.",
                    )
                    continue
                fi = build_retrieval_index(embedder, chunks)
                bm25 = (
                    build_bm25_resources(chunks)
                    if config.use_hybrid
                    else None
                )
                fusion = config.fusion_list_k
                if bm25 is not None and fusion is None:
                    fusion = len(chunks)
                t0_retrieve = perf_counter()
                if cached_prediction is not None:
                    passages = []
                else:
                    passages = _run_retrieval_for_question(
                        ex.question,
                        force_plain=False,
                        generator=generator,
                        config=config,
                        embedder=embedder,
                        corpus_chunks=chunks,
                        faiss_index=fi,
                        reranker=reranker,
                        bm25_resources=bm25,
                        fusion_list_k=fusion,
                    )
                if (
                    cached_prediction is None
                    and config.rewrite_on_empty_retrieval
                    and not passages
                ):
                    t0_rewrite = perf_counter()
                    rewritten = _rewrite_query(generator, ex.question)
                    rewrite_ms = (perf_counter() - t0_rewrite) * 1000.0
                    passages = _run_retrieval_for_question(
                        rewritten,
                        force_plain=True,
                        generator=generator,
                        config=config,
                        embedder=embedder,
                        corpus_chunks=chunks,
                        faiss_index=fi,
                        reranker=reranker,
                        bm25_resources=bm25,
                        fusion_list_k=fusion,
                    )
                retrieve_ms = (perf_counter() - t0_retrieve) * 1000.0
                lat_retrieve_ms.append(retrieve_ms)
                lat_rewrite_ms.append(rewrite_ms)
            else:
                t0_retrieve = perf_counter()
                if cached_prediction is not None:
                    passages = []
                else:
                    passages = _run_retrieval_for_question(
                        ex.question,
                        force_plain=False,
                        generator=generator,
                        config=config,
                        embedder=embedder,
                        corpus_chunks=corpus_chunks,
                        faiss_index=faiss_index,  # type: ignore[arg-type]
                        reranker=reranker,
                        bm25_resources=None,
                        fusion_list_k=None,
                    )
                if (
                    cached_prediction is None
                    and config.rewrite_on_empty_retrieval
                    and not passages
                ):
                    t0_rewrite = perf_counter()
                    rewritten = _rewrite_query(generator, ex.question)
                    rewrite_ms = (perf_counter() - t0_rewrite) * 1000.0
                    passages = _run_retrieval_for_question(
                        rewritten,
                        force_plain=True,
                        generator=generator,
                        config=config,
                        embedder=embedder,
                        corpus_chunks=corpus_chunks,
                        faiss_index=faiss_index,  # type: ignore[arg-type]
                        reranker=reranker,
                        bm25_resources=None,
                        fusion_list_k=None,
                    )
                retrieve_ms = (perf_counter() - t0_retrieve) * 1000.0
                lat_retrieve_ms.append(retrieve_ms)
                lat_rewrite_ms.append(rewrite_ms)
            raw_context = passages_to_context(passages)
            context = truncate_context(
                raw_context, config.max_context_chars, config.truncation
            )
            prompt = format_rag_prompt(
                config.prompt_template, context=context, question=ex.question
            )
            t0_generate = perf_counter()
            prediction = cached_prediction if cached_prediction is not None else generator.generate(prompt)
            generate_ms = (perf_counter() - t0_generate) * 1000.0
            lat_generate_ms.append(generate_ms)

            if config.use_semantic_cache and cached_prediction is None:
                if q_emb is None:
                    q_text = prepare_query(embedder.name, ex.question)
                    q_emb = embedder.encode([q_text])[0]
                if len(semantic_cache) >= config.semantic_cache_max_entries:
                    semantic_cache.pop(0)
                semantic_cache.append((q_emb, prediction))

            golds = _gold_strings(ex)
            ems.append(max(exact_match(prediction, g) for g in golds))
            f1s.append(max(token_f1(prediction, g) for g in golds))
            hits.append(max(gold_answer_hit(prediction, g) for g in golds))
            lat_total_ms.append((perf_counter() - t0_total) * 1000.0)
            if return_per_example:
                tok = max(1, (len(prompt) + len(prediction)) // 4)
                per_example_rows.append(
                    {
                        "question": ex.question,
                        "reference_answer": ex.answer,
                        "prediction": prediction,
                        "retrieved_passages": list(passages),
                        "token_f1": f1s[-1],
                        "gold_hit": hits[-1],
                        "exact_match": ems[-1],
                        "latency_total_ms": lat_total_ms[-1],
                        "token_count": tok,
                    }
                )
        except Exception as e:
            if not config.continue_on_error:
                raise
            _record_failure(
                ex,
                passages=passages,
                retrieve_ms=retrieve_ms,
                rewrite_ms=rewrite_ms,
                generate_ms=generate_ms,
                error=str(e),
                error_traceback=format_exc(),
            )

    out: Dict[str, Any] = {
        "exact_match": mean(ems),
        "token_f1": mean(f1s),
        "gold_hit": mean(hits),
        "latency_total_ms": mean(lat_total_ms),
        "latency_retrieve_ms": mean(lat_retrieve_ms),
        "latency_rewrite_ms": mean(lat_rewrite_ms),
        "latency_generate_ms": mean(lat_generate_ms),
        "semantic_cache_hit_rate": (cache_hits / cache_lookups) if cache_lookups > 0 else 0.0,
        "n_questions": float(len(examples)),
        "n_failures": failures,
    }
    if return_per_example:
        out["per_example"] = per_example_rows
        out["approx_total_tokens"] = float(
            sum(int(x.get("token_count") or 0) for x in per_example_rows)
        )
    return out
