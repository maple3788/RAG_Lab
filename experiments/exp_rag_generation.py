from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any, Dict, Optional, cast

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.context_truncation import TruncationStrategy
from src.embedder import load_embedding_model
from src.generator import (
    GeminiGenerator,
    MockGenerator,
    OllamaGenerator,
    OpenAICompatibleGenerator,
)
from src.loader import load_qa_jsonl
from src.prompts import PROMPT_TEMPLATES
from src.rag_generation import RAGGenerationConfig, evaluate_rag_answer_quality
from src.rag_pipeline import build_retrieval_corpus, build_retrieval_index
from src.reranker import load_reranker


def _metrics_for_table(m: Dict[str, Any]) -> Dict[str, Any]:
    return {
        k: v for k, v in m.items() if k not in ("per_example", "approx_total_tokens")
    }


def _log_run(
    experiment: str,
    setting: str,
    cfg: RAGGenerationConfig,
    m: Dict[str, Any],
    log_meta: Dict[str, str],
) -> None:
    from src.experiment_tracking import log_evaluation_batch

    flat = _metrics_for_table(m)
    pe = m.get("per_example") or []
    config = {**asdict(cfg), "embedding_model": log_meta["embedding_model"]}
    log_evaluation_batch(
        source="exp_rag_generation.py",
        experiment_name=experiment,
        setting_label=setting,
        embedding_model=log_meta["embedding_model"],
        llm_backend=log_meta["llm_backend"],
        llm_model=log_meta["llm_model"],
        config=config,
        metrics=flat,
        per_example=pe,
    )


def _make_generator(args: argparse.Namespace):
    if args.mock_generation:
        return MockGenerator()
    if args.llm_backend == "openai":
        return OpenAICompatibleGenerator(model=args.llm_model)
    if args.llm_backend == "ollama":
        return OllamaGenerator(model=args.llm_model)
    return GeminiGenerator(model=args.llm_model)


def run_compare_rerank(
    embedder,
    corpus_chunks,
    examples,
    generator,
    base: RAGGenerationConfig,
    reranker,
    faiss_index,
    log_meta: Optional[Dict[str, str]] = None,
):
    rows = []
    for use_rerank, label in [(False, "no_rerank"), (True, "with_rerank")]:
        cfg = RAGGenerationConfig(
            retrieve_k=base.retrieve_k,
            final_k=base.final_k,
            use_rerank=use_rerank,
            prompt_template=base.prompt_template,
            max_context_chars=base.max_context_chars,
            truncation=base.truncation,
            per_example_retrieval=base.per_example_retrieval,
            chunk_size=base.chunk_size,
            chunk_overlap=base.chunk_overlap,
            use_hybrid=base.use_hybrid,
            fusion_list_k=base.fusion_list_k,
            rrf_k=base.rrf_k,
            use_semantic_cache=base.use_semantic_cache,
            semantic_cache_threshold=base.semantic_cache_threshold,
            semantic_cache_max_entries=base.semantic_cache_max_entries,
            rewrite_on_empty_retrieval=base.rewrite_on_empty_retrieval,
        )
        m = evaluate_rag_answer_quality(
            examples,
            embedder=embedder,
            corpus_chunks=corpus_chunks,
            generator=generator,
            config=cfg,
            reranker=reranker if use_rerank else None,
            faiss_index=faiss_index,
            return_per_example=True,
        )
        rows.append({"experiment": "rerank", "setting": label, **_metrics_for_table(m)})
        if log_meta:
            _log_run("rerank", label, cfg, m, log_meta)
        print(label, _metrics_for_table(m))
    return rows


def run_compare_topk(
    embedder,
    corpus_chunks,
    examples,
    generator,
    base: RAGGenerationConfig,
    faiss_index,
    log_meta: Optional[Dict[str, str]] = None,
):
    rows = []
    for fk in [1, 3, 5]:
        cfg = RAGGenerationConfig(
            retrieve_k=max(base.retrieve_k, fk),
            final_k=fk,
            use_rerank=False,
            prompt_template=base.prompt_template,
            max_context_chars=base.max_context_chars,
            truncation=base.truncation,
            per_example_retrieval=base.per_example_retrieval,
            chunk_size=base.chunk_size,
            chunk_overlap=base.chunk_overlap,
            use_hybrid=base.use_hybrid,
            fusion_list_k=base.fusion_list_k,
            rrf_k=base.rrf_k,
            use_semantic_cache=base.use_semantic_cache,
            semantic_cache_threshold=base.semantic_cache_threshold,
            semantic_cache_max_entries=base.semantic_cache_max_entries,
            rewrite_on_empty_retrieval=base.rewrite_on_empty_retrieval,
        )
        m = evaluate_rag_answer_quality(
            examples,
            embedder=embedder,
            corpus_chunks=corpus_chunks,
            generator=generator,
            config=cfg,
            reranker=None,
            faiss_index=faiss_index,
            return_per_example=True,
        )
        rows.append(
            {"experiment": "topk", "setting": f"final_k={fk}", **_metrics_for_table(m)}
        )
        if log_meta:
            _log_run("topk", f"final_k={fk}", cfg, m, log_meta)
        print(f"final_k={fk}", _metrics_for_table(m))
    return rows


def run_compare_prompts(
    embedder,
    corpus_chunks,
    examples,
    generator,
    base: RAGGenerationConfig,
    faiss_index,
    log_meta: Optional[Dict[str, str]] = None,
):
    rows = []
    for name, tmpl in PROMPT_TEMPLATES.items():
        cfg = RAGGenerationConfig(
            retrieve_k=base.retrieve_k,
            final_k=base.final_k,
            use_rerank=base.use_rerank,
            prompt_template=tmpl,
            max_context_chars=base.max_context_chars,
            truncation=base.truncation,
            per_example_retrieval=base.per_example_retrieval,
            chunk_size=base.chunk_size,
            chunk_overlap=base.chunk_overlap,
            use_hybrid=base.use_hybrid,
            fusion_list_k=base.fusion_list_k,
            rrf_k=base.rrf_k,
            use_semantic_cache=base.use_semantic_cache,
            semantic_cache_threshold=base.semantic_cache_threshold,
            semantic_cache_max_entries=base.semantic_cache_max_entries,
            rewrite_on_empty_retrieval=base.rewrite_on_empty_retrieval,
        )
        m = evaluate_rag_answer_quality(
            examples,
            embedder=embedder,
            corpus_chunks=corpus_chunks,
            generator=generator,
            config=cfg,
            reranker=None,
            faiss_index=faiss_index,
            return_per_example=True,
        )
        rows.append({"experiment": "prompt", "setting": name, **_metrics_for_table(m)})
        if log_meta:
            _log_run("prompt", name, cfg, m, log_meta)
        print(name, _metrics_for_table(m))
    return rows


def run_compare_truncation(
    embedder,
    corpus_chunks,
    examples,
    generator,
    base: RAGGenerationConfig,
    *,
    max_context_chars: int,
    faiss_index,
    log_meta: Optional[Dict[str, str]] = None,
):
    rows = []
    for strat in ("head", "tail", "middle"):
        cfg = RAGGenerationConfig(
            retrieve_k=base.retrieve_k,
            final_k=base.final_k,
            use_rerank=False,
            prompt_template=base.prompt_template,
            max_context_chars=max_context_chars,
            truncation=cast(TruncationStrategy, strat),
            per_example_retrieval=base.per_example_retrieval,
            chunk_size=base.chunk_size,
            chunk_overlap=base.chunk_overlap,
            use_hybrid=base.use_hybrid,
            fusion_list_k=base.fusion_list_k,
            rrf_k=base.rrf_k,
            use_semantic_cache=base.use_semantic_cache,
            semantic_cache_threshold=base.semantic_cache_threshold,
            semantic_cache_max_entries=base.semantic_cache_max_entries,
            rewrite_on_empty_retrieval=base.rewrite_on_empty_retrieval,
        )
        m = evaluate_rag_answer_quality(
            examples,
            embedder=embedder,
            corpus_chunks=corpus_chunks,
            generator=generator,
            config=cfg,
            reranker=None,
            faiss_index=faiss_index,
            return_per_example=True,
        )
        rows.append(
            {"experiment": "truncation", "setting": strat, **_metrics_for_table(m)}
        )
        if log_meta:
            _log_run("truncation", strat, cfg, m, log_meta)
        print(strat, _metrics_for_table(m))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG end-to-end answer quality (EM, F1, gold hit)"
    )
    parser.add_argument(
        "--mode",
        choices=[
            "compare-rerank",
            "compare-topk",
            "compare-prompts",
            "compare-truncation",
            "all",
        ],
        default="all",
    )
    parser.add_argument(
        "--data-path", type=Path, default=ROOT / "datasets" / "qa_dataset.jsonl"
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--rerank-model", default="BAAI/bge-reranker-base")
    parser.add_argument(
        "--retrieve-k", type=int, default=10, help="FAISS top-k when reranking"
    )
    parser.add_argument(
        "--final-k",
        type=int,
        default=3,
        help="Passages fed to the LLM after retrieve/rerank",
    )
    parser.add_argument("--max-context-chars", type=int, default=6000)
    parser.add_argument("--use-semantic-cache", action="store_true")
    parser.add_argument("--semantic-cache-threshold", type=float, default=0.93)
    parser.add_argument("--semantic-cache-max-entries", type=int, default=512)
    parser.add_argument("--rewrite-on-empty-retrieval", action="store_true")
    parser.add_argument(
        "--truncation-chars",
        type=int,
        default=1200,
        help="For compare-truncation: tight char budget so head/tail/middle differ",
    )
    parser.add_argument(
        "--truncation", default="head", choices=("head", "tail", "middle")
    )
    parser.add_argument(
        "--llm-backend",
        choices=("gemini", "openai", "ollama"),
        default="gemini",
        help="gemini=Google AI; openai=OpenAI-compatible; ollama=local Ollama (no cloud LLM key)",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Model id (defaults: gemini-2.5-flash / gpt-4o-mini / llama3.2)",
    )
    parser.add_argument(
        "--mock-generation",
        action="store_true",
        help="Use MockGenerator (no API; for pipeline smoke tests only)",
    )
    parser.add_argument(
        "--no-experiment-db",
        action="store_true",
        help="Skip writing per-run rows to results/experiment_db.sqlite",
    )
    args = parser.parse_args()
    if args.llm_model is None:
        if args.llm_backend == "openai":
            args.llm_model = "gpt-4o-mini"
        elif args.llm_backend == "ollama":
            args.llm_model = "llama3.2"
        else:
            args.llm_model = "gemini-2.5-flash"

    examples = load_qa_jsonl(args.data_path)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_size = 512
    chunk_overlap = 64
    corpus_chunks = build_retrieval_corpus(
        examples, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    embedder = load_embedding_model(args.embedding_model, normalize=True)
    generator = _make_generator(args)

    base_tmpl = PROMPT_TEMPLATES["default"]
    trunc: TruncationStrategy = args.truncation  # type: ignore[assignment]

    base = RAGGenerationConfig(
        retrieve_k=args.retrieve_k,
        final_k=args.final_k,
        use_rerank=False,
        prompt_template=base_tmpl,
        max_context_chars=args.max_context_chars,
        truncation=trunc,
        per_example_retrieval=False,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        use_semantic_cache=bool(args.use_semantic_cache),
        semantic_cache_threshold=float(args.semantic_cache_threshold),
        semantic_cache_max_entries=int(args.semantic_cache_max_entries),
        rewrite_on_empty_retrieval=bool(args.rewrite_on_empty_retrieval),
    )

    reranker = None
    if args.mode in ("compare-rerank", "all"):
        reranker = load_reranker(args.rerank_model)

    if base.per_example_retrieval:
        faiss_index = None
    else:
        print("Building FAISS index (reused across all runs in this process)…")
        faiss_index = build_retrieval_index(embedder, corpus_chunks)

    log_meta: Optional[Dict[str, str]] = None
    if not args.no_experiment_db:
        log_meta = {
            "embedding_model": args.embedding_model,
            "llm_backend": args.llm_backend,
            "llm_model": args.llm_model,
        }

    all_rows = []
    if args.mode in ("compare-rerank", "all"):
        all_rows.extend(
            run_compare_rerank(
                embedder,
                corpus_chunks,
                examples,
                generator,
                base,
                reranker,
                faiss_index,
                log_meta,
            )
        )
    if args.mode in ("compare-topk", "all"):
        all_rows.extend(
            run_compare_topk(
                embedder,
                corpus_chunks,
                examples,
                generator,
                base,
                faiss_index,
                log_meta,
            )
        )
    if args.mode in ("compare-prompts", "all"):
        all_rows.extend(
            run_compare_prompts(
                embedder,
                corpus_chunks,
                examples,
                generator,
                base,
                faiss_index,
                log_meta,
            )
        )
    if args.mode in ("compare-truncation", "all"):
        all_rows.extend(
            run_compare_truncation(
                embedder,
                corpus_chunks,
                examples,
                generator,
                base,
                max_context_chars=args.truncation_chars,
                faiss_index=faiss_index,
                log_meta=log_meta,
            )
        )

    df = pd.DataFrame(all_rows)
    csv_path = out_dir / "rag_generation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
