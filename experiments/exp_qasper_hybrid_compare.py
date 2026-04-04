"""
QASPER: compare **dense-only** vs **BM25+dense (RRF)** retrieval on real examples.

**Mode ``retrieval`` (default):** no LLM — measures whether any gold alias appears in the
top-``retrieve_k`` pool (and in the first ``final_k`` chunks without rerank). Fast.

**Mode ``generation``:** full RAG with ``evaluate_rag_answer_quality`` twice (dense vs hybrid).
Requires a generator (use ``--mock-generation`` for a quick smoke test).

Examples::

    # Retrieval-only, 200 questions (recommended first)
    python experiments/exp_qasper_hybrid_compare.py --mode retrieval --max-examples 200

    # End-to-end on 20 examples with mock LLM
    python experiments/exp_qasper_hybrid_compare.py --mode generation --max-examples 20 --mock-generation
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.exp_rag_generation import _make_generator
from src.prompts import PROMPT_TEMPLATES
from src.embedder import load_embedding_model
from src.error_analysis import gold_in_any_chunk
from src.hybrid_retrieval import build_bm25_resources
from src.loader import QAExample
from src.qasper_hf import DEFAULT_QASPER_REVISION, load_qasper_hf
from src.rag_generation import RAGGenerationConfig, evaluate_rag_answer_quality
from src.rag_pipeline import (
    build_corpus_chunks_from_documents,
    build_retrieval_index,
    retrieve_passages_pool_and_final,
)
from src.reranker import load_reranker


def _aliases(ex: QAExample) -> list[str]:
    if ex.answer_aliases:
        return list(ex.answer_aliases)
    return [ex.answer]


def run_retrieval_compare(
    examples: list[QAExample],
    *,
    embedder,
    retrieve_k: int,
    final_k: int,
    chunk_size: int,
    chunk_overlap: int,
    fusion_list_k: int | None,
    rrf_k: int,
    use_rerank: bool,
    reranker,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows: list[dict] = []
    n = 0
    gd_pool = gh_pool = 0
    gd_final = gh_final = 0
    hybrid_saves = hybrid_hurts = 0

    for ex in examples:
        chunks = build_corpus_chunks_from_documents(
            list(ex.contexts),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        if not chunks:
            continue
        aliases = _aliases(ex)
        if not aliases:
            continue

        fi = build_retrieval_index(embedder, chunks)
        bm25 = build_bm25_resources(chunks)
        fusion = fusion_list_k
        if fusion is None:
            fusion = len(chunks)
        fusion = min(fusion, len(chunks))

        pool_d, final_d = retrieve_passages_pool_and_final(
            ex.question,
            embedder,
            chunks,
            fi,
            retrieve_k=retrieve_k,
            reranker=reranker if use_rerank else None,
            final_k=final_k,
        )
        pool_h, final_h = retrieve_passages_pool_and_final(
            ex.question,
            embedder,
            chunks,
            fi,
            retrieve_k=retrieve_k,
            reranker=reranker if use_rerank else None,
            final_k=final_k,
            bm25_resources=bm25,
            rrf_k=rrf_k,
            fusion_list_k=fusion,
        )

        in_d = gold_in_any_chunk(pool_d, aliases)
        in_h = gold_in_any_chunk(pool_h, aliases)
        in_fd = gold_in_any_chunk(final_d, aliases)
        in_fh = gold_in_any_chunk(final_h, aliases)

        n += 1
        gd_pool += int(in_d)
        gh_pool += int(in_h)
        gd_final += int(in_fd)
        gh_final += int(in_fh)
        if in_h and not in_d:
            hybrid_saves += 1
        if in_d and not in_h:
            hybrid_hurts += 1

        rows.append(
            {
                "id": ex.id,
                "question": ex.question[:200],
                "n_chunks": len(chunks),
                "gold_in_pool_dense": in_d,
                "gold_in_pool_hybrid": in_h,
                "gold_in_final_dense": in_fd,
                "gold_in_final_hybrid": in_fh,
            }
        )

    summary = {
        "n_questions": float(n),
        "gold_hit_pool_dense": float(gd_pool / n) if n else 0.0,
        "gold_hit_pool_hybrid": float(gh_pool / n) if n else 0.0,
        "gold_hit_final_dense": float(gd_final / n) if n else 0.0,
        "gold_hit_final_hybrid": float(gh_final / n) if n else 0.0,
        "n_hybrid_recovered": float(hybrid_saves),
        "n_hybrid_lost": float(hybrid_hurts),
    }
    return pd.DataFrame(rows), summary


def main() -> None:
    p = argparse.ArgumentParser(description="QASPER dense vs hybrid retrieval / RAG compare")
    p.add_argument(
        "--mode",
        choices=("retrieval", "generation"),
        default="retrieval",
    )
    p.add_argument("--out-dir", type=Path, default=ROOT / "results")
    p.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5")
    p.add_argument("--retrieve-k", type=int, default=10)
    p.add_argument("--final-k", type=int, default=3)
    p.add_argument("--chunk-size", type=int, default=384)
    p.add_argument("--chunk-overlap", type=int, default=48)
    p.add_argument(
        "--fusion-list-k",
        type=int,
        default=None,
        help="BM25/dense list depth before RRF; default = min(n_chunks) per paper",
    )
    p.add_argument("--rrf-k", type=int, default=60)
    p.add_argument("--split", default="validation")
    p.add_argument("--max-examples", type=int, default=200)
    p.add_argument("--dataset-id", default="allenai/qasper")
    p.add_argument("--dataset-revision", default=DEFAULT_QASPER_REVISION)
    p.add_argument("--use-rerank", action="store_true")
    p.add_argument("--rerank-model", default="BAAI/bge-reranker-base")
    p.add_argument("--llm-backend", choices=("gemini", "openai", "ollama"), default="gemini")
    p.add_argument("--llm-model", default=None)
    p.add_argument("--mock-generation", action="store_true")
    p.add_argument("--max-context-chars", type=int, default=8000)
    args = p.parse_args()

    if args.llm_model is None:
        if args.llm_backend == "openai":
            args.llm_model = "gpt-4o-mini"
        elif args.llm_backend == "ollama":
            args.llm_model = "llama3.2"
        else:
            args.llm_model = "gemini-2.5-flash"

    print("Loading QASPER…")
    examples = load_qasper_hf(
        split=args.split,
        max_examples=args.max_examples,
        dataset_id=args.dataset_id,
        revision=args.dataset_revision,
    )
    print(f"Loaded {len(examples)} QA pairs")
    if not examples:
        raise SystemExit("No examples")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    embedder = load_embedding_model(args.embedding_model, normalize=True)
    reranker = load_reranker(args.rerank_model) if args.use_rerank else None

    if args.mode == "retrieval":
        df, summary = run_retrieval_compare(
            examples,
            embedder=embedder,
            retrieve_k=args.retrieve_k,
            final_k=args.final_k,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            fusion_list_k=args.fusion_list_k,
            rrf_k=args.rrf_k,
            use_rerank=args.use_rerank,
            reranker=reranker,
        )
        per_path = out_dir / "qasper_dense_hybrid_retrieval_per_question.csv"
        sum_path = out_dir / "qasper_dense_hybrid_retrieval_summary.csv"
        df.to_csv(per_path, index=False)
        pd.DataFrame([summary]).to_csv(sum_path, index=False)
        print()
        print("Summary (gold alias substring in retrieved chunks):")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        print()
        print(f"Per-question: {per_path}")
        print(f"Summary:      {sum_path}")
        return

    # generation
    generator = _make_generator(args)
    base = RAGGenerationConfig(
        retrieve_k=args.retrieve_k,
        final_k=args.final_k,
        use_rerank=args.use_rerank,
        prompt_template=PROMPT_TEMPLATES["default"],
        max_context_chars=args.max_context_chars,
        truncation="head",
        per_example_retrieval=True,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        use_hybrid=False,
        fusion_list_k=args.fusion_list_k,
        rrf_k=args.rrf_k,
    )

    print("Running dense RAG…")
    m_dense = evaluate_rag_answer_quality(
        examples,
        embedder=embedder,
        corpus_chunks=[],
        generator=generator,
        config=replace(base, use_hybrid=False),
        reranker=reranker if args.use_rerank else None,
        faiss_index=None,
    )
    print("Running hybrid RAG…")
    m_hybrid = evaluate_rag_answer_quality(
        examples,
        embedder=embedder,
        corpus_chunks=[],
        generator=generator,
        config=replace(base, use_hybrid=True),
        reranker=reranker if args.use_rerank else None,
        faiss_index=None,
    )

    rows = [
        {"setting": "dense", **m_dense},
        {"setting": "hybrid_bm25_rrf", **m_hybrid},
    ]
    gen_path = out_dir / "qasper_dense_hybrid_generation.csv"
    pd.DataFrame(rows).to_csv(gen_path, index=False)
    print()
    print("Dense: ", m_dense)
    print("Hybrid:", m_hybrid)
    print(f"Saved: {gen_path}")


if __name__ == "__main__":
    main()
