"""
RAG ablations on **QASPER** (``allenai/qasper``): QA over **full academic papers**.

This is the “**application / long-document**” track: same metrics as TriviaQA, but each example
chunks a **multi-page paper** (abstract + body), closer to **PDF / technical manual** workflows than
short web snippets.

Uses HF ``datasets`` with revision ``refs/convert/parquet`` (required for modern ``datasets``).

Requires: ``pip install datasets`` (see ``requirements.txt``).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.exp_rag_generation import (
    _make_generator,
    run_compare_prompts,
    run_compare_rerank,
    run_compare_topk,
    run_compare_truncation,
)
from src.rag.context_truncation import TruncationStrategy
from src.llm.embedder import load_embedding_model
from src.llm.prompts import PROMPT_TEMPLATES
from src.datasets.qasper_hf import DEFAULT_QASPER_REVISION, load_qasper_hf
from src.rag.rag_generation import RAGGenerationConfig
from src.llm.reranker import load_reranker


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG ablations on QASPER (long-document QA, HF)"
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
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--rerank-model", default="BAAI/bge-reranker-base")
    parser.add_argument("--retrieve-k", type=int, default=10)
    parser.add_argument("--final-k", type=int, default=3)
    parser.add_argument("--max-context-chars", type=int, default=8000)
    parser.add_argument("--truncation-chars", type=int, default=2000)
    parser.add_argument(
        "--truncation", default="head", choices=("head", "tail", "middle")
    )
    parser.add_argument("--chunk-size", type=int, default=384)
    parser.add_argument("--chunk-overlap", type=int, default=48)
    parser.add_argument(
        "--split", default="validation", help="HF split: train / validation / test"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Max QA pairs after filtering (default: all loaded)",
    )
    parser.add_argument("--dataset-id", default="allenai/qasper")
    parser.add_argument(
        "--dataset-revision",
        default=DEFAULT_QASPER_REVISION,
        help="HF git revision (Parquet branch for allenai/qasper)",
    )
    parser.add_argument(
        "--llm-backend",
        choices=("gemini", "openai", "ollama"),
        default="gemini",
    )
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--mock-generation", action="store_true")
    args = parser.parse_args()

    if args.llm_model is None:
        if args.llm_backend == "openai":
            args.llm_model = "gpt-4o-mini"
        elif args.llm_backend == "ollama":
            args.llm_model = "llama3.2"
        else:
            args.llm_model = "gemini-2.5-flash"

    print("Loading QASPER (HF, Parquet revision)…")
    examples = load_qasper_hf(
        split=args.split,
        max_examples=args.max_examples,
        dataset_id=args.dataset_id,
        revision=args.dataset_revision,
    )
    print(f"QA examples loaded: {len(examples)}")
    if not examples:
        raise SystemExit("No examples loaded (need document text + answer aliases).")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

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
        per_example_retrieval=True,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    reranker = None
    if args.mode in ("compare-rerank", "all"):
        reranker = load_reranker(args.rerank_model)

    corpus_chunks: list = []
    faiss_index = None

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
            )
        )
    if args.mode in ("compare-topk", "all"):
        all_rows.extend(
            run_compare_topk(
                embedder, corpus_chunks, examples, generator, base, faiss_index
            )
        )
    if args.mode in ("compare-prompts", "all"):
        all_rows.extend(
            run_compare_prompts(
                embedder, corpus_chunks, examples, generator, base, faiss_index
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
            )
        )

    df = pd.DataFrame(all_rows)
    csv_path = out_dir / "qasper_rag_generation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
