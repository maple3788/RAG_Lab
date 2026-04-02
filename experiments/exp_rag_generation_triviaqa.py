"""
RAG generation ablations on **TriviaQA** (``mandarjoshi/trivia_qa``, config ``rc``).

Uses HF ``datasets``; first run downloads the split. Retrieval is **per question** over that
row's ``entity_pages`` / ``search_results`` only (standard RC setting). Metrics use **any**
official answer **alias**.

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
from src.context_truncation import TruncationStrategy
from src.embedder import load_embedding_model
from src.prompts import PROMPT_TEMPLATES
from src.rag_generation import RAGGenerationConfig
from src.reranker import load_reranker
from src.triviaqa_hf import load_triviaqa_rc_hf


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG ablations on TriviaQA (RC, HuggingFace)")
    parser.add_argument(
        "--mode",
        choices=["compare-rerank", "compare-topk", "compare-prompts", "compare-truncation", "all"],
        default="all",
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--rerank-model", default="BAAI/bge-reranker-base")
    parser.add_argument("--retrieve-k", type=int, default=10)
    parser.add_argument("--final-k", type=int, default=3)
    parser.add_argument("--max-context-chars", type=int, default=6000)
    parser.add_argument("--truncation-chars", type=int, default=1200)
    parser.add_argument("--truncation", default="head", choices=("head", "tail", "middle"))
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    parser.add_argument("--split", default="validation", help="HF split: train / validation / test")
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Cap examples after skipping rows without passages (default: all loaded)",
    )
    parser.add_argument("--dataset-id", default="mandarjoshi/trivia_qa")
    parser.add_argument("--dataset-config", default="rc")
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

    print("Loading TriviaQA (HF)…")
    examples = load_triviaqa_rc_hf(
        split=args.split,
        max_examples=args.max_examples,
        dataset_id=args.dataset_id,
        config=args.dataset_config,
    )
    print(f"Examples with passages: {len(examples)}")
    if not examples:
        raise SystemExit("No examples loaded (need passages + answers).")

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
            run_compare_rerank(embedder, corpus_chunks, examples, generator, base, reranker, faiss_index)
        )
    if args.mode in ("compare-topk", "all"):
        all_rows.extend(run_compare_topk(embedder, corpus_chunks, examples, generator, base, faiss_index))
    if args.mode in ("compare-prompts", "all"):
        all_rows.extend(run_compare_prompts(embedder, corpus_chunks, examples, generator, base, faiss_index))
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
    csv_path = out_dir / "triviaqa_rag_generation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
