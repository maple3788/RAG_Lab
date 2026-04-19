"""
RAG end-to-end **generation** ablations on **BEIR TREC-COVID** (same spirit as ``exp_rag_generation.py``,
but corpus = full CORD-19 slice, queries = official topics).

**Gold label for EM / F1 / gold_hit:** the short string in ``metadata.query`` in ``queries.jsonl``
(``keywords`` field in TREC topics). This is **not** an extractive answer span; use metrics only as a
rough signal. For retrieval-only benchmarking, use ``exp_trec_covid.py`` + ir-measures.

Requires: ``data/trec-covid/`` (or ``--data-dir``) with ``corpus.jsonl``, ``queries.jsonl``, ``qrels/test.tsv``.
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
from src.beir_io import load_beir_corpus_ordered
from src.context_truncation import TruncationStrategy
from src.embedder import load_embedding_model
from src.loader import load_beir_queries_as_qa_examples
from src.prompts import PROMPT_TEMPLATES
from src.rag_generation import RAGGenerationConfig
from src.rag_pipeline import build_corpus_chunks_from_documents, build_retrieval_index
from src.reranker import load_reranker


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG generation ablations on TREC-COVID (BEIR); gold = metadata.query proxy"
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
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "trec-covid",
        help="Folder with corpus.jsonl, queries.jsonl, qrels/test.tsv",
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--rerank-model", default="BAAI/bge-reranker-base")
    parser.add_argument("--retrieve-k", type=int, default=10)
    parser.add_argument("--final-k", type=int, default=3)
    parser.add_argument("--max-context-chars", type=int, default=6000)
    parser.add_argument("--truncation-chars", type=int, default=1200)
    parser.add_argument(
        "--truncation", default="head", choices=("head", "tail", "middle")
    )
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Cap number of topics (order follows qrels); default = all with judgments",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Cap corpus documents for debugging (default: full corpus — long encode)",
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

    corpus_path = args.data_dir / "corpus.jsonl"
    queries_path = args.data_dir / "queries.jsonl"
    qrels_path = args.data_dir / "qrels" / "test.tsv"
    if not corpus_path.is_file():
        raise FileNotFoundError(f"Missing {corpus_path}")
    if not queries_path.is_file():
        raise FileNotFoundError(f"Missing {queries_path}")
    if not qrels_path.is_file():
        raise FileNotFoundError(f"Missing {qrels_path}")

    print("Loading corpus…")
    corpus_rows = load_beir_corpus_ordered(corpus_path)
    if args.max_docs is not None:
        corpus_rows = corpus_rows[: args.max_docs]
    doc_texts = [body for _, body in corpus_rows]
    print(f"Documents: {len(doc_texts)}")

    print("Chunking…")
    corpus_chunks = build_corpus_chunks_from_documents(
        doc_texts,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(f"Chunks: {len(corpus_chunks)}")

    examples = load_beir_queries_as_qa_examples(
        queries_path,
        qrels_path=qrels_path,
        max_queries=args.max_queries,
    )
    print(f"Queries (with qrels order): {len(examples)}")

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
    )

    reranker = None
    if args.mode in ("compare-rerank", "all"):
        reranker = load_reranker(args.rerank_model)

    print("Building FAISS index (one encode; reused for all ablations)…")
    faiss_index = build_retrieval_index(embedder, corpus_chunks)

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
    csv_path = out_dir / "trec_rag_generation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
