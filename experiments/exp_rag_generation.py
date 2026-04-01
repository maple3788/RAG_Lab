from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import cast

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.context_truncation import TruncationStrategy
from src.embedder import load_embedding_model
from src.generator import GeminiGenerator, MockGenerator
from src.loader import load_qa_jsonl
from src.prompts import PROMPT_TEMPLATES
from src.rag_generation import RAGGenerationConfig, evaluate_rag_answer_quality
from src.rag_pipeline import build_retrieval_corpus
from src.reranker import load_reranker


def _make_generator(args: argparse.Namespace):
    if args.mock_generation:
        return MockGenerator()
    return GeminiGenerator(model=args.llm_model)


def run_compare_rerank(embedder, corpus_chunks, examples, generator, base: RAGGenerationConfig, reranker):
    rows = []
    for use_rerank, label in [(False, "no_rerank"), (True, "with_rerank")]:
        cfg = RAGGenerationConfig(
            retrieve_k=base.retrieve_k,
            final_k=base.final_k,
            use_rerank=use_rerank,
            prompt_template=base.prompt_template,
            max_context_chars=base.max_context_chars,
            truncation=base.truncation,
        )
        m = evaluate_rag_answer_quality(
            examples,
            embedder=embedder,
            corpus_chunks=corpus_chunks,
            generator=generator,
            config=cfg,
            reranker=reranker if use_rerank else None,
        )
        rows.append({"experiment": "rerank", "setting": label, **m})
        print(label, m)
    return rows


def run_compare_topk(embedder, corpus_chunks, examples, generator, base: RAGGenerationConfig):
    rows = []
    for fk in [1, 3, 5]:
        cfg = RAGGenerationConfig(
            retrieve_k=max(base.retrieve_k, fk),
            final_k=fk,
            use_rerank=False,
            prompt_template=base.prompt_template,
            max_context_chars=base.max_context_chars,
            truncation=base.truncation,
        )
        m = evaluate_rag_answer_quality(
            examples,
            embedder=embedder,
            corpus_chunks=corpus_chunks,
            generator=generator,
            config=cfg,
            reranker=None,
        )
        rows.append({"experiment": "topk", "setting": f"final_k={fk}", **m})
        print(f"final_k={fk}", m)
    return rows


def run_compare_prompts(embedder, corpus_chunks, examples, generator, base: RAGGenerationConfig):
    rows = []
    for name, tmpl in PROMPT_TEMPLATES.items():
        cfg = RAGGenerationConfig(
            retrieve_k=base.retrieve_k,
            final_k=base.final_k,
            use_rerank=base.use_rerank,
            prompt_template=tmpl,
            max_context_chars=base.max_context_chars,
            truncation=base.truncation,
        )
        m = evaluate_rag_answer_quality(
            examples,
            embedder=embedder,
            corpus_chunks=corpus_chunks,
            generator=generator,
            config=cfg,
            reranker=None,
        )
        rows.append({"experiment": "prompt", "setting": name, **m})
        print(name, m)
    return rows


def run_compare_truncation(
    embedder,
    corpus_chunks,
    examples,
    generator,
    base: RAGGenerationConfig,
    *,
    max_context_chars: int,
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
        )
        m = evaluate_rag_answer_quality(
            examples,
            embedder=embedder,
            corpus_chunks=corpus_chunks,
            generator=generator,
            config=cfg,
            reranker=None,
        )
        rows.append({"experiment": "truncation", "setting": strat, **m})
        print(strat, m)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG end-to-end answer quality (EM, F1, gold hit)")
    parser.add_argument(
        "--mode",
        choices=["compare-rerank", "compare-topk", "compare-prompts", "compare-truncation", "all"],
        default="all",
    )
    parser.add_argument("--data-path", type=Path, default=ROOT / "datasets" / "qa_dataset.jsonl")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--rerank-model", default="BAAI/bge-reranker-base")
    parser.add_argument("--retrieve-k", type=int, default=10, help="FAISS top-k when reranking")
    parser.add_argument("--final-k", type=int, default=3, help="Passages fed to the LLM after retrieve/rerank")
    parser.add_argument("--max-context-chars", type=int, default=6000)
    parser.add_argument(
        "--truncation-chars",
        type=int,
        default=1200,
        help="For compare-truncation: tight char budget so head/tail/middle differ",
    )
    parser.add_argument("--truncation", default="head", choices=("head", "tail", "middle"))
    parser.add_argument("--llm-model", default="gemini-2.0-flash", help="Gemini model id")
    parser.add_argument(
        "--mock-generation",
        action="store_true",
        help="Use MockGenerator (no API; for pipeline smoke tests only)",
    )
    args = parser.parse_args()

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
    )

    reranker = None
    if args.mode in ("compare-rerank", "all"):
        reranker = load_reranker(args.rerank_model)

    all_rows = []
    if args.mode in ("compare-rerank", "all"):
        all_rows.extend(run_compare_rerank(embedder, corpus_chunks, examples, generator, base, reranker))
    if args.mode in ("compare-topk", "all"):
        all_rows.extend(run_compare_topk(embedder, corpus_chunks, examples, generator, base))
    if args.mode in ("compare-prompts", "all"):
        all_rows.extend(run_compare_prompts(embedder, corpus_chunks, examples, generator, base))
    if args.mode in ("compare-truncation", "all"):
        all_rows.extend(
            run_compare_truncation(
                embedder,
                corpus_chunks,
                examples,
                generator,
                base,
                max_context_chars=args.truncation_chars,
            )
        )

    df = pd.DataFrame(all_rows)
    csv_path = out_dir / "rag_generation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
