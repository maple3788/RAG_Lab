#!/usr/bin/env python3
"""
Export per-question JSONL traces for QASPER error analysis (same schema spirit as TriviaQA export).

Example::

    python tools/export_qasper_traces.py --out analysis/qasper_traces.jsonl --max-examples 40 --llm-backend ollama
    python tools/export_qasper_traces.py --out analysis/qasper_traces.jsonl --max-examples 30 --skip-generation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.embedder import load_embedding_model
from src.error_analysis import (
    classify_failure_bucket,
    gold_in_any_chunk,
    gold_lost_to_truncation,
)
from src.generator import GeminiGenerator, MockGenerator, OllamaGenerator, OpenAICompatibleGenerator
from src.loader import QAExample
from src.prompts import PROMPT_TEMPLATES, format_rag_prompt
from src.rag_generation import passages_to_context
from src.rag_pipeline import (
    build_corpus_chunks_from_documents,
    build_retrieval_index,
    retrieve_passages_pool_and_final,
)
from src.context_truncation import TruncationStrategy, truncate_context
from src.answer_metrics import exact_match, gold_answer_hit, token_f1
from src.reranker import load_reranker
from src.qasper_hf import DEFAULT_QASPER_REVISION, load_qasper_hf


def _make_generator(args: argparse.Namespace):
    if args.mock_generation:
        return MockGenerator()
    if args.llm_backend == "openai":
        return OpenAICompatibleGenerator(model=args.llm_model)
    if args.llm_backend == "ollama":
        return OllamaGenerator(model=args.llm_model)
    return GeminiGenerator(model=args.llm_model)


def _gold_list(ex: QAExample) -> list[str]:
    if ex.answer_aliases:
        return list(ex.answer_aliases)
    return [ex.answer]


def main() -> None:
    p = argparse.ArgumentParser(description="Export QASPER traces for error analysis")
    p.add_argument("--out", type=Path, required=True, help="Output JSONL path")
    p.add_argument("--split", default="validation")
    p.add_argument("--max-examples", type=int, default=50)
    p.add_argument("--dataset-id", default="allenai/qasper")
    p.add_argument("--dataset-revision", default=DEFAULT_QASPER_REVISION)
    p.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5")
    p.add_argument("--rerank-model", default="BAAI/bge-reranker-base")
    p.add_argument("--retrieve-k", type=int, default=10)
    p.add_argument("--final-k", type=int, default=3)
    p.add_argument("--chunk-size", type=int, default=384)
    p.add_argument("--chunk-overlap", type=int, default=48)
    p.add_argument("--max-context-chars", type=int, default=8000)
    p.add_argument("--truncation", default="head", choices=("head", "tail", "middle"))
    p.add_argument("--prompt-name", default="default", choices=tuple(PROMPT_TEMPLATES.keys()))
    p.add_argument("--use-rerank", action="store_true")
    p.add_argument("--skip-generation", action="store_true")
    p.add_argument("--mock-generation", action="store_true")
    p.add_argument("--llm-backend", choices=("gemini", "openai", "ollama"), default="ollama")
    p.add_argument("--llm-model", default=None)
    p.add_argument("--preview-chars", type=int, default=400)
    args = p.parse_args()

    if args.llm_model is None:
        if args.llm_backend == "openai":
            args.llm_model = "gpt-4o-mini"
        elif args.llm_backend == "ollama":
            args.llm_model = "llama3.2"
        else:
            args.llm_model = "gemini-2.5-flash"

    examples = load_qasper_hf(
        split=args.split,
        max_examples=args.max_examples,
        dataset_id=args.dataset_id,
        revision=args.dataset_revision,
    )
    if not examples:
        raise SystemExit("No examples loaded.")

    embedder = load_embedding_model(args.embedding_model, normalize=True)
    reranker = load_reranker(args.rerank_model) if args.use_rerank else None
    generator = None if args.skip_generation else _make_generator(args)
    tmpl = PROMPT_TEMPLATES[args.prompt_name]
    trunc: TruncationStrategy = args.truncation  # type: ignore[assignment]
    prev = args.preview_chars

    def shorten(text: str) -> str:
        t = text or ""
        return t if len(t) <= prev else t[:prev] + "…"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.out.open("w", encoding="utf-8") as f:
        for ex in examples:
            golds = _gold_list(ex)
            chunks = build_corpus_chunks_from_documents(
                list(ex.contexts),
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
            if not chunks:
                row = {
                    "question_id": ex.id,
                    "question": ex.question,
                    "error": "no_chunks_after_splitting",
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
                continue

            fi = build_retrieval_index(embedder, chunks)
            pool, final = retrieve_passages_pool_and_final(
                ex.question,
                embedder,
                chunks,
                fi,
                retrieve_k=args.retrieve_k,
                reranker=reranker,
                final_k=args.final_k,
            )

            any_pool = gold_in_any_chunk(pool, golds)
            any_final = gold_in_any_chunk(final, golds)

            raw_ctx = passages_to_context(final)
            trunc_ctx = truncate_context(raw_ctx, args.max_context_chars, trunc)
            lost_trunc = gold_lost_to_truncation(raw_ctx, trunc_ctx, golds)

            prediction = ""
            if not args.skip_generation and generator is not None:
                prompt = format_rag_prompt(tmpl, context=trunc_ctx, question=ex.question)
                prediction = generator.generate(prompt)

            em = max(exact_match(prediction, g) for g in golds) if prediction else 0.0
            f1 = max(token_f1(prediction, g) for g in golds) if prediction else 0.0
            gh = max(gold_answer_hit(prediction, g) for g in golds) if prediction else 0.0

            if not any_pool:
                stage = "retrieval"
            elif not any_final:
                stage = "ranking"
            else:
                stage = "gold_in_final"

            if args.skip_generation:
                bucket = None
            else:
                bucket = classify_failure_bucket(
                    any_gold_in_pool=any_pool,
                    any_gold_in_final=any_final,
                    exact_match_score=em,
                )

            row = {
                "question_id": ex.id,
                "question": ex.question,
                "gold_aliases": golds[:20],
                "config": {
                    "retrieve_k": args.retrieve_k,
                    "final_k": args.final_k,
                    "use_rerank": args.use_rerank,
                    "prompt": args.prompt_name,
                    "truncation": args.truncation,
                    "max_context_chars": args.max_context_chars,
                },
                "any_gold_in_pool": any_pool,
                "any_gold_in_final_passages": any_final,
                "gold_lost_to_truncation": lost_trunc,
                "retrieval_stage": stage,
                "failure_bucket": bucket,
                "llm_called": not args.skip_generation,
                "pool_previews": [shorten(t) for t in pool],
                "final_previews": [shorten(t) for t in final],
                "prediction": prediction[:2000] if prediction else "",
                "exact_match": em,
                "token_f1": f1,
                "gold_hit": gh,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1

    print(f"Wrote {n} lines to {args.out}")


if __name__ == "__main__":
    main()
