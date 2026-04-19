"""
Run RAGAS evaluation on FinanceBench open-source examples.

Prereqs:
  - Clone https://github.com/patronus-ai/financebench locally
  - pip install ragas datasets
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.exp_rag_generation import _make_generator
from src.context_truncation import TruncationStrategy, truncate_context
from src.embedder import load_embedding_model
from src.financebench_loader import load_financebench_open_source
from src.prompts import PROMPT_TEMPLATES, format_rag_prompt
from src.rag_generation import passages_to_context
from src.rag_pipeline import (
    build_corpus_chunks_from_documents,
    build_retrieval_index,
    retrieve_passages_for_query,
)
from src.reranker import load_reranker


def _resolve_metric(m, name: str):
    """
    Normalize ragas metric imports across versions.
    - If already a metric object, return as-is.
    - If module-like export (e.g., ragas.metrics.collections), extract named object.
    - If class export, instantiate it.
    """
    if m is None:
        raise ValueError(f"Metric {name} is None")
    # module-like export
    if hasattr(m, "__dict__") and hasattr(m, "__name__") and hasattr(m, name):
        obj = getattr(m, name)
        return obj() if isinstance(obj, type) else obj
    # class export
    if isinstance(m, type):
        return m()
    return m


def _make_ragas_eval_stack(args):
    """
    Build explicit RAGAS evaluator LLM+embeddings to avoid hidden defaults.
    """
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
    except Exception as e:
        raise SystemExit(
            "Missing evaluator deps. Install with: pip install langchain-openai ragas datasets\n"
            f"Import error: {e}"
        )

    eval_backend = args.ragas_eval_backend or args.llm_backend
    eval_model = args.ragas_eval_model or args.llm_model
    embed_model = args.ragas_embed_model or (
        "nomic-embed-text" if eval_backend == "ollama" else "text-embedding-3-small"
    )
    use_eval_embeddings = not bool(args.no_ragas_embeddings)

    if eval_backend == "ollama":
        base = (
            os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
            .strip()
            .rstrip("/")
        )
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        lc_llm = ChatOpenAI(
            model=eval_model,
            base_url=base,
            api_key="ollama",
            temperature=0.0,
        )
        lc_emb = (
            OpenAIEmbeddings(
                model=embed_model,
                base_url=base,
                api_key="ollama",
            )
            if use_eval_embeddings
            else None
        )
    elif eval_backend == "gemini":
        try:
            from langchain_google_genai import (
                ChatGoogleGenerativeAI,
                GoogleGenerativeAIEmbeddings,
            )
        except Exception as e:
            raise SystemExit(
                "Gemini evaluator requires langchain-google-genai.\n"
                "Install with: pip install langchain-google-genai\n"
                f"Import error: {e}"
            )
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise SystemExit("GEMINI_API_KEY is required for gemini RAGAS evaluator.")
        gemini_embed_model = args.ragas_embed_model or "gemini-embedding-001"
        lc_llm = ChatGoogleGenerativeAI(
            model=eval_model,
            google_api_key=api_key,
            temperature=0.0,
        )
        lc_emb = (
            GoogleGenerativeAIEmbeddings(
                model=gemini_embed_model,
                google_api_key=api_key,
            )
            if use_eval_embeddings
            else None
        )
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL")
        if not api_key:
            raise SystemExit(
                "OPENAI_API_KEY is required for non-ollama RAGAS evaluator."
            )
        lc_llm = ChatOpenAI(
            model=eval_model,
            api_key=api_key,
            base_url=base_url,
            temperature=0.0,
        )
        lc_emb = (
            OpenAIEmbeddings(
                model=embed_model,
                api_key=api_key,
                base_url=base_url,
            )
            if use_eval_embeddings
            else None
        )

    return (
        LangchainLLMWrapper(lc_llm),
        LangchainEmbeddingsWrapper(lc_emb) if lc_emb is not None else None,
    )


def _predict_examples(
    examples,
    *,
    embedder,
    generator,
    retrieve_k: int,
    final_k: int,
    prompt_template: str,
    max_context_chars: int,
    truncation: TruncationStrategy,
    chunk_size: int,
    chunk_overlap: int,
    reranker=None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ex in examples:
        chunks = build_corpus_chunks_from_documents(
            list(ex.contexts), chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        if not chunks:
            continue
        faiss_index = build_retrieval_index(embedder, chunks)
        passages = retrieve_passages_for_query(
            ex.question,
            embedder,
            chunks,
            faiss_index,
            retrieve_k=retrieve_k,
            reranker=reranker,
            final_k=final_k,
        )
        raw_context = passages_to_context(passages)
        context = truncate_context(raw_context, max_context_chars, truncation)
        prompt = format_rag_prompt(
            prompt_template,
            context=context,
            question=ex.question,
            history="None",
        )
        prediction = generator.generate(prompt)
        out.append(
            {
                "id": ex.id,
                "question": ex.question,
                "answer": prediction,
                "contexts": list(passages),
                "ground_truth": ex.answer,
                "source": ex.source,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAGAS on FinanceBench open-source examples."
    )
    parser.add_argument(
        "--financebench-root",
        type=Path,
        required=True,
        help="Path to local clone of patronus-ai/financebench",
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--max-examples", type=int, default=100)
    parser.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--rerank-model", default="BAAI/bge-reranker-base")
    parser.add_argument("--use-rerank", action="store_true")
    parser.add_argument("--retrieve-k", type=int, default=10)
    parser.add_argument("--final-k", type=int, default=3)
    parser.add_argument("--max-context-chars", type=int, default=7000)
    parser.add_argument(
        "--truncation", default="head", choices=("head", "tail", "middle")
    )
    parser.add_argument("--chunk-size", type=int, default=384)
    parser.add_argument("--chunk-overlap", type=int, default=48)
    parser.add_argument(
        "--prompt-template",
        default="default",
        choices=tuple(PROMPT_TEMPLATES.keys()),
    )
    parser.add_argument(
        "--llm-backend",
        choices=("gemini", "openai", "ollama"),
        default="gemini",
    )
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--mock-generation", action="store_true")
    parser.add_argument(
        "--ragas-eval-backend",
        choices=("gemini", "openai", "ollama"),
        default=None,
        help="Backend used by RAGAS evaluator (defaults to --llm-backend).",
    )
    parser.add_argument(
        "--ragas-eval-model",
        default=None,
        help="RAGAS evaluator LLM model (default: --llm-model).",
    )
    parser.add_argument(
        "--ragas-embed-model",
        default=None,
        help="RAGAS evaluator embedding model id (Ollama: nomic-embed-text; Gemini: gemini-embedding-001).",
    )
    parser.add_argument(
        "--no-ragas-embeddings",
        action="store_true",
        help="Disable evaluator embeddings and run LLM-only RAGAS metrics.",
    )
    args = parser.parse_args()

    if args.llm_model is None:
        if args.llm_backend == "openai":
            args.llm_model = "gpt-4o-mini"
        elif args.llm_backend == "ollama":
            args.llm_model = "llama3.2"
        else:
            args.llm_model = "gemini-2.5-flash"

    eval_backend = args.ragas_eval_backend or args.llm_backend
    # RAGAS evaluator defaults to OpenAI-compatible clients for several metrics.
    # For local Ollama evaluator runs, provide OpenAI-compatible env defaults.
    if eval_backend == "ollama":
        os.environ.setdefault("OPENAI_API_KEY", "ollama")
        base = (
            os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
            .strip()
            .rstrip("/")
        )
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        os.environ.setdefault("OPENAI_BASE_URL", base)
        os.environ.setdefault("OPENAI_MODEL", args.ragas_eval_model or args.llm_model)
        os.environ.setdefault(
            "OPENAI_EMBEDDING_MODEL", args.ragas_embed_model or "nomic-embed-text"
        )

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except Exception as e:
        raise SystemExit(
            "RAGAS dependencies missing. Install with: pip install ragas datasets\n"
            f"Import error: {e}"
        )

    examples = load_financebench_open_source(
        financebench_root=args.financebench_root,
        max_examples=args.max_examples,
    )
    if not examples:
        raise SystemExit("No FinanceBench examples loaded.")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    embedder = load_embedding_model(args.embedding_model, normalize=True)
    generator = _make_generator(args)
    reranker = load_reranker(args.rerank_model) if args.use_rerank else None

    rows = _predict_examples(
        examples,
        embedder=embedder,
        generator=generator,
        retrieve_k=args.retrieve_k,
        final_k=args.final_k,
        prompt_template=PROMPT_TEMPLATES[args.prompt_template],
        max_context_chars=int(args.max_context_chars),
        truncation=args.truncation,  # type: ignore[arg-type]
        chunk_size=int(args.chunk_size),
        chunk_overlap=int(args.chunk_overlap),
        reranker=reranker,
    )
    if not rows:
        raise SystemExit("No predictions generated.")

    pred_csv = out_dir / "financebench_ragas_predictions.csv"
    pd.DataFrame(rows).to_csv(pred_csv, index=False)
    print(f"Saved predictions: {pred_csv}")

    ds = Dataset.from_list(rows)
    eval_llm, eval_embeddings = _make_ragas_eval_stack(args)
    if args.no_ragas_embeddings:
        metrics = [_resolve_metric(faithfulness, "faithfulness")]
        result = evaluate(
            ds,
            metrics=metrics,
            llm=eval_llm,
        )
    else:
        metrics = [
            _resolve_metric(faithfulness, "faithfulness"),
            _resolve_metric(answer_relevancy, "answer_relevancy"),
            _resolve_metric(context_precision, "context_precision"),
            _resolve_metric(context_recall, "context_recall"),
        ]
        result = evaluate(
            ds,
            metrics=metrics,
            llm=eval_llm,
            embeddings=eval_embeddings,
        )
    if hasattr(result, "to_pandas"):
        score_df = result.to_pandas()
    else:
        score_df = pd.DataFrame([dict(result)])

    metric_cols = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]
    present = [c for c in metric_cols if c in score_df.columns]
    if present and score_df[present].isna().all().all():
        raise SystemExit(
            "RAGAS returned only empty metric values. "
            "Check evaluator LLM credentials/config (OpenAI key or Ollama OpenAI-compatible endpoint)."
        )

    score_csv = out_dir / "financebench_ragas_scores.csv"
    score_df.to_csv(score_csv, index=False)
    print(f"Saved RAGAS scores: {score_csv}")
    print(score_df.to_string(index=False))


if __name__ == "__main__":
    main()
