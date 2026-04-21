from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Dict, List, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.loader import QAExample, load_qa_jsonl
from src.eval.answer_metrics import exact_match, gold_answer_hit, mean, token_f1
from src.llm.embedder import load_embedding_model
from src.llm.generator import OllamaGenerator
from src.llm.prompts import PROMPT_TEMPLATES, format_rag_prompt
from src.rag.context_truncation import truncate_context
from src.rag.rag_pipeline import build_retrieval_corpus, build_retrieval_index, retrieve_passages_for_query


def passages_to_context(passages: Sequence[str], *, separator: str = "\n\n") -> str:
    return separator.join(f"[{i + 1}] {p}" for i, p in enumerate(passages))


def parse_binary_score(text: str) -> bool:
    """Parse loose LLM grader output into yes/no."""
    t = (text or "").strip().lower()
    if not t:
        return False
    first = t.split()[0]
    if first in {"yes", "true", "1"}:
        return True
    if first in {"no", "false", "0"}:
        return False
    return "yes" in t and "no" not in t


def grade_doc_relevance(generator: OllamaGenerator, *, question: str, document: str) -> bool:
    prompt = (
        "You are a strict relevance grader for retrieval.\n"
        "Answer ONLY 'yes' or 'no'.\n"
        "Return 'yes' if the document is useful for answering the question.\n\n"
        f"Question: {question}\n\n"
        f"Document:\n{document}\n\n"
        "Binary score:"
    )
    return parse_binary_score(generator.generate(prompt))


def grade_grounding(
    generator: OllamaGenerator, *, question: str, documents: Sequence[str], answer: str
) -> bool:
    context = passages_to_context(documents)
    prompt = (
        "You are a strict hallucination checker.\n"
        "Answer ONLY 'yes' or 'no'.\n"
        "Return 'yes' only if the answer is fully supported by the provided passages.\n\n"
        f"Question: {question}\n\n"
        f"Passages:\n{context}\n\n"
        f"Answer:\n{answer}\n\n"
        "Grounded:"
    )
    return parse_binary_score(generator.generate(prompt))


def grade_answer_usefulness(generator: OllamaGenerator, *, question: str, answer: str) -> bool:
    prompt = (
        "You are a strict answer-quality grader.\n"
        "Answer ONLY 'yes' or 'no'.\n"
        "Return 'yes' only if the answer directly resolves the question.\n\n"
        f"Question: {question}\n\n"
        f"Answer:\n{answer}\n\n"
        "Useful:"
    )
    return parse_binary_score(generator.generate(prompt))


def rewrite_query(generator: OllamaGenerator, *, question: str) -> str:
    prompt = (
        "Rewrite the user question for retrieval only.\n"
        "Keep intent unchanged, include key entities/terms, output one line only.\n\n"
        f"Question: {question}\n"
        "Rewritten query:"
    )
    rewritten = generator.generate(prompt).strip()
    return rewritten if rewritten else question


@dataclass
class RunConfig:
    retrieve_k: int
    final_k: int
    max_context_chars: int
    max_loops: int
    prompt_template_name: str


def generate_answer(
    generator: OllamaGenerator,
    *,
    question: str,
    passages: Sequence[str],
    prompt_template_name: str,
    max_context_chars: int,
) -> str:
    template = PROMPT_TEMPLATES[prompt_template_name]
    context = truncate_context(passages_to_context(passages), max_context_chars, "head")
    prompt = format_rag_prompt(template, context=context, question=question, history="None")
    return generator.generate(prompt).strip()


def gold_strings(ex: QAExample) -> List[str]:
    aliases = ex.answer_aliases
    if aliases:
        return list(aliases)
    return [ex.answer]


def score_prediction(pred: str, ex: QAExample) -> Dict[str, float]:
    golds = gold_strings(ex)
    return {
        "em": max(exact_match(pred, g) for g in golds),
        "f1": max(token_f1(pred, g) for g in golds),
        "gold_hit": max(gold_answer_hit(pred, g) for g in golds),
    }


def run_baseline(
    *,
    examples: Sequence[QAExample],
    generator: OllamaGenerator,
    embedder,
    corpus_chunks: Sequence[str],
    faiss_index,
    cfg: RunConfig,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    ems: List[float] = []
    f1s: List[float] = []
    hits: List[float] = []
    lat_ms: List[float] = []

    for ex in examples:
        t0 = perf_counter()
        passages = retrieve_passages_for_query(
            ex.question,
            embedder,
            corpus_chunks,
            faiss_index,
            retrieve_k=cfg.retrieve_k,
            final_k=cfg.final_k,
        )
        answer = generate_answer(
            generator,
            question=ex.question,
            passages=passages,
            prompt_template_name=cfg.prompt_template_name,
            max_context_chars=cfg.max_context_chars,
        )
        m = score_prediction(answer, ex)
        elapsed = (perf_counter() - t0) * 1000.0
        ems.append(m["em"])
        f1s.append(m["f1"])
        hits.append(m["gold_hit"])
        lat_ms.append(elapsed)
        rows.append(
            {
                "id": ex.id,
                "question": ex.question,
                "answer_pred": answer,
                "answer_gold": ex.answer,
                "em": m["em"],
                "f1": m["f1"],
                "gold_hit": m["gold_hit"],
                "latency_ms": elapsed,
            }
        )

    return {
        "method": "baseline",
        "n": len(rows),
        "em": mean(ems),
        "f1": mean(f1s),
        "gold_hit": mean(hits),
        "latency_ms": mean(lat_ms),
        "rows": rows,
    }


def run_self_rag(
    *,
    examples: Sequence[QAExample],
    generator: OllamaGenerator,
    embedder,
    corpus_chunks: Sequence[str],
    faiss_index,
    cfg: RunConfig,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    ems: List[float] = []
    f1s: List[float] = []
    hits: List[float] = []
    lat_ms: List[float] = []
    loops_used: List[float] = []
    rewrites: List[float] = []

    for ex in examples:
        t0 = perf_counter()
        question_cur = ex.question
        final_answer = "unknown"
        final_passages: List[str] = []
        used_rewrites = 0
        used_loops = 0

        for loop_idx in range(cfg.max_loops):
            used_loops = loop_idx + 1
            passages = retrieve_passages_for_query(
                question_cur,
                embedder,
                corpus_chunks,
                faiss_index,
                retrieve_k=cfg.retrieve_k,
                final_k=cfg.final_k,
            )
            filtered = [
                p for p in passages if grade_doc_relevance(generator, question=question_cur, document=p)
            ]
            if not filtered:
                question_cur = rewrite_query(generator, question=question_cur)
                used_rewrites += 1
                continue

            candidate = generate_answer(
                generator,
                question=question_cur,
                passages=filtered,
                prompt_template_name=cfg.prompt_template_name,
                max_context_chars=cfg.max_context_chars,
            )

            grounded = grade_grounding(
                generator,
                question=question_cur,
                documents=filtered,
                answer=candidate,
            )
            if not grounded:
                # Retry generation once with stricter prompt before full loop retry.
                strict_candidate = generate_answer(
                    generator,
                    question=question_cur,
                    passages=filtered,
                    prompt_template_name="strict_cite",
                    max_context_chars=cfg.max_context_chars,
                )
                grounded = grade_grounding(
                    generator,
                    question=question_cur,
                    documents=filtered,
                    answer=strict_candidate,
                )
                if grounded:
                    candidate = strict_candidate

            useful = grounded and grade_answer_usefulness(
                generator, question=question_cur, answer=candidate
            )
            final_answer = candidate
            final_passages = filtered
            if useful:
                break

            question_cur = rewrite_query(generator, question=question_cur)
            used_rewrites += 1

        m = score_prediction(final_answer, ex)
        elapsed = (perf_counter() - t0) * 1000.0
        ems.append(m["em"])
        f1s.append(m["f1"])
        hits.append(m["gold_hit"])
        lat_ms.append(elapsed)
        loops_used.append(float(used_loops))
        rewrites.append(float(used_rewrites))
        rows.append(
            {
                "id": ex.id,
                "question": ex.question,
                "question_final": question_cur,
                "answer_pred": final_answer,
                "answer_gold": ex.answer,
                "passages_used": len(final_passages),
                "loops": used_loops,
                "rewrites": used_rewrites,
                "em": m["em"],
                "f1": m["f1"],
                "gold_hit": m["gold_hit"],
                "latency_ms": elapsed,
            }
        )

    return {
        "method": "self_rag",
        "n": len(rows),
        "em": mean(ems),
        "f1": mean(f1s),
        "gold_hit": mean(hits),
        "latency_ms": mean(lat_ms),
        "avg_loops": mean(loops_used),
        "avg_rewrites": mean(rewrites),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Self-RAG experiment with Ollama (default: qwen3:8b)."
    )
    parser.add_argument("--data-path", type=Path, default=ROOT / "datasets" / "qa_dataset.jsonl")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--llm-model", default="qwen3:8b")
    parser.add_argument("--retrieve-k", type=int, default=8)
    parser.add_argument("--final-k", type=int, default=4)
    parser.add_argument("--max-context-chars", type=int, default=6000)
    parser.add_argument("--max-loops", type=int, default=3)
    parser.add_argument(
        "--prompt-template",
        choices=tuple(PROMPT_TEMPLATES.keys()),
        default="default",
    )
    parser.add_argument("--max-examples", type=int, default=30)
    args = parser.parse_args()

    examples = load_qa_jsonl(args.data_path)
    if args.max_examples > 0:
        examples = examples[: args.max_examples]
    if not examples:
        raise ValueError("No examples loaded. Check --data-path.")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(examples)} examples from {args.data_path}")
    print("Building retrieval corpus and index...")
    corpus_chunks = build_retrieval_corpus(examples, chunk_size=512, chunk_overlap=64)
    embedder = load_embedding_model(args.embedding_model, normalize=True)
    faiss_index = build_retrieval_index(embedder, corpus_chunks)

    generator = OllamaGenerator(model=args.llm_model)
    cfg = RunConfig(
        retrieve_k=args.retrieve_k,
        final_k=args.final_k,
        max_context_chars=args.max_context_chars,
        max_loops=args.max_loops,
        prompt_template_name=args.prompt_template,
    )

    print(f"Running baseline with Ollama model: {args.llm_model}")
    baseline = run_baseline(
        examples=examples,
        generator=generator,
        embedder=embedder,
        corpus_chunks=corpus_chunks,
        faiss_index=faiss_index,
        cfg=cfg,
    )
    print(f"Running self-rag with max_loops={args.max_loops}")
    self_rag = run_self_rag(
        examples=examples,
        generator=generator,
        embedder=embedder,
        corpus_chunks=corpus_chunks,
        faiss_index=faiss_index,
        cfg=cfg,
    )

    summary_rows = [
        {k: v for k, v in baseline.items() if k != "rows"},
        {k: v for k, v in self_rag.items() if k != "rows"},
    ]
    df_summary = pd.DataFrame(summary_rows)
    df_details = pd.concat(
        [
            pd.DataFrame(baseline["rows"]).assign(method="baseline"),
            pd.DataFrame(self_rag["rows"]).assign(method="self_rag"),
        ],
        ignore_index=True,
    )

    stem = f"self_rag_ollama_{args.llm_model.replace(':', '_')}"
    summary_path = out_dir / f"{stem}_summary.csv"
    details_path = out_dir / f"{stem}_details.csv"
    json_path = out_dir / f"{stem}.json"
    df_summary.to_csv(summary_path, index=False)
    df_details.to_csv(details_path, index=False)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary_rows, "baseline": baseline, "self_rag": self_rag}, f, indent=2)

    print("\n=== Self-RAG Experiment Summary ===")
    print(df_summary.to_string(index=False))
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved details: {details_path}")
    print(f"Saved JSON:   {json_path}")


if __name__ == "__main__":
    main()
