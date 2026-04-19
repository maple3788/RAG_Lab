#!/usr/bin/env python3
"""
Sanity test for hybrid BM25 + dense (RRF) retrieval, inspired by ERROR_ANALYSIS.md
retrieval failure: dense–sparse mismatch on metrics / tables.

Uses a small synthetic corpus and compares **dense-only** vs **hybrid** pool ordering.
Optionally uses a deeper candidate list per retriever before RRF (``--fusion-list-k``) so
a chunk that is mid-ranked on one list can still rise after fusion.

Run from repo root::

    pip install -r requirements.txt
    python tools/test_hybrid_retrieval_example.py

Optional::

    python tools/test_hybrid_retrieval_example.py --embedding-model BAAI/bge-base-en-v1.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.llm.embedder import load_embedding_model, prepare_query
from src.retrieval.hybrid_retrieval import (
    BM25Resources,
    build_bm25_resources,
    fused_top_indices,
    tokenize_for_bm25,
)
from src.rag.rag_pipeline import build_retrieval_index
from src.retrieval.retriever import search


GOLD_SUBSTRING = "85.99"

# Six chunks: only [1] contains the benchmark numbers (oracle substring).
CHUNKS: list[tuple[str, str]] = [
    (
        "theory_saddle",
        """\
We study saddle-point optimization for the training objective. The update rules follow
from variational inequalities; this subsection contains no empirical benchmark results.
""",
    ),
    (
        "results_table_GOLD",
        """\
Table 2: Main results on the DL-PS data. F1 scores of 85.99 on the DL-PS data, 75.15 on
the EC-MT data and 71.53 on the EC-UQ data. FLOAT SELECTED: Table 2: Main results on DL-PS.
""",
    ),
    (
        "bleu_europarl",
        """\
We use BLEU as the automatic metric for translation evaluation. Europarl and MultiUN
corpus statistics appear in Table 18. We evaluate French-English and German-English pairs.
""",
    ),
    (
        "related_pivot",
        """\
We compare against pivot-based baselines and multilingual NMT. Cross-lingual transfer
experiments use the same preprocessing pipeline as the main model.
""",
    ),
    (
        "accuracy_discussion",
        """\
What accuracy means here: we first analyze whether the proposed system achieves stable
convergence. The proposed system's accuracy in the theoretical sense is about optimization,
not test-set accuracy; numerical scores are reported separately in the results tables.
""",
    ),
    (
        "data_stats",
        """\
The training data contains millions of tokens. We shuffle shards and filter duplicates.
Dataset licensing and splits are described in the supplementary material.
""",
    ),
]

# Mirrors ERROR_ANALYSIS.md (semantic query; table may not win dense or BM25 alone).
QUESTION_ERROR_ANALYSIS = "What accuracy does the proposed system achieve?"

# Lexical query where BM25 should strongly prefer the table chunk (hybrid motivation).
QUESTION_METRIC_LOOKUP = "What is the F1 score on the DL-PS dataset in Table 2?"


def _dense_order(
    question: str,
    embedder,
    faiss_index,
    n_chunks: int,
    *,
    top_k: int,
) -> list[int]:
    k = min(top_k, n_chunks)
    q = prepare_query(embedder.name, question)
    q_emb = embedder.encode([q])
    _, idx = search(faiss_index, q_emb, top_k=k)
    return [int(i) for i in idx[0].tolist() if i >= 0]


def _pool_has_gold(pool_texts: list[str], gold: str) -> bool:
    return any(gold in t for t in pool_texts)


def _rank_of_gold(ordered_indices: list[int], gold_index: int) -> int | None:
    if gold_index not in ordered_indices:
        return None
    return ordered_indices.index(gold_index) + 1


def run_case(
    name: str,
    question: str,
    *,
    corpus_chunks: list[str],
    labels: list[str],
    gold_index: int,
    embedder,
    faiss_index,
    bm25_resources: BM25Resources,
    retrieve_k: int,
    fusion_list_k: int | None,
    rrf_k: int,
) -> None:
    print(f"=== {name} ===")
    print("Question:", question)
    print()

    dense_order = _dense_order(
        question, embedder, faiss_index, len(corpus_chunks), top_k=retrieve_k
    )
    hybrid_indices = fused_top_indices(
        question,
        embedder=embedder,
        corpus_chunks=corpus_chunks,
        faiss_index=faiss_index,
        bm25_resources=bm25_resources,
        retrieve_k=retrieve_k,
        rrf_k=rrf_k,
        fusion_list_k=fusion_list_k,
    )

    dense_pool = [corpus_chunks[i] for i in dense_order]
    hybrid_pool = [corpus_chunks[i] for i in hybrid_indices]

    q_tokens = tokenize_for_bm25(question)
    scores = bm25_resources.bm25.get_scores(q_tokens)
    bm25_order = [int(i) for i in np.argsort(-np.asarray(scores, dtype=np.float64))]

    print(
        f"Dense top-{retrieve_k}:", dense_order, "→", [labels[i] for i in dense_order]
    )
    print(f"BM25 order (full):", bm25_order, "→", [labels[i] for i in bm25_order])
    print(
        f"Hybrid top-{retrieve_k} (fusion_list_k={fusion_list_k}):",
        hybrid_indices,
        "→",
        [labels[i] for i in hybrid_indices],
    )
    print(
        f"  gold substring in dense pool: {_pool_has_gold(dense_pool, GOLD_SUBSTRING)}"
    )
    print(
        f"  gold substring in hybrid pool: {_pool_has_gold(hybrid_pool, GOLD_SUBSTRING)}"
    )

    dr = _rank_of_gold(dense_order, gold_index)
    hr = _rank_of_gold(hybrid_indices, gold_index)
    print(f"  Gold chunk [{gold_index}] rank — dense: {dr}, hybrid: {hr}")

    if dr is None and hr is not None and _pool_has_gold(hybrid_pool, GOLD_SUBSTRING):
        print(
            "  → Dense top-k missed the gold chunk; hybrid RRF still surfaces it "
            "(typical motivation for BM25+dense on tables / numbers)."
        )
    elif dr is not None and hr is not None and hr < dr:
        print("  → Hybrid improved gold rank vs dense-only.")
    elif dr is not None and hr is not None and hr == dr == 1:
        print(
            "  → Gold already top-1 for both (embedding + sparse agree on this corpus)."
        )
    print()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--embedding-model",
        default="BAAI/bge-small-en-v1.5",
        help="Sentence-Transformer name",
    )
    p.add_argument("--retrieve-k", type=int, default=3)
    p.add_argument(
        "--fusion-list-k",
        type=int,
        default=None,
        help="Candidates per list before RRF (default: same as retrieve-k). "
        "Try 6 with --retrieve-k 3 to mimic pool>>final fusion.",
    )
    p.add_argument("--rrf-k", type=int, default=60)
    args = p.parse_args()

    labels = [t[0] for t in CHUNKS]
    corpus_chunks = [t[1] for t in CHUNKS]
    gold_index = 1

    print("Synthetic corpus (gold substring in chunk [1] only):", repr(GOLD_SUBSTRING))
    for i, lab in enumerate(labels):
        print(f"  [{i}] {lab}")
    print()

    embedder = load_embedding_model(args.embedding_model, device="cpu")
    faiss_index = build_retrieval_index(embedder, corpus_chunks)
    bm25_resources = build_bm25_resources(corpus_chunks)

    fusion_list_k = args.fusion_list_k
    if fusion_list_k is None:
        fusion_list_k = min(args.retrieve_k, len(corpus_chunks))

    run_case(
        "A — ERROR_ANALYSIS-style question (semantic)",
        QUESTION_ERROR_ANALYSIS,
        corpus_chunks=corpus_chunks,
        labels=labels,
        gold_index=gold_index,
        embedder=embedder,
        faiss_index=faiss_index,
        bm25_resources=bm25_resources,
        retrieve_k=args.retrieve_k,
        fusion_list_k=fusion_list_k,
        rrf_k=args.rrf_k,
    )
    run_case(
        "B — Metric / table lookup (lexical)",
        QUESTION_METRIC_LOOKUP,
        corpus_chunks=corpus_chunks,
        labels=labels,
        gold_index=gold_index,
        embedder=embedder,
        faiss_index=faiss_index,
        bm25_resources=bm25_resources,
        retrieve_k=args.retrieve_k,
        fusion_list_k=fusion_list_k,
        rrf_k=args.rrf_k,
    )


if __name__ == "__main__":
    main()
