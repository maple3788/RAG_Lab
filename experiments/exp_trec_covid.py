"""
Dense retrieval on BEIR TREC-COVID with optional **comparisons** (like the small QA exps):

- ``compare-embeddings``: several bi-encoders → ``results/trec_covid_compare_embeddings.csv``
- ``compare-chunks``: chunk sizes (chunk index → max-pool to doc ids) → ``trec_covid_compare_chunks.csv``
- ``compare-rerank``: bi-encoder only vs bi-encoder + cross-encoder rerank → ``trec_covid_compare_rerank.csv``
- ``single`` / ``compare-all``: see ``--mode``

Metrics: nDCG@10, P@10, MAP (AP), R@100 when retrieve_k >= 100.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import ir_measures as ir
import numpy as np
import pandas as pd
from ir_measures import ScoredDoc
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.beir_io import (
    corpus_list_to_dict,
    load_beir_corpus_ordered,
    load_beir_qrels,
    load_beir_queries_ordered,
    ordered_qids_from_qrels,
)
from src.chunker import chunk_text
from src.embedder import (
    EmbeddingModel,
    load_embedding_model,
    prepare_passages,
    prepare_query,
)
from src.faiss_cache import (
    corpus_fingerprint,
    save_chunk_cache,
    save_doc_level_cache,
    stem_chunked,
    stem_doc_level,
    try_load_chunk_cache,
    try_load_doc_level_cache,
)
from src.reranker import load_reranker
from src.retriever import FaissIndex, build_faiss_index, search


def _default_measures(retrieve_k: int) -> list:
    ms = [
        ir.parse_measure("nDCG@10"),
        ir.parse_measure("P@10"),
        ir.parse_measure("MAP"),
    ]
    if retrieve_k >= 100:
        ms.append(ir.parse_measure("R@100"))
    return ms


def _metrics_to_row(label: Dict[str, object], results: dict) -> Dict[str, object]:
    row = dict(label)
    for k, v in results.items():
        row[str(k)] = float(v)
    return row


def _build_or_load_doc_index(
    embedder: EmbeddingModel,
    doc_ids: Sequence[str],
    corpus_dict: Dict[str, str],
    *,
    batch_size: int,
    corpus_path: Path,
    max_docs: int | None,
    cache_dir: Path | None,
) -> FaissIndex:
    fp = corpus_fingerprint(corpus_path, max_docs)
    stem = stem_doc_level(embedder.name, fp)
    normalize = True
    if cache_dir is not None:
        loaded = try_load_doc_level_cache(
            cache_dir,
            stem,
            embedding_model=embedder.name,
            corpus_fingerprint_str=fp,
            doc_ids=doc_ids,
            normalize_embeddings=normalize,
        )
        if loaded is not None:
            print(f"Loaded doc-level FAISS cache ({stem}) — skipping encoding.")
            return loaded

    texts = prepare_passages(embedder.name, [corpus_dict[d] for d in doc_ids])
    print(f"Encoding {len(texts)} documents…")
    doc_emb = embedder.encode(texts, batch_size=batch_size, show_progress_bar=True)
    faiss_index = build_faiss_index(doc_emb)
    if cache_dir is not None:
        save_doc_level_cache(
            cache_dir,
            stem,
            faiss_index=faiss_index,
            doc_ids=doc_ids,
            embedding_model=embedder.name,
            corpus_fingerprint_str=fp,
            max_docs=max_docs,
            normalize_embeddings=normalize,
        )
        print(f"Saved doc-level FAISS cache ({stem}).")
    return faiss_index


def run_doc_level(
    embedder: EmbeddingModel,
    doc_ids: Sequence[str],
    corpus_dict: Dict[str, str],
    qtext: Dict[str, str],
    qid_order: Sequence[str],
    retrieve_k: int,
    *,
    batch_size: int = 64,
    corpus_path: Path | None = None,
    max_docs: int | None = None,
    cache_dir: Path | None = None,
) -> List[ScoredDoc]:
    if corpus_path is None:
        raise ValueError(
            "corpus_path is required for indexing (use data-dir / corpus.jsonl)"
        )
    faiss_index = _build_or_load_doc_index(
        embedder,
        doc_ids,
        corpus_dict,
        batch_size=batch_size,
        corpus_path=corpus_path,
        max_docs=max_docs,
        cache_dir=cache_dir,
    )
    rk = max(1, min(retrieve_k, len(doc_ids)))
    run: List[ScoredDoc] = []
    for qid in tqdm(qid_order, desc="queries (doc-level)"):
        if qid not in qtext:
            continue
        q = prepare_query(embedder.name, qtext[qid])
        q_emb = embedder.encode([q], batch_size=1)
        _, idx = search(faiss_index, q_emb, top_k=rk)
        for rank, j in enumerate(idx[0].tolist()):
            if j < 0 or j >= len(doc_ids):
                continue
            did = doc_ids[j]
            run.append(ScoredDoc(query_id=qid, doc_id=did, score=float(rk - rank)))
    return run


def run_chunked_doc_pool(
    embedder: EmbeddingModel,
    corpus_rows: Sequence[Tuple[str, str]],
    *,
    chunk_size: int,
    chunk_overlap: int,
    chunk_search_k: int,
    qtext: Dict[str, str],
    qid_order: Sequence[str],
    retrieve_k: int,
    batch_size: int = 64,
    corpus_path: Path | None = None,
    max_docs: int | None = None,
    cache_dir: Path | None = None,
) -> List[ScoredDoc]:
    """Index chunks; search top chunks; max-pool scores to parent doc_ids (TREC qrels are doc-level)."""
    chunk_texts: List[str] = []
    chunk_parents: List[str] = []
    for i, (did, text) in enumerate(corpus_rows):
        for ch in chunk_text(
            text,
            doc_id=i,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        ):
            chunk_texts.append(ch.text)
            chunk_parents.append(did)

    if not chunk_texts:
        raise ValueError("No chunks produced (empty corpus?)")

    if corpus_path is None:
        raise ValueError("corpus_path is required for chunk indexing / cache")

    fp = corpus_fingerprint(corpus_path, max_docs)
    stem = stem_chunked(embedder.name, fp, chunk_size, chunk_overlap)
    normalize = True
    faiss_index: FaissIndex
    if cache_dir is not None:
        loaded = try_load_chunk_cache(
            cache_dir,
            stem,
            embedding_model=embedder.name,
            corpus_fingerprint_str=fp,
            chunk_parents=chunk_parents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            normalize_embeddings=normalize,
        )
        if loaded is not None:
            print(f"Loaded chunk FAISS cache ({stem}) — skipping chunk encoding.")
            faiss_index = loaded
        else:
            passages = prepare_passages(embedder.name, chunk_texts)
            print(
                f"Encoding {len(passages)} chunks (size={chunk_size}, overlap={chunk_overlap})…"
            )
            emb = embedder.encode(
                passages, batch_size=batch_size, show_progress_bar=True
            )
            faiss_index = build_faiss_index(emb)
            save_chunk_cache(
                cache_dir,
                stem,
                faiss_index=faiss_index,
                chunk_parents=chunk_parents,
                embedding_model=embedder.name,
                corpus_fingerprint_str=fp,
                max_docs=max_docs,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                normalize_embeddings=normalize,
            )
            print(f"Saved chunk FAISS cache ({stem}).")
    else:
        passages = prepare_passages(embedder.name, chunk_texts)
        print(
            f"Encoding {len(passages)} chunks (size={chunk_size}, overlap={chunk_overlap})…"
        )
        emb = embedder.encode(passages, batch_size=batch_size, show_progress_bar=True)
        faiss_index = build_faiss_index(emb)

    n_chunks = len(chunk_parents)
    k_search = max(1, min(chunk_search_k, n_chunks))
    rk = max(1, retrieve_k)

    run: List[ScoredDoc] = []
    for qid in tqdm(qid_order, desc=f"queries (chunks {chunk_size})"):
        if qid not in qtext:
            continue
        q = prepare_query(embedder.name, qtext[qid])
        q_emb = embedder.encode([q], batch_size=1)
        scores, idx = search(faiss_index, q_emb, top_k=k_search)
        best: Dict[str, float] = defaultdict(float)
        for sc, j in zip(scores[0].tolist(), idx[0].tolist()):
            if j < 0 or j >= n_chunks:
                continue
            pid = chunk_parents[j]
            best[pid] = max(best[pid], float(sc))
        ranked = sorted(best.items(), key=lambda x: -x[1])[:rk]
        for did, sc in ranked:
            run.append(ScoredDoc(query_id=qid, doc_id=did, score=sc))
    return run


def run_with_rerank(
    embedder: EmbeddingModel,
    reranker,
    doc_ids: Sequence[str],
    corpus_dict: Dict[str, str],
    qtext: Dict[str, str],
    qid_order: Sequence[str],
    first_stage_k: int,
    retrieve_k: int,
    *,
    batch_size: int = 64,
    corpus_path: Path | None = None,
    max_docs: int | None = None,
    cache_dir: Path | None = None,
) -> List[ScoredDoc]:
    if corpus_path is None:
        raise ValueError(
            "corpus_path is required for indexing (use data-dir / corpus.jsonl)"
        )
    faiss_index = _build_or_load_doc_index(
        embedder,
        doc_ids,
        corpus_dict,
        batch_size=batch_size,
        corpus_path=corpus_path,
        max_docs=max_docs,
        cache_dir=cache_dir,
    )
    fs = max(1, min(first_stage_k, len(doc_ids)))
    rk = max(1, min(retrieve_k, fs))

    run: List[ScoredDoc] = []
    for qid in tqdm(qid_order, desc="queries (rerank)"):
        if qid not in qtext:
            continue
        q_raw = qtext[qid]
        q = prepare_query(embedder.name, q_raw)
        q_emb = embedder.encode([q], batch_size=1)
        _, idx = search(faiss_index, q_emb, top_k=fs)
        cand_ids = [doc_ids[j] for j in idx[0].tolist() if 0 <= j < len(doc_ids)]
        cand_texts = [corpus_dict[d] for d in cand_ids]
        if not cand_ids:
            continue
        pairs = [(q_raw, t) for t in cand_texts]
        scores = np.asarray(reranker.model.predict(pairs), dtype=np.float32)
        order = np.argsort(-scores)
        for rank in range(min(rk, len(order))):
            j = int(order[rank])
            run.append(
                ScoredDoc(
                    query_id=qid,
                    doc_id=cand_ids[j],
                    score=float(scores[j]),
                )
            )
    return run


def evaluate_run(
    qrels: list,
    run: List[ScoredDoc],
    retrieve_k: int,
) -> dict:
    measures = _default_measures(retrieve_k)
    return ir.calc_aggregate(measures, qrels, run)


def load_beir_split(args: argparse.Namespace) -> Tuple:
    data_dir: Path = args.data_dir
    corpus_path = data_dir / "corpus.jsonl"
    queries_path = data_dir / "queries.jsonl"
    qrels_path = data_dir / args.qrels_file

    if (
        not corpus_path.is_file()
        or not queries_path.is_file()
        or not qrels_path.is_file()
    ):
        print(
            "Missing BEIR TREC-COVID files.\n"
            f"Expected:\n  {corpus_path}\n  {queries_path}\n  {qrels_path}\n\n"
            "Download the dataset (see README), unzip, and set --data-dir."
        )
        raise SystemExit(1)

    corpus_rows = load_beir_corpus_ordered(corpus_path)
    if args.max_docs is not None:
        corpus_rows = corpus_rows[: args.max_docs]

    corpus_dict = corpus_list_to_dict(corpus_rows)
    doc_ids = [did for did, _ in corpus_rows]
    if not doc_ids:
        print("Corpus is empty.")
        raise SystemExit(1)

    corpus_ids = set(corpus_dict.keys())
    qrels = load_beir_qrels(qrels_path)
    qrels = [q for q in qrels if q.doc_id in corpus_ids]
    if not qrels:
        print(
            "No qrels left after intersecting with corpus doc_ids. "
            "If you used --max-docs, increase it."
        )
        raise SystemExit(1)

    qid_order = ordered_qids_from_qrels(qrels)
    if args.max_queries is not None:
        keep = set(qid_order[: args.max_queries])
        qrels = [q for q in qrels if q.query_id in keep]
        qid_order = [q for q in qid_order if q in keep]

    queries_rows = load_beir_queries_ordered(queries_path)
    qtext = {qid: t for qid, t in queries_rows}

    qrels = [q for q in qrels if q.query_id in qtext]
    qid_order = [qid for qid in qid_order if qid in qtext]
    if not qrels or not qid_order:
        print("No overlapping queries between qrels and queries.jsonl.")
        raise SystemExit(1)

    return corpus_rows, corpus_dict, doc_ids, qrels, qid_order, qtext


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BEIR TREC-COVID: single run or compare embeddings / chunks / rerank"
    )
    parser.add_argument(
        "--mode",
        choices=[
            "single",
            "compare-embeddings",
            "compare-chunks",
            "compare-rerank",
            "compare-all",
        ],
        default="single",
        help="single: one model; compare-*: ablation tables; compare-all runs all three comparisons",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "trec-covid",
    )
    parser.add_argument(
        "--embedding-model",
        default="BAAI/bge-base-en-v1.5",
        help="Bi-encoder for single / chunk / rerank baseline and for compare-chunks",
    )
    parser.add_argument(
        "--embedding-models",
        nargs="+",
        default=[
            "BAAI/bge-small-en-v1.5",
            "BAAI/bge-base-en-v1.5",
            "intfloat/e5-small-v2",
        ],
        help="Models for --mode compare-embeddings",
    )
    parser.add_argument(
        "--chunk-sizes",
        nargs="+",
        type=int,
        default=[256, 512, 1024],
        help="Chunk sizes (word-token-ish) for compare-chunks",
    )
    parser.add_argument(
        "--chunk-search-k",
        type=int,
        default=500,
        help="Top chunks to retrieve before max-pooling to documents",
    )
    parser.add_argument(
        "--rerank-model",
        default="BAAI/bge-reranker-base",
        help="Cross-encoder for compare-rerank",
    )
    parser.add_argument(
        "--first-stage-k",
        type=int,
        default=100,
        help="Bi-encoder retrieve this many docs before reranking",
    )
    parser.add_argument("--retrieve-k", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--qrels-file", default="qrels/test.tsv")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory for FAISS + id caches (default: <data-dir>/.rag_lab_cache)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Do not read or write on-disk index; always re-encode the corpus",
    )
    args = parser.parse_args()

    retrieve_k = max(1, args.retrieve_k)
    corpus_path = args.data_dir / "corpus.jsonl"
    cache_dir: Path | None = (
        None if args.no_cache else (args.cache_dir or args.data_dir / ".rag_lab_cache")
    )

    corpus_rows, corpus_dict, doc_ids, qrels, qid_order, qtext = load_beir_split(args)
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    def save_df(df: pd.DataFrame, name: str) -> None:
        path = out_dir / name
        df.to_csv(path, index=False)
        print(f"Saved: {path}")

    modes = (
        ["compare-embeddings", "compare-chunks", "compare-rerank"]
        if args.mode == "compare-all"
        else [args.mode]
    )

    for mode in modes:
        if mode == "single":
            emb = load_embedding_model(
                args.embedding_model, normalize=True, device=args.device
            )
            run = run_doc_level(
                emb,
                doc_ids,
                corpus_dict,
                qtext,
                qid_order,
                retrieve_k,
                batch_size=args.batch_size,
                corpus_path=corpus_path,
                max_docs=args.max_docs,
                cache_dir=cache_dir,
            )
            res = evaluate_run(qrels, run, retrieve_k)
            row = _metrics_to_row(
                {"mode": "single", "embedding_model": args.embedding_model}, res
            )
            save_df(pd.DataFrame([row]), "trec_covid_beir_results.csv")
            for m, v in res.items():
                print(f"  {m}: {v:.6f}")

        elif mode == "compare-embeddings":
            rows = []
            for model_name in args.embedding_models:
                print(f"\n=== Embedding model: {model_name} ===")
                emb = load_embedding_model(
                    model_name, normalize=True, device=args.device
                )
                run = run_doc_level(
                    emb,
                    doc_ids,
                    corpus_dict,
                    qtext,
                    qid_order,
                    retrieve_k,
                    batch_size=args.batch_size,
                    corpus_path=corpus_path,
                    max_docs=args.max_docs,
                    cache_dir=cache_dir,
                )
                res = evaluate_run(qrels, run, retrieve_k)
                rows.append(_metrics_to_row({"embedding_model": model_name}, res))
                print(res)
            save_df(pd.DataFrame(rows), "trec_covid_compare_embeddings.csv")

        elif mode == "compare-chunks":
            emb = load_embedding_model(
                args.embedding_model, normalize=True, device=args.device
            )
            rows = []
            for cs in args.chunk_sizes:
                ov = min(64, max(0, cs // 4))
                print(f"\n=== Chunk size: {cs}, overlap: {ov} ===")
                run = run_chunked_doc_pool(
                    emb,
                    corpus_rows,
                    chunk_size=cs,
                    chunk_overlap=ov,
                    chunk_search_k=args.chunk_search_k,
                    qtext=qtext,
                    qid_order=qid_order,
                    retrieve_k=retrieve_k,
                    batch_size=args.batch_size,
                    corpus_path=corpus_path,
                    max_docs=args.max_docs,
                    cache_dir=cache_dir,
                )
                res = evaluate_run(qrels, run, retrieve_k)
                rows.append(
                    _metrics_to_row(
                        {
                            "embedding_model": args.embedding_model,
                            "chunk_size": cs,
                            "chunk_overlap": ov,
                            "chunk_search_k": args.chunk_search_k,
                        },
                        res,
                    )
                )
                print(res)
            save_df(pd.DataFrame(rows), "trec_covid_compare_chunks.csv")

        elif mode == "compare-rerank":
            emb = load_embedding_model(
                args.embedding_model, normalize=True, device=args.device
            )
            print("\n=== No rerank (bi-encoder only) ===")
            run_base = run_doc_level(
                emb,
                doc_ids,
                corpus_dict,
                qtext,
                qid_order,
                retrieve_k,
                batch_size=args.batch_size,
                corpus_path=corpus_path,
                max_docs=args.max_docs,
                cache_dir=cache_dir,
            )
            res_base = evaluate_run(qrels, run_base, retrieve_k)
            print(res_base)

            print(f"\n=== Rerank: {args.rerank_model} ===")
            rr = load_reranker(args.rerank_model, device=args.device)
            run_rr = run_with_rerank(
                emb,
                rr,
                doc_ids,
                corpus_dict,
                qtext,
                qid_order,
                first_stage_k=args.first_stage_k,
                retrieve_k=retrieve_k,
                batch_size=args.batch_size,
                corpus_path=corpus_path,
                max_docs=args.max_docs,
                cache_dir=cache_dir,
            )
            res_rr = evaluate_run(qrels, run_rr, retrieve_k)
            print(res_rr)

            rows = [
                _metrics_to_row(
                    {
                        "setting": "bi_encoder_only",
                        "embedding_model": args.embedding_model,
                    },
                    res_base,
                ),
                _metrics_to_row(
                    {
                        "setting": "bi_encoder_plus_reranker",
                        "embedding_model": args.embedding_model,
                        "rerank_model": args.rerank_model,
                        "first_stage_k": args.first_stage_k,
                    },
                    res_rr,
                ),
            ]
            save_df(pd.DataFrame(rows), "trec_covid_compare_rerank.csv")

    if args.mode == "compare-all":
        print(
            "\ncompare-all finished: trec_covid_compare_embeddings.csv, "
            "trec_covid_compare_chunks.csv, trec_covid_compare_rerank.csv"
        )


if __name__ == "__main__":
    main()
