# RAG-Lab

## Overview

This repository is an experiment framework for studying how retrieval components affect RAG performance.
It is structured like a small research project: modular pipeline code, reproducible experiments, and saved results.

## Project structure

```text
rag-lab
├── datasets
│   └── qa_dataset.jsonl
├── src
│   ├── loader.py
│   ├── chunker.py
│   ├── embedder.py
│   ├── retriever.py
│   ├── reranker.py
│   ├── metrics.py
│   ├── rag_pipeline.py
│   ├── beir_io.py
│   └── faiss_cache.py
├── data
│   └── trec-covid/          # optional: BEIR TREC-COVID (download, see below)
├── experiments
│   ├── exp_embedding.py
│   ├── exp_chunk_size.py
│   ├── exp_rerank.py
│   └── exp_trec_covid.py
└── results
    ├── embedding_results.csv
    └── *.png
```

## Experiments

1. **Embedding model comparison** (`experiments/exp_embedding.py`)
2. **Chunk size analysis** (`experiments/exp_chunk_size.py`)
3. **Reranking evaluation** (`experiments/exp_rerank.py`)
4. **TREC-COVID (BEIR format)** (`experiments/exp_trec_covid.py`) — standard IR metrics on an official benchmark

The first three experiments use **recall@k** on the small custom QA JSONL. The TREC-COVID experiment uses **graded qrels** and reports **nDCG@10, P@10, MAP, R@100** (via `ir-measures`), which matches how many retrieval papers report BEIR / TREC-style results.

## Quickstart

Create env and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run experiments:

```bash
python experiments/exp_embedding.py
python experiments/exp_chunk_size.py
python experiments/exp_rerank.py
```

Results will be saved to `results/` as CSV + plots.

## TREC-COVID (formal benchmark, BEIR packaging)

Many papers evaluate on **TREC-COVID** using the **BEIR** release: fixed corpus, topics, and qrels so numbers are comparable across systems.

1. Download and unzip (large, ~hundreds of MB):

   - [BEIR `trec-covid.zip`](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip)

2. You should get a folder containing at least:

   - `corpus.jsonl`
   - `queries.jsonl`
   - `qrels/test.tsv`

3. Point the experiment at that folder. **Single run** (one bi-encoder):

```bash
python experiments/exp_trec_covid.py --data-dir data/trec-covid --device mps
```

**Ablations** (same spirit as `exp_embedding.py` / `exp_chunk_size.py` / `exp_rerank.py`):

```bash
# Several embedding models → results/trec_covid_compare_embeddings.csv
python experiments/exp_trec_covid.py --data-dir data/trec-covid --mode compare-embeddings

# Chunk sizes (chunk index → max-pool to doc ids, matches doc-level qrels) → trec_covid_compare_chunks.csv
python experiments/exp_trec_covid.py --data-dir data/trec-covid --mode compare-chunks --embedding-model BAAI/bge-base-en-v1.5

# Bi-encoder only vs bi-encoder + cross-encoder rerank → trec_covid_compare_rerank.csv
python experiments/exp_trec_covid.py --data-dir data/trec-covid --mode compare-rerank --first-stage-k 100

# Run all three comparisons in one go (long)
python experiments/exp_trec_covid.py --data-dir data/trec-covid --mode compare-all
```

Optional flags: `--embedding-model`, `--embedding-models`, `--chunk-sizes`, `--chunk-search-k`, `--rerank-model`, `--retrieve-k` (use **≥ 100** for **R@100**), `--max-queries` / `--max-docs` for debugging only.

**Index cache:** FAISS indices and id lists are stored under `<data-dir>/.rag_lab_cache/` (override with `--cache-dir`) so reruns with the **same** corpus fingerprint, embedding model, and (for chunk mode) chunk settings **reuse** the index instead of re-encoding. Use `--no-cache` to force a full rebuild.

Outputs: `results/trec_covid_beir_results.csv` for `--mode single`; comparison CSVs as above. Column **`AP`** is **MAP** (mean average precision).

Background: [NIST TREC-COVID overview](https://ir.nist.gov/covidSubmit/index.html).

## Metric

**Custom QA experiments:** **recall@k** — for each question, if at least one of the top-k retrieved chunks contains the ground-truth answer string, it counts as 1; otherwise 0. The final score is the mean over all questions.

**TREC-COVID:** standard IR metrics from judged qrels (see experiment script), not substring overlap.

## Notes

- Models are loaded via `sentence-transformers` so you can swap in BGE / E5 / rerankers easily.
- Dataset is JSONL for easy extension.
- For leaderboard-comparable TREC-COVID runs, use the full corpus and official qrels; avoid `--max-docs` except for smoke tests.

