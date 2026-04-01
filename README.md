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

## Results (TREC-COVID, BEIR)

Full BEIR **trec-covid** corpus and official **qrels**; dense retrieval with FAISS (`IndexFlatIP`, normalized embeddings); **`retrieve_k` = 100** so **R@100** is reported. Chunk experiments use **chunk index → max-pool to document ids** (doc-level qrels). Rerank experiment: bi-encoder retrieves **top 100** docs, **BGE-reranker-base** reranks; **R@100** is unchanged when reranking only **reorders** candidates inside that pool.

### Embedding models (`compare-embeddings`)

| Model | P@10 | nDCG@10 | R@100 | AP (MAP) |
|-------|------|---------|-------|----------|
| intfloat/e5-small-v2 | **0.784** | **0.744** | **0.136** | **0.105** |
| BAAI/bge-base-en-v1.5 | 0.730 | 0.672 | 0.133 | 0.096 |
| BAAI/bge-small-en-v1.5 | 0.708 | 0.666 | 0.123 | 0.087 |

**E5-small-v2** scored best on all four metrics. **BGE-base** beat **BGE-small**; ranking still depends on model family and training, not only parameter count.

### Chunk sizes (`compare-chunks`, bi-encoder BGE-base)

| chunk_size | P@10 | nDCG@10 | R@100 | AP |
|------------|------|---------|-------|-----|
| 256 | **0.738** | **0.682** | **0.133** | **0.097** |
| 512 | 0.730 | 0.672 | 0.133 | 0.096 |
| 1024 | 0.730 | 0.672 | 0.133 | 0.096 |

**256** (word-token–style chunks) slightly improved **P@10** and **nDCG@10** vs **512 / 1024**; the latter two were nearly identical. Effect size is modest.

### Reranking (`compare-rerank`, bi-encoder BGE-base)

| Setting | P@10 | nDCG@10 | R@100 | AP |
|---------|------|---------|-------|-----|
| Bi-encoder only | 0.730 | 0.672 | 0.133 | 0.096 |
| + BGE-reranker-base | **0.818** | **0.762** | 0.133 | **0.102** |

**Cross-encoder reranking** improved **P@10** and **nDCG@10** strongly. **R@100** did not change, which is expected when **R@100** is evaluated on the **same first-stage top-100** candidate set. **AP** increased slightly.

### One-line summary

On BEIR TREC-COVID, **E5-small** achieved the best dense retrieval scores among the three bi-encoders; **chunk size 256** slightly improved **BGE-base** over 512/1024; **BGE-reranker** improved **P@10** and **nDCG@10** with **no R@100 gain** under a fixed top-100 pool.

Raw CSVs: `results/trec_covid_compare_embeddings.csv`, `trec_covid_compare_chunks.csv`, `trec_covid_compare_rerank.csv` (paths may differ if copied elsewhere).

## Metric

**Custom QA experiments:** **recall@k** — for each question, if at least one of the top-k retrieved chunks contains the ground-truth answer string, it counts as 1; otherwise 0. The final score is the mean over all questions.

**TREC-COVID:** standard IR metrics from judged qrels (see experiment script), not substring overlap.

## Notes

- Models are loaded via `sentence-transformers` so you can swap in BGE / E5 / rerankers easily.
- Dataset is JSONL for easy extension.
- For leaderboard-comparable TREC-COVID runs, use the full corpus and official qrels; avoid `--max-docs` except for smoke tests.

