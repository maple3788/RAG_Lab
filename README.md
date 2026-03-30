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
│   └── rag_pipeline.py
├── experiments
│   ├── exp_embedding.py
│   ├── exp_chunk_size.py
│   └── exp_rerank.py
└── results
    ├── embedding_results.csv
    └── *.png
```

## Experiments

1. **Embedding model comparison** (`experiments/exp_embedding.py`)
2. **Chunk size analysis** (`experiments/exp_chunk_size.py`)
3. **Reranking evaluation** (`experiments/exp_rerank.py`)

All experiments evaluate retrieval quality using **recall@k** on a small QA dataset.

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

## Metric

We use **recall@k**: for each question, if at least one of the top-k retrieved chunks contains the ground-truth answer string, it counts as 1; otherwise 0. The final score is the mean over all questions.

## Notes

- Models are loaded via `sentence-transformers` so you can swap in BGE / E5 / rerankers easily.
- Dataset is JSONL for easy extension.

