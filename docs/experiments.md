# Experiments

This document lists the main evaluation scripts and expected outputs.

## Retrieval-centric experiments

- `experiments/exp_embedding.py`
- `experiments/exp_chunk_size.py`
- `experiments/exp_rerank.py`

Metric family: `recall@k` on QA JSONL data.

## IR benchmark (BEIR / TREC-COVID)

- `experiments/exp_trec_covid.py`

Metric family:
- `nDCG@10`
- `P@10`
- `MAP`
- `R@100`

Typical usage:

```bash
python experiments/exp_trec_covid.py --data-dir data/trec-covid --mode compare-embeddings
python experiments/exp_trec_covid.py --data-dir data/trec-covid --mode compare-all
```

## Generation-centric experiments

- `experiments/exp_rag_generation.py`
- `experiments/exp_rag_generation_trec.py`
- `experiments/exp_rag_generation_triviaqa.py`
- `experiments/exp_rag_generation_qasper.py`
- `experiments/exp_rag_generation_financebench.py`

Metric family:
- Exact Match
- Token F1
- Gold Hit
- Latency

Main ablations in `exp_rag_generation.py`:
- reranker on/off
- final context size (`final_k`)
- prompt template
- truncation strategy
- query expansion mode (`none`, `multi_query`, `hyde`)
- expansion query count cap (`expansion_max_queries`)
- rewrite-on-empty fallback (`rewrite_on_empty_retrieval`)

Query expansion notes:
- `multi_query`: generate several paraphrase queries and merge/dedupe retrieval pools before final rerank.
- `hyde`: generate a hypothetical passage and retrieve via dense passage embedding (FAISS path).
- If rewrite-on-empty triggers, retry uses one rewritten query (no second expansion pass).

## Hybrid retrieval comparison

- `experiments/exp_qasper_hybrid_compare.py`

Modes:
- `retrieval`: dense vs BM25+dense (RRF), no LLM required
- `generation`: end-to-end answer metrics

## RAGAS evaluation

- `experiments/exp_ragas_eval.py`
- `experiments/exp_ragas_financebench.py`

Typical usage:

```bash
python experiments/exp_ragas_eval.py --data-path datasets/qa_dataset.jsonl --max-examples 100 --llm-backend ollama
```

## Outputs

Most scripts write CSV files under `results/`.

For dashboard compatibility, `exp_rag_generation.py` also logs aggregate and per-query rows into:
- `results/experiment_db.sqlite`

