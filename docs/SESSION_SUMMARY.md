# Session summary — RAG-Lab

**Goal:** Evolve the project into a small **research-style** RAG benchmark: modular code, **TREC-COVID (BEIR)** evaluation, ablations, caching, and a README suitable for GitHub and interviews.

---

## What was built or changed

| Area | Details |
|------|---------|
| **Layout** | `datasets/`, `src/`, `experiments/`, `results/`, `README.md`, `requirements.txt` |
| **Core `src/`** | `loader`, `chunker`, `embedder` (BGE/E5 prefixes, optional `device`), `retriever` (FAISS `IndexFlatIP`), `reranker`, `metrics`, `rag_pipeline` |
| **TREC / BEIR** | `beir_io.py` — loads `corpus.jsonl`, `queries.jsonl`, qrels TSV |
| **Formal metrics** | `ir-measures`: nDCG@10, P@10, AP (MAP), R@100 |
| **`exp_trec_covid.py`** | `--mode single` \| `compare-embeddings` \| `compare-chunks` \| `compare-rerank` \| `compare-all`; `--cache-dir` / default `.rag_lab_cache`; `--no-cache` |
| **`faiss_cache.py`** | Save/load FAISS + id lists keyed by corpus fingerprint, model, chunk settings — skip re-encode on reruns when cache hits |
| **Bugfix** | `rag_pipeline.py`: rerank path reported wrong `recall@*` key → **KeyError**; fixed by aligning metric label with evaluated list size |
| **`reranker`** | Optional `device=` for cross-encoder |
| **README** | Quickstart, TREC download, ablation commands, **Results (TREC-COVID)** tables |
| **`.gitignore`** | `.venv`, `__pycache__`, `.rag_lab_cache`, `data/`, `*.log`, `RAG_Lab_results_from_server/` |

---

## Benchmark results (BEIR TREC-COVID, full run)

- **Embeddings:** **E5-small-v2** best overall; **BGE-base** > **BGE-small** on P@10, nDCG@10, R@100, AP.
- **Chunks (BGE-base):** **256** slightly better than **512 / 1024** (small gap).
- **Rerank:** **BGE-reranker** improved **P@10** and **nDCG@10**; **R@100** unchanged (same top-100 candidate pool, reorder only).

Canonical tables: **`README.md` → Results (TREC-COVID, BEIR)**.

---

## Operations notes

- **Local (e.g. Apple Silicon):** Full `compare-all` over ~171k docs is **slow** for large bi-encoders; manage **sleep vs lock**, or use **`caffeinate`**, or **rent a GPU** for full runs.
- **Remote GPU (e.g. GPUHub):** Prefer **`/root/autodl-tmp`** for code, data, `HF_HOME`, and `--cache-dir`; **`source .venv`** in each new shell/tmux or use **`.venv/bin/python`**; **`tmux`** / **`nohup`** for detach; **`scp`** / **`rsync`** to pull `results/` and logs.

---

## Key paths

| Path | Role |
|------|------|
| `experiments/exp_trec_covid.py` | Main TREC script, ablations, cache wiring |
| `src/faiss_cache.py` | Disk cache for indices |
| `src/beir_io.py` | BEIR loaders |
| `results/trec_covid_compare_*.csv` | TREC CSV outputs |

---

## One-line pitch (resume / interview)

*Implemented a modular dense-retrieval RAG lab with **FAISS**, evaluated on **BEIR TREC-COVID** using **nDCG / P / MAP / R@100**, compared **embedding models**, **chunk sizes**, and **cross-encoder reranking**, and documented results on GitHub.*
