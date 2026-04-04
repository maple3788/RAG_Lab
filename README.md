# RAG-Lab

How **retrieval** choices affect **RAG**: one modular pipeline, benchmarks, CSVs/plots.

**Tracks:** (1) **IR / RC benchmarks** — TREC-COVID, TriviaQA RC. (2) **Long docs** — [QASPER](https://arxiv.org/abs/2105.03011) (full papers → PDF/manual-like QA; same retrieval path as TriviaQA with `per_example_retrieval`).

## Layout

```text
rag-lab
├── datasets/qa_dataset.jsonl
├── demo/app.py                 # Streamlit: upload → FAISS + rerank → LLM (see below)
├── src/                        # loader, chunker, embedder, retriever, hybrid_retrieval (BM25+RRF),
│                               # reranker, metrics, generator, rag_pipeline, rag_generation, …
├── data/trec-covid/            # optional BEIR
├── experiments/                # exp_embedding, exp_chunk_size, exp_rerank, exp_trec_covid,
│                               # exp_rag_generation*.py, exp_qasper_hybrid_compare.py
├── analysis/  assets/  scripts/  tools/  results/
```

## Architecture

```mermaid
flowchart TD
    A[PDF / Text Corpus] --> B(Chunking)
    B --> C{Embedding Model\nE5 / BGE}
    C --> D[(FAISS Vector Index)]
    
    Q[User Question] --> E{Embedding Model}
    E --> F[Top-K Retrieval]
    D --> F
    
    F -->|Initial k=100 chunks| G[Cross-Encoder Reranker\ne.g., BGE-Reranker]
    G -->|Final k=5 chunks| H[Prompt Template + Truncation]
    H --> I[LLM Generation\nLlama 3 / Gemini]
    I --> J[Final Output & Evaluation Metrics]
```

### Hybrid retrieval (BM25 + dense)

For long documents, **dense-only** FAISS search can miss chunks that match lexically (tables, numbers, rare tokens). Optional **hybrid** retrieval builds a **BM25** index over the same chunks and merges dense + sparse rankings with **Reciprocal Rank Fusion (RRF)** (`src/hybrid_retrieval.py`, dependency `rank-bm25`).

- **Pipeline:** `retrieve_passages_pool_and_final(..., bm25_resources=...)` or set `RAGGenerationConfig(use_hybrid=True)` when `per_example_retrieval=True` (QASPER / TriviaQA RC style).
- **`fusion_list_k`:** how many candidates each retriever contributes before RRF; default `None` uses the full chunk list for that document (capped by corpus size). Larger lists help when gold is mid-ranked on one side only.
- **Sanity check (no dataset):** `python tools/test_hybrid_retrieval_example.py --fusion-list-k 6`
- **QASPER benchmark:** `python experiments/exp_qasper_hybrid_compare.py --mode retrieval --max-examples 200` → writes `results/qasper_dense_hybrid_retrieval_*.csv`.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env    # optional: GEMINI_API_KEY, OPENAI_*, OLLAMA_BASE_URL
```

**Streamlit demo** (upload PDF/text, ask questions, tabs: FAISS chunks → cross-encoder rerank → prompt + answer):

```bash
.venv/bin/streamlit run demo/app.py
```

Use the **project venv** so `sentence_transformers` / `torch` match. Index is **in-memory only**: each new `streamlit run` or server restart needs **upload + “Build index”** again. Removing the file in the widget does **not** clear the index until you rebuild or reload the session. Persistence: not implemented (would need disk cache or a vector DB). `.streamlit/config.toml` turns off file-watching to avoid Hugging Face `transformers` log spam; refresh the browser after code edits.

**Batch experiments:**

```bash
python experiments/exp_embedding.py
python experiments/exp_chunk_size.py
python experiments/exp_rerank.py
```

**RAG generation** (needs LLM, Ollama, or `--mock-generation`):

```bash
python experiments/exp_rag_generation.py --mode all
# ollama pull llama3.2 && python experiments/exp_rag_generation.py --mode all --llm-backend ollama
```

Modes → `results/rag_generation_results.csv`: `compare-rerank`, `compare-topk`, `compare-prompts`, `compare-truncation`, `all`.

```bash
python experiments/exp_rag_generation_trec.py --data-dir data/trec-covid --mode all --llm-backend ollama
python experiments/exp_rag_generation_triviaqa.py --split validation --max-examples 200 --mode all --llm-backend ollama
python experiments/exp_rag_generation_qasper.py --split validation --max-examples 200 --mode all --llm-backend ollama

# QASPER: dense FAISS vs BM25+dense (RRF) — retrieval oracle (gold alias in chunks), no LLM by default
python experiments/exp_qasper_hybrid_compare.py --mode retrieval --max-examples 200
# End-to-end EM/F1/gold_hit vs same with hybrid (needs LLM or --mock-generation)
# python experiments/exp_qasper_hybrid_compare.py --mode generation --max-examples 50 --mock-generation
```

`.env` loads from repo root (`python-dotenv`); do not commit `.env`.

## Experiments (summary)

| # | Script | Measures |
|---|--------|----------|
| 1–3 | `exp_embedding.py`, `exp_chunk_size.py`, `exp_rerank.py` | recall@k (custom QA JSONL) |
| 4 | `exp_trec_covid.py` | nDCG@10, P@10, MAP, R@100 (BEIR qrels) |
| 5 | `exp_rag_generation.py` | EM, F1, gold_hit; ablations: rerank, `final_k`, prompts, truncation. Default **Gemini**; **`--llm-backend ollama` / `openai`**; region issues → `docs/gemini-region-restriction.md`; **`--mock-generation`** = no API |
| 6–8 | `exp_rag_generation_trec.py`, `_triviaqa.py`, `_qasper.py` | Same style on TREC corpus / TriviaQA RC / QASPER (long papers). Outputs under `results/` |
| 9 | `exp_qasper_hybrid_compare.py` | QASPER **dense vs hybrid** (BM25+RRF): oracle gold substring in `retrieve_k` pool / `final_k` slice; optional `--mode generation` |

## TREC-COVID (BEIR)

1. Download [trec-covid.zip](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip) → `corpus.jsonl`, `queries.jsonl`, `qrels/test.tsv` under `data/trec-covid/`.

```bash
python experiments/exp_trec_covid.py --data-dir data/trec-covid --device mps
python experiments/exp_trec_covid.py --data-dir data/trec-covid --mode compare-embeddings
python experiments/exp_trec_covid.py --data-dir data/trec-covid --mode compare-all   # long
```

Flags: `--embedding-model(s)`, `--chunk-sizes`, `--chunk-search-k`, `--rerank-model`, `--retrieve-k` (≥100 for R@100), `--max-queries` / `--max-docs` (smoke only). **Cache:** `<data-dir>/.rag_lab_cache/` (`--cache-dir`, `--no-cache`). Outputs: `results/trec_covid_*.csv`; **AP** = MAP. [NIST background](https://ir.nist.gov/covidSubmit/index.html).

## Results (TREC-COVID)

Setup: full corpus + qrels; FAISS `IndexFlatIP`, normalized embeddings; `retrieve_k` = 100; rerank = bi-encoder top-100 → **BGE-reranker-base** (reorders inside pool; R@100 unchanged).

**Embeddings (`compare-embeddings`)**

| Model | P@10 | nDCG@10 | R@100 | AP |
|-------|------|---------|-------|-----|
| intfloat/e5-small-v2 | **0.784** | **0.744** | **0.136** | **0.105** |
| BAAI/bge-base-en-v1.5 | 0.730 | 0.672 | 0.133 | 0.096 |
| BAAI/bge-small-en-v1.5 | 0.708 | 0.666 | 0.123 | 0.087 |

**Chunks (`compare-chunks`, BGE-base)** — 256 slightly best; 512/1024 tie.

**Rerank (`compare-rerank`)** — +rerank: P@10 **0.818**, nDCG@10 **0.762** vs bi-encoder-only 0.730 / 0.672.

Raw: `results/trec_covid_compare_embeddings.csv`, `*_chunks.csv`, `*_rerank.csv`.

## Results (TriviaQA RAG, n=200)

Ollama `llama3.2`, BGE + reranker, `validation`, `--mode all`. **Rerank** ↑ EM/F1/gold_hit. **Prompts:** `bullets` best F1. **Truncation** (1200 chars): **head** ≫ tail/middle. **final_k** 3 ≈ 5 > 1. Server CSV copy may live under `RAG_Lab_results_from_server/triviaqa_rag_generation_results.csv`.

## Results (QASPER, n=200)

Long document QA evaluation. Metrics focus on Token F1 and Gold Hit (whether the answer contains the exact gold alias).

**Retrieval oracle (dense vs hybrid, validation, n=200)** — no LLM; **gold hit** = any answer alias appears as a substring in a retrieved chunk (`--retrieve-k 10`, `--final-k 3`, no rerank, BGE-base, chunk 384/48). Slight gains from BM25+RRF on pool and top-`final_k` slices; see `results/qasper_dense_hybrid_retrieval_summary.csv`.

| Setting | Gold in top-10 pool | Gold in first 3 chunks |
|---------|----------------------|-------------------------|
| Dense (FAISS) | 94.5% | 73.5% |
| Hybrid (BM25 + RRF) | 95.5% | 75.5% |

**End-to-end RAG** (table below): LLM + rerank / prompts / truncation ablations.

| Experiment | Setting | Exact Match | Token F1 | Gold Hit |
|------------|---------|-------------|----------|----------|
| **Reranker** | No Rerank | 0.015 | 0.368 | 0.325 |
| | With Rerank | **0.035** | **0.396** | **0.345** |
| **Top-K** | k=1 | 0.025 | 0.311 | 0.255 |
| | k=3 | 0.015 | 0.368 | 0.325 |
| | k=5 | **0.030** | **0.386** | **0.340** |
| **Prompt** | Default | 0.015 | **0.368** | 0.325 |
| | Bullets | **0.050** | 0.367 | **0.330** |
| | Strict Cite | 0.015 | 0.354 | 0.285 |
| **Truncation**| Head | **0.030** | **0.314** | **0.250** |
| | Tail | 0.030 | 0.265 | 0.180 |
| | Middle | 0.025 | 0.252 | 0.195 |

**Key Findings**
1. **Reranking is Crucial for Long Docs:** Adding a Cross-Encoder reranker increased Token F1 from 0.368 to 0.396 and improved the gold hit rate by 2%.
2. **Context Volume vs. Noise:** Increasing the final context window to `k=5` yielded the best F1 scores (0.386). The model successfully navigated the extra context without hallucinating heavily.
3. **Prompt Engineering:** Asking the LLM to output in "bullets" slightly improved exact match and gold hit metrics, though F1 remained comparable.
4. **Context Truncation:** When hitting strict context limits, keeping the `head` (start) of the retrieved chunks vastly outperformed keeping the `tail` or `middle`.

**Limitations**
- **Exact Match (EM) is Harsh:** EM scores are very low (3-5%) because QASPER answers are often long aliases or phrases, and LLMs rarely generate the *exact* string without surrounding conversational text. Token F1 and Gold Hit are better signals here.
- **In-Memory Indexing:** The current Streamlit demo uses an in-memory FAISS index, meaning the index resets on server restarts. A production version would require a persistent Vector DB (e.g., Milvus, Pinecone).

Long papers (10k–30k+ chars); defaults e.g. `--chunk-size 384`, `--max-context-chars 8000`. **Rerank** helps; **k=5 > k=3 > k=1** on F1/gold hit; **bullets** best EM; **strict_cite** lowers gold hit; **head > tail > middle** under truncation. Figures: `python scripts/generate_charts.py` → `assets/qasper_*.png`. CSV: `results/qasper_rag_generation_results.csv`.

**Caveats:** EM is harsh on long aliases (prefer F1 / gold_hit); gold heuristics in `qasper_hf.py`; truncation rows use a **tight budget**—not comparable to non-truncation runs.

## Traces & errors

Qualitative QASPER success/failure writeup: `analysis/ERROR_ANALYSIS.md`. Export schema and field meanings: `analysis/TRACES_JSONL_FIELDS.md`. Failure **taxonomy** (retrieval vs ranking vs generation) is described there and in exported JSONL traces.

- `--skip-generation`: pool vs final labels only (no LLM cost).
- Real LLM: `prediction`, F1, `failure_bucket` for case studies.
- `--mock-generation`: stub only; do not interpret “generation” failures literally.

```bash
python tools/export_triviaqa_traces.py --out analysis/triviaqa_traces.jsonl --max-examples 100 --skip-generation
python tools/export_qasper_traces.py --out analysis/qasper_traces.jsonl --max-examples 60 --llm-backend ollama
python scripts/build_error_analysis_draft.py --in analysis/qasper_traces.jsonl --out analysis/error_analysis_draft.md
```

## Metrics & notes

Custom QA: **recall@k**. TREC: qrels IR metrics. RAG: **EM**, **token F1**, **gold_hit** (alias substring in output).

Models are swappable (`sentence-transformers`). TREC: use full corpus + qrels; `--max-docs` is for smoke tests only.
