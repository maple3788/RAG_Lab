# RAG-Lab

RAG-Lab is a modular retrieval-augmented generation system for experimentation and production-oriented benchmarking.

It includes:
- document ingest (`PDF/TXT/MD`) -> chunking -> embedding -> FAISS artifact persistence
- interactive Streamlit app for ingest/query/library/benchmark
- experiment runners for retrieval and generation quality
- production API path (`FastAPI + Nginx + Redis cache + Milvus + Ollama`)
- observability stack (`Prometheus + Grafana`)
- advanced retrieval options: multi-job querying, hybrid dense+BM25 retrieval, and query expansion (multi-query / HyDE)

## Start Here

- Documentation index: [`docs/README.md`](docs/README.md)
- Environment setup and local app usage: [`docs/quickstart.md`](docs/quickstart.md)
- Architecture and design principles: [`docs/architecture.md`](docs/architecture.md)
- API + deployment (high-concurrency path): [`docs/api-deployment.md`](docs/api-deployment.md)
- Experiments and evaluation scripts: [`docs/experiments.md`](docs/experiments.md)
- Key benchmark snapshots and interpretation: [`docs/results.md`](docs/results.md)
- Traces, error analysis, and ops notes: [`docs/operations.md`](docs/operations.md)
- Retrieval expansion references: [`docs/knowledge/advanced-rag-techniques-summary.md`](docs/knowledge/advanced-rag-techniques-summary.md)

## Minimal Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Run Streamlit:

```bash
streamlit run demo/app.py
```

Run FastAPI (API + built-in tester UI):

```bash
uvicorn src.api.server:app --reload --port 8000
```

Then open:
- `http://localhost:8000/docs` (Swagger UI)
- `http://localhost:8000/v1/rag/ui` (lightweight custom API tester)

Optional local services (MinIO + Redis + monitoring):

```bash
docker compose up -d
```

## Repository Map

```text
rag-lab
├── demo/app.py
├── src/
│   ├── api/           # FastAPI service
│   ├── config/        # Pydantic + YAML pipeline config
│   ├── datasets/      # QA / BEIR / Hugging Face loaders
│   ├── eval/          # metrics, RAGAS UI, experiment tracking
│   ├── ingestion/     # extract, chunk, ingest pipeline
│   ├── llm/           # embedder, reranker, generators, prompts
│   ├── rag/           # retrieval + generation orchestration
│   ├── retrieval/     # FAISS/Milvus, hybrid, metadata filters
│   └── storage/       # MinIO, Redis, Milvus clients
├── experiments/
├── docs/
├── tools/
├── scripts/
├── analysis/
├── assets/
└── results/
```

## Notes

- Scanned or image-only PDFs need **OCR**: `pypdfium2` and `rapidocr-onnxruntime` from `requirements.txt`. On **Python 3.13**, install `rapidocr-onnxruntime` **1.2.x** (1.4+ does not publish wheels for 3.13 yet); `pip install -r requirements.txt` picks a compatible version.
- Keep secrets in `.env` only. Do not commit it.
- For Gemini region/API constraints, see [`docs/gemini-region-restriction.md`](docs/gemini-region-restriction.md).
