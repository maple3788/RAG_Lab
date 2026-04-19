# Quickstart

This guide is for local development with the Streamlit app.

For production API serving (`FastAPI + Nginx + Milvus + Redis`), see [`api-deployment.md`](api-deployment.md).

## 1) Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Scanned or image-only PDFs need OCR libraries (`pypdfium2`, `rapidocr-onnxruntime`) pulled in by `requirements.txt`. On **Python 3.13**, use a `rapidocr-onnxruntime` release that publishes wheels for your platform (the README Notes section summarizes version constraints).

Recommended `.env` variables for local runs:
- `OLLAMA_BASE_URL`
- `MILVUS_URI`, `MILVUS_COLLECTION`
- `REDIS_URL`
- `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `MINIO_BUCKET`

## 2) Start Local Services (Optional)

```bash
docker compose up -d
```

This starts MinIO, Redis, Prometheus, and Grafana from `docker-compose.yml`.

## 3) Start Streamlit

```bash
streamlit run demo/app.py
```

Main app views:
- **Ingest**: upload docs and create retrieval artifacts
- **Query**: run retrieval + generation over a loaded job
- **Library**: browse and remove stored jobs
- **Benchmark**: leaderboard, failure analysis, and cost telemetry

## 4) Typical Usage Flow

1. Ingest a document.
2. Open Query and load the ingested `job_id`.
3. (Optional) Sync loaded chunks to Milvus.
4. Ask questions and inspect traces/latency.
5. Review run logs in Benchmark.

## 5) Verify the Setup

Check these endpoints:
- Streamlit: `http://localhost:8501`
- Streamlit metrics: `http://localhost:8000/metrics`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`

