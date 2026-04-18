# API and Deployment

This page describes the high-concurrency service path:
- **Ollama** for generation
- **Milvus** for vector retrieval
- **Redis** for caching
- **Nginx** for rate-limiting/reverse proxy
- **Prometheus + Grafana** for observability

## Prerequisites

- Ollama is running and the target model is available (default `llama3.2`).
- Milvus is reachable from Docker (the default Compose stack includes Milvus services).
- Redis is available (provided by Compose in this project).

## API server

Module: `src/api/server.py`

Endpoints:
- `POST /v1/rag/query`
- `POST /v1/rag/batch`
- `POST /v1/rag/load`
- `POST /v1/rag/session`
- `DELETE /v1/rag/session`
- `POST /v1/ingest`
- `GET /v1/jobs`
- `GET /healthz`
- `GET /readyz`
- `GET /metrics`

## Run with Docker Compose

Start the Milvus stack first (included in `docker-compose.yml`):

```bash
docker compose up -d milvus-etcd milvus-minio milvus-standalone
```

Then start API and gateway:

```bash
docker compose up -d redis rag-api nginx
```

Gateway URL:
- `http://localhost:8080/v1/rag/query`

Quick health checks:

```bash
docker compose ps
curl http://localhost:8080/healthz
```

### Optional monitoring services

```bash
docker compose up -d prometheus grafana
```

## Example request

```bash
curl -X POST "http://localhost:8080/v1/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question":"What is this document about?",
    "job_ids":["<job_id_a>", "<job_id_b>"],
    "retrieve_k":12,
    "final_k":4,
    "use_rerank":true,
    "retrieval_mode":"hybrid"
  }'
```

Session-based target binding (avoid sending `job_id` each call):

```bash
curl -X POST "http://localhost:8080/v1/rag/session" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo-session","job_id":"<your_job_id>"}'

curl -X POST "http://localhost:8080/v1/rag/query" \
  -H "Content-Type: application/json" \
  -H "X-RAG-Session-Id: demo-session" \
  -d '{"question":"Summarize key findings."}'
```

## Concurrency and cache controls

Environment variables:
- `RAG_MAX_GENERATION_CONCURRENCY`
- `RAG_L1_CACHE_MAX_ITEMS`
- `RAG_L1_CACHE_TTL_SEC`
- `RAG_SEMANTIC_CACHE_MAX_ENTRIES`
- `RAG_SEMANTIC_CACHE_TTL_SEC`

Model and retrieval backends:
- `OLLAMA_BASE_URL`, `OLLAMA_MODEL`
- `MILVUS_URI`, `MILVUS_TOKEN`, `MILVUS_COLLECTION`
- `REDIS_URL`
- `RAG_DEFAULT_JOB_ID` (optional fallback when no explicit `job_id`/`job_ids`/session)
- `RAG_HYBRID_MAX_CHUNKS` (cap lexical corpus size for API hybrid retrieval)

## Monitoring stack

- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`
- Dashboard: **RAG Lab · Prometheus**

Key API metrics:
- `rag_api_requests_total`
- `rag_api_request_seconds`
- `rag_api_cache_hits_total`
- `rag_retrieval_seconds`
- `rag_generation_seconds`

