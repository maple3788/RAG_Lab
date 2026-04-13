from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import JSONResponse, Response

from src.context_truncation import truncate_context
from src.embedder import EmbeddingModel, load_embedding_model, prepare_query
from src.generator import OllamaGenerator
from src.metrics import approx_token_count
from src.prompts import PROMPT_TEMPLATES, format_rag_prompt
from src.rag_generation import passages_to_context
from src.rag_pipeline import record_rag_generation_latency, record_rag_retrieval_latency
from src.reranker import Reranker, load_reranker
from src.storage.milvus_store import MilvusChunkStore
from src.storage.redis_semantic_cache import RedisSemanticCache


DEFAULT_EMBED = os.environ.get("RAG_EMBED_MODEL", "BAAI/bge-base-en-v1.5")
DEFAULT_RERANK = os.environ.get("RAG_RERANK_MODEL", "BAAI/bge-reranker-base")
RAG_SYSTEM_PROMPT = (
    "You are a grounded RAG assistant. Use only provided context. "
    "If evidence is insufficient, say unknown."
)

REQ_TOTAL = Counter("rag_api_requests_total", "RAG API request count", ["endpoint", "status"])
REQ_LAT = Histogram(
    "rag_api_request_seconds",
    "RAG API request latency",
    ["endpoint"],
    buckets=(0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60),
)
CACHE_HIT = Counter("rag_api_cache_hits_total", "RAG API cache hits", ["layer"])


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    job_id: str = Field(min_length=1)
    retrieve_k: int = Field(default=12, ge=1, le=64)
    final_k: int = Field(default=4, ge=1, le=16)
    use_rerank: bool = True
    use_semantic_cache: bool = True
    semantic_cache_threshold: float = Field(default=0.93, ge=0.5, le=0.999)
    max_context_chars: int = Field(default=6000, ge=200, le=50000)
    truncation: str = Field(default="head", pattern="^(head|tail|middle)$")
    prompt_template: str = Field(default="default")
    include_debug: bool = False


class QueryResponse(BaseModel):
    answer: str
    latency_ms: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    token_count: int
    cache_hit: str
    retrieved_chunks: list[str] = []


@dataclass
class CacheItem:
    value: QueryResponse
    expires_at: float


class TTLCache:
    def __init__(self, max_items: int, ttl_seconds: int):
        self.max_items = max(1, max_items)
        self.ttl_seconds = max(1, ttl_seconds)
        self._store: OrderedDict[str, CacheItem] = OrderedDict()

    def get(self, key: str) -> Optional[QueryResponse]:
        now = time.time()
        item = self._store.get(key)
        if item is None:
            return None
        if item.expires_at < now:
            self._store.pop(key, None)
            return None
        self._store.move_to_end(key, last=True)
        return item.value

    def set(self, key: str, value: QueryResponse) -> None:
        self._store[key] = CacheItem(value=value, expires_at=time.time() + self.ttl_seconds)
        self._store.move_to_end(key, last=True)
        while len(self._store) > self.max_items:
            self._store.popitem(last=False)


app = FastAPI(title="RAG Lab API", version="0.1.0")
_embedder: Optional[EmbeddingModel] = None
_reranker: Optional[Reranker] = None
_milvus: Optional[MilvusChunkStore] = None
_gen: Optional[OllamaGenerator] = None
_l1_cache = TTLCache(
    max_items=int(os.environ.get("RAG_L1_CACHE_MAX_ITEMS", "512")),
    ttl_seconds=int(os.environ.get("RAG_L1_CACHE_TTL_SEC", "120")),
)
_key_locks: dict[str, asyncio.Lock] = {}
_generation_sem = asyncio.Semaphore(int(os.environ.get("RAG_MAX_GENERATION_CONCURRENCY", "8")))


def _embed() -> EmbeddingModel:
    global _embedder
    if _embedder is None:
        _embedder = load_embedding_model(DEFAULT_EMBED, normalize=True)
    return _embedder


def _rr() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = load_reranker(DEFAULT_RERANK)
    return _reranker


def _milvus_store() -> MilvusChunkStore:
    global _milvus
    if _milvus is None:
        _milvus = MilvusChunkStore()
    return _milvus


def _generator() -> OllamaGenerator:
    global _gen
    if _gen is None:
        _gen = OllamaGenerator(model=os.environ.get("OLLAMA_MODEL", "llama3.2"), max_tokens=512)
    return _gen


def _cache_key(req: QueryRequest) -> str:
    payload = {
        "q": req.question.strip(),
        "job_id": req.job_id.strip(),
        "retrieve_k": req.retrieve_k,
        "final_k": req.final_k,
        "use_rerank": req.use_rerank,
        "max_context_chars": req.max_context_chars,
        "truncation": req.truncation,
        "prompt_template": req.prompt_template,
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _retrieve(req: QueryRequest) -> tuple[list[dict[str, Any]], float]:
    t0 = time.perf_counter()
    rows = _milvus_store().search_job_chunks(
        job_id=req.job_id.strip(),
        query=req.question.strip(),
        embedder=_embed(),
        top_k=req.retrieve_k,
    )
    if req.use_rerank and rows:
        texts = [r["text"] for r in rows]
        reranked, _ = _rr().rerank(req.question.strip(), texts, top_k=min(req.final_k, len(texts)))
        rank_by_text = {t: i for i, t in enumerate(reranked)}
        rows = sorted(
            (r for r in rows if r["text"] in rank_by_text),
            key=lambda r: rank_by_text[r["text"]],
        )
        rows = rows[: req.final_k]
    else:
        rows = rows[: req.final_k]
    dt = time.perf_counter() - t0
    record_rag_retrieval_latency(dt)
    return rows, dt


def _build_prompt(req: QueryRequest, passages: list[str]) -> str:
    context = truncate_context(passages_to_context(passages), req.max_context_chars, req.truncation)  # type: ignore[arg-type]
    template = PROMPT_TEMPLATES.get(req.prompt_template, PROMPT_TEMPLATES["default"])
    user = format_rag_prompt(template, context=context, question=req.question.strip(), history="None")
    return f"SYSTEM_PROMPT:\n{RAG_SYSTEM_PROMPT}\n\nUSER_PROMPT:\n{user}"


def _generate(prompt: str) -> tuple[str, float]:
    t0 = time.perf_counter()
    out = _generator().generate(prompt)
    dt = time.perf_counter() - t0
    record_rag_generation_latency(dt)
    return out, dt


async def _run_query(req: QueryRequest) -> QueryResponse:
    key = _cache_key(req)
    hit = _l1_cache.get(key)
    if hit is not None:
        CACHE_HIT.labels(layer="l1").inc()
        return hit.model_copy(update={"cache_hit": "l1"})

    lock = _key_locks.setdefault(key, asyncio.Lock())
    async with lock:
        hit2 = _l1_cache.get(key)
        if hit2 is not None:
            CACHE_HIT.labels(layer="l1").inc()
            return hit2.model_copy(update={"cache_hit": "l1"})

        emb = _embed()
        semantic = RedisSemanticCache(namespace=req.job_id.strip()) if req.use_semantic_cache else None
        sem_answer: Optional[str] = None
        if semantic is not None:
            q_emb = emb.encode([prepare_query(emb.name, req.question.strip())])[0]
            sem_answer = semantic.lookup(q_emb, threshold=req.semantic_cache_threshold)
            if sem_answer:
                CACHE_HIT.labels(layer="semantic").inc()
                resp = QueryResponse(
                    answer=sem_answer,
                    latency_ms=0.0,
                    retrieval_latency_ms=0.0,
                    generation_latency_ms=0.0,
                    token_count=max(1, len(sem_answer) // 4),
                    cache_hit="semantic",
                    retrieved_chunks=[],
                )
                _l1_cache.set(key, resp)
                return resp

        rows, retrieval_s = await asyncio.to_thread(_retrieve, req)
        passages = [r["text"] for r in rows]
        prompt = _build_prompt(req, passages)
        async with _generation_sem:
            answer, gen_s = await asyncio.to_thread(_generate, prompt)

        total_ms = (retrieval_s + gen_s) * 1000.0
        token_count = approx_token_count(prompt, answer)
        resp = QueryResponse(
            answer=answer,
            latency_ms=round(total_ms, 2),
            retrieval_latency_ms=round(retrieval_s * 1000.0, 2),
            generation_latency_ms=round(gen_s * 1000.0, 2),
            token_count=token_count,
            cache_hit="none",
            retrieved_chunks=passages if req.include_debug else [],
        )
        _l1_cache.set(key, resp)

        if semantic is not None:
            q_emb = emb.encode([prepare_query(emb.name, req.question.strip())])[0]
            semantic.write(
                question=req.question.strip(),
                answer=answer,
                query_embedding=q_emb,
                max_entries=int(os.environ.get("RAG_SEMANTIC_CACHE_MAX_ENTRIES", "512")),
                ttl_seconds=int(os.environ.get("RAG_SEMANTIC_CACHE_TTL_SEC", "86400")),
            )
        return resp


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
async def readyz() -> dict[str, str]:
    try:
        _milvus_store()
        return {"status": "ready"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"milvus unavailable: {e}")


@app.get("/metrics")
async def metrics() -> Response:
    return Response(content=generate_latest(), media_type="text/plain; version=0.0.4")


@app.post("/v1/rag/query", response_model=QueryResponse)
async def query_rag(req: QueryRequest) -> QueryResponse:
    t0 = time.perf_counter()
    status = "ok"
    try:
        result = await _run_query(req)
        return result
    except Exception as e:
        status = "error"
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        REQ_TOTAL.labels(endpoint="/v1/rag/query", status=status).inc()
        REQ_LAT.labels(endpoint="/v1/rag/query").observe(time.perf_counter() - t0)


class BatchRequest(BaseModel):
    queries: list[QueryRequest]


@app.post("/v1/rag/batch")
async def batch_rag(req: BatchRequest) -> JSONResponse:
    if len(req.queries) > 64:
        raise HTTPException(status_code=400, detail="too many queries; max 64")
    t0 = time.perf_counter()
    status = "ok"
    try:
        results = await asyncio.gather(*(_run_query(q) for q in req.queries))
        return JSONResponse({"results": [r.model_dump() for r in results]})
    except Exception as e:
        status = "error"
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        REQ_TOTAL.labels(endpoint="/v1/rag/batch", status=status).inc()
        REQ_LAT.labels(endpoint="/v1/rag/batch").observe(time.perf_counter() - t0)
