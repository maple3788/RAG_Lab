from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Literal, Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import JSONResponse, Response

from src.context_truncation import truncate_context
from src.document_ingest_pipeline import IngestPipelineConfig, load_ingest_from_minio, run_document_ingest
from src.embedder import EmbeddingModel, load_embedding_model, prepare_query
from src.hybrid_retrieval import fuse_milvus_dense_order_with_bm25
from src.generator import OllamaGenerator
from src.metrics import approx_token_count
from src.prompts import PROMPT_TEMPLATES, format_rag_prompt
from src.rag_generation import passages_to_context
from src.rag_pipeline import record_rag_generation_latency, record_rag_retrieval_latency
from src.reranker import Reranker, load_reranker
from src.storage.milvus_store import MilvusChunkStore, MilvusSearchConfig
from src.storage.minio_artifacts import MinioArtifactStore
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
    #: Search across multiple ingest jobs (multiple docs). If set (non-empty), overrides singular ``job_id`` for resolution.
    job_ids: Optional[list[str]] = Field(default=None, max_length=32)
    #: If omitted, use ``rag_session_id`` / ``X-RAG-Session-Id`` (see ``POST /v1/rag/session``) or ``RAG_DEFAULT_JOB_ID``.
    job_id: Optional[str] = Field(default=None, max_length=256)
    #: Client-generated id; server maps it to ``job_id`` via ``POST /v1/rag/session`` or ``POST /v1/rag/load`` (optional).
    rag_session_id: Optional[str] = Field(default=None, max_length=128)
    #: Must match ingest metadata ``milvus_collection`` when using the pipeline (e.g. ``rag_chunks_…``). If unset, auto-resolve from metadata by ``job_id``.
    milvus_collection: Optional[str] = Field(default=None, max_length=256)
    #: Align with the index used for that collection (affects search params).
    milvus_index_type: str = Field(default="AUTOINDEX")
    #: Optional search metric override.
    milvus_metric_type: Optional[str] = Field(default=None)
    #: IVF query breadth; used when index is IVF_FLAT.
    milvus_ivf_nprobe: int = Field(default=32, ge=1, le=256)
    #: HNSW query breadth; used for HNSW/AUTOINDEX.
    milvus_hnsw_ef: int = Field(default=64, ge=8, le=512)
    #: ``dense`` = Milvus ANN only; ``hybrid`` = RRF(BM25 over MinIO chunks + Milvus ANN order).
    retrieval_mode: Literal["dense", "hybrid"] = "dense"
    #: Candidates per retriever list before RRF (hybrid only).
    fusion_list_k: int = Field(default=30, ge=5, le=100)
    #: RRF smoothing constant (hybrid only).
    rrf_k: int = Field(default=60, ge=10, le=200)
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


class QueryErrorResponse(BaseModel):
    error: str
    stage: str
    latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    cache_hit: str = "none"
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
_minio: Optional[MinioArtifactStore] = None
_gen: Optional[OllamaGenerator] = None
_l1_cache = TTLCache(
    max_items=int(os.environ.get("RAG_L1_CACHE_MAX_ITEMS", "512")),
    ttl_seconds=int(os.environ.get("RAG_L1_CACHE_TTL_SEC", "120")),
)
_key_locks: dict[str, asyncio.Lock] = {}
_generation_sem = asyncio.Semaphore(int(os.environ.get("RAG_MAX_GENERATION_CONCURRENCY", "8")))


class QueryStageError(Exception):
    def __init__(
        self,
        stage: str,
        message: str,
        *,
        status_code: int = 500,
        retrieval_latency_ms: float = 0.0,
        generation_latency_ms: float = 0.0,
        cache_hit: str = "none",
        retrieved_chunks: Optional[list[str]] = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.status_code = status_code
        self.retrieval_latency_ms = retrieval_latency_ms
        self.generation_latency_ms = generation_latency_ms
        self.cache_hit = cache_hit
        self.retrieved_chunks = retrieved_chunks or []

    def to_dict(self) -> dict[str, Any]:
        total_ms = self.retrieval_latency_ms + self.generation_latency_ms
        return {
            "error": str(self),
            "stage": self.stage,
            "latency_ms": round(total_ms, 2),
            "retrieval_latency_ms": round(self.retrieval_latency_ms, 2),
            "generation_latency_ms": round(self.generation_latency_ms, 2),
            "cache_hit": self.cache_hit,
            "retrieved_chunks": self.retrieved_chunks,
        }


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


def _minio_store() -> MinioArtifactStore:
    global _minio
    if _minio is None:
        _minio = MinioArtifactStore()
    return _minio


def _generator() -> OllamaGenerator:
    global _gen
    if _gen is None:
        _gen = OllamaGenerator(model=os.environ.get("OLLAMA_MODEL", "llama3.2"), max_tokens=512)
    return _gen


def _normalized_milvus_collection(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    s = name.strip()
    return s if s else None


_redis_cli: Any = None


def _api_redis():
    global _redis_cli
    if _redis_cli is None:
        import redis

        url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        _redis_cli = redis.from_url(url, decode_responses=True)
    return _redis_cli


def _session_ttl_sec() -> int:
    return int(os.environ.get("RAG_API_SESSION_TTL_SEC", "86400"))


def _session_key(session_id: str) -> str:
    return f"rag:api:active-job:{session_id.strip()}"


def _session_set_job(session_id: str, job_id: str) -> None:
    _api_redis().setex(_session_key(session_id), _session_ttl_sec(), job_id.strip())


def _session_get_job(session_id: str) -> Optional[str]:
    try:
        return _api_redis().get(_session_key(session_id))
    except Exception:
        return None


def _session_clear(session_id: str) -> None:
    try:
        _api_redis().delete(_session_key(session_id))
    except Exception:
        pass


def _normalize_job_id_list(raw: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for j in raw:
        s = (j or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _resolve_job_ids_for_query(req: QueryRequest, x_rag_session_id: Optional[str]) -> list[str]:
    if req.job_ids is not None:
        ids = _normalize_job_id_list(list(req.job_ids))
        if ids:
            return ids
    if req.job_id and str(req.job_id).strip():
        return [str(req.job_id).strip()]
    sid = (req.rag_session_id or "").strip() or (x_rag_session_id or "").strip()
    if sid:
        jid = _session_get_job(sid)
        if jid:
            return [jid.strip()]
        raise HTTPException(
            status_code=400,
            detail="no active job bound to this session; call POST /v1/rag/session or POST /v1/rag/load?session_id=…",
        )
    default = (os.environ.get("RAG_DEFAULT_JOB_ID") or "").strip()
    if default:
        return [default]
    raise HTTPException(
        status_code=400,
        detail=(
            "job_id or job_ids required, or bind a session (rag_session_id / X-RAG-Session-Id) via POST /v1/rag/session, "
            "or set RAG_DEFAULT_JOB_ID"
        ),
    )


def _resolve_query_target(req: QueryRequest) -> QueryRequest:
    # Explicit request params always win.
    if _normalized_milvus_collection(req.milvus_collection) and (req.milvus_index_type or "").strip():
        return req
    try:
        meta = _minio_store().get_job_metadata(req.job_id.strip()) or {}
    except Exception:
        meta = {}
    resolved_collection = _normalized_milvus_collection(req.milvus_collection) or _normalized_milvus_collection(
        meta.get("milvus_collection")
    )
    resolved_index_type = (req.milvus_index_type or "").strip() or str(meta.get("milvus_index_type", "")).strip()
    return req.model_copy(
        update={
            "milvus_collection": resolved_collection,
            "milvus_index_type": resolved_index_type or "AUTOINDEX",
            "milvus_metric_type": req.milvus_metric_type or str(meta.get("milvus_metric_type", "")).strip() or None,
        }
    )


def _cache_key(req: QueryRequest) -> str:
    if req.job_ids:
        job_key: Any = sorted(_normalize_job_id_list(list(req.job_ids)))
    elif req.job_id and str(req.job_id).strip():
        job_key = [str(req.job_id).strip()]
    else:
        job_key = []
    payload = {
        "q": req.question.strip(),
        "job_ids": job_key,
        "milvus_collection": _normalized_milvus_collection(req.milvus_collection),
        "milvus_index_type": (req.milvus_index_type or "AUTOINDEX").strip(),
        "retrieve_k": req.retrieve_k,
        "final_k": req.final_k,
        "use_rerank": req.use_rerank,
        "max_context_chars": req.max_context_chars,
        "truncation": req.truncation,
        "prompt_template": req.prompt_template,
        "retrieval_mode": getattr(req, "retrieval_mode", "dense"),
        "fusion_list_k": int(req.fusion_list_k),
        "rrf_k": int(req.rrf_k),
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _hybrid_lexical_corpus(job_ids: list[str]) -> tuple[list[str], list[str], list[int], dict[tuple[str, int], int]]:
    """MinIO chunk texts aligned with Milvus rows (job_id + chunk_index)."""
    corpus: list[str] = []
    job_at: list[str] = []
    cidx_at: list[int] = []
    key_to_gi: dict[tuple[str, int], int] = {}
    max_c = int(os.environ.get("RAG_HYBRID_MAX_CHUNKS", "8000"))
    for jid in job_ids:
        j = jid.strip()
        texts, _ = load_ingest_from_minio(j)
        for i, t in enumerate(texts):
            if len(corpus) >= max_c:
                raise ValueError(
                    f"hybrid lexical corpus exceeds RAG_HYBRID_MAX_CHUNKS ({max_c})"
                )
            gi = len(corpus)
            corpus.append(t)
            job_at.append(j)
            cidx_at.append(i)
            key_to_gi[(j, i)] = gi
    return corpus, job_at, cidx_at, key_to_gi


def _milvus_dense_candidates(req: QueryRequest, job_ids: list[str], dense_cap: int) -> list[dict[str, Any]]:
    """Milvus ANN hits (no rerank), up to ``dense_cap`` rows in relevance order."""
    search_cfg = MilvusSearchConfig(
        metric_type=(req.milvus_metric_type or "COSINE").strip(),
        ivf_nprobe=int(req.milvus_ivf_nprobe),
        hnsw_ef=int(req.milvus_hnsw_ef),
    )
    store = _milvus_store()
    if len(job_ids) == 1:
        return store.search_job_chunks(
            job_id=job_ids[0].strip(),
            query=req.question.strip(),
            embedder=_embed(),
            top_k=dense_cap,
            index_type=(req.milvus_index_type or "AUTOINDEX").strip(),
            search_config=search_cfg,
            collection_name=_normalized_milvus_collection(req.milvus_collection),
        )
    groups: dict[str, list[str]] = {}
    metas: dict[str, dict[str, Any]] = {}
    for jid in job_ids:
        j = jid.strip()
        meta = _minio_store().get_job_metadata(j) or {}
        coll = _normalized_milvus_collection(meta.get("milvus_collection"))
        if not coll:
            raise ValueError(f"missing milvus_collection for job_id={j}")
        groups.setdefault(coll, []).append(j)
        metas[j] = meta
    all_rows: list[dict[str, Any]] = []
    for coll, jids_in in groups.items():
        m0 = metas.get(jids_in[0], {})
        idx_type = (req.milvus_index_type or "").strip() or str(m0.get("milvus_index_type", "AUTOINDEX"))
        mtype = (req.milvus_metric_type or "").strip() or str(m0.get("milvus_metric_type", "COSINE"))
        sc = MilvusSearchConfig(
            metric_type=mtype,
            ivf_nprobe=int(req.milvus_ivf_nprobe),
            hnsw_ef=int(req.milvus_hnsw_ef),
        )
        part = store.search_multi_job_chunks(
            job_ids=jids_in,
            query=req.question.strip(),
            embedder=_embed(),
            top_k=dense_cap,
            index_type=str(idx_type or "AUTOINDEX").strip(),
            search_config=sc,
            collection_name=coll,
        )
        all_rows.extend(part)
    all_rows.sort(key=lambda r: float(r.get("faiss_score", 0.0)))
    return all_rows[:dense_cap]


def _rows_rerank_final(req: QueryRequest, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if req.use_rerank and rows:
        texts = [str(r["text"]) for r in rows]
        reranked, _ = _rr().rerank(req.question.strip(), texts, top_k=min(req.final_k, len(texts)))
        rank_by_text = {t: i for i, t in enumerate(reranked)}
        rows = sorted(
            (r for r in rows if r["text"] in rank_by_text),
            key=lambda r: rank_by_text[r["text"]],
        )
        return rows[: req.final_k]
    return rows[: req.final_k]


def _retrieve_hybrid_milvus(req: QueryRequest, job_ids: list[str]) -> tuple[list[dict[str, Any]], float]:
    """BM25 (lexical over MinIO chunks) + Milvus dense order, merged via RRF; then cross-encoder to ``final_k``."""
    t0 = time.perf_counter()
    try:
        corpus, job_at, cidx_at, key_to_gi = _hybrid_lexical_corpus(job_ids)
    except ValueError as e:
        raise QueryStageError("retrieval", str(e), status_code=400) from e
    if not corpus:
        raise QueryStageError(
            "retrieval",
            "empty lexical corpus — ingest jobs need MinIO chunks.json",
            status_code=400,
        )
    dense_cap = min(max(int(req.retrieve_k), int(req.fusion_list_k)), 64)
    dense_rows = _milvus_dense_candidates(req, job_ids, dense_cap)
    ordered_gi: list[int] = []
    seen_gi: set[int] = set()
    for row in dense_rows:
        jid = str(row.get("job_id", "")).strip()
        ci = int(row.get("chunk_index", -1))
        if not jid or ci < 0:
            continue
        gi = key_to_gi.get((jid, ci))
        if gi is None or gi in seen_gi:
            continue
        seen_gi.add(gi)
        ordered_gi.append(gi)
    fused_idx = fuse_milvus_dense_order_with_bm25(
        req.question.strip(),
        corpus_chunks=corpus,
        dense_global_indices_ordered=ordered_gi,
        retrieve_k=int(req.retrieve_k),
        fusion_list_k=int(req.fusion_list_k),
        rrf_k=int(req.rrf_k),
    )
    rows: list[dict[str, Any]] = []
    for rank, gi in enumerate(fused_idx, start=1):
        rows.append(
            {
                "query_used": req.question.strip(),
                "faiss_rank": rank,
                "chunk_index": cidx_at[gi],
                "faiss_score": 1.0 / float(rank),
                "text": corpus[gi],
                "job_id": job_at[gi],
                "retrieval": "hybrid_rrf",
            }
        )
    rows = _rows_rerank_final(req, rows)
    dt = time.perf_counter() - t0
    record_rag_retrieval_latency(dt)
    return rows, dt


def _retrieve(req: QueryRequest) -> tuple[list[dict[str, Any]], float]:
    t0 = time.perf_counter()
    search_cfg = MilvusSearchConfig(
        metric_type=(req.milvus_metric_type or "COSINE").strip(),
        ivf_nprobe=int(req.milvus_ivf_nprobe),
        hnsw_ef=int(req.milvus_hnsw_ef),
    )
    rows = _milvus_store().search_job_chunks(
        job_id=req.job_id.strip(),
        query=req.question.strip(),
        embedder=_embed(),
        top_k=req.retrieve_k,
        index_type=(req.milvus_index_type or "AUTOINDEX").strip(),
        search_config=search_cfg,
        collection_name=_normalized_milvus_collection(req.milvus_collection),
    )
    rows = _rows_rerank_final(req, rows)
    dt = time.perf_counter() - t0
    record_rag_retrieval_latency(dt)
    return rows, dt


def _retrieve_multi(req: QueryRequest, job_ids: list[str]) -> tuple[list[dict[str, Any]], float]:
    """Retrieve across multiple jobs, possibly spanning multiple Milvus collections."""
    t0 = time.perf_counter()
    store = _milvus_store()
    groups: dict[str, list[str]] = {}
    metas: dict[str, dict[str, Any]] = {}
    for jid in job_ids:
        j = jid.strip()
        meta = _minio_store().get_job_metadata(j) or {}
        coll = _normalized_milvus_collection(meta.get("milvus_collection"))
        if not coll:
            raise ValueError(f"missing milvus_collection in MinIO metadata for job_id={j}")
        groups.setdefault(coll, []).append(j)
        metas[j] = meta
    all_rows: list[dict[str, Any]] = []
    for coll, jids_in in groups.items():
        m0 = metas.get(jids_in[0], {})
        idx_type = (req.milvus_index_type or "").strip() or str(m0.get("milvus_index_type", "AUTOINDEX"))
        mtype = (req.milvus_metric_type or "").strip() or str(m0.get("milvus_metric_type", "COSINE"))
        sc = MilvusSearchConfig(
            metric_type=mtype,
            ivf_nprobe=int(req.milvus_ivf_nprobe),
            hnsw_ef=int(req.milvus_hnsw_ef),
        )
        part = store.search_multi_job_chunks(
            job_ids=jids_in,
            query=req.question.strip(),
            embedder=_embed(),
            top_k=req.retrieve_k,
            index_type=str(idx_type or "AUTOINDEX").strip(),
            search_config=sc,
            collection_name=coll,
        )
        all_rows.extend(part)
    all_rows.sort(key=lambda r: float(r.get("faiss_score", 0.0)))
    all_rows = all_rows[: req.retrieve_k]
    all_rows = _rows_rerank_final(req, all_rows)
    dt = time.perf_counter() - t0
    record_rag_retrieval_latency(dt)
    return all_rows, dt


def _passages_for_prompt(rows: list[dict[str, Any]]) -> list[str]:
    job_ids_present = {str(r.get("job_id", "")) for r in rows if r.get("job_id")}
    if len(job_ids_present) <= 1:
        return [str(r["text"]) for r in rows]
    out: list[str] = []
    for r in rows:
        jid = str(r.get("job_id", "")).strip()
        t = str(r["text"])
        out.append(f"[doc:{jid}] {t}" if jid else t)
    return out


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
    jids = list(req.job_ids) if req.job_ids else []
    if not jids and req.job_id and str(req.job_id).strip():
        jids = [str(req.job_id).strip()]
    if not jids:
        raise QueryStageError("query", "missing job_ids / job_id", status_code=400)
    if len(jids) == 1:
        req_eff = _resolve_query_target(req.model_copy(update={"job_id": jids[0], "job_ids": jids}))
    else:
        req_eff = req.model_copy(update={"job_ids": jids, "job_id": jids[0] if jids else req.job_id})
    key = _cache_key(req_eff)
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
        multi = len(jids) > 1
        hybrid = getattr(req_eff, "retrieval_mode", "dense") == "hybrid"
        try:
            if multi or hybrid:
                semantic = None
            elif not req_eff.use_semantic_cache:
                semantic = None
            else:
                semantic = RedisSemanticCache(namespace=req_eff.job_id.strip())
        except Exception as e:
            raise QueryStageError("semantic_cache_init", f"semantic cache init failed: {e}") from e
        sem_answer: Optional[str] = None
        q_emb: Optional[Any] = None
        if semantic is not None:
            try:
                q_emb = emb.encode([prepare_query(emb.name, req_eff.question.strip())])[0]
                sem_answer = semantic.lookup(q_emb, threshold=req_eff.semantic_cache_threshold)
            except Exception as e:
                raise QueryStageError("semantic_cache_lookup", f"semantic cache lookup failed: {e}") from e
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

        try:
            if hybrid:
                rows, retrieval_s = await asyncio.to_thread(_retrieve_hybrid_milvus, req_eff, jids)
            elif len(jids) <= 1:
                rows, retrieval_s = await asyncio.to_thread(_retrieve, req_eff)
            else:
                rows, retrieval_s = await asyncio.to_thread(_retrieve_multi, req_eff, jids)
        except Exception as e:
            raise QueryStageError("retrieval", f"retrieval failed: {e}") from e
        passages = _passages_for_prompt(rows)
        try:
            prompt = _build_prompt(req_eff, passages)
        except Exception as e:
            raise QueryStageError(
                "prompt_build",
                f"prompt build failed: {e}",
                retrieval_latency_ms=retrieval_s * 1000.0,
                retrieved_chunks=(passages if req_eff.include_debug else []),
            ) from e
        async with _generation_sem:
            try:
                answer, gen_s = await asyncio.to_thread(_generate, prompt)
            except Exception as e:
                raise QueryStageError(
                    "generation",
                    f"generation failed: {e}",
                    retrieval_latency_ms=retrieval_s * 1000.0,
                    retrieved_chunks=passages if req_eff.include_debug else [],
                ) from e

        total_ms = (retrieval_s + gen_s) * 1000.0
        token_count = approx_token_count(prompt, answer)
        resp = QueryResponse(
            answer=answer,
            latency_ms=round(total_ms, 2),
            retrieval_latency_ms=round(retrieval_s * 1000.0, 2),
            generation_latency_ms=round(gen_s * 1000.0, 2),
            token_count=token_count,
            cache_hit="none",
            retrieved_chunks=passages if req_eff.include_debug else [],
        )
        _l1_cache.set(key, resp)

        if semantic is not None:
            try:
                if q_emb is None:
                    q_emb = emb.encode([prepare_query(emb.name, req_eff.question.strip())])[0]
                semantic.write(
                    question=req_eff.question.strip(),
                    answer=answer,
                    query_embedding=q_emb,
                    max_entries=int(os.environ.get("RAG_SEMANTIC_CACHE_MAX_ENTRIES", "512")),
                    ttl_seconds=int(os.environ.get("RAG_SEMANTIC_CACHE_TTL_SEC", "86400")),
                )
            except Exception as e:
                raise QueryStageError(
                    "semantic_cache_write",
                    f"semantic cache write failed: {e}",
                    retrieval_latency_ms=round(retrieval_s * 1000.0, 2),
                    generation_latency_ms=round(gen_s * 1000.0, 2),
                    cache_hit="none",
                    retrieved_chunks=passages if req_eff.include_debug else [],
                ) from e
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
async def query_rag(
    req: QueryRequest,
    x_rag_session_id: Optional[str] = Header(default=None, alias="X-RAG-Session-Id"),
) -> QueryResponse:
    t0 = time.perf_counter()
    status = "ok"
    try:
        job_ids = _resolve_job_ids_for_query(req, x_rag_session_id)
        result = await _run_query(req.model_copy(update={"job_ids": job_ids, "job_id": job_ids[0]}))
        return result
    except QueryStageError as e:
        status = "error"
        raise HTTPException(status_code=e.status_code, detail=e.to_dict())
    except Exception as e:
        status = "error"
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        REQ_TOTAL.labels(endpoint="/v1/rag/query", status=status).inc()
        REQ_LAT.labels(endpoint="/v1/rag/query").observe(time.perf_counter() - t0)


class BatchRequest(BaseModel):
    queries: list[QueryRequest]


class LoadJobRequest(BaseModel):
    job_id: str = Field(min_length=1)
    release_others: bool = False
    #: If set, bind this session id to ``job_id`` in Redis (same as ``POST /v1/rag/session``).
    session_id: Optional[str] = Field(default=None, max_length=128)


class SetSessionJobRequest(BaseModel):
    session_id: str = Field(min_length=4, max_length=128)
    job_id: str = Field(min_length=1, max_length=256)


class LoadJobResponse(BaseModel):
    job_id: str
    loaded_job_id: str
    milvus_collection: Optional[str] = None
    milvus_index_type: str = "AUTOINDEX"
    milvus_metric_type: str = "COSINE"
    metadata: dict[str, Any]
    load_result: dict[str, Any]


@app.post("/v1/rag/batch")
async def batch_rag(
    req: BatchRequest,
    x_rag_session_id: Optional[str] = Header(default=None, alias="X-RAG-Session-Id"),
) -> JSONResponse:
    if len(req.queries) > 64:
        raise HTTPException(status_code=400, detail="too many queries; max 64")
    t0 = time.perf_counter()
    status = "ok"
    try:
        async def _one(q: QueryRequest) -> QueryResponse:
            job_ids = _resolve_job_ids_for_query(q, x_rag_session_id)
            return await _run_query(q.model_copy(update={"job_ids": job_ids, "job_id": job_ids[0]}))

        raw_results = await asyncio.gather(*(_one(q) for q in req.queries), return_exceptions=True)
        results: list[dict[str, Any]] = []
        had_error = False
        for item in raw_results:
            if isinstance(item, QueryResponse):
                results.append(item.model_dump())
            elif isinstance(item, QueryStageError):
                had_error = True
                results.append(QueryErrorResponse(**item.to_dict()).model_dump())
            elif isinstance(item, Exception):
                had_error = True
                results.append(
                    QueryErrorResponse(error=str(item), stage="unknown").model_dump()
                )
            else:
                had_error = True
                results.append(
                    QueryErrorResponse(
                        error="unknown batch result type",
                        stage="unknown",
                    ).model_dump()
                )
        if had_error:
            status = "partial_error"
        return JSONResponse({"results": results})
    except Exception as e:
        status = "error"
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        REQ_TOTAL.labels(endpoint="/v1/rag/batch", status=status).inc()
        REQ_LAT.labels(endpoint="/v1/rag/batch").observe(time.perf_counter() - t0)


@app.get("/v1/rag/jobs")
async def list_ingest_jobs() -> JSONResponse:
    """API equivalent of Streamlit's job picker list."""
    try:
        rows = await asyncio.to_thread(_minio_store().list_ingest_jobs_table)
        return JSONResponse({"jobs": rows})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"list jobs failed: {e}") from e


@app.post("/v1/rag/load", response_model=LoadJobResponse)
async def load_rag_job(req: LoadJobRequest) -> LoadJobResponse:
    """
    API equivalent of Streamlit's Load action:
    - resolve ingest metadata by job_id
    - load corresponding Milvus collection into memory
    """
    jid = req.job_id.strip()
    try:
        meta = await asyncio.to_thread(_minio_store().get_job_metadata, jid)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"metadata lookup failed: {e}") from e
    if not meta:
        raise HTTPException(status_code=404, detail=f"job not found in MinIO metadata: {jid}")

    coll = _normalized_milvus_collection(str(meta.get("milvus_collection", "")))
    index_type = str(meta.get("milvus_index_type", "AUTOINDEX") or "AUTOINDEX")
    metric_type = str(meta.get("milvus_metric_type", "COSINE") or "COSINE")
    try:
        load_result = await asyncio.to_thread(
            _milvus_store().load_collection,
            collection_name=coll,
            release_others=bool(req.release_others),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"milvus load failed: {e}") from e

    sid = (req.session_id or "").strip()
    if sid:
        try:
            await asyncio.to_thread(_session_set_job, sid, jid)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"session bind failed (Redis?): {e}") from e

    return LoadJobResponse(
        job_id=jid,
        loaded_job_id=jid,
        milvus_collection=coll,
        milvus_index_type=index_type,
        milvus_metric_type=metric_type,
        metadata=meta,
        load_result=load_result,
    )


@app.post("/v1/rag/session")
async def set_rag_session(body: SetSessionJobRequest) -> JSONResponse:
    """Bind a client ``session_id`` to an ingest ``job_id`` so queries can omit ``job_id``."""
    jid = body.job_id.strip()
    sid = body.session_id.strip()
    try:
        meta = await asyncio.to_thread(_minio_store().get_job_metadata, jid)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"metadata lookup failed: {e}") from e
    if not meta:
        raise HTTPException(status_code=404, detail=f"job not found in MinIO metadata: {jid}")
    try:
        await asyncio.to_thread(_session_set_job, sid, jid)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"session store failed (Redis?): {e}") from e
    return JSONResponse({"ok": True, "session_id": sid, "job_id": jid})


@app.delete("/v1/rag/session")
async def clear_rag_session(session_id: str = Query(..., min_length=4, max_length=128)) -> JSONResponse:
    """Remove the active-job binding for ``session_id``."""
    try:
        await asyncio.to_thread(_session_clear, session_id.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"session clear failed: {e}") from e
    return JSONResponse({"ok": True})


def _ingest_config_from_form(
    *,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
    extraction: str,
    summarization: str,
    llm_min_interval_seconds: float,
    milvus_index_type: str,
    milvus_metric_type: str,
    milvus_ivf_nlist: int,
    milvus_hnsw_m: int,
    milvus_hnsw_ef_construction: int,
    milvus_upsert_batch_size: int,
) -> IngestPipelineConfig:
    ex = extraction.strip().lower()
    if ex not in ("shallow", "full"):
        raise HTTPException(status_code=400, detail="extraction must be shallow or full")
    sm = summarization.strip().lower()
    if sm not in ("single", "hierarchical", "iterative"):
        raise HTTPException(status_code=400, detail="summarization must be single, hierarchical, or iterative")
    return IngestPipelineConfig(
        chunk_size=int(chunk_size),
        chunk_overlap=int(chunk_overlap),
        embedding_model=embedding_model.strip(),
        extraction=ex,  # type: ignore[arg-type]
        summarization=sm,  # type: ignore[arg-type]
        llm_min_interval_seconds=float(llm_min_interval_seconds),
        milvus_index_type=milvus_index_type.strip(),
        milvus_metric_type=milvus_metric_type.strip(),
        milvus_ivf_nlist=int(milvus_ivf_nlist),
        milvus_hnsw_m=int(milvus_hnsw_m),
        milvus_hnsw_ef_construction=int(milvus_hnsw_ef_construction),
        milvus_upsert_batch_size=int(milvus_upsert_batch_size),
    )


@app.post("/v1/rag/ingest")
async def ingest_rag(
    file: UploadFile = File(..., description="PDF, TXT, or MD"),
    page_filter: str = Form(""),
    chunk_size: int = Form(512),
    chunk_overlap: int = Form(64),
    embedding_model: str = Form(DEFAULT_EMBED),
    extraction: str = Form("shallow"),
    summarization: str = Form("single"),
    llm_min_interval_seconds: float = Form(0.75),
    milvus_index_type: str = Form("AUTOINDEX"),
    milvus_metric_type: str = Form("COSINE"),
    milvus_ivf_nlist: int = Form(1024),
    milvus_hnsw_m: int = Form(16),
    milvus_hnsw_ef_construction: int = Form(200),
    milvus_upsert_batch_size: int = Form(256),
    job_id: Optional[str] = Form(None),
) -> JSONResponse:
    """
    Same pipeline as the Streamlit ingest tab: Milvus upsert + MinIO artifacts + Redis status.
    Use returned ``job_id`` and ``milvus_collection`` with ``POST /v1/rag/query`` (same ``MILVUS_URI``).
    """
    t0 = time.perf_counter()
    status = "ok"
    try:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="empty upload")
        name = (file.filename or "upload.bin").strip() or "upload.bin"
        cfg = _ingest_config_from_form(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model,
            extraction=extraction,
            summarization=summarization,
            llm_min_interval_seconds=llm_min_interval_seconds,
            milvus_index_type=milvus_index_type,
            milvus_metric_type=milvus_metric_type,
            milvus_ivf_nlist=milvus_ivf_nlist,
            milvus_hnsw_m=milvus_hnsw_m,
            milvus_hnsw_ef_construction=milvus_hnsw_ef_construction,
            milvus_upsert_batch_size=milvus_upsert_batch_size,
        )
        jid = (job_id or "").strip() or None

        def _run() -> dict[str, Any]:
            return run_document_ingest(
                filename=name,
                raw_bytes=raw,
                page_filter_spec=page_filter,
                config=cfg,
                summarizer=None,
                job_id=jid,
            )

        meta = await asyncio.to_thread(_run)
        return JSONResponse(meta)
    except HTTPException:
        status = "error"
        raise
    except Exception as e:
        status = "error"
        raise HTTPException(status_code=500, detail=f"ingest failed: {e}") from e
    finally:
        REQ_TOTAL.labels(endpoint="/v1/rag/ingest", status=status).inc()
        REQ_LAT.labels(endpoint="/v1/rag/ingest").observe(time.perf_counter() - t0)
