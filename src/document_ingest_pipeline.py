"""
Multi-stage ingest aligned with the product flowchart:

upload → (optional page filter) → extraction (shallow vs full text) → chunking +
summarization strategy → embed + Milvus upsert → **store artifacts in MinIO** → **Redis status**.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Callable, List, Literal, Optional, Sequence, Tuple

from src.document_extract import (
    extract_pdf_text,
    extract_pdf_text_full,
    extract_text_from_bytes,
    parse_page_list,
)
from src.embedder import load_embedding_model
from src.ingest_rate_limit import RateLimiter
from src.rag_pipeline import build_corpus_chunks_from_documents
from src.storage.minio_artifacts import MinioArtifactStore, load_minio_settings
from src.storage.milvus_store import MilvusChunkStore, MilvusIndexConfig
from src.storage.redis_jobs import JobStatus, RedisJobStore

SummarizationStrategy = Literal["single", "hierarchical", "iterative"]
ExtractionMode = Literal["shallow", "full"]


@dataclass(frozen=True)
class IngestPipelineConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    extraction: ExtractionMode = "shallow"
    summarization: SummarizationStrategy = "single"
    #: Min seconds between LLM calls when summarization needs the model
    llm_min_interval_seconds: float = 0.75
    milvus_index_type: str = "AUTOINDEX"
    milvus_metric_type: str = "COSINE"
    milvus_ivf_nlist: int = 1024
    milvus_hnsw_m: int = 16
    milvus_hnsw_ef_construction: int = 200
    milvus_upsert_batch_size: int = 256


def _chunks_to_records(
    texts: Sequence[str],
    summaries: Optional[Sequence[str]],
) -> List[dict[str, Any]]:
    out: List[dict[str, Any]] = []
    for i, t in enumerate(texts):
        row: dict[str, Any] = {"index": i, "text": t}
        if summaries is not None and i < len(summaries):
            row["summary"] = summaries[i]
        out.append(row)
    return out


def _stub_summaries(texts: Sequence[str]) -> List[str]:
    """Deterministic placeholder when no LLM summarizer is provided."""
    return [(t[:240] + "…") if len(t) > 240 else t for t in texts]


def _run_summarization(
    strategy: SummarizationStrategy,
    chunk_texts: List[str],
    *,
    rate_limiter: RateLimiter,
    summarizer: Optional[Callable[[str], str]],
) -> Tuple[List[str], Optional[str]]:
    """
    Returns (per_chunk_summaries_or_empty, optional_iterative_global_summary).
    """
    if strategy == "single":
        return [], None

    if summarizer is None:
        return _stub_summaries(chunk_texts), (
            "iterative: " + " | ".join(_stub_summaries(chunk_texts)[:3])
            if strategy == "iterative"
            else None
        )

    summaries: List[str] = []
    for ch in chunk_texts:
        rate_limiter.acquire()
        prompt = (
            "Summarize the following passage in 2-3 sentences for retrieval indexing. "
            "Passage:\n\n"
            + ch[:8000]
        )
        summaries.append(summarizer(prompt))

    global_sum: Optional[str] = None
    if strategy == "iterative":
        rate_limiter.acquire()
        joined = "\n\n".join(f"[{i}] {s}" for i, s in enumerate(summaries))
        global_sum = summarizer(
            "Given these chunk summaries, write one short abstract of the whole document:\n\n"
            + joined[:12000]
        )
    return summaries, global_sum


def run_document_ingest(
    *,
    filename: str,
    raw_bytes: bytes,
    page_filter_spec: str = "",
    config: IngestPipelineConfig,
    summarizer: Optional[Callable[[str], str]] = None,
    minio: Optional[MinioArtifactStore] = None,
    milvus_store: Optional[MilvusChunkStore] = None,
    redis_store: Optional[RedisJobStore] = None,
    job_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Run full pipeline and persist under ``{job_id}/`` in MinIO.

    Returns metadata dict including ``job_id`` and object keys.
    """
    jid = job_id or str(uuid.uuid4())
    store = minio or MinioArtifactStore(load_minio_settings())
    rds = redis_store or RedisJobStore()

    def _status(stage: str, msg: str = "", error: Optional[str] = None, **meta: Any) -> None:
        st: Any = stage
        rds.set_status(
            JobStatus(
                job_id=jid,
                stage=st,  # type: ignore[arg-type]
                message=msg,
                error=error,
                meta=meta or None,
            )
        )

    _status("queued", "starting")
    try:
        _status("extracting", "reading document")
        pages = parse_page_list(page_filter_spec)
        lower = filename.lower()
        if lower.endswith(".pdf"):
            if config.extraction == "shallow":
                text = extract_pdf_text(raw_bytes, page_indices=pages)
            else:
                text = extract_pdf_text_full(raw_bytes, page_indices=pages)
        else:
            text = extract_text_from_bytes(filename, raw_bytes)
            if pages:
                # page filter ignored for non-PDF
                pass

        if not text.strip():
            raise ValueError("No text extracted from document")

        _status("chunking", "chunking + summarization")
        rate = RateLimiter(min_interval_seconds=config.llm_min_interval_seconds)
        chunks = build_corpus_chunks_from_documents(
            [text],
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        summaries, global_summary = _run_summarization(
            config.summarization,
            chunks,
            rate_limiter=rate,
            summarizer=summarizer,
        )

        _status("embedding", "embedding + upsert to Milvus")
        embedder = load_embedding_model(config.embedding_model, normalize=True)
        milvus = milvus_store or MilvusChunkStore()
        index_cfg = MilvusIndexConfig(
            index_type=config.milvus_index_type,
            metric_type=config.milvus_metric_type,
            ivf_nlist=int(config.milvus_ivf_nlist),
            hnsw_m=int(config.milvus_hnsw_m),
            hnsw_ef_construction=int(config.milvus_hnsw_ef_construction),
        )
        n_upserted = milvus.upsert_job_chunks(
            job_id=jid,
            chunk_texts=chunks,
            embedder=embedder,
            batch_size=int(config.milvus_upsert_batch_size),
            index_config=index_cfg,
        )

        _status("storing", "uploading to MinIO")
        prefix = f"{jid}/"
        store.put_bytes(prefix + "source.bin", raw_bytes, content_type="application/octet-stream")
        store.put_json(
            prefix + "extracted_text.json",
            {"filename": filename, "chars": len(text), "preview": text[:2000]},
        )
        records = _chunks_to_records(chunks, summaries if summaries else None)
        store.put_json(prefix + "chunks.json", {"chunks": records})
        meta = {
            "job_id": jid,
            "filename": filename,
            "embedding_model": config.embedding_model,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "n_chunks": len(chunks),
            "extraction": config.extraction,
            "summarization": config.summarization,
            "milvus_collection": milvus.settings.collection,
            "milvus_rows_upserted": int(n_upserted),
            "milvus_index_type": index_cfg.index_type,
            "milvus_metric_type": index_cfg.metric_type,
            "milvus_ivf_nlist": int(index_cfg.ivf_nlist),
            "milvus_hnsw_m": int(index_cfg.hnsw_m),
            "milvus_hnsw_ef_construction": int(index_cfg.hnsw_ef_construction),
            "minio_bucket": store.bucket_name,
            "prefix": prefix,
        }
        if global_summary:
            meta["iterative_global_summary"] = global_summary
            store.put_json(prefix + "summary.json", {"global": global_summary})
        store.put_json(prefix + "metadata.json", meta)

        _status("completed", "done", meta=meta)
        return meta
    except Exception as e:
        _status("failed", str(e), error=repr(e))
        raise


def load_ingest_from_minio(
    job_id: str,
    *,
    minio: Optional[MinioArtifactStore] = None,
) -> tuple[list[str], dict[str, Any]]:
    """
    Download chunks + metadata from MinIO.

    Returns ``(chunk_texts, metadata)``.
    """
    store = minio or MinioArtifactStore(load_minio_settings())
    prefix = f"{job_id}/"
    meta = store.get_json(prefix + "metadata.json")
    chunks_payload = store.get_json(prefix + "chunks.json")
    texts = [c["text"] for c in chunks_payload["chunks"]]
    return texts, meta
