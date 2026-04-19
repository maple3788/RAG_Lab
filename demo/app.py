"""
Streamlit: **Ingest** | **Query** | **Library** | **Benchmark** (SQLite experiment DB, Plotly, Redis).

Run::

    streamlit run demo/app.py
"""

from __future__ import annotations

from dataclasses import asdict
import logging
import html
import math
import os
import re
import time
import warnings
from time import perf_counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Quiet Python warnings (deprecations, ragas, torch, etc.) in the Streamlit terminal.
warnings.simplefilter("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

if not logging.root.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
# Third-party chatter → errors only; app/RAGAS loggers stay at INFO via basicConfig.
for _noisy in (
    "transformers",
    "transformers.models",
    "urllib3",
    "urllib3.connectionpool",
    "httpx",
    "httpcore",
    "asyncio",
    "PIL",
    "faiss",
):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

import sys

import json

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.rag.context_truncation import TruncationStrategy, truncate_context
from src.ingestion.document_ingest_pipeline import (
    load_ingest_from_minio,
    run_document_ingest,
)
from src.config import (
    MetadataFilterConfig,
    MilvusRuntimeConfig,
    PipelineFeaturesConfig,
    PromptConfig,
    RAGPipelineConfig,
    RetrievalConfig,
    SemanticCacheConfig,
    load_ingest_config,
    merge_ingest_with_dict,
)
from src.retrieval.milvus_metadata import metadata_filter_to_milvus_expr
from src.llm.embedder import EmbeddingModel, load_embedding_model, prepare_query
from src.llm.generator import (
    ChatTextGenerator,
    GeminiGenerator,
    MockGenerator,
    OllamaGenerator,
    OpenAICompatibleGenerator,
    StreamingChatTextGenerator,
    StreamingTextGenerator,
    TextGenerator,
)
from src.llm.prompts import PROMPT_TEMPLATES, format_rag_prompt
from src.rag.rag_generation import passages_to_context
from src.retrieval.hybrid_retrieval import (
    BM25Resources,
    build_bm25_resources,
    fused_top_indices,
)
from src.llm.reranker import Reranker, load_reranker
from src.retrieval.retriever import FaissIndex, search
from src.storage.minio_artifacts import MinioArtifactStore, load_minio_settings
from src.storage.milvus_store import (
    MilvusChunkStore,
    MilvusSearchConfig,
)
from src.storage.redis_jobs import RedisJobStore
from src.storage.redis_semantic_cache import RedisSemanticCache
from src.ingestion.streaming_parser import (
    HiddenReasoningStreamParser,
    strip_hidden_reasoning_text,
)
from src.eval.ragas_ui_metrics import run_ragas_legacy_evaluate
from src.rag.rag_pipeline import (
    record_rag_generation_latency,
    record_rag_retrieval_latency,
)
from src.eval.experiment_tracking import (
    FAILURE_FEEDBACK_LABELS,
    aggregate_token_totals,
    fetch_queries_dataframe,
    fetch_runs_dataframe,
    init_db,
    log_experiment_run,
    log_query_event,
    update_query_feedback,
)
from src.eval.metrics import (
    approx_token_count,
    composite_ragas_score,
    compute_answer_metrics,
)


DEFAULT_EMBED = "BAAI/bge-base-en-v1.5"
DEFAULT_RERANK = "BAAI/bge-reranker-base"
MILVUS_INDEX_TYPES = ["AUTOINDEX", "IVF_FLAT", "HNSW"]
MILVUS_METRICS = ["COSINE", "IP", "L2"]
MILVUS_PRESETS: Dict[str, Dict[str, Any]] = {
    "fast": {
        "index_type": "AUTOINDEX",
        "metric_type": "COSINE",
        "upsert_batch_size": 512,
        "ivf_nlist": 256,
        "hnsw_m": 8,
        "hnsw_ef_construction": 100,
        "ivf_nprobe": 8,
        "hnsw_ef": 32,
    },
    "balanced": {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "upsert_batch_size": 256,
        "ivf_nlist": 1024,
        "hnsw_m": 16,
        "hnsw_ef_construction": 200,
        "ivf_nprobe": 32,
        "hnsw_ef": 64,
    },
    "high_recall": {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "upsert_batch_size": 128,
        "ivf_nlist": 2048,
        "hnsw_m": 32,
        "hnsw_ef_construction": 320,
        "ivf_nprobe": 96,
        "hnsw_ef": 128,
    },
}


def _milvus_preset_values(name: str) -> Dict[str, Any]:
    return dict(MILVUS_PRESETS.get(name, MILVUS_PRESETS["balanced"]))


RAG_SYSTEM_PROMPT = (
    "You are a grounded RAG assistant. Use only provided context. "
    "If evidence is insufficient, say unknown."
)
REWRITE_SYSTEM_PROMPT = "You rewrite user questions for retrieval quality."
DECOMPOSE_SYSTEM_PROMPT = "You decompose questions into focused retrieval sub-queries."
FILTER_SYSTEM_PROMPT = "You generate retrieval include/exclude keyword filters."
REFLECTION_SYSTEM_PROMPT = (
    "You audit RAG answers and request follow-up retrieval only when needed."
)


@st.cache_resource
def cached_embedder(model_name: str) -> EmbeddingModel:
    return load_embedding_model(model_name, normalize=True)


@st.cache_resource
def cached_reranker(model_name: str) -> Reranker:
    return load_reranker(model_name)


def _minio_store() -> MinioArtifactStore:
    return MinioArtifactStore(load_minio_settings())


def _milvus_store() -> MilvusChunkStore:
    """
    Do not cache_resource: Milvus may be offline; cached init would show
    "Running _milvus_store()" and block until connect completes.
    Connect timeout is set via MILVUS_CONNECT_TIMEOUT (see MilvusSettings).
    """
    return MilvusChunkStore()


def make_generator(backend: str) -> TextGenerator:
    if backend == "gemini":
        return GeminiGenerator()
    if backend == "openai":
        return OpenAICompatibleGenerator()
    if backend == "ollama":
        return OllamaGenerator()
    return MockGenerator()


def ingest_summarizer(backend: str):
    if backend == "mock":

        def _fn(text: str) -> str:
            return (text[:300] + "…") if len(text) > 300 else text

        return _fn
    gen = make_generator(backend)

    def _fn(text: str) -> str:
        return gen.generate(text)

    return _fn


def _retrieval_merge_key(row: dict) -> Union[int, Tuple[str, str, int]]:
    """Deduplicate merged hits: chunk_index alone collides across different Milvus jobs."""
    jid = row.get("job_id")
    if jid is not None and str(jid).strip() != "":
        return ("jid", str(jid).strip(), int(row["chunk_index"]))
    return int(row["chunk_index"])


def _apply_multi_doc_prefixes(
    passages: List[str], retrieved_rows: List[dict]
) -> List[str]:
    """Prefix passages when chunks come from more than one ingest job."""
    tj = {str(r["text"]): r.get("job_id") for r in retrieved_rows if r.get("text")}
    jobs = {tj.get(p) for p in passages if tj.get(p)}
    jobs.discard(None)
    if len(jobs) <= 1:
        return passages
    out: List[str] = []
    for p in passages:
        jid = tj.get(p)
        out.append(f"[doc:{jid}] {p}" if jid else p)
    return out


def _milvus_search_multi_jobs(
    job_ids: List[str],
    query: str,
    embedder: EmbeddingModel,
    milvus_store: MilvusChunkStore,
    top_k: int,
    index_type_default: str,
    search_config: MilvusSearchConfig,
    metadata_filter_expr: Optional[str] = None,
) -> List[dict]:
    """Search across multiple jobs (possibly multiple Milvus collections)."""
    minio = MinioArtifactStore()
    groups: Dict[str, List[str]] = {}
    metas: Dict[str, Dict[str, Any]] = {}
    for jid in job_ids:
        j = jid.strip()
        meta = minio.get_job_metadata(j) or {}
        coll = str(meta.get("milvus_collection") or "").strip() or None
        if not coll:
            raise ValueError(
                f"missing milvus_collection in MinIO metadata for job_id={j}"
            )
        groups.setdefault(coll, []).append(j)
        metas[j] = meta
    all_rows: List[dict] = []
    for coll, jids_in in groups.items():
        m0 = metas.get(jids_in[0], {})
        idx_t = (index_type_default or "").strip() or str(
            m0.get("milvus_index_type", "AUTOINDEX")
        )
        mtype = str(m0.get("milvus_metric_type", "COSINE") or "COSINE")
        sc = MilvusSearchConfig(
            metric_type=mtype,
            ivf_nprobe=int(search_config.ivf_nprobe),
            hnsw_ef=int(search_config.hnsw_ef),
        )
        part = milvus_store.search_multi_job_chunks(
            job_ids=jids_in,
            query=query,
            embedder=embedder,
            top_k=top_k,
            index_type=str(idx_t or "AUTOINDEX").strip(),
            search_config=sc,
            collection_name=coll,
            metadata_filter_expr=metadata_filter_expr,
        )
        all_rows.extend(part)
    all_rows.sort(key=lambda r: float(r.get("faiss_score", 0.0)))
    return all_rows[:top_k]


def cross_encoder_rerank_trace(
    reranker: Reranker, query: str, pool: List[str]
) -> List[dict]:
    if not pool:
        return []
    pairs = [(query, c) for c in pool]
    scores = np.asarray(reranker.model.predict(pairs), dtype=np.float32)
    order = np.argsort(-scores)
    rows: List[dict] = []
    for new_rank, orig_i in enumerate(order, start=1):
        oi = int(orig_i)
        rows.append(
            {
                "new_rank": new_rank,
                "faiss_rank": oi + 1,
                "cross_encoder_score": float(scores[oi]),
                "text": pool[oi],
            }
        )
    return rows


def _sanitize_singleline(text: str) -> str:
    s = " ".join((text or "").strip().split())
    return s


def _trace_text_area_height(text: str, *, cap: int = 240) -> int:
    """Streamlit ``st.text_area`` requires height >= 68px."""
    h = 24 + 12 * (text or "").count("\n")
    return max(68, min(cap, h))


def _compose_chat_prompt(system_prompt: str, user_prompt: str) -> str:
    return (
        f"SYSTEM_PROMPT:\n{system_prompt.strip()}\n\n"
        f"USER_PROMPT:\n{user_prompt.strip()}"
    )


def llm_generate_text(
    *,
    generator: TextGenerator,
    system_prompt: str,
    user_prompt: str,
    combined_prompt: str,
) -> str:
    if isinstance(generator, ChatTextGenerator):
        return generator.generate_chat(
            system_prompt=system_prompt, user_prompt=user_prompt
        )
    return generator.generate(combined_prompt)


def _rewrite_user_prompt(question: str) -> str:
    return (
        "Rewrite the user's question to improve retrieval quality.\n"
        "Keep intent and constraints exactly the same.\n"
        "Return one short standalone question only (no explanation).\n\n"
        f"Question: {question}\n"
        "Rewritten:"
    )


def _decompose_user_prompt(question: str) -> str:
    return (
        "Break the question into 2-4 focused retrieval sub-queries.\n"
        "Rules:\n"
        "- each line must be a standalone search query\n"
        "- keep entities, dates, and numeric constraints\n"
        "- no explanations, only the list\n\n"
        f"Question: {question}\n"
        "Sub-queries:"
    )


def _filter_user_prompt(question: str) -> str:
    return (
        "Create retrieval filter keywords for this question.\n"
        "Return exactly two lines:\n"
        "INCLUDE: comma-separated short terms\n"
        "EXCLUDE: comma-separated short terms\n"
        "If none, keep line but leave blank after colon.\n\n"
        f"Question: {question}\n"
    )


def _reflection_user_prompt(question: str, answer: str) -> str:
    return (
        "You are checking whether a RAG answer is complete and grounded.\n"
        "If the answer is sufficient, output exactly: DONE\n"
        "If more retrieval is needed, output exactly one line:\n"
        "FOLLOWUP: <one focused search query>\n\n"
        f"Question: {question}\n"
        f"Current answer: {answer}\n"
    )


def _history_to_text(entries: List[dict], *, max_turns: int = 3) -> str:
    if not entries:
        return "None"
    rows: List[str] = []
    for e in entries[-max_turns:]:
        q = (e.get("question") or "").strip()
        rw = (e.get("rewritten_question") or "").strip()
        sq = e.get("subqueries") or []
        ans = (e.get("answer") or "").strip()
        lines = [f"Q: {q}"]
        if rw:
            lines.append(f"Rewritten: {rw}")
        if sq:
            lines.append("Subqueries: " + " | ".join(str(x) for x in sq))
        if ans:
            lines.append(f"A: {ans[:220]}")
        rows.append("\n".join(lines))
    return "\n\n---\n\n".join(rows) if rows else "None"


def _render_qa_scroll_block(messages: List[dict[str, str]]) -> None:
    rows: List[str] = []
    for m in messages:
        q = html.escape((m.get("q") or "").strip())
        a = html.escape((m.get("a") or "").strip())
        rows.append(
            f"""
            <div style="padding:10px 12px; margin-bottom:10px; border:1px solid #e6e6e6; border-radius:10px; background:#ffffff;">
              <div style="font-weight:600; margin-bottom:6px;">You</div>
              <div style="white-space:pre-wrap; margin-bottom:10px;">{q}</div>
              <div style="font-weight:600; margin-bottom:6px;">Assistant</div>
              <div style="white-space:pre-wrap;">{a}</div>
            </div>
            """
        )
    body = "".join(rows) if rows else "<div style='opacity:0.7;'>No messages yet.</div>"
    html_block = (
        '<div style="max-height:380px; overflow-y:auto; border:1px solid #ddd; '
        'border-radius:12px; padding:10px; background:#fafafa;">'
        f"{body}</div>"
    )
    components.html(html_block, height=410, scrolling=True)


def _safe_format_rag_prompt(
    template: str, *, context: str, question: str, history: str
) -> str:
    try:
        return format_rag_prompt(
            template, context=context, question=question, history=history
        )
    except TypeError:
        # Backward compatibility if an older prompts module (without `history`) is loaded.
        return format_rag_prompt(template, context=context, question=question)


def get_generator_details(generator: Optional[TextGenerator]) -> dict[str, Any]:
    if generator is None:
        return {"backend": None, "model": None}
    details: dict[str, Any] = {
        "backend": type(generator).__name__,
        "model": getattr(generator, "model", None),
    }
    for name in ("temperature", "max_tokens", "max_output_tokens"):
        if hasattr(generator, name):
            details[name] = getattr(generator, name)
    return details


def rewrite_query_with_llm(
    question: str,
    *,
    generator: Optional[TextGenerator],
    history: str = "None",
    debug: bool = False,
) -> str | tuple[str, dict[str, Any]]:
    if generator is None:
        out = question
        info = {
            "system_prompt": REWRITE_SYSTEM_PROMPT,
            "user_prompt": None,
            "prompt": None,
            "raw_response": None,
            "parsed": out,
        }
        return (out, info) if debug else out
    user_prompt = _rewrite_user_prompt(question) + "\n\nHistory:\n" + history
    prompt = _compose_chat_prompt(REWRITE_SYSTEM_PROMPT, user_prompt)
    raw = llm_generate_text(
        generator=generator,
        system_prompt=REWRITE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        combined_prompt=prompt,
    )
    out = _sanitize_singleline(raw) or question
    info = {
        "system_prompt": REWRITE_SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "prompt": prompt,
        "raw_response": raw,
        "parsed": out,
    }
    return (out, info) if debug else out


def decompose_query_with_llm(
    question: str,
    *,
    generator: Optional[TextGenerator],
    max_subqueries: int = 3,
    history: str = "None",
    debug: bool = False,
) -> List[str] | tuple[List[str], dict[str, Any]]:
    if generator is None or max_subqueries <= 0:
        out: List[str] = []
        info = {
            "system_prompt": DECOMPOSE_SYSTEM_PROMPT,
            "user_prompt": None,
            "prompt": None,
            "raw_response": None,
            "parsed": out,
        }
        return (out, info) if debug else out
    user_prompt = _decompose_user_prompt(question) + "\n\nHistory:\n" + history
    prompt = _compose_chat_prompt(DECOMPOSE_SYSTEM_PROMPT, user_prompt)
    raw = llm_generate_text(
        generator=generator,
        system_prompt=DECOMPOSE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        combined_prompt=prompt,
    )
    lines: List[str] = []
    for ln in raw.splitlines():
        cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", ln).strip()
        cleaned = _sanitize_singleline(cleaned)
        if cleaned:
            lines.append(cleaned)
    uniq: List[str] = []
    seen = set()
    for q in lines:
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(q)
        if len(uniq) >= max_subqueries:
            break
    info = {
        "system_prompt": DECOMPOSE_SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "prompt": prompt,
        "raw_response": raw,
        "parsed": uniq,
    }
    return (uniq, info) if debug else uniq


def _extract_citation_ids(answer: str) -> List[int]:
    ids = {int(m) for m in re.findall(r"\[(\d+)\]", answer or "")}
    return sorted(i for i in ids if i > 0)


def generate_filter_with_llm(
    question: str,
    *,
    generator: Optional[TextGenerator],
) -> dict | tuple[dict, dict[str, Any]]:
    if generator is None:
        filt = {"include": [], "exclude": []}
        info = {
            "system_prompt": FILTER_SYSTEM_PROMPT,
            "user_prompt": None,
            "prompt": None,
            "raw_response": None,
            "parsed": filt,
        }
        return (filt, info)
    user_prompt = _filter_user_prompt(question)
    prompt = _compose_chat_prompt(FILTER_SYSTEM_PROMPT, user_prompt)
    out = llm_generate_text(
        generator=generator,
        system_prompt=FILTER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        combined_prompt=prompt,
    )
    include: List[str] = []
    exclude: List[str] = []
    for ln in out.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.lower().startswith("include:"):
            vals = s.split(":", 1)[1].strip()
            include = [
                _sanitize_singleline(v)
                for v in vals.split(",")
                if _sanitize_singleline(v)
            ]
        elif s.lower().startswith("exclude:"):
            vals = s.split(":", 1)[1].strip()
            exclude = [
                _sanitize_singleline(v)
                for v in vals.split(",")
                if _sanitize_singleline(v)
            ]
    filt = {"include": include, "exclude": exclude}
    info = {
        "system_prompt": FILTER_SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "prompt": prompt,
        "raw_response": out,
        "parsed": filt,
    }
    return filt, info


def apply_keyword_filter(rows: List[dict], filt: dict, *, min_keep: int) -> List[dict]:
    include = [
        t.lower() for t in filt.get("include", []) if isinstance(t, str) and t.strip()
    ]
    exclude = [
        t.lower() for t in filt.get("exclude", []) if isinstance(t, str) and t.strip()
    ]
    if not include and not exclude:
        return rows
    kept: List[dict] = []
    for row in rows:
        text = str(row.get("text", "")).lower()
        if include and not any(t in text for t in include):
            continue
        if exclude and any(t in text for t in exclude):
            continue
        kept.append(row)
    if len(kept) < min_keep:
        return rows
    return kept


def reflection_followup_query(
    *,
    question: str,
    answer: str,
    generator: Optional[TextGenerator],
) -> Optional[str] | tuple[Optional[str], dict[str, Any]]:
    if generator is None:
        info = {
            "system_prompt": REFLECTION_SYSTEM_PROMPT,
            "user_prompt": None,
            "prompt": None,
            "raw_response": None,
            "parsed": None,
        }
        return None, info
    user_prompt = _reflection_user_prompt(question, answer)
    prompt = _compose_chat_prompt(REFLECTION_SYSTEM_PROMPT, user_prompt)
    raw = llm_generate_text(
        generator=generator,
        system_prompt=REFLECTION_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        combined_prompt=prompt,
    )
    out = _sanitize_singleline(raw)
    if not out or out.upper() == "DONE":
        return None, {
            "system_prompt": REFLECTION_SYSTEM_PROMPT,
            "user_prompt": user_prompt,
            "prompt": prompt,
            "raw_response": raw,
            "parsed": None,
        }
    m = re.match(r"(?i)^FOLLOWUP:\s*(.+)$", out)
    if not m:
        return None, {
            "system_prompt": REFLECTION_SYSTEM_PROMPT,
            "user_prompt": user_prompt,
            "prompt": prompt,
            "raw_response": raw,
            "parsed": None,
        }
    q = _sanitize_singleline(m.group(1))
    parsed = q or None
    return parsed, {
        "system_prompt": REFLECTION_SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "prompt": prompt,
        "raw_response": raw,
        "parsed": parsed,
    }


def generate_visible_answer(
    *,
    generator: TextGenerator,
    system_prompt: str,
    user_prompt: str,
    prompt: str,
    on_visible_text: Optional[Callable[[str], None]] = None,
    on_raw_text: Optional[Callable[[str], None]] = None,
) -> str:
    t_gen = perf_counter()
    try:
        parser = HiddenReasoningStreamParser()
        out_parts: List[str] = []
        raw_parts: List[str] = []

        if on_visible_text is not None and isinstance(
            generator, (StreamingTextGenerator, StreamingChatTextGenerator)
        ):
            if isinstance(generator, StreamingChatTextGenerator):
                token_iter = generator.generate_chat_stream(
                    system_prompt=system_prompt, user_prompt=user_prompt
                )
            else:
                token_iter = generator.generate_stream(prompt)
            for token in token_iter:
                raw_parts.append(token)
                if on_raw_text is not None:
                    on_raw_text("".join(raw_parts))
                visible = parser.feed(token)
                if visible:
                    out_parts.append(visible)
                    on_visible_text("".join(out_parts))
            tail = parser.flush()
            if tail:
                out_parts.append(tail)
                on_visible_text("".join(out_parts))
            return "".join(out_parts).strip()

        raw = llm_generate_text(
            generator=generator,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            combined_prompt=prompt,
        )
        if on_raw_text is not None:
            on_raw_text(raw)
        visible = strip_hidden_reasoning_text(raw)
        if on_visible_text is not None:
            on_visible_text(visible)
        return visible
    finally:
        record_rag_generation_latency(perf_counter() - t_gen)


def _retrieve_rows_for_query(
    query: str,
    *,
    embedder: EmbeddingModel,
    corpus_chunks: List[str],
    faiss_index: FaissIndex,
    config: RAGPipelineConfig,
    top_k: int,
    milvus_store: Optional[MilvusChunkStore] = None,
    bm25_resources: Optional[BM25Resources] = None,
    metadata_filter_expr: Optional[str] = None,
) -> List[dict]:
    r = config.retrieval
    mctx = config.milvus
    milvus_search_config = mctx.search_config()
    mids = mctx.resolved_job_ids()
    vdb_backend = r.vdb_backend
    if vdb_backend == "milvus" and milvus_store is not None and mids:
        if len(mids) > 1:
            return _milvus_search_multi_jobs(
                mids,
                query,
                embedder,
                milvus_store,
                top_k,
                mctx.index_type,
                milvus_search_config,
                metadata_filter_expr=metadata_filter_expr,
            )
        return milvus_store.search_job_chunks(
            job_id=mids[0],
            query=query,
            embedder=embedder,
            top_k=top_k,
            search_config=milvus_search_config,
            index_type=mctx.index_type,
            collection_name=mctx.collection,
            metadata_filter_expr=metadata_filter_expr,
        )
    # FAISS backend: allow dense-only or hybrid BM25+dense (RRF).
    if r.mode == "hybrid" and bm25_resources is not None:
        idx_row = fused_top_indices(
            query,
            embedder=embedder,
            corpus_chunks=corpus_chunks,
            faiss_index=faiss_index,
            bm25_resources=bm25_resources,
            retrieve_k=top_k,
            rrf_k=int(r.rrf_k),
            fusion_list_k=r.fusion_list_k,
        )
        rows: List[dict] = []
        for rank, ci in enumerate(idx_row, start=1):
            rows.append(
                {
                    "query_used": query,
                    "faiss_rank": rank,
                    "chunk_index": int(ci),
                    # In hybrid path this is RRF-ranked, not raw FAISS score.
                    "faiss_score": float(0.0),
                    "text": corpus_chunks[int(ci)],
                }
            )
        return rows

    q = prepare_query(embedder.name, query)
    q_emb = embedder.encode([q])
    faiss_scores, idx = search(faiss_index, q_emb, top_k=top_k)
    idx_row = idx[0].tolist()
    score_row = faiss_scores[0].tolist()
    rows: List[dict] = []
    rank = 0
    for i, chunk_idx in enumerate(idx_row):
        if chunk_idx < 0:
            continue
        rank += 1
        ci = int(chunk_idx)
        rows.append(
            {
                "query_used": query,
                "faiss_rank": rank,
                "chunk_index": ci,
                "faiss_score": float(score_row[i]),
                "text": corpus_chunks[ci],
            }
        )
    return rows


def run_pipeline(
    question: str,
    *,
    embedder: EmbeddingModel,
    corpus_chunks: List[str],
    faiss_index: FaissIndex,
    config: RAGPipelineConfig,
    generator: TextGenerator,
    reranker: Optional[Reranker],
    milvus_store: Optional[MilvusChunkStore] = None,
    bm25_resources: Optional[BM25Resources] = None,
    semantic_cache: Optional[RedisSemanticCache] = None,
    history: str = "None",
) -> dict:
    feat = config.features
    r = config.retrieval
    mv = config.milvus
    pc = config.prompt
    scf = config.semantic_cache
    meta = config.metadata
    mids = mv.resolved_job_ids()
    milvus_search_config = mv.search_config()
    use_milvus_vec = bool(
        r.vdb_backend == "milvus"
        and milvus_store is not None
        and len(mids) >= 1
        and milvus_search_config is not None
    )
    if use_milvus_vec:
        k = r.retrieve_k
    else:
        k = min(r.retrieve_k, len(corpus_chunks))
    prompt_template = PROMPT_TEMPLATES.get(pc.template_key, PROMPT_TEMPLATES["default"])
    max_context_chars = int(pc.max_context_chars)
    truncation = pc.truncation  # type: ignore[assignment]
    retrieve_k = r.retrieve_k
    final_k = r.final_k
    retrieval_mode = r.mode
    milvus_collection = mv.collection
    milvus_index_type = mv.index_type
    fusion_list_k = r.fusion_list_k
    rrf_k = r.rrf_k
    vdb_backend = r.vdb_backend
    use_query_rewrite = feat.use_query_rewrite
    use_query_decomposition = feat.use_query_decomposition
    max_subqueries = feat.max_subqueries
    use_filter_generation = feat.use_filter_generation
    min_filter_keep = feat.min_filter_keep
    use_reflection_loops = feat.use_reflection_loops
    max_reflection_loops = feat.max_reflection_loops
    require_citations = feat.require_citations
    use_rerank = feat.use_rerank
    filter_doc_date_min = meta.doc_date_min
    filter_doc_date_max = meta.doc_date_max
    filter_source_type = meta.source_type
    filter_section = meta.section
    semantic_cache_threshold = float(scf.threshold)
    semantic_cache_max_entries = int(scf.max_entries)
    llm_details = get_generator_details(generator)
    stage_trace: List[dict] = []
    stage_trace.append(
        {
            "stage": "input",
            "title": "Input question",
            "latency_ms": 0.0,
            "input": {"question": question},
            "output": {
                "llm": llm_details,
                "retrieve_k": retrieve_k,
                "final_k": final_k,
                "use_rerank": use_rerank,
                "use_query_rewrite": use_query_rewrite,
                "use_query_decomposition": use_query_decomposition,
                "use_filter_generation": use_filter_generation,
                "use_reflection_loops": use_reflection_loops,
                "require_citations": require_citations,
                "history_chars": len(history or ""),
                "retrieval_mode": retrieval_mode,
                "milvus_index_type": milvus_index_type,
                "milvus_collection": milvus_collection,
                "milvus_job_ids": mids if len(mids) > 1 else None,
                "filter_doc_date_min": (filter_doc_date_min or "").strip() or None,
                "filter_doc_date_max": (filter_doc_date_max or "").strip() or None,
                "filter_source_type": (filter_source_type or "").strip() or None,
                "filter_section": (filter_section or "").strip() or None,
                "config_source": "RAGPipelineConfig",
            },
        }
    )
    if k <= 0:
        return {
            "error": "No job loaded.",
            "retrieved": [],
            "rerank_rows": [],
            "final_passages": [],
            "prompt": "",
            "answer": "",
            "raw_context": "",
        }

    metadata_filter_expr: Optional[str] = None
    if any(
        (
            (filter_doc_date_min or "").strip(),
            (filter_doc_date_max or "").strip(),
            (filter_source_type or "").strip(),
            (filter_section or "").strip(),
        )
    ):
        try:
            metadata_filter_expr = metadata_filter_to_milvus_expr(
                doc_date_min=filter_doc_date_min,
                doc_date_max=filter_doc_date_max,
                source_type=filter_source_type,
                section=filter_section,
            )
        except ValueError as e:
            return {
                "error": f"metadata filter: {e}",
                "retrieved": [],
                "rerank_rows": [],
                "final_passages": [],
                "prompt": "",
                "answer": "",
                "raw_context": "",
            }

    t_all = perf_counter()
    stage_latencies: Dict[str, float] = {}

    t_stage = perf_counter()
    cache_hit = False
    cached_answer: Optional[str] = None
    if semantic_cache is not None:
        q0 = prepare_query(embedder.name, question)
        q0_emb = embedder.encode([q0])[0]
        cached_answer = semantic_cache.lookup(
            q0_emb, threshold=float(semantic_cache_threshold)
        )
        if cached_answer:
            cache_hit = True
    stage_latencies["semantic_cache_lookup_ms"] = (perf_counter() - t_stage) * 1000.0

    t_stage = perf_counter()
    rewritten_question = question
    rewrite_debug: dict[str, Any] = {
        "prompt": None,
        "raw_response": None,
        "parsed": question,
    }
    if use_query_rewrite:
        rewritten_question, rewrite_debug = rewrite_query_with_llm(
            question, generator=generator, history=history, debug=True
        )  # type: ignore[assignment]
    stage_latencies["query_rewrite_ms"] = (perf_counter() - t_stage) * 1000.0
    stage_trace.append(
        {
            "stage": "query_rewrite",
            "title": "Query rewrite",
            "latency_ms": round(stage_latencies["query_rewrite_ms"], 2),
            "input": {"question": question, "enabled": use_query_rewrite},
            "output": {
                "rewritten_question": rewritten_question,
                "llm_call": {"system_prompt": None, **rewrite_debug},
            },
        }
    )

    t_stage = perf_counter()
    retrieval_queries: List[str] = [rewritten_question]
    decompose_debug: dict[str, Any] = {
        "prompt": None,
        "raw_response": None,
        "parsed": [],
    }
    if use_query_decomposition:
        subs, decompose_debug = decompose_query_with_llm(
            rewritten_question,
            generator=generator,
            max_subqueries=max_subqueries,
            history=history,
            debug=True,
        )  # type: ignore[assignment]
        for sq in subs:
            if sq.lower() == rewritten_question.lower():
                continue
            retrieval_queries.append(sq)
    stage_latencies["query_decomposition_ms"] = (perf_counter() - t_stage) * 1000.0
    stage_trace.append(
        {
            "stage": "query_decomposition",
            "title": "Query decomposition",
            "latency_ms": round(stage_latencies["query_decomposition_ms"], 2),
            "input": {
                "rewritten_question": rewritten_question,
                "enabled": use_query_decomposition,
                "max_subqueries": max_subqueries,
            },
            "output": {
                "retrieval_queries": retrieval_queries,
                "llm_call": {"system_prompt": None, **decompose_debug},
            },
        }
    )

    t_stage = perf_counter()
    merged: Dict[Any, dict] = {}
    for rq in retrieval_queries:
        rows = _retrieve_rows_for_query(
            rq,
            embedder=embedder,
            corpus_chunks=corpus_chunks,
            faiss_index=faiss_index,
            config=config,
            top_k=k,
            milvus_store=milvus_store,
            bm25_resources=bm25_resources,
            metadata_filter_expr=metadata_filter_expr,
        )
        for row in rows:
            mk = _retrieval_merge_key(row)
            ci = int(row["chunk_index"])
            if mk not in merged:
                merged[mk] = {
                    "chunk_index": ci,
                    "faiss_score": float(row["faiss_score"]),
                    "best_faiss_rank": int(row["faiss_rank"]),
                    "matched_queries": [rq],
                    "text": row["text"],
                    "job_id": row.get("job_id"),
                }
            else:
                cur = merged[mk]
                cur["faiss_score"] = max(
                    float(cur["faiss_score"]), float(row["faiss_score"])
                )
                cur["best_faiss_rank"] = min(
                    int(cur["best_faiss_rank"]), int(row["faiss_rank"])
                )
                if rq not in cur["matched_queries"]:
                    cur["matched_queries"].append(rq)

    retrieved_all = sorted(
        merged.values(),
        key=lambda r: (-float(r["faiss_score"]), int(r["best_faiss_rank"])),
    )
    for i, row in enumerate(retrieved_all, start=1):
        row["faiss_rank"] = i
    stage_latencies["dense_retrieval_ms"] = (perf_counter() - t_stage) * 1000.0
    stage_trace.append(
        {
            "stage": "dense_retrieval",
            "title": "Retrieval + merge",
            "latency_ms": round(stage_latencies["dense_retrieval_ms"], 2),
            "input": {
                "retrieval_queries": retrieval_queries,
                "top_k_each": k,
                "retrieval_mode": retrieval_mode,
                "fusion_list_k": fusion_list_k,
                "rrf_k": rrf_k,
            },
            "output": {
                "n_unique_chunks": len(retrieved_all),
                "top_preview": [
                    {
                        "faiss_rank": r.get("faiss_rank"),
                        "chunk_index": r.get("chunk_index"),
                        "faiss_score": r.get("faiss_score"),
                        "matched_queries": r.get("matched_queries", []),
                    }
                    for r in retrieved_all[: min(5, len(retrieved_all))]
                ],
            },
        }
    )

    t_stage = perf_counter()
    retrieval_filter = {"include": [], "exclude": []}
    filter_debug: dict[str, Any] = {
        "prompt": None,
        "raw_response": None,
        "parsed": retrieval_filter,
    }
    retrieved = retrieved_all
    if use_filter_generation:
        retrieval_filter, filter_debug = generate_filter_with_llm(
            rewritten_question, generator=generator
        )  # type: ignore[assignment]
        retrieved = apply_keyword_filter(
            retrieved_all,
            retrieval_filter,
            min_keep=max(1, int(min_filter_keep)),
        )
        for i, row in enumerate(retrieved, start=1):
            row["faiss_rank"] = i
    stage_latencies["filter_generation_ms"] = (perf_counter() - t_stage) * 1000.0
    stage_trace.append(
        {
            "stage": "filter_generation",
            "title": "Filter generation",
            "latency_ms": round(stage_latencies["filter_generation_ms"], 2),
            "input": {
                "enabled": use_filter_generation,
                "query": rewritten_question,
                "min_filter_keep": min_filter_keep,
            },
            "output": {
                "filter": retrieval_filter,
                "before_count": len(retrieved_all),
                "after_count": len(retrieved),
                "llm_call": {"system_prompt": None, **filter_debug},
            },
        }
    )
    pool = [r["text"] for r in retrieved]

    rerank_rows: List[dict] = []
    final_passages: List[str] = []

    t_stage = perf_counter()
    if use_rerank and reranker is not None and pool:
        rerank_rows = cross_encoder_rerank_trace(reranker, rewritten_question, pool)
        final_passages = [
            r["text"] for r in rerank_rows[: min(final_k, len(rerank_rows))]
        ]
    else:
        final_passages = pool[: min(final_k, len(pool))]
    stage_latencies["rerank_ms"] = (perf_counter() - t_stage) * 1000.0
    stage_trace.append(
        {
            "stage": "rerank",
            "title": "Rerank / final passage selection",
            "latency_ms": round(stage_latencies["rerank_ms"], 2),
            "input": {
                "enabled": use_rerank,
                "pool_size": len(pool),
                "final_k": final_k,
            },
            "output": {
                "used_reranker": bool(use_rerank and reranker is not None and pool),
                "final_passage_count": len(final_passages),
                "final_passage_preview": final_passages[: min(3, len(final_passages))],
            },
        }
    )
    record_rag_retrieval_latency(
        (
            float(stage_latencies.get("dense_retrieval_ms", 0.0))
            + float(stage_latencies.get("rerank_ms", 0.0))
        )
        / 1000.0
    )

    def _compose_rag_prompt_parts(
        passages: List[str],
        q_for_prompt: str,
    ) -> tuple[str, str, str]:
        """Build the same user prompt + combined prompt string used for generation (for display)."""
        raw_ctx = passages_to_context(passages)
        context = truncate_context(raw_ctx, max_context_chars, truncation)
        effective_user_template = prompt_template
        if require_citations:
            effective_user_template += (
                "\n\nWhen answering, include supporting citations using bracket numbers "
                "like [1], [2]. Use only numbers that correspond to the provided passages."
            )
        user_prompt = _safe_format_rag_prompt(
            effective_user_template,
            context=context,
            question=q_for_prompt,
            history=history,
        )
        prompt_val = _compose_chat_prompt(RAG_SYSTEM_PROMPT, user_prompt)
        return raw_ctx, user_prompt, prompt_val

    def _build_answer(
        passages: List[str],
        q_for_prompt: str,
        *,
        on_stream: Optional[Callable[[str], None]] = None,
        on_raw_stream: Optional[Callable[[str], None]] = None,
    ) -> tuple[str, str, str, str]:
        raw_ctx, user_prompt, prompt_val = _compose_rag_prompt_parts(
            _apply_multi_doc_prefixes(passages, retrieved), q_for_prompt
        )
        answer_val = generate_visible_answer(
            generator=generator,
            system_prompt=RAG_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            prompt=prompt_val,
            on_visible_text=on_stream,
            on_raw_text=on_raw_stream,
        )
        return raw_ctx, user_prompt, prompt_val, answer_val

    t_stage = perf_counter()
    raw_context, user_prompt, prompt = _compose_rag_prompt_parts(
        _apply_multi_doc_prefixes(final_passages, retrieved), rewritten_question
    )
    stage_latencies["prompt_build_ms"] = (perf_counter() - t_stage) * 1000.0

    t_stage = perf_counter()
    if cache_hit and cached_answer is not None:
        answer = cached_answer
    else:
        answer = generate_visible_answer(
            generator=generator,
            system_prompt=RAG_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            prompt=prompt,
            on_visible_text=None,
            on_raw_text=None,
        )
    stage_latencies["generation_llm_ms"] = (perf_counter() - t_stage) * 1000.0
    gen_lat = round(
        stage_latencies["prompt_build_ms"] + stage_latencies["generation_llm_ms"], 2
    )
    stage_trace.append(
        {
            "stage": "generation",
            "title": "Prompt + generation",
            "latency_ms": gen_lat,
            "input": {
                "question_for_prompt": rewritten_question,
                "prompt_template": prompt_template[:180],
                "max_context_chars": max_context_chars,
                "truncation": truncation,
            },
            "output": {
                "cache_hit": cache_hit,
                "prompt_build_ms": round(stage_latencies["prompt_build_ms"], 2),
                "generation_llm_ms": round(stage_latencies["generation_llm_ms"], 2),
                "raw_context_chars": len(raw_context),
                "prompt_chars": len(prompt),
                "answer_preview": answer[:500],
                "llm_call": {
                    "system_prompt": RAG_SYSTEM_PROMPT,
                    "user_prompt": user_prompt,
                    "prompt": prompt,
                    "raw_response": answer,
                    "parsed": answer,
                },
            },
        }
    )

    reflection_steps: List[dict] = []
    seen_reflection_queries = {q.lower() for q in retrieval_queries}
    t_refl = perf_counter()
    if use_reflection_loops and max_reflection_loops > 0:
        for i in range(int(max_reflection_loops)):
            followup, refl_debug = reflection_followup_query(
                question=rewritten_question,
                answer=answer,
                generator=generator,
            )  # type: ignore[assignment]
            if not followup:
                reflection_steps.append(
                    {
                        "loop": i + 1,
                        "decision": "done",
                        "llm_call": {"system_prompt": None, **refl_debug},
                    }
                )
                break
            if followup.lower() in seen_reflection_queries:
                reflection_steps.append(
                    {
                        "loop": i + 1,
                        "decision": "duplicate_followup",
                        "query": followup,
                        "llm_call": {"system_prompt": None, **refl_debug},
                    }
                )
                break

            seen_reflection_queries.add(followup.lower())
            retrieval_queries.append(followup)
            new_rows = _retrieve_rows_for_query(
                followup,
                embedder=embedder,
                corpus_chunks=corpus_chunks,
                faiss_index=faiss_index,
                config=config,
                top_k=k,
                milvus_store=milvus_store,
                bm25_resources=bm25_resources,
                metadata_filter_expr=metadata_filter_expr,
            )
            added = 0
            for row in new_rows:
                mk = _retrieval_merge_key(row)
                ci = int(row["chunk_index"])
                if mk not in merged:
                    merged[mk] = {
                        "chunk_index": ci,
                        "faiss_score": float(row["faiss_score"]),
                        "best_faiss_rank": int(row["faiss_rank"]),
                        "matched_queries": [followup],
                        "text": row["text"],
                        "job_id": row.get("job_id"),
                    }
                    added += 1
                else:
                    cur = merged[mk]
                    cur["faiss_score"] = max(
                        float(cur["faiss_score"]), float(row["faiss_score"])
                    )
                    cur["best_faiss_rank"] = min(
                        int(cur["best_faiss_rank"]), int(row["faiss_rank"])
                    )
                    if followup not in cur["matched_queries"]:
                        cur["matched_queries"].append(followup)

            retrieved_all = sorted(
                merged.values(),
                key=lambda r: (-float(r["faiss_score"]), int(r["best_faiss_rank"])),
            )
            if use_filter_generation:
                retrieved = apply_keyword_filter(
                    retrieved_all,
                    retrieval_filter,
                    min_keep=max(1, int(min_filter_keep)),
                )
            else:
                retrieved = retrieved_all
            for r_i, row in enumerate(retrieved, start=1):
                row["faiss_rank"] = r_i
            pool = [r["text"] for r in retrieved]

            if use_rerank and reranker is not None and pool:
                rerank_rows = cross_encoder_rerank_trace(
                    reranker, rewritten_question, pool
                )
                final_passages = [
                    r["text"] for r in rerank_rows[: min(final_k, len(rerank_rows))]
                ]
            else:
                final_passages = pool[: min(final_k, len(pool))]
            raw_context, user_prompt, prompt, answer = _build_answer(
                final_passages, rewritten_question
            )
            reflection_steps.append(
                {
                    "loop": i + 1,
                    "decision": "followup",
                    "query": followup,
                    "new_chunks_added": added,
                    "llm_call": {"system_prompt": None, **refl_debug},
                }
            )
        stage_latencies["reflection_ms"] = (perf_counter() - t_refl) * 1000.0
        stage_trace.append(
            {
                "stage": "reflection",
                "title": "Reflection loops",
                "latency_ms": round(stage_latencies["reflection_ms"], 2),
                "input": {
                    "enabled": use_reflection_loops,
                    "max_reflection_loops": max_reflection_loops,
                },
                "output": {"steps": reflection_steps},
            }
        )
    else:
        stage_latencies["reflection_ms"] = 0.0
        stage_trace.append(
            {
                "stage": "reflection",
                "title": "Reflection loops",
                "latency_ms": 0.0,
                "input": {
                    "enabled": use_reflection_loops,
                    "max_reflection_loops": max_reflection_loops,
                },
                "output": {"steps": []},
            }
        )

    passage_to_meta = {r["text"]: r for r in retrieved}
    citation_table: List[dict] = []
    for i, p in enumerate(final_passages, start=1):
        meta = passage_to_meta.get(p, {})
        citation_table.append(
            {
                "id": i,
                "chunk_index": meta.get("chunk_index"),
                "faiss_score": meta.get("faiss_score"),
                "text": p,
            }
        )

    t_stage = perf_counter()
    cited_ids = _extract_citation_ids(answer)
    cited_rows = [r for r in citation_table if int(r["id"]) in cited_ids]
    stage_latencies["citations_ms"] = (perf_counter() - t_stage) * 1000.0
    stage_trace.append(
        {
            "stage": "semantic_cache",
            "title": "Semantic cache",
            "latency_ms": round(stage_latencies["semantic_cache_lookup_ms"], 2),
            "input": {
                "enabled": semantic_cache is not None,
                "threshold": semantic_cache_threshold,
            },
            "output": {
                "cache_hit": cache_hit,
                "lookup_ms": round(stage_latencies["semantic_cache_lookup_ms"], 2),
            },
        }
    )
    stage_trace.append(
        {
            "stage": "citations",
            "title": "Citation extraction",
            "latency_ms": round(stage_latencies["citations_ms"], 2),
            "input": {"answer_preview": answer[:500]},
            "output": {
                "citation_ids": cited_ids,
                "citation_table": [
                    {
                        "id": r.get("id"),
                        "chunk_index": r.get("chunk_index"),
                        "faiss_score": r.get("faiss_score"),
                    }
                    for r in citation_table
                ],
            },
        }
    )

    t_stage = perf_counter()
    if semantic_cache is not None and not cache_hit:
        q0 = prepare_query(embedder.name, question)
        q0_emb = embedder.encode([q0])[0]
        semantic_cache.write(
            question=question,
            answer=answer,
            query_embedding=q0_emb,
            max_entries=int(semantic_cache_max_entries),
        )
    stage_latencies["semantic_cache_write_ms"] = (perf_counter() - t_stage) * 1000.0

    latency_total_ms = (perf_counter() - t_all) * 1000.0

    return {
        "error": None,
        "retrieved": retrieved,
        "retrieved_before_filter": retrieved_all,
        "retrieval_filter": retrieval_filter,
        "rerank_rows": rerank_rows,
        "final_passages": final_passages,
        "citation_table": citation_table,
        "cited_rows": cited_rows,
        "retrieval_queries": retrieval_queries,
        "reflection_steps": reflection_steps,
        "original_question": question,
        "rewritten_question": rewritten_question,
        "stage_trace": stage_trace,
        "stage_latencies": {k: round(float(v), 2) for k, v in stage_latencies.items()},
        "latency_total_ms": round(latency_total_ms, 2),
        "prompt": prompt,
        "answer": answer,
        "raw_context": raw_context,
        "truncation": truncation,
        "max_context_chars": max_context_chars,
        "use_rerank": use_rerank,
        "retrieval_mode": retrieval_mode,
        "cache_hit": cache_hit,
    }


def render_rag_trace(result: dict, *, ui_id: int, final_k: int) -> None:
    fk = int(final_k)
    tab_chunks, tab_rerank, tab_prompt = st.tabs(
        [
            "Retrieved chunks (vector search)",
            "Reranking (cross-encoder)",
            "Final prompt & answer",
        ]
    )

    with tab_chunks:
        mode = result.get("retrieval_mode", "dense")
        if mode == "hybrid":
            st.caption("Retrieval mode: **Hybrid BM25 + dense (RRF fusion)**.")
        if result.get("rewritten_question") and result.get(
            "rewritten_question"
        ) != result.get("original_question"):
            st.caption(
                f"Query rewrite: `{result['original_question']}` → `{result['rewritten_question']}`"
            )
        queries = result.get("retrieval_queries", [])
        if len(queries) > 1:
            st.caption(f"Query decomposition used {len(queries)} retrieval queries.")
        filt = result.get("retrieval_filter", {})
        if isinstance(filt, dict) and (filt.get("include") or filt.get("exclude")):
            st.caption(
                f"LLM filter include={filt.get('include', [])} exclude={filt.get('exclude', [])}"
            )
        st.markdown("Top-K from bi-encoder vector search (cosine-style similarity).")
        for row in result["retrieved"]:
            _jid = row.get("job_id")
            _jl = f" · job `{_jid[:10]}…`" if _jid else ""
            with st.expander(
                f"Rank {row['faiss_rank']} · chunk #{row['chunk_index']}{_jl} · {row['faiss_score']:.4f}"
            ):
                mqs = row.get("matched_queries")
                if isinstance(mqs, list) and mqs:
                    st.caption("Matched queries: " + " | ".join(mqs))
                st.text_area(
                    "Chunk",
                    row["text"],
                    height=_trace_text_area_height(row["text"], cap=240),
                    key=f"{ui_id}_c_{row['faiss_rank']}_{row['chunk_index']}",
                    label_visibility="collapsed",
                )

    with tab_rerank:
        if not result["use_rerank"]:
            st.info("Reranking off — using retrieval order for final passages.")
        elif not result["rerank_rows"]:
            st.warning("Empty pool.")
        else:
            for row in result["rerank_rows"]:
                with st.expander(
                    f"CE rank {row['new_rank']} (was retrieval #{row['faiss_rank']}) · {row['cross_encoder_score']:.4f}"
                ):
                    st.text_area(
                        "Passage",
                        row["text"],
                        height=_trace_text_area_height(row["text"], cap=240),
                        key=f"{ui_id}_r_{row['new_rank']}",
                        label_visibility="collapsed",
                    )
            st.caption(
                f"Passages 1–{min(fk, len(result['rerank_rows']))} → LLM context."
            )

    with tab_prompt:
        if result.get("cache_hit"):
            st.caption(
                "Answer from **Redis semantic cache** (no LLM call). "
                "Prompt below is the same RAG prompt built from retrieved context."
            )
        st.caption(
            f"Context ~{len(result.get('raw_context', ''))} chars · "
            f"truncation **{result['truncation']}** → **{result['max_context_chars']}**"
        )
        st.text_area("Prompt", result.get("prompt") or "", height=360, key=f"{ui_id}_p")
        st.write("**Answer (repeat)**")
        st.write(result["answer"])
        steps = result.get("reflection_steps", [])
        if steps:
            st.write("**Reflection loop trace**")
            for step in steps:
                if step.get("decision") == "followup":
                    st.caption(
                        f"Loop {step['loop']}: follow-up query `{step.get('query', '')}` "
                        f"(new chunks: {step.get('new_chunks_added', 0)})"
                    )
                else:
                    st.caption(f"Loop {step.get('loop')}: {step.get('decision')}")
        cited = result.get("cited_rows", [])
        if cited:
            st.write("**Citations used in answer**")
            for row in cited:
                with st.expander(f"[{row['id']}] chunk #{row.get('chunk_index')}"):
                    st.text_area(
                        "Source",
                        row["text"],
                        height=_trace_text_area_height(row["text"], cap=220),
                        key=f"{ui_id}_cite_{row['id']}",
                        label_visibility="collapsed",
                    )


def _default_ragas_eval_model(backend: str) -> str:
    """Default evaluator model id (LangChain chat; match typical chat defaults in this app)."""
    if backend == "openai":
        return "gpt-4o-mini"
    if backend == "ollama":
        return "llama3.2"
    if backend == "gemini":
        return "gemini-2.5-flash"
    return "gpt-4o-mini"


def _ragas_score_cell(raw: Any, err: Optional[str]) -> str:
    """Format a metric value; show N/A + reason when missing or non-finite (e.g. NaN)."""
    if raw is None:
        return f"N/A ({err})" if err else "N/A"
    try:
        x = float(raw)
    except (TypeError, ValueError):
        return f"N/A ({err or 'bad value'})"
    if not math.isfinite(x):
        return f"N/A ({err or 'non-finite score'})"
    return f"{x:.3f}"


def render_ragas_metrics_block(result: dict) -> None:
    """RAGAS via ``evaluate()`` (faithfulness, context precision, answer correctness) — see exp_ragas_financebench."""
    if not result or result.get("error"):
        return
    current_ui = int(st.session_state.get("last_rag_ui", 0))
    lr = st.session_state.get("last_rag_backend", "gemini")
    if lr == "mock":
        lr = "gemini"
    if "ragas_eval_backend" not in st.session_state:
        st.session_state.ragas_eval_backend = (
            lr if lr in ("gemini", "openai", "ollama") else "gemini"
        )

    with st.expander(
        "RAGAS scores (context precision · faithfulness · answer correctness)",
        expanded=False,
    ):
        st.caption(
            "Optional **post-hoc** evaluation (same stack as `experiments/exp_ragas_financebench.py`: "
            "`ragas.evaluate` + LangChain LLM/embeddings). "
            "**Answer accuracy** uses `answer_correctness` when a reference is set; without a reference, "
            "context relevance uses **LLM context precision (no reference)**. "
            "Install: `pip install ragas langchain-openai` (Gemini: also `langchain-google-genai`). "
            "If scores show **N/A**, try another evaluator model or check API keys."
        )
        eb_col, em_col = st.columns(2)
        with eb_col:
            st.selectbox(
                "RAGAS evaluator backend",
                options=["gemini", "openai", "ollama"],
                key="ragas_eval_backend",
                help="Gemini: `GEMINI_API_KEY`. OpenAI: `OPENAI_API_KEY` (+ optional `OPENAI_BASE_URL`). "
                "Ollama: local OpenAI-compatible API.",
            )
        with em_col:
            st.text_input(
                "RAGAS evaluator model id",
                key="ragas_eval_model",
                placeholder="Empty = use defaults (see caption)",
            )
        st.caption(
            f"Defaults if model empty: Gemini → `{_default_ragas_eval_model('gemini')}`, "
            f"OpenAI → `{_default_ragas_eval_model('openai')}`, "
            f"Ollama → `{_default_ragas_eval_model('ollama')}`"
        )
        ref = st.text_input(
            "Reference answer (optional, for Answer accuracy)",
            key="ragas_reference_input",
            placeholder="Ground truth answer; leave empty to skip Answer accuracy",
        )
        if st.button("Compute RAGAS scores", type="secondary", key="ragas_compute_btn"):
            backend = str(st.session_state.get("ragas_eval_backend") or "gemini")
            model_raw = (st.session_state.get("ragas_eval_model") or "").strip()
            model = model_raw if model_raw else _default_ragas_eval_model(backend)
            try:
                scores = run_ragas_legacy_evaluate(
                    backend=backend,
                    model=model,
                    user_input=str(result.get("original_question") or ""),
                    response=str(result.get("answer") or ""),
                    retrieved_contexts=list(result.get("final_passages") or []),
                    reference=ref.strip() if ref else None,
                )
                scores["evaluator_backend"] = backend
                scores["evaluator_model"] = model
                st.session_state["last_ragas_scores"] = scores
                st.session_state["ragas_scores_ui_version"] = current_ui
            except Exception as e:
                st.session_state["last_ragas_scores"] = {"error": str(e)}
                st.session_state["ragas_scores_ui_version"] = current_ui
            st.rerun()

        scores = st.session_state.get("last_ragas_scores")
        ver = st.session_state.get("ragas_scores_ui_version")
        if scores and ver is not None and int(ver) == current_ui:
            if scores.get("error"):
                st.error(scores["error"])
            else:
                eb = scores.get("evaluator_backend", "—")
                em = scores.get("evaluator_model", "—")
                st.caption(f"Last run: evaluator **{eb}** / `{em}`")
                lines = ["| Metric | Score |", "| --- | --- |"]
                lines.append(
                    "| Context relevance | "
                    f"{_ragas_score_cell(scores.get('context_relevance'), scores.get('context_relevance_error'))} |"
                )
                lines.append(
                    "| Response groundedness | "
                    f"{_ragas_score_cell(scores.get('response_groundedness'), scores.get('response_groundedness_error'))} |"
                )
                aa = scores.get("answer_accuracy")
                if aa is not None or scores.get("answer_accuracy_error"):
                    lines.append(
                        "| Answer accuracy | "
                        f"{_ragas_score_cell(aa, scores.get('answer_accuracy_error'))} |"
                    )
                elif scores.get("answer_accuracy_note") == "skipped":
                    lines.append("| Answer accuracy | — (no reference) |")
                st.markdown("\n".join(lines))
                if scores.get("context_relevance_note"):
                    st.caption(scores["context_relevance_note"])
                hints = []
                for label, key in (
                    ("Context relevance", "context_relevance_error"),
                    ("Response groundedness", "response_groundedness_error"),
                    ("Answer accuracy", "answer_accuracy_error"),
                ):
                    if scores.get(key):
                        hints.append(f"**{label}:** {scores[key]}")
                if hints:
                    st.info("Details:\n\n" + "\n\n".join(hints))


def render_latency_summary(result: dict) -> None:
    lat = result.get("stage_latencies") or {}
    total = result.get("latency_total_ms")
    if not lat and total is None:
        return
    st.subheader("Stage latency")
    order = [
        ("semantic_cache_lookup_ms", "Semantic cache lookup"),
        ("query_rewrite_ms", "Query rewrite"),
        ("query_decomposition_ms", "Query decomposition"),
        ("dense_retrieval_ms", "Dense retrieval"),
        ("filter_generation_ms", "Filter generation"),
        ("rerank_ms", "Rerank"),
        ("prompt_build_ms", "Prompt build"),
        ("generation_llm_ms", "Generation (LLM)"),
        ("reflection_ms", "Reflection loops"),
        ("citations_ms", "Citation extraction"),
        ("semantic_cache_write_ms", "Semantic cache write"),
    ]
    lines = ["| Stage | ms |", "| --- | --- |"]
    for key, label in order:
        if key in lat:
            lines.append(f"| {label} | {lat[key]:.2f} |")
    if total is not None:
        lines.append(f"| **Total (wall clock)** | **{total:.2f}** |")
    st.markdown("\n".join(lines))
    st.caption(
        "Sequential per-stage timings. **Total** is end-to-end for the whole `run_pipeline` call."
    )


def render_stage_trace(result: dict) -> None:
    stages = result.get("stage_trace", [])
    if not stages:
        return
    st.subheader("Query process trace")
    st.caption(
        "Click each stage to inspect input/output. Titles include **latency_ms** when available."
    )
    for i, stg in enumerate(stages, start=1):
        title = stg.get("title") or stg.get("stage") or f"Stage {i}"
        lm = stg.get("latency_ms")
        if lm is not None:
            title = f"{title} · {lm} ms"
        with st.expander(f"{i}. {title}"):
            st.write("**Input**")
            st.json(stg.get("input", {}), expanded=False)
            st.write("**Output**")
            st.json(stg.get("output", {}), expanded=False)


# --- Benchmarking & observability (``results/experiment_db.sqlite``) ---
GEMINI_25_FLASH_USD_PER_1M = (0.30, 2.50)
GLM45_CLASS_USD_PER_1M = (0.60, 2.20)


def _safe_json_loads(val: Any) -> dict[str, Any]:
    if val is None:
        return {}
    if isinstance(val, dict):
        return val
    if isinstance(val, float) and not math.isfinite(val):
        return {}
    try:
        return json.loads(str(val))
    except Exception:
        return {}


def _runs_leaderboard_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    parsed: List[dict[str, Any]] = []
    for _, row in df.iterrows():
        parsed.append(_safe_json_loads(row.get("metrics_json")))
    mdf = pd.DataFrame(parsed)
    drop_cols = [c for c in ("metrics_json", "config_json") if c in df.columns]
    out = df.drop(columns=drop_cols, errors="ignore").copy()
    if not mdf.empty:
        for c in mdf.columns:
            out[c] = mdf[c].values
    return out


def _log_ui_rag_observation(
    *,
    question: str,
    answer: str,
    result: dict,
    model_config: dict[str, Any],
    reference: str,
    backend: str,
    job_id: Optional[str],
    llm_model: Optional[str] = None,
) -> None:
    init_db()
    prompt = str(result.get("prompt") or "")
    tok = approx_token_count(prompt, answer)
    lat = float(result.get("latency_total_ms") or 0.0)
    ref = (reference or "").strip()
    metrics_computed: dict[str, Any] = {
        "latency_total_ms": lat,
        "n_questions": 1.0,
    }
    gh: Optional[float] = None
    f1v: Optional[float] = None
    em: Optional[float] = None
    if ref:
        am = compute_answer_metrics(answer, ref)
        f1v = float(am["token_f1"])
        gh = float(am["gold_hit"])
        em = float(am["exact_match"])
        metrics_computed["token_f1"] = f1v
        metrics_computed["gold_hit"] = gh
        metrics_computed["exact_match"] = em

    ragas = st.session_state.get("last_ragas_scores")
    ui_ver = st.session_state.get("ragas_scores_ui_version")
    cur_ui = int(st.session_state.get("last_rag_ui", 0))
    if (
        isinstance(ragas, dict)
        and not ragas.get("error")
        and ui_ver is not None
        and int(ui_ver) == cur_ui
    ):
        for k in ("context_relevance", "response_groundedness", "answer_accuracy"):
            if ragas.get(k) is not None:
                try:
                    metrics_computed[k] = float(ragas[k])
                except (TypeError, ValueError):
                    pass
        comp = composite_ragas_score(ragas)
        if comp is not None:
            metrics_computed["ragas_composite"] = float(comp)

    run_id = log_experiment_run(
        source="demo/app.py",
        run_label=f"ui_query:{(job_id or 'local')[:24]}",
        experiment_tag="ui_query",
        embedding_model=str(model_config.get("embedding_model") or DEFAULT_EMBED),
        model_config=model_config,
        latency_ms=lat,
        token_count=tok,
        metrics=metrics_computed,
        llm_backend=backend,
        llm_model=llm_model,
        job_id=job_id,
    )
    log_query_event(
        source="demo/app.py",
        question=question,
        llm_output=answer,
        retrieved_chunks=list(result.get("final_passages") or []),
        latency_ms=lat,
        token_count=tok,
        model_config=model_config,
        reference_answer=ref or None,
        gold_hit=gh,
        token_f1=f1v,
        exact_match=em,
        ragas=ragas if isinstance(ragas, dict) else None,
        run_id=run_id,
        stage_trace=result.get("stage_trace"),
    )


def _ui_benchmark() -> None:
    st.title("RAG experiment observability")
    st.caption(
        "Leaderboard and traces are backed by **results/experiment_db.sqlite** "
        "(populated by **Benchmark** UI logging, **Query** logging, and **experiments/exp_rag_generation.py**)."
    )
    init_db()
    df_runs = fetch_runs_dataframe()
    totals = aggregate_token_totals()
    nq = len(fetch_queries_dataframe())

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Logged aggregate runs", f"{len(df_runs):,}")
    with m2:
        st.metric("Logged query events", f"{nq:,}")
    with m3:
        st.metric("Tokens (runs table)", f"{int(totals['run_table_tokens']):,}")
    with m4:
        st.metric("Tokens (query table)", f"{int(totals['query_table_tokens']):,}")

    tab_lb, tab_fail, tab_cost, tab_jobs = st.tabs(
        [
            "Performance Leaderboard",
            "Failure analysis",
            "Cost calculator",
            "Job metadata (Redis)",
        ]
    )

    with tab_lb:
        st.subheader("Performance Leaderboard")
        st.caption(
            "Compare configurations from scripted sweeps and UI runs. "
            "Mean **latency_ms** on the X-axis is wall-clock for the logged unit (batch mean or single query)."
        )
        flat = _runs_leaderboard_frame(df_runs)
        if flat.empty:
            st.info(
                "No runs yet. Run **exp_rag_generation.py** or enable logging on the **Query** page."
            )
        else:
            show_cols = [
                c
                for c in (
                    "id",
                    "created_at",
                    "source",
                    "run_label",
                    "embedding_model",
                    "chunk_size",
                    "retrieve_k",
                    "final_k",
                    "use_hybrid",
                    "use_rerank",
                    "latency_ms",
                    "token_count",
                    "token_f1",
                    "gold_hit",
                    "exact_match",
                    "latency_total_ms",
                    "n_questions",
                    "ragas_composite",
                    "semantic_cache_hit_rate",
                )
                if c in flat.columns
            ]
            st.dataframe(flat[show_cols], use_container_width=True, hide_index=True)

            plot_rows: List[dict[str, Any]] = []
            for _, row in df_runs.iterrows():
                m = _safe_json_loads(row.get("metrics_json"))
                y_r = m.get("ragas_composite")
                y_f = m.get("token_f1")
                y_plot = y_r if y_r is not None else y_f
                if y_plot is None:
                    continue
                try:
                    yf = float(y_plot)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(yf):
                    continue
                lx = row.get("latency_ms")
                if lx is None:
                    continue
                try:
                    xf = float(lx)
                except (TypeError, ValueError):
                    continue
                cfg = _safe_json_loads(row.get("config_json"))
                label = str(row.get("run_label") or row.get("source") or "")
                rr = int(row.get("use_rerank") or 0)
                plot_rows.append(
                    {
                        "latency_ms": xf,
                        "accuracy_y": yf,
                        "score_kind": (
                            "RAGAS composite" if y_r is not None else "Token F1"
                        ),
                        "label": label[:80],
                        "rerank": "rerank on" if rr else "rerank off",
                    }
                )
            if plot_rows:
                pdf = pd.DataFrame(plot_rows)
                fig = px.scatter(
                    pdf,
                    x="latency_ms",
                    y="accuracy_y",
                    color="rerank",
                    hover_data=["label", "score_kind"],
                    labels={
                        "latency_ms": "Latency (ms)",
                        "accuracy_y": "Accuracy (RAGAS composite or token F1)",
                    },
                    title="Accuracy vs speed trade-off",
                )
                fig.update_layout(height=520)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(
                    "Need runs with both **latency_ms** and **token_f1** or **ragas_composite** to plot."
                )

    with tab_fail:
        st.subheader("Bad-case review")
        st.caption(
            "Rows where **gold hit** missed (gold_hit below 0.5) or **token F1** is below 0.2. "
            "Provide a reference answer on the **Query** page to populate these fields for UI runs."
        )
        df_fail = fetch_queries_dataframe(failed_only=True)
        if not df_fail.empty:
            mask = (df_fail["gold_hit"].fillna(1.0) < 0.5) | (
                df_fail["token_f1"].fillna(1.0) < 0.2
            )
            df_fail = df_fail[mask]
        if df_fail.empty:
            st.info(
                "No failed queries logged yet (or no reference / metrics captured)."
            )
        else:
            ids = [int(x) for x in df_fail["id"].tolist()]
            pick = st.selectbox("Select query id", ids, format_func=lambda i: f"#{i}")
            row = df_fail[df_fail["id"] == pick].iloc[0]
            q1, q2, q3 = st.columns(3)
            with q1:
                st.markdown("**User query**")
                st.text_area(
                    "q",
                    str(row.get("question") or ""),
                    height=220,
                    key="bad_q",
                    label_visibility="collapsed",
                )
            with q2:
                st.markdown("**Retrieved chunks (final context)**")
                chunks = _safe_json_loads(row.get("retrieved_chunks_json"))
                if isinstance(chunks, list):
                    blob = "\n\n---\n\n".join(str(c) for c in chunks)
                else:
                    blob = str(chunks)
                st.text_area(
                    "ctx", blob, height=220, key="bad_ctx", label_visibility="collapsed"
                )
            with q3:
                st.markdown("**LLM output**")
                st.text_area(
                    "out",
                    str(row.get("llm_output") or ""),
                    height=220,
                    key="bad_out",
                    label_visibility="collapsed",
                )
            st.caption(
                f"gold_hit={row.get('gold_hit')} · token_f1={row.get('token_f1')} · "
                f"latency_ms={row.get('latency_ms')} · tokens={row.get('token_count')}"
            )
            with st.expander("Full trace (stage_trace_json)"):
                st.json(_safe_json_loads(row.get("stage_trace_json")), expanded=False)

            fb_col, _ = st.columns([1, 2])
            with fb_col:
                tag = st.selectbox(
                    "Human feedback",
                    options=["(none)"] + list(FAILURE_FEEDBACK_LABELS),
                    key="fail_fb_pick",
                )
                if st.button("Save feedback tag", key="fail_fb_save"):
                    if tag == "(none)":
                        st.warning("Pick a label first.")
                    else:
                        update_query_feedback(int(pick), tag)
                        st.success("Saved.")
                        st.rerun()

    with tab_cost:
        st.subheader("Cost calculator (RMB estimates)")
        st.caption(
            "Rates default to **Gemini 2.5 Flash** (~$0.30 / $2.50 per 1M tok) and **GLM-4.5-class** (~$0.60 / $2.20 per 1M tok on Z.AI). "
            "Adjust FX and split to match your billing."
        )
        fx = st.number_input(
            "USD → CNY", min_value=0.01, value=7.20, step=0.05, format="%.2f"
        )
        in_frac = st.slider(
            "Assumed share of tokens that are prompt (input)", 0.0, 1.0, 0.72, 0.01
        )
        out_frac = max(0.0, 1.0 - in_frac)
        gem_in = GEMINI_25_FLASH_USD_PER_1M[0] * fx
        gem_out = GEMINI_25_FLASH_USD_PER_1M[1] * fx
        glm_in = GLM45_CLASS_USD_PER_1M[0] * fx
        glm_out = GLM45_CLASS_USD_PER_1M[1] * fx

        def _cost_million_scale(total_tokens: float, rin: float, rout: float) -> float:
            t = max(0.0, float(total_tokens))
            return (t * in_frac / 1e6) * rin + (t * out_frac / 1e6) * rout

        tr = totals["run_table_tokens"] + totals["query_table_tokens"]
        g_cost = _cost_million_scale(tr, gem_in, gem_out)
        z_cost = _cost_million_scale(tr, glm_in, glm_out)
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Est. Gemini 2.5 Flash (RMB)", f"¥{g_cost:.4f}")
        with c2:
            st.metric("Est. GLM-4.5-class (RMB)", f"¥{z_cost:.4f}")
        st.caption(
            f"Combined token basis: **{int(tr):,}** (runs + query tables; heuristic tokens from char/4 estimates)."
        )

    with tab_jobs:
        st.subheader("Ingest job metadata (Redis + MinIO)")
        st.caption(
            "Redis: live pipeline status (``ingest:job:{{id}}``). "
            "MinIO: persisted artifact index (same as **Library**)."
        )
        jid = st.text_input(
            "Job ID",
            value=st.session_state.get("last_job_id", ""),
            key="bench_redis_jid",
        )
        if st.button("Fetch Redis status", key="bench_redis_btn"):
            if not (jid or "").strip():
                st.warning("Enter a job id.")
            else:
                try:
                    job_st = RedisJobStore().get_status(jid.strip())
                    if job_st is None:
                        st.warning("No key found for this job id.")
                    else:
                        st.json(asdict(job_st))
                except Exception as e:
                    st.error(str(e))
        st.divider()
        st.markdown("**MinIO ingest jobs (metadata table)**")
        try:
            mrows = _minio_store().list_ingest_jobs_table()
            st.dataframe(
                mrows[:200] if len(mrows) > 200 else mrows,
                use_container_width=True,
                hide_index=True,
            )
        except Exception as e:
            st.warning(f"MinIO: {e}")


def unload_index() -> None:
    st.session_state.corpus_chunks = []
    st.session_state.faiss_index = None
    st.session_state.source_label = None
    st.session_state.ingest_meta = {}
    st.session_state.loaded_job_id = None
    st.session_state.loaded_job_ids = []
    st.session_state.loaded_milvus_collection = None
    st.session_state.pop("last_rag", None)
    st.session_state.pop("last_question", None)
    st.session_state.query_history = []
    st.session_state.qa_messages = []


def load_job(job_id: str) -> None:
    chunks, meta = load_ingest_from_minio(job_id.strip())
    st.session_state.corpus_chunks = chunks
    st.session_state.faiss_index = None
    st.session_state.ingest_meta = meta
    fn = meta.get("filename") or "document"
    st.session_state.source_label = f"{job_id.strip()} · {fn}"
    st.session_state.loaded_job_id = job_id.strip()
    st.session_state.loaded_job_ids = [job_id.strip()]
    st.session_state.loaded_milvus_collection = (
        str(meta.get("milvus_collection") or "").strip() or None
    )
    st.session_state.last_job_id = job_id.strip()
    st.session_state.pop("last_rag", None)
    st.session_state.pop("last_question", None)
    st.session_state.query_history = []
    st.session_state.qa_messages = []
    if st.session_state.loaded_milvus_collection:
        load_res = _milvus_store().load_collection(
            collection_name=st.session_state.loaded_milvus_collection,
            release_others=True,
        )
        if not load_res.get("loaded"):
            raise RuntimeError(f"Milvus collection load failed: {load_res}")


def load_jobs_multi(job_ids: List[str]) -> None:
    """Load several ingest jobs so retrieval can query across Milvus with multiple job_id filters."""
    ids = [j.strip() for j in job_ids if j and j.strip()]
    if not ids:
        raise ValueError("no job ids")
    if len(ids) > 32:
        raise ValueError("at most 32 jobs")
    all_chunks: List[str] = []
    metas: List[dict[str, Any]] = []
    cols_seen: set[str] = set()
    for jid in ids:
        chunks, meta = load_ingest_from_minio(jid)
        all_chunks.extend(chunks)
        metas.append(meta)
        coll = str(meta.get("milvus_collection") or "").strip()
        if coll and coll not in cols_seen:
            load_res = _milvus_store().load_collection(
                collection_name=coll,
                release_others=False,
            )
            if not load_res.get("loaded") and load_res.get("exists", True):
                raise RuntimeError(f"Milvus collection load failed: {load_res}")
            cols_seen.add(coll)
    st.session_state.corpus_chunks = all_chunks
    st.session_state.faiss_index = None
    st.session_state.ingest_meta = metas[0] if metas else {}
    st.session_state.loaded_job_ids = ids
    st.session_state.loaded_job_id = ids[0]
    st.session_state.loaded_milvus_collection = None
    short = [f"{j[:8]}…" if len(j) > 12 else j for j in ids]
    st.session_state.source_label = f"{len(ids)} jobs · " + ", ".join(short)
    st.session_state.last_job_id = ids[-1]
    st.session_state.pop("last_rag", None)
    st.session_state.pop("last_question", None)
    st.session_state.query_history = []
    st.session_state.qa_messages = []


def _init_session() -> None:
    if "corpus_chunks" not in st.session_state:
        st.session_state.corpus_chunks = []
    if "faiss_index" not in st.session_state:
        st.session_state.faiss_index = None
    if "source_label" not in st.session_state:
        st.session_state.source_label = None
    if "ingest_meta" not in st.session_state:
        st.session_state.ingest_meta = {}
    if "last_job_id" not in st.session_state:
        st.session_state.last_job_id = ""
    if "loaded_job_id" not in st.session_state:
        st.session_state.loaded_job_id = None
    if "loaded_job_ids" not in st.session_state:
        st.session_state.loaded_job_ids = []
    if "loaded_milvus_collection" not in st.session_state:
        st.session_state.loaded_milvus_collection = None
    if "nav" not in st.session_state:
        st.session_state.nav = "ingest"
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    if "qa_messages" not in st.session_state:
        st.session_state.qa_messages = []


def _sidebar_nav() -> str:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] { min-width: 14rem; }
        section[data-testid="stSidebar"] > div { padding-top: 1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.sidebar:
        st.markdown("### RAG Lab")
        nav = st.session_state.nav
        b1 = st.button(
            "Ingest",
            key="nav_ingest",
            use_container_width=True,
            type="primary" if nav == "ingest" else "secondary",
        )
        b2 = st.button(
            "Query",
            key="nav_query",
            use_container_width=True,
            type="primary" if nav == "query" else "secondary",
        )
        b3 = st.button(
            "Library",
            key="nav_library",
            use_container_width=True,
            type="primary" if nav == "library" else "secondary",
        )
        b4 = st.button(
            "Benchmark",
            key="nav_benchmark",
            use_container_width=True,
            type="primary" if nav == "benchmark" else "secondary",
        )
        if b1:
            st.session_state.nav = "ingest"
        if b2:
            st.session_state.nav = "query"
        if b3:
            st.session_state.nav = "library"
        if b4:
            st.session_state.nav = "benchmark"
        st.divider()
        st.caption("MinIO + Redis · SQLite experiments")
    return st.session_state.nav


def main() -> None:
    st.set_page_config(
        page_title="RAG Lab", layout="wide", initial_sidebar_state="expanded"
    )
    _init_session()
    if not st.session_state.get("_prometheus_http_started"):
        try:
            from prometheus_client import start_http_server

            _prom_port = int(os.environ.get("PROMETHEUS_METRICS_PORT", "8000"))
            start_http_server(_prom_port)
            st.session_state._prometheus_http_started = True
            logging.getLogger(__name__).info(
                "Prometheus metrics: http://0.0.0.0:%s/metrics", _prom_port
            )
        except OSError as e:
            st.session_state._prometheus_http_started = True
            logging.getLogger(__name__).warning(
                "Prometheus metrics server not started (port may be in use): %s", e
            )
    _sidebar_nav()

    nav = st.session_state.nav
    if nav == "benchmark":
        _ui_benchmark()
        return
    st.title("Document pipeline")
    if nav == "ingest":
        _ui_ingest_pipeline()
    elif nav == "query":
        _ui_query()
    else:
        _ui_library()


def _ui_ingest_pipeline() -> None:
    st.caption(
        "Upload → filter → extract → chunk & summarize → embed/upsert (**Milvus**) → **MinIO** + **Redis**"
    )

    st.subheader("1 — Upload & page filter")
    up_col, filt_col = st.columns(2)
    with up_col:
        uploaded_files = st.file_uploader(
            "Documents",
            type=["pdf", "txt", "md"],
            key="ing_upl",
            accept_multiple_files=True,
            help=(
                "Upload one or more documents; **each file is a separate ingest job** (its own job_id). "
                "To query several together, use Query → select multiple jobs → Load."
            ),
        )
    with filt_col:
        page_spec = st.text_input(
            "Page filter (PDF)",
            "",
            placeholder="1,3,5-8 — empty = all",
            help="Optional PDF page selection. Empty means all pages. Ignored for non-PDF files.",
        )

    st.subheader("2 — Extraction")
    extraction = st.radio(
        "Depth",
        ("shallow", "full"),
        horizontal=True,
        help="shallow is faster extraction; full is more thorough but slower.",
    )

    st.subheader("3 — Chunking & summarization")
    md1, md2 = st.columns(2)
    with md1:
        ingest_doc_date = st.text_input(
            "Document date (optional, YYYY-MM-DD)",
            value="",
            key="ing_doc_date",
            help="Stored per chunk as Milvus `doc_date`. Overrides PDF metadata when set.",
        )
    with md2:
        ingest_doc_section = st.text_input(
            "Section label",
            value="document",
            key="ing_doc_section",
            help='Logical section for all chunks (flat ingest), e.g. "document" or "methods".',
        )
    c1, c2, c3 = st.columns(3)
    with c1:
        chunk_size = st.number_input(
            "Chunk size",
            128,
            2048,
            512,
            64,
            help="Approximate characters/tokens per chunk. Larger chunks carry more context but reduce granularity.",
        )
        chunk_overlap = st.number_input(
            "Chunk overlap",
            0,
            512,
            64,
            32,
            help="Shared text between adjacent chunks to reduce boundary information loss.",
        )
    with c2:
        strat = st.selectbox(
            "Summarization strategy",
            ("single", "hierarchical", "iterative"),
            help="single: no summaries; hierarchical: per-chunk summaries; iterative: per-chunk + global abstract.",
        )
    with c3:
        summ_backend = st.selectbox(
            "Summarizer LLM (if not single)",
            ["mock", "gemini", "openai", "ollama"],
            help="LLM backend used for summary generation when strategy is not single.",
        )
    with st.expander("Milvus index options", expanded=False):
        ingest_preset = st.selectbox(
            "Milvus preset",
            ["fast", "balanced", "high_recall", "custom"],
            index=1,
            help="Preset fills sensible defaults; choose custom to tune each option manually.",
        )
        pvals = _milvus_preset_values(
            ingest_preset if ingest_preset != "custom" else "balanced"
        )
        mi1, mi2, mi3 = st.columns(3)
        with mi1:
            milvus_index_type = st.selectbox(
                "Index type",
                MILVUS_INDEX_TYPES,
                index=MILVUS_INDEX_TYPES.index(str(pvals["index_type"])),
                help="AUTOINDEX lets Milvus choose; IVF_FLAT uses cluster lists; HNSW uses graph search.",
            )
            milvus_metric_type = st.selectbox(
                "Metric",
                MILVUS_METRICS,
                index=MILVUS_METRICS.index(str(pvals["metric_type"])),
                help="Similarity metric for vector scoring: COSINE, IP (dot product), or L2 (Euclidean distance).",
            )
            milvus_upsert_batch_size = st.slider(
                "Upsert batch size",
                32,
                1024,
                int(pvals["upsert_batch_size"]),
                32,
                help="Number of rows inserted per upsert batch. Larger batches are faster but use more memory.",
            )
        with mi2:
            milvus_ivf_nlist = st.slider(
                "IVF nlist",
                64,
                4096,
                int(pvals["ivf_nlist"]),
                64,
                disabled=milvus_index_type != "IVF_FLAT",
                help="IVF cluster count. Larger values can improve recall but increase build and memory cost.",
            )
        with mi3:
            milvus_hnsw_m = st.slider(
                "HNSW M",
                4,
                64,
                int(pvals["hnsw_m"]),
                2,
                disabled=milvus_index_type != "HNSW",
                help="HNSW connectivity. Higher M often improves quality with higher memory/build cost.",
            )
            milvus_hnsw_ef_construction = st.slider(
                "HNSW efConstruction",
                32,
                512,
                int(pvals["hnsw_ef_construction"]),
                8,
                disabled=milvus_index_type != "HNSW",
                help="HNSW build-time search depth. Higher values usually improve recall but slow indexing.",
            )

    st.subheader("4 — Run → MinIO + Redis")
    if st.button("Run full pipeline", type="primary"):
        if not uploaded_files:
            st.error("Upload at least one document first.")
        else:
            try:
                cfg = merge_ingest_with_dict(
                    load_ingest_config(),
                    {
                        "chunk_size": int(chunk_size),
                        "chunk_overlap": int(chunk_overlap),
                        "extraction": extraction,
                        "summarization": strat,
                        "milvus_index_type": str(milvus_index_type),
                        "milvus_metric_type": str(milvus_metric_type),
                        "milvus_ivf_nlist": int(milvus_ivf_nlist),
                        "milvus_hnsw_m": int(milvus_hnsw_m),
                        "milvus_hnsw_ef_construction": int(milvus_hnsw_ef_construction),
                        "milvus_upsert_batch_size": int(milvus_upsert_batch_size),
                        "doc_date": (ingest_doc_date or "").strip() or None,
                        "doc_section": (ingest_doc_section or "").strip(),
                    },
                ).to_pipeline_dataclass()
                summ = ingest_summarizer(summ_backend) if strat != "single" else None
                successes: List[dict[str, Any]] = []
                failures: List[dict[str, str]] = []
                total = len(uploaded_files)
                progress_bar = st.progress(
                    0.0, text=f"Starting ingest for {total} file(s)…"
                )
                status_box = st.empty()
                for i, up in enumerate(uploaded_files, start=1):
                    status_box.info(f"Running pipeline ({i}/{total}): {up.name}")
                    try:
                        meta = run_document_ingest(
                            filename=up.name,
                            raw_bytes=up.getvalue(),
                            page_filter_spec=page_spec,
                            config=cfg,
                            summarizer=summ,
                        )
                        successes.append(meta)
                        st.session_state["last_job_id"] = meta["job_id"]
                    except Exception as e:
                        failures.append({"filename": up.name, "error": str(e)})
                    progress_bar.progress(
                        i / max(1, total),
                        text=f"Completed {i}/{total} file(s)",
                    )

                if failures:
                    status_box.warning(
                        f"Finished with errors: {len(successes)} succeeded, {len(failures)} failed."
                    )
                else:
                    status_box.success(f"Finished ingest for {len(successes)} file(s).")

                if successes:
                    st.success(f"Ingested {len(successes)} file(s) successfully.")
                    st.json(
                        [
                            {
                                "job_id": m.get("job_id"),
                                "filename": m.get("filename"),
                                "n_chunks": m.get("n_chunks"),
                                "milvus_collection": m.get("milvus_collection"),
                            }
                            for m in successes
                        ]
                    )
                if failures:
                    st.error(f"{len(failures)} file(s) failed.")
                    st.json(failures)
            except Exception as e:
                st.exception(e)

    st.divider()
    st.subheader("Redis job status")
    jid = st.text_input("Job ID", value=st.session_state.get("last_job_id", ""))
    if jid and st.button("Refresh Redis status"):
        try:
            st.write(RedisJobStore().get_status(jid.strip()))
        except Exception as e:
            st.warning(str(e))


def _ui_library() -> None:
    st.caption("Jobs stored in MinIO (prefix = **job_id**).")
    if st.button("Refresh list"):
        st.session_state.library_refresh = (
            st.session_state.get("library_refresh", 0) + 1
        )

    try:
        rows = _minio_store().list_ingest_jobs_table()
    except Exception as e:
        st.error(f"MinIO: {e}")
        return

    if not rows:
        st.info("No jobs yet. Run an **Ingest** first.")
        return

    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
        column_config={
            "job_id": st.column_config.TextColumn("job_id", width="large"),
            "filename": st.column_config.TextColumn("filename"),
            "n_chunks": st.column_config.NumberColumn("chunks"),
            "embedding_model": st.column_config.TextColumn("embedding"),
            "summarization": st.column_config.TextColumn("summary strat"),
            "extraction": st.column_config.TextColumn("extraction"),
            "milvus_collection": st.column_config.TextColumn(
                "milvus collection", width="large"
            ),
            "milvus_index_type": st.column_config.TextColumn("index"),
            "milvus_metric_type": st.column_config.TextColumn("metric"),
        },
    )
    st.caption(f"**{len(rows)}** job(s) in bucket `{load_minio_settings().bucket}`.")

    st.divider()
    st.subheader("Remove from library")
    st.caption(
        "Deletes the job prefix in MinIO (chunks + metadata). "
        "Optional: clears Redis ingest status. If this job is loaded in **Query**, memory is unloaded."
    )
    options = [r["job_id"] for r in rows]
    label_by_id = {r["job_id"]: _job_label_for_select(r) for r in rows}
    to_remove = st.selectbox(
        "Job to remove",
        options=options,
        format_func=lambda jid: label_by_id.get(jid, jid),
        key="library_remove_select",
    )
    confirm = st.checkbox(
        "I understand this permanently deletes stored artifacts for this job",
        key="library_remove_confirm",
    )
    if st.button("Remove from library", type="primary", disabled=not confirm):
        try:
            store = _minio_store()
            n_obj = store.delete_job(to_remove)
            still_there = to_remove in store.list_job_ids()
            if n_obj == 0 and still_there:
                st.error(
                    "No objects were deleted but the job prefix still exists. "
                    "Check MinIO permissions or console."
                )
            else:
                try:
                    RedisJobStore().delete_status(to_remove)
                except Exception as redis_err:
                    st.caption(f"Redis cleanup skipped: {redis_err}")
                if st.session_state.get("loaded_job_id") == to_remove:
                    unload_index()
                if n_obj > 0:
                    st.success(
                        f"Removed **`{to_remove}`** from MinIO ({n_obj} object(s)). "
                        "Milvus vectors are not deleted automatically."
                    )
                else:
                    st.info(
                        f"No MinIO objects for **`{to_remove}`** (already removed). "
                        "Redis status cleared if present."
                    )
                st.rerun()
        except Exception as e:
            st.exception(e)


def _job_label_for_select(row: dict[str, Any]) -> str:
    jid = row["job_id"]
    fn = row.get("filename", "?")
    nc = row.get("n_chunks", "?")
    short = f"{jid[:8]}…" if len(jid) > 12 else jid
    return f"{short} · {fn} · {nc} chunks"


def _ui_query() -> None:
    st.caption(
        "Attach one or more ingest jobs from MinIO, then ask questions. "
        "Multi-select + **Load** queries across all selected docs (Milvus). Retrieval is Milvus-only."
    )

    try:
        jobs = _minio_store().list_ingest_jobs_table()
    except Exception as e:
        st.error(f"MinIO: {e}")
        return

    st.subheader("Loaded job context")
    _lj = st.session_state.get("loaded_job_ids") or []
    if not _lj and st.session_state.get("loaded_job_id"):
        _lj = [st.session_state.loaded_job_id]
    if st.session_state.loaded_job_id and len(st.session_state.corpus_chunks) > 0:
        if len(_lj) > 1:
            st.success(
                f"Loaded **{len(_lj)} jobs** — {len(st.session_state.corpus_chunks)} chunks total · "
                f"{st.session_state.source_label}"
            )
            st.caption(
                "Queries use **combined Milvus retrieval** across the selected jobs."
            )
        else:
            st.success(
                f"Loaded **{st.session_state.loaded_job_id}** — "
                f"{len(st.session_state.corpus_chunks)} chunks · {st.session_state.source_label}"
            )
            if st.session_state.get("loaded_milvus_collection"):
                st.caption(
                    f"Milvus collection loaded: `{st.session_state.loaded_milvus_collection}`"
                )
    else:
        st.warning("No job loaded.")

    st.subheader("Attach & load")
    attach = st.checkbox(
        "Attach job for querying",
        value=True,
        help="Checked: **Load** downloads the selected job metadata/chunks. Unchecked: **Load** clears loaded job context.",
    )

    options: List[str] = [r["job_id"] for r in jobs]
    label_by_id = {r["job_id"]: _job_label_for_select(r) for r in jobs}

    if not options:
        st.info("No jobs in MinIO yet — use **Ingest** first.")
        selected: List[str] = []
    else:
        default_sel: List[str] = []
        prev_multi = st.session_state.get("loaded_job_ids") or []
        if prev_multi and isinstance(prev_multi, list):
            default_sel = [x for x in prev_multi if x in options]
        elif st.session_state.get("loaded_job_id") in options:
            default_sel = [st.session_state.loaded_job_id]
        selected = st.multiselect(
            "Job(s)",
            options=options,
            default=default_sel,
            format_func=lambda jid: label_by_id.get(jid, jid),
            help="Select one job, or multiple jobs to query across all of them (Milvus).",
            max_selections=32,
        )

    if st.button("Load", type="primary"):
        if attach:
            if not selected:
                st.error("Select at least one job to load.")
            else:
                try:
                    if len(selected) == 1:
                        load_job(selected[0])
                        st.success(f"Loaded `{selected[0]}`")
                    else:
                        load_jobs_multi(selected)
                        st.success(f"Loaded {len(selected)} jobs for multi-doc query.")
                    st.rerun()
                except Exception as e:
                    st.exception(e)
        else:
            unload_index()
            st.success("Job unloaded from memory.")
            st.rerun()

    ready = (
        st.session_state.loaded_job_id is not None
        and len(st.session_state.corpus_chunks) > 0
    )
    ingest_meta = st.session_state.get("ingest_meta", {}) or {}
    default_milvus_index_type = str(ingest_meta.get("milvus_index_type", "AUTOINDEX"))
    default_milvus_metric_type = str(ingest_meta.get("milvus_metric_type", "COSINE"))
    query_preset_default = "balanced"

    filter_doc_date_min_ui = ""
    filter_doc_date_max_ui = ""
    filter_source_type_ui = ""
    filter_section_ui = ""

    chat_bar_options_col, _ = st.columns([1, 6])
    with chat_bar_options_col:
        with st.popover("Options", disabled=not ready):
            st.caption("Attached chat options")
            vdb_backend = "milvus"
            retrieval_mode = "dense"
            st.caption("Vector backend: **milvus**")
            query_preset = st.selectbox(
                "Milvus search preset",
                ["fast", "balanced", "high_recall", "custom"],
                index=["fast", "balanced", "high_recall", "custom"].index(
                    query_preset_default
                ),
                disabled=not ready,
                help="Preset sets query-time ANN params. Custom lets you override manually.",
            )
            qvals = _milvus_preset_values(
                query_preset if query_preset != "custom" else "balanced"
            )
            milvus_index_type = (
                default_milvus_index_type
                if default_milvus_index_type in MILVUS_INDEX_TYPES
                else "AUTOINDEX"
            )
            milvus_metric_type = (
                default_milvus_metric_type
                if default_milvus_metric_type in MILVUS_METRICS
                else "COSINE"
            )
            st.caption(
                f"Loaded collection index: **{milvus_index_type}** · metric: **{milvus_metric_type}**"
            )
            milvus_ivf_nprobe = st.slider(
                "Milvus IVF nprobe",
                1,
                256,
                int(qvals["ivf_nprobe"]),
                1,
                disabled=not ready or milvus_index_type != "IVF_FLAT",
                help="IVF query breadth (number of clusters searched). Higher improves recall but adds latency.",
            )
            milvus_hnsw_ef = st.slider(
                "Milvus HNSW ef",
                8,
                512,
                int(qvals["hnsw_ef"]),
                8,
                disabled=not ready or milvus_index_type not in ("HNSW", "AUTOINDEX"),
                help="HNSW query breadth. Higher ef improves recall at the cost of slower search.",
            )
            dbg_col, _ = st.columns([1.8, 4.2])
            with dbg_col:
                if st.button("Show active Milvus index config", disabled=not ready):
                    try:
                        info = _milvus_store().describe_collection(
                            collection_name=st.session_state.get(
                                "loaded_milvus_collection"
                            )
                        )
                        st.json(info)
                    except Exception as e:
                        st.error(f"Failed to read Milvus index config: {e}")
            fusion_list_k = st.slider(
                "Hybrid fusion list K",
                5,
                100,
                30,
                5,
                disabled=True,
                help="Candidates taken from each retriever (BM25 and dense) before RRF merge.",
            )
            rrf_k = st.slider(
                "RRF k",
                10,
                200,
                60,
                5,
                disabled=True,
                help="RRF smoothing constant; larger values flatten rank contribution.",
            )
            st.divider()
            st.caption(
                "Metadata filters (same fields as ``POST /v1/rag/query`` ``filter_*``)"
            )
            filter_doc_date_min_ui = st.text_input(
                "filter_doc_date_min",
                value="",
                disabled=not ready,
                help="ISO date YYYY-MM-DD — lower bound on stored ``doc_date`` (excludes ``unknown``).",
            )
            filter_doc_date_max_ui = st.text_input(
                "filter_doc_date_max",
                value="",
                disabled=not ready,
                help="ISO date YYYY-MM-DD — upper bound on stored ``doc_date``.",
            )
            filter_source_type_ui = st.text_input(
                "filter_source_type",
                value="",
                disabled=not ready,
                help="Exact match, e.g. pdf, txt, md (from ingest filename).",
            )
            filter_section_ui = st.text_input(
                "filter_section",
                value="",
                disabled=not ready,
                help='Exact section label (ingest default is usually "document").',
            )
            retrieve_k = st.slider(
                "Retrieval top-K",
                3,
                30,
                10,
                disabled=not ready,
                help="How many chunks to retrieve before rerank/filter stages.",
            )
            final_k = st.slider(
                "Final passages to LLM",
                1,
                10,
                3,
                disabled=not ready,
                help="How many passages are included in the final LLM prompt context.",
            )
            use_semantic_cache = st.checkbox(
                "Redis semantic cache",
                value=True,
                disabled=not ready,
                help="Reuse previous answers for semantically similar questions to reduce latency and cost.",
            )
            semantic_cache_threshold = st.slider(
                "Semantic cache threshold",
                0.70,
                0.99,
                0.93,
                0.01,
                disabled=not ready or not use_semantic_cache,
                help="Minimum cosine similarity to treat a cached answer as a hit.",
            )
            use_filter_generation = st.checkbox(
                "Filter generation (LLM include/exclude keywords)",
                value=False,
                disabled=not ready,
                help="Ask an LLM to generate include/exclude keyword filters before final context selection.",
            )
            min_filter_keep = st.slider(
                "Min chunks to keep after filter",
                1,
                10,
                3,
                disabled=not ready or not use_filter_generation,
                help="Guarantees at least this many chunks survive filtering.",
            )
            use_rerank = st.checkbox(
                "Cross-encoder rerank",
                value=True,
                disabled=not ready,
                help="Re-score retrieved chunks with a cross-encoder for better ordering.",
            )
            use_query_rewrite = st.checkbox(
                "Query rewrite",
                value=False,
                disabled=not ready,
                help="Rewrite the user query for better retrieval quality.",
            )
            use_query_decomposition = st.checkbox(
                "Query decomposition",
                value=False,
                disabled=not ready,
                help="Split complex questions into multiple focused retrieval sub-queries.",
            )
            max_subqueries = st.slider(
                "Max sub-queries",
                2,
                6,
                3,
                disabled=not ready or not use_query_decomposition,
                help="Maximum number of generated sub-queries.",
            )
            use_reflection_loops = st.checkbox(
                "Reflection loops (answer critique + follow-up retrieval)",
                value=False,
                disabled=not ready,
                help="Iteratively critique answers and run additional retrieval if needed.",
            )
            max_reflection_loops = st.slider(
                "Max reflection loops",
                1,
                4,
                2,
                disabled=not ready or not use_reflection_loops,
                help="Upper bound on reflection iterations.",
            )
            require_citations = st.checkbox(
                "Require citation markers in answer ([1], [2])",
                value=True,
                disabled=not ready,
                help="Encourage answers to include citation markers tied to retrieved passages.",
            )
            use_session_history = st.checkbox(
                "Use session history in rewrite/decomposition",
                value=True,
                disabled=not ready,
                help="Include recent chat turns when rewriting/decomposing the query.",
            )
            history_turns = st.slider(
                "History turns",
                1,
                8,
                3,
                disabled=not ready or not use_session_history,
                help="How many recent turns to include as history context.",
            )
            template_key = st.selectbox(
                "Prompt template",
                list(PROMPT_TEMPLATES.keys()),
                disabled=not ready,
                help="Prompt formatting strategy used for final answer generation.",
            )
            max_context_chars = st.number_input(
                "Max context chars",
                500,
                32000,
                6000,
                step=500,
                disabled=not ready,
                help="Maximum total context length sent to the LLM after retrieval/rerank.",
            )
            truncation = st.selectbox(
                "Truncation",
                ["head", "tail", "middle"],
                disabled=not ready,
                help="How to trim context if it exceeds max context chars.",
            )
            backend = st.selectbox(
                "LLM",
                ["mock", "gemini", "openai", "ollama"],
                disabled=not ready,
                help="Answer-generation backend used for final response.",
            )

    # Defaults when options popover is disabled (no index loaded yet)
    if not ready:
        vdb_backend = "milvus"
        retrieval_mode = "dense"
        milvus_index_type = (
            default_milvus_index_type
            if default_milvus_index_type in MILVUS_INDEX_TYPES
            else "AUTOINDEX"
        )
        milvus_metric_type = (
            default_milvus_metric_type
            if default_milvus_metric_type in MILVUS_METRICS
            else "COSINE"
        )
        milvus_ivf_nprobe = 32
        milvus_hnsw_ef = 64
        fusion_list_k = 30
        rrf_k = 60
        retrieve_k = 10
        final_k = 3
        use_semantic_cache = False
        semantic_cache_threshold = 0.93
        use_filter_generation = False
        min_filter_keep = 3
        use_rerank = True
        use_query_rewrite = False
        use_query_decomposition = False
        max_subqueries = 3
        use_reflection_loops = False
        max_reflection_loops = 2
        require_citations = True
        use_session_history = True
        history_turns = 3
        template_key = list(PROMPT_TEMPLATES.keys())[0]
        max_context_chars = 6000
        truncation = "head"
        backend = "mock"
        filter_doc_date_min_ui = ""
        filter_doc_date_max_ui = ""
        filter_source_type_ui = ""
        filter_section_ui = ""

    if not ready:
        st.info("Attach a job and click **Load** to enable questions.")
        return

    st.divider()
    with st.expander("Experiment logging (SQLite benchmark DB)", expanded=False):
        st.caption(
            "Appends to **results/experiment_db.sqlite** for the **Benchmark** leaderboard, cost view, and bad-case review."
        )
        st.checkbox(
            "Log each query to the experiment DB", value=True, key="exp_log_queries"
        )
        st.text_area(
            "Reference answer (optional — enables token F1, gold hit, and failure filters)",
            key="exp_reference_answer",
            height=72,
        )

    result = st.session_state.get("last_rag")
    convo_col, trace_col = st.columns([1.35, 1.0])
    with convo_col:
        st.subheader("Conversation")
        _render_qa_scroll_block(st.session_state.get("qa_messages", []))
    with trace_col:
        st.subheader("Trace & diagnostics")
        if result:
            ui_id = st.session_state.get("last_rag_ui", 0)
            fk = int(st.session_state.get("last_final_k", 3))
            render_rag_trace(result, ui_id=ui_id, final_k=fk)
            render_latency_summary(result)
            render_stage_trace(result)
            render_ragas_metrics_block(result)
        else:
            st.info("Trace appears after the first answer.")

    question = st.chat_input("Ask about the document…")

    if question and question.strip():
        embedder = cached_embedder(DEFAULT_EMBED)
        reranker = cached_reranker(DEFAULT_RERANK) if use_rerank else None
        try:
            gen = make_generator(backend)
        except Exception as e:
            st.error(f"Generator: {e}")
            return
        milvus_store = None
        try:
            with st.spinner("Connecting to Milvus…"):
                milvus_store = _milvus_store()
        except Exception as e:
            st.error(
                f"Milvus backend unavailable: {e}\n\n"
                "Start Milvus (e.g. docker) and set `MILVUS_URI` in `.env`."
            )
            return
        semantic_cache = None
        _loaded_ids = st.session_state.get("loaded_job_ids") or []
        if not _loaded_ids and st.session_state.get("loaded_job_id"):
            _loaded_ids = [st.session_state.loaded_job_id]
        if use_semantic_cache and len(_loaded_ids) <= 1:
            try:
                ns = st.session_state.loaded_job_id or "default"
                semantic_cache = RedisSemanticCache(namespace=ns)
            except Exception as e:
                st.warning(f"Redis semantic cache disabled: {e}")
                semantic_cache = None
        elif use_semantic_cache and len(_loaded_ids) > 1:
            st.caption("Semantic cache off when multiple jobs are loaded.")
        hist = (
            _history_to_text(
                st.session_state.get("query_history", []), max_turns=int(history_turns)
            )
            if use_session_history
            else "None"
        )
        bm25_resources: Optional[BM25Resources] = None
        if vdb_backend == "faiss" and retrieval_mode == "hybrid":
            with st.spinner("Building BM25 resources for hybrid retrieval…"):
                bm25_resources = build_bm25_resources(st.session_state.corpus_chunks)

        with st.spinner("Retrieving & generating…"):
            pipe_cfg = RAGPipelineConfig(
                retrieval=RetrievalConfig(
                    retrieve_k=int(retrieve_k),
                    final_k=int(final_k),
                    mode=retrieval_mode,  # type: ignore[arg-type]
                    fusion_list_k=int(fusion_list_k),
                    rrf_k=int(rrf_k),
                    vdb_backend=vdb_backend,  # type: ignore[arg-type]
                ),
                milvus=MilvusRuntimeConfig(
                    job_id=st.session_state.loaded_job_id,
                    job_ids=_loaded_ids if len(_loaded_ids) > 1 else None,
                    collection=(
                        None
                        if len(_loaded_ids) > 1
                        else st.session_state.get("loaded_milvus_collection")
                    ),
                    index_type=str(milvus_index_type),
                    metric_type=str(milvus_metric_type),
                    ivf_nprobe=int(milvus_ivf_nprobe),
                    hnsw_ef=int(milvus_hnsw_ef),
                ),
                metadata=MetadataFilterConfig(
                    doc_date_min=(filter_doc_date_min_ui or "").strip() or None,
                    doc_date_max=(filter_doc_date_max_ui or "").strip() or None,
                    source_type=(filter_source_type_ui or "").strip() or None,
                    section=(filter_section_ui or "").strip() or None,
                ),
                features=PipelineFeaturesConfig(
                    use_query_rewrite=use_query_rewrite,
                    use_query_decomposition=use_query_decomposition,
                    max_subqueries=int(max_subqueries),
                    use_filter_generation=use_filter_generation,
                    min_filter_keep=int(min_filter_keep),
                    use_reflection_loops=use_reflection_loops,
                    max_reflection_loops=int(max_reflection_loops),
                    require_citations=require_citations,
                    use_rerank=use_rerank,
                ),
                prompt=PromptConfig(
                    template_key=template_key,
                    max_context_chars=int(max_context_chars),
                    truncation=str(truncation),  # type: ignore[arg-type]
                ),
                semantic_cache=SemanticCacheConfig(
                    enabled=bool(semantic_cache is not None and use_semantic_cache),
                    threshold=float(semantic_cache_threshold),
                    max_entries=512,
                ),
            )
            result = run_pipeline(
                question.strip(),
                embedder=embedder,
                corpus_chunks=st.session_state.corpus_chunks,
                faiss_index=st.session_state.faiss_index,
                config=pipe_cfg,
                generator=gen,
                reranker=reranker,
                milvus_store=milvus_store,
                bm25_resources=bm25_resources,
                semantic_cache=semantic_cache,
                history=hist,
            )
        if result.get("error"):
            st.error(result["error"])
            return
        st.session_state["last_rag"] = result
        st.session_state["last_question"] = question.strip()
        st.session_state["last_rag_backend"] = backend
        st.session_state["last_rag_ui"] = st.session_state.get("last_rag_ui", 0) + 1
        st.session_state["last_final_k"] = final_k
        st.session_state.query_history.append(
            {
                "ts": time.time(),
                "question": question.strip(),
                "rewritten_question": result.get("rewritten_question", ""),
                "subqueries": result.get("retrieval_queries", []),
                "answer": result.get("answer", ""),
            }
        )
        st.session_state.qa_messages.append(
            {
                "q": question.strip(),
                "a": result.get("answer", ""),
            }
        )
        if st.session_state.get("exp_log_queries", True):
            try:
                ref = (st.session_state.get("exp_reference_answer") or "").strip()
                mc: dict[str, Any] = {
                    "embedding_model": DEFAULT_EMBED,
                    "retrieve_k": int(retrieve_k),
                    "final_k": int(final_k),
                    "use_rerank": bool(use_rerank),
                    "use_hybrid": retrieval_mode == "hybrid",
                    "fusion_list_k": int(fusion_list_k),
                    "rrf_k": int(rrf_k),
                    "retrieval_mode": retrieval_mode,
                    "vdb_backend": vdb_backend,
                    "milvus_index_type": str(milvus_index_type),
                    "milvus_metric_type": str(milvus_metric_type),
                    "milvus_ivf_nprobe": int(milvus_ivf_nprobe),
                    "milvus_hnsw_ef": int(milvus_hnsw_ef),
                    "max_context_chars": int(max_context_chars),
                    "truncation": str(truncation),
                    "template_key": template_key,
                    "use_query_rewrite": use_query_rewrite,
                    "use_query_decomposition": use_query_decomposition,
                    "use_filter_generation": use_filter_generation,
                    "use_reflection_loops": use_reflection_loops,
                }
                lm = getattr(gen, "model", None)
                _log_ui_rag_observation(
                    question=question.strip(),
                    answer=str(result.get("answer") or ""),
                    result=result,
                    model_config=mc,
                    reference=ref,
                    backend=backend,
                    job_id=st.session_state.get("loaded_job_id"),
                    llm_model=str(lm) if lm is not None else None,
                )
            except Exception as e:
                logging.getLogger(__name__).warning("experiment log failed: %s", e)
        st.rerun()


if __name__ == "__main__":
    main()
