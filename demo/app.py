"""
Streamlit: sidebar nav (**Ingest** | **Query** | **Library**), MinIO-backed ingest, job browser.

Run::

    streamlit run demo/app.py
"""

from __future__ import annotations

import logging
import html
import math
import os
import re
import time
import warnings
from time import perf_counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

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

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.context_truncation import TruncationStrategy, truncate_context
from src.document_ingest_pipeline import (
    IngestPipelineConfig,
    load_ingest_from_minio,
    run_document_ingest,
)
from src.embedder import EmbeddingModel, load_embedding_model, prepare_query
from src.generator import (
    ChatTextGenerator,
    GeminiGenerator,
    MockGenerator,
    OllamaGenerator,
    OpenAICompatibleGenerator,
    StreamingChatTextGenerator,
    StreamingTextGenerator,
    TextGenerator,
)
from src.prompts import PROMPT_TEMPLATES, format_rag_prompt
from src.rag_generation import passages_to_context
from src.hybrid_retrieval import BM25Resources, build_bm25_resources, fused_top_indices
from src.reranker import Reranker, load_reranker
from src.retriever import FaissIndex, search
from src.storage.minio_artifacts import MinioArtifactStore, load_minio_settings
from src.storage.milvus_store import MilvusChunkStore
from src.storage.redis_jobs import RedisJobStore
from src.storage.redis_semantic_cache import RedisSemanticCache
from src.streaming_parser import HiddenReasoningStreamParser, strip_hidden_reasoning_text
from src.ragas_ui_metrics import run_ragas_legacy_evaluate


DEFAULT_EMBED = "BAAI/bge-base-en-v1.5"
DEFAULT_RERANK = "BAAI/bge-reranker-base"
RAG_SYSTEM_PROMPT = (
    "You are a grounded RAG assistant. Use only provided context. "
    "If evidence is insufficient, say unknown."
)
REWRITE_SYSTEM_PROMPT = "You rewrite user questions for retrieval quality."
DECOMPOSE_SYSTEM_PROMPT = "You decompose questions into focused retrieval sub-queries."
FILTER_SYSTEM_PROMPT = "You generate retrieval include/exclude keyword filters."
REFLECTION_SYSTEM_PROMPT = "You audit RAG answers and request follow-up retrieval only when needed."


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
        "<div style=\"max-height:380px; overflow-y:auto; border:1px solid #ddd; "
        "border-radius:12px; padding:10px; background:#fafafa;\">"
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
    user_prompt = (
        _rewrite_user_prompt(question)
        + "\n\nHistory:\n"
        + history
    )
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
    user_prompt = (
        _decompose_user_prompt(question)
        + "\n\nHistory:\n"
        + history
    )
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
        t.lower()
        for t in filt.get("include", [])
        if isinstance(t, str) and t.strip()
    ]
    exclude = [
        t.lower()
        for t in filt.get("exclude", [])
        if isinstance(t, str) and t.strip()
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


def _retrieve_rows_for_query(
    query: str,
    *,
    embedder: EmbeddingModel,
    corpus_chunks: List[str],
    faiss_index: FaissIndex,
    top_k: int,
    vdb_backend: str = "faiss",
    milvus_store: Optional[MilvusChunkStore] = None,
    milvus_job_id: Optional[str] = None,
    retrieval_mode: str = "dense",
    bm25_resources: Optional[BM25Resources] = None,
    fusion_list_k: Optional[int] = None,
    rrf_k: int = 60,
) -> List[dict]:
    if (
        vdb_backend == "milvus"
        and milvus_store is not None
        and milvus_job_id
    ):
        return milvus_store.search_job_chunks(
            job_id=milvus_job_id,
            query=query,
            embedder=embedder,
            top_k=top_k,
        )
    # FAISS backend: allow dense-only or hybrid BM25+dense (RRF).
    if retrieval_mode == "hybrid" and bm25_resources is not None:
        idx_row = fused_top_indices(
            query,
            embedder=embedder,
            corpus_chunks=corpus_chunks,
            faiss_index=faiss_index,
            bm25_resources=bm25_resources,
            retrieve_k=top_k,
            rrf_k=int(rrf_k),
            fusion_list_k=fusion_list_k,
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
    retrieve_k: int,
    final_k: int,
    use_rerank: bool,
    reranker: Optional[Reranker],
    prompt_template: str,
    max_context_chars: int,
    truncation: TruncationStrategy,
    generator: TextGenerator,
    use_query_rewrite: bool = False,
    use_query_decomposition: bool = False,
    max_subqueries: int = 3,
    use_filter_generation: bool = False,
    min_filter_keep: int = 3,
    use_reflection_loops: bool = False,
    max_reflection_loops: int = 2,
    require_citations: bool = False,
    history: str = "None",
    vdb_backend: str = "faiss",
    milvus_store: Optional[MilvusChunkStore] = None,
    milvus_job_id: Optional[str] = None,
    semantic_cache: Optional[RedisSemanticCache] = None,
    semantic_cache_threshold: float = 0.93,
    semantic_cache_max_entries: int = 512,
    retrieval_mode: str = "dense",
    bm25_resources: Optional[BM25Resources] = None,
    fusion_list_k: Optional[int] = None,
    rrf_k: int = 60,
) -> dict:
    k = min(retrieve_k, len(corpus_chunks))
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
            },
        }
    )
    if k <= 0:
        return {
            "error": "No index loaded.",
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
    merged: Dict[int, dict] = {}
    for rq in retrieval_queries:
        rows = _retrieve_rows_for_query(
            rq,
            embedder=embedder,
            corpus_chunks=corpus_chunks,
            faiss_index=faiss_index,
            top_k=k,
            vdb_backend=vdb_backend,
            milvus_store=milvus_store,
            milvus_job_id=milvus_job_id,
            retrieval_mode=retrieval_mode,
            bm25_resources=bm25_resources,
            fusion_list_k=fusion_list_k,
            rrf_k=rrf_k,
        )
        for row in rows:
            ci = int(row["chunk_index"])
            if ci not in merged:
                merged[ci] = {
                    "chunk_index": ci,
                    "faiss_score": float(row["faiss_score"]),
                    "best_faiss_rank": int(row["faiss_rank"]),
                    "matched_queries": [rq],
                    "text": row["text"],
                }
            else:
                cur = merged[ci]
                cur["faiss_score"] = max(float(cur["faiss_score"]), float(row["faiss_score"]))
                cur["best_faiss_rank"] = min(int(cur["best_faiss_rank"]), int(row["faiss_rank"]))
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
        final_passages = [r["text"] for r in rerank_rows[: min(final_k, len(rerank_rows))]]
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
        raw_ctx, user_prompt, prompt_val = _compose_rag_prompt_parts(passages, q_for_prompt)
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
        final_passages, rewritten_question
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
                top_k=k,
                vdb_backend=vdb_backend,
                milvus_store=milvus_store,
                milvus_job_id=milvus_job_id,
                retrieval_mode=retrieval_mode,
                bm25_resources=bm25_resources,
                fusion_list_k=fusion_list_k,
                rrf_k=rrf_k,
            )
            added = 0
            for row in new_rows:
                ci = int(row["chunk_index"])
                if ci not in merged:
                    merged[ci] = {
                        "chunk_index": ci,
                        "faiss_score": float(row["faiss_score"]),
                        "best_faiss_rank": int(row["faiss_rank"]),
                        "matched_queries": [followup],
                        "text": row["text"],
                    }
                    added += 1
                else:
                    cur = merged[ci]
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
                rerank_rows = cross_encoder_rerank_trace(reranker, rewritten_question, pool)
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
        ["Retrieved chunks (FAISS)", "Reranking (cross-encoder)", "Final prompt & answer"]
    )

    with tab_chunks:
        mode = result.get("retrieval_mode", "dense")
        if mode == "hybrid":
            st.caption("Retrieval mode: **Hybrid BM25 + dense (RRF fusion)**.")
        if result.get("rewritten_question") and result.get("rewritten_question") != result.get("original_question"):
            st.caption(f"Query rewrite: `{result['original_question']}` → `{result['rewritten_question']}`")
        queries = result.get("retrieval_queries", [])
        if len(queries) > 1:
            st.caption(f"Query decomposition used {len(queries)} retrieval queries.")
        filt = result.get("retrieval_filter", {})
        if isinstance(filt, dict) and (filt.get("include") or filt.get("exclude")):
            st.caption(
                f"LLM filter include={filt.get('include', [])} exclude={filt.get('exclude', [])}"
            )
        st.markdown(
            "Top-K from bi-encoder + FAISS (inner product on normalized vectors ≈ cosine similarity)."
        )
        for row in result["retrieved"]:
            with st.expander(
                f"Rank {row['faiss_rank']} · chunk #{row['chunk_index']} · {row['faiss_score']:.4f}"
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
            st.info("Reranking off — using FAISS order for final passages.")
        elif not result["rerank_rows"]:
            st.warning("Empty pool.")
        else:
            for row in result["rerank_rows"]:
                with st.expander(f"CE rank {row['new_rank']} (was FAISS #{row['faiss_rank']}) · {row['cross_encoder_score']:.4f}"):
                    st.text_area(
                        "Passage",
                        row["text"],
                        height=_trace_text_area_height(row["text"], cap=240),
                        key=f"{ui_id}_r_{row['new_rank']}",
                        label_visibility="collapsed",
                    )
            st.caption(f"Passages 1–{min(fk, len(result['rerank_rows']))} → LLM context.")

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
        st.session_state.ragas_eval_backend = lr if lr in ("gemini", "openai", "ollama") else "gemini"

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
        if (
            scores
            and ver is not None
            and int(ver) == current_ui
        ):
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
    st.caption("Click each stage to inspect input/output. Titles include **latency_ms** when available.")
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


def unload_index() -> None:
    st.session_state.corpus_chunks = []
    st.session_state.faiss_index = None
    st.session_state.source_label = None
    st.session_state.ingest_meta = {}
    st.session_state.loaded_job_id = None
    st.session_state.pop("last_rag", None)
    st.session_state.pop("last_question", None)
    st.session_state.query_history = []
    st.session_state.qa_messages = []


def load_job(job_id: str) -> None:
    chunks, faiss_index, meta = load_ingest_from_minio(job_id.strip())
    st.session_state.corpus_chunks = chunks
    st.session_state.faiss_index = faiss_index
    st.session_state.ingest_meta = meta
    fn = meta.get("filename") or "document"
    st.session_state.source_label = f"{job_id.strip()} · {fn}"
    st.session_state.loaded_job_id = job_id.strip()
    st.session_state.last_job_id = job_id.strip()
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
        if b1:
            st.session_state.nav = "ingest"
        if b2:
            st.session_state.nav = "query"
        if b3:
            st.session_state.nav = "library"
        st.divider()
        st.caption("MinIO + Redis")
    return st.session_state.nav


def main() -> None:
    st.set_page_config(page_title="RAG Lab", layout="wide", initial_sidebar_state="expanded")
    _init_session()
    _sidebar_nav()

    st.title("Document pipeline")
    nav = st.session_state.nav
    if nav == "ingest":
        _ui_ingest_pipeline()
    elif nav == "query":
        _ui_query()
    else:
        _ui_library()


def _ui_ingest_pipeline() -> None:
    st.caption("Upload → filter → extract → chunk & summarize → embed → **MinIO** + **Redis**")

    st.subheader("1 — Upload & page filter")
    up_col, filt_col = st.columns(2)
    with up_col:
        uploaded = st.file_uploader("Document", type=["pdf", "txt", "md"], key="ing_upl")
    with filt_col:
        page_spec = st.text_input(
            "Page filter (PDF)",
            "",
            placeholder="1,3,5-8 — empty = all",
        )

    st.subheader("2 — Extraction")
    extraction = st.radio("Depth", ("shallow", "full"), horizontal=True)

    st.subheader("3 — Chunking & summarization")
    c1, c2, c3 = st.columns(3)
    with c1:
        chunk_size = st.number_input("Chunk size", 128, 2048, 512, 64)
        chunk_overlap = st.number_input("Chunk overlap", 0, 512, 64, 32)
    with c2:
        strat = st.selectbox(
            "Summarization strategy",
            ("single", "hierarchical", "iterative"),
        )
    with c3:
        summ_backend = st.selectbox(
            "Summarizer LLM (if not single)",
            ["mock", "gemini", "openai", "ollama"],
        )

    st.subheader("4 — Run → MinIO + Redis")
    if st.button("Run full pipeline", type="primary"):
        if uploaded is None:
            st.error("Upload a document first.")
        else:
            try:
                cfg = IngestPipelineConfig(
                    chunk_size=int(chunk_size),
                    chunk_overlap=int(chunk_overlap),
                    extraction=extraction,  # type: ignore[arg-type]
                    summarization=strat,  # type: ignore[arg-type]
                )
                summ = ingest_summarizer(summ_backend) if strat != "single" else None
                with st.spinner("Running pipeline…"):
                    meta = run_document_ingest(
                        filename=uploaded.name,
                        raw_bytes=uploaded.getvalue(),
                        page_filter_spec=page_spec,
                        config=cfg,
                        summarizer=summ,
                    )
                st.success(f"**job_id:** `{meta['job_id']}`")
                st.json(meta)
                st.session_state["last_job_id"] = meta["job_id"]
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
        st.session_state.library_refresh = st.session_state.get("library_refresh", 0) + 1

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
        },
    )
    st.caption(f"**{len(rows)}** job(s) in bucket `{load_minio_settings().bucket}`.")


def _job_label_for_select(row: dict[str, Any]) -> str:
    jid = row["job_id"]
    fn = row.get("filename", "?")
    nc = row.get("n_chunks", "?")
    short = f"{jid[:8]}…" if len(jid) > 12 else jid
    return f"{short} · {fn} · {nc} chunks"


def _ui_query() -> None:
    st.caption("Attach an index from MinIO, then ask questions. **Attach** checked + **Load** = fetch index; **Attach** unchecked + **Load** = unload.")

    try:
        jobs = _minio_store().list_ingest_jobs_table()
    except Exception as e:
        st.error(f"MinIO: {e}")
        return

    st.subheader("Index in memory")
    if st.session_state.loaded_job_id and st.session_state.faiss_index is not None:
        st.success(
            f"Loaded **{st.session_state.loaded_job_id}** — "
            f"{len(st.session_state.corpus_chunks)} chunks · {st.session_state.source_label}"
        )
    else:
        st.warning("No index loaded.")

    st.subheader("Attach & load")
    attach = st.checkbox(
        "Attach index for querying",
        value=True,
        help="Checked: **Load** downloads the selected job into memory. Unchecked: **Load** clears the index (unload).",
    )

    options: List[str] = [r["job_id"] for r in jobs]
    label_by_id = {r["job_id"]: _job_label_for_select(r) for r in jobs}

    if not options:
        st.info("No jobs in MinIO yet — use **Ingest** first.")
        selected = None
    else:
        default_ix = 0
        lid = st.session_state.get("loaded_job_id")
        if lid and lid in options:
            default_ix = options.index(lid)
        selected = st.selectbox(
            "Job",
            options=options,
            index=default_ix,
            format_func=lambda jid: label_by_id.get(jid, jid),
        )

    if st.button("Load", type="primary"):
        if attach:
            if not selected:
                st.error("No job to load.")
            else:
                try:
                    load_job(selected)
                    st.success(f"Loaded `{selected}`")
                    st.rerun()
                except Exception as e:
                    st.exception(e)
        else:
            unload_index()
            st.success("Index unloaded from memory.")
            st.rerun()

    ready = (
        st.session_state.faiss_index is not None
        and len(st.session_state.corpus_chunks) > 0
    )

    chat_bar_options_col, _ = st.columns([1, 6])
    with chat_bar_options_col:
        with st.popover("Options", disabled=not ready):
            st.caption("Attached chat options")
            vdb_backend = st.selectbox(
                "Vector DB backend",
                ["faiss", "milvus"],
                disabled=not ready,
                help="faiss = in-memory local index, milvus = remote vector database",
            )
            retrieval_mode = st.selectbox(
                "ANN retrieval mode",
                ["dense", "hybrid"],
                disabled=not ready or vdb_backend != "faiss",
                help="dense = embedding-only FAISS; hybrid = BM25 + dense merged with RRF (FAISS backend only).",
            )
            fusion_list_k = st.slider(
                "Hybrid fusion list K",
                5,
                100,
                30,
                5,
                disabled=not ready or vdb_backend != "faiss" or retrieval_mode != "hybrid",
                help="Candidates taken from each retriever (BM25 and dense) before RRF merge.",
            )
            rrf_k = st.slider(
                "RRF k",
                10,
                200,
                60,
                5,
                disabled=not ready or vdb_backend != "faiss" or retrieval_mode != "hybrid",
                help="RRF smoothing constant; larger values flatten rank contribution.",
            )
            retrieve_k = st.slider("FAISS top-K", 3, 30, 10, disabled=not ready)
            final_k = st.slider("Final passages to LLM", 1, 10, 3, disabled=not ready)
            use_semantic_cache = st.checkbox(
                "Redis semantic cache",
                value=True,
                disabled=not ready,
            )
            semantic_cache_threshold = st.slider(
                "Semantic cache threshold",
                0.70,
                0.99,
                0.93,
                0.01,
                disabled=not ready or not use_semantic_cache,
            )
            use_filter_generation = st.checkbox(
                "Filter generation (LLM include/exclude keywords)",
                value=False,
                disabled=not ready,
            )
            min_filter_keep = st.slider(
                "Min chunks to keep after filter",
                1,
                10,
                3,
                disabled=not ready or not use_filter_generation,
            )
            use_rerank = st.checkbox("Cross-encoder rerank", value=True, disabled=not ready)
            use_query_rewrite = st.checkbox("Query rewrite", value=False, disabled=not ready)
            use_query_decomposition = st.checkbox("Query decomposition", value=False, disabled=not ready)
            max_subqueries = st.slider(
                "Max sub-queries",
                2,
                6,
                3,
                disabled=not ready or not use_query_decomposition,
            )
            use_reflection_loops = st.checkbox(
                "Reflection loops (answer critique + follow-up retrieval)",
                value=False,
                disabled=not ready,
            )
            max_reflection_loops = st.slider(
                "Max reflection loops",
                1,
                4,
                2,
                disabled=not ready or not use_reflection_loops,
            )
            require_citations = st.checkbox(
                "Require citation markers in answer ([1], [2])",
                value=True,
                disabled=not ready,
            )
            use_session_history = st.checkbox(
                "Use session history in rewrite/decomposition",
                value=True,
                disabled=not ready,
            )
            history_turns = st.slider(
                "History turns",
                1,
                8,
                3,
                disabled=not ready or not use_session_history,
            )
            template_key = st.selectbox(
                "Prompt template",
                list(PROMPT_TEMPLATES.keys()),
                disabled=not ready,
            )
            max_context_chars = st.number_input(
                "Max context chars", 500, 32000, 6000, step=500, disabled=not ready
            )
            truncation = st.selectbox(
                "Truncation", ["head", "tail", "middle"], disabled=not ready
            )
            backend = st.selectbox(
                "LLM",
                ["mock", "gemini", "openai", "ollama"],
                disabled=not ready,
            )

    # Defaults when options popover is disabled (no index loaded yet)
    if not ready:
        vdb_backend = "faiss"
        retrieval_mode = "dense"
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

    if not ready:
        st.info("Attach a job and click **Load** to enable questions.")
        return

    st.divider()
    if st.session_state.loaded_job_id:
        sync_col, _ = st.columns([1.2, 4.8])
        with sync_col:
            if st.button("Sync loaded job to Milvus"):
                try:
                    with st.spinner("Connecting to Milvus (set MILVUS_URI; MILVUS_CONNECT_TIMEOUT)…"):
                        embedder_sync = cached_embedder(DEFAULT_EMBED)
                        n = _milvus_store().upsert_job_chunks(
                            job_id=st.session_state.loaded_job_id,
                            chunk_texts=st.session_state.corpus_chunks,
                            embedder=embedder_sync,
                        )
                    st.success(f"Upserted {n} chunks to Milvus.")
                except Exception as e:
                    st.error(f"Milvus sync failed: {e}")

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
        if vdb_backend == "milvus":
            try:
                with st.spinner("Connecting to Milvus…"):
                    milvus_store = _milvus_store()
            except Exception as e:
                st.error(
                    f"Milvus backend unavailable: {e}\n\n"
                    "Start Milvus (e.g. docker) and set `MILVUS_URI` in `.env`. "
                    "Use **faiss** backend if Milvus is not running."
                )
                return
        semantic_cache = None
        if use_semantic_cache:
            try:
                ns = st.session_state.loaded_job_id or "default"
                semantic_cache = RedisSemanticCache(namespace=ns)
            except Exception as e:
                st.warning(f"Redis semantic cache disabled: {e}")
                semantic_cache = None
        hist = (
            _history_to_text(st.session_state.get("query_history", []), max_turns=int(history_turns))
            if use_session_history
            else "None"
        )
        bm25_resources: Optional[BM25Resources] = None
        if vdb_backend == "faiss" and retrieval_mode == "hybrid":
            with st.spinner("Building BM25 resources for hybrid retrieval…"):
                bm25_resources = build_bm25_resources(st.session_state.corpus_chunks)

        with st.spinner("Retrieving & generating…"):
            result = run_pipeline(
                question.strip(),
                embedder=embedder,
                corpus_chunks=st.session_state.corpus_chunks,
                faiss_index=st.session_state.faiss_index,
                retrieve_k=retrieve_k,
                final_k=final_k,
                use_rerank=use_rerank,
                reranker=reranker,
                prompt_template=PROMPT_TEMPLATES[template_key],
                max_context_chars=int(max_context_chars),
                truncation=truncation,  # type: ignore[arg-type]
                generator=gen,
                use_query_rewrite=use_query_rewrite,
                use_query_decomposition=use_query_decomposition,
                max_subqueries=int(max_subqueries),
                use_filter_generation=use_filter_generation,
                min_filter_keep=int(min_filter_keep),
                use_reflection_loops=use_reflection_loops,
                max_reflection_loops=int(max_reflection_loops),
                require_citations=require_citations,
                history=hist,
                vdb_backend=vdb_backend,
                milvus_store=milvus_store,
                milvus_job_id=st.session_state.loaded_job_id,
                semantic_cache=semantic_cache,
                semantic_cache_threshold=float(semantic_cache_threshold),
                retrieval_mode=retrieval_mode,
                bm25_resources=bm25_resources,
                fusion_list_k=int(fusion_list_k),
                rrf_k=int(rrf_k),
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
        st.rerun()


if __name__ == "__main__":
    main()
