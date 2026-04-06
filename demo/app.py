"""
Streamlit: sidebar nav (**Ingest** | **Query** | **Library**), MinIO-backed ingest, job browser.

Run::

    streamlit run demo/app.py
"""

from __future__ import annotations

import logging
import html
import re
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

warnings.filterwarnings("ignore", message=r"Accessing `__path__`")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.models").setLevel(logging.ERROR)

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
from src.reranker import Reranker, load_reranker
from src.retriever import FaissIndex, search
from src.storage.minio_artifacts import MinioArtifactStore, load_minio_settings
from src.storage.redis_jobs import RedisJobStore
from src.streaming_parser import HiddenReasoningStreamParser, strip_hidden_reasoning_text


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
) -> List[dict]:
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
) -> dict:
    k = min(retrieve_k, len(corpus_chunks))
    llm_details = get_generator_details(generator)
    stage_trace: List[dict] = []
    stage_trace.append(
        {
            "stage": "input",
            "title": "Input question",
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
    stage_trace.append(
        {
            "stage": "query_rewrite",
            "title": "Query rewrite",
            "input": {"question": question, "enabled": use_query_rewrite},
            "output": {
                "rewritten_question": rewritten_question,
                "llm_call": {"system_prompt": None, **rewrite_debug},
            },
        }
    )

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
    stage_trace.append(
        {
            "stage": "query_decomposition",
            "title": "Query decomposition",
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

    merged: Dict[int, dict] = {}
    for rq in retrieval_queries:
        rows = _retrieve_rows_for_query(
            rq,
            embedder=embedder,
            corpus_chunks=corpus_chunks,
            faiss_index=faiss_index,
            top_k=k,
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
    stage_trace.append(
        {
            "stage": "dense_retrieval",
            "title": "Dense retrieval + merge",
            "input": {"retrieval_queries": retrieval_queries, "top_k_each": k},
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
    stage_trace.append(
        {
            "stage": "filter_generation",
            "title": "Filter generation",
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

    if use_rerank and reranker is not None and pool:
        rerank_rows = cross_encoder_rerank_trace(reranker, rewritten_question, pool)
        final_passages = [r["text"] for r in rerank_rows[: min(final_k, len(rerank_rows))]]
    else:
        final_passages = pool[: min(final_k, len(pool))]
    stage_trace.append(
        {
            "stage": "rerank",
            "title": "Rerank / final passage selection",
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

    def _build_answer(
        passages: List[str],
        q_for_prompt: str,
        *,
        on_stream: Optional[Callable[[str], None]] = None,
        on_raw_stream: Optional[Callable[[str], None]] = None,
    ) -> tuple[str, str, str, str]:
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
        answer_val = generate_visible_answer(
            generator=generator,
            system_prompt=RAG_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            prompt=prompt_val,
            on_visible_text=on_stream,
            on_raw_text=on_raw_stream,
        )
        return raw_ctx, user_prompt, prompt_val, answer_val

    raw_context, user_prompt, prompt, answer = _build_answer(final_passages, rewritten_question)
    stage_trace.append(
        {
            "stage": "generation",
            "title": "Prompt + generation",
            "input": {
                "question_for_prompt": rewritten_question,
                "prompt_template": prompt_template[:180],
                "max_context_chars": max_context_chars,
                "truncation": truncation,
            },
            "output": {
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
        stage_trace.append(
            {
                "stage": "reflection",
                "title": "Reflection loops",
                "input": {
                    "enabled": use_reflection_loops,
                    "max_reflection_loops": max_reflection_loops,
                },
                "output": {"steps": reflection_steps},
            }
        )
    else:
        stage_trace.append(
            {
                "stage": "reflection",
                "title": "Reflection loops",
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

    cited_ids = _extract_citation_ids(answer)
    cited_rows = [r for r in citation_table if int(r["id"]) in cited_ids]
    stage_trace.append(
        {
            "stage": "citations",
            "title": "Citation extraction",
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
        "prompt": prompt,
        "answer": answer,
        "raw_context": raw_context,
        "truncation": truncation,
        "max_context_chars": max_context_chars,
        "use_rerank": use_rerank,
    }


def render_rag_trace(result: dict, *, ui_id: int, final_k: int) -> None:
    fk = int(final_k)
    tab_chunks, tab_rerank, tab_prompt = st.tabs(
        ["Retrieved chunks (FAISS)", "Reranking (cross-encoder)", "Final prompt & answer"]
    )

    with tab_chunks:
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
                    height=min(240, 24 + 12 * row["text"].count("\n")),
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
                        height=min(240, 24 + 12 * row["text"].count("\n")),
                        key=f"{ui_id}_r_{row['new_rank']}",
                        label_visibility="collapsed",
                    )
            st.caption(f"Passages 1–{min(fk, len(result['rerank_rows']))} → LLM context.")

    with tab_prompt:
        st.caption(
            f"Context ~{len(result.get('raw_context', ''))} chars · "
            f"truncation **{result['truncation']}** → **{result['max_context_chars']}**"
        )
        st.text_area("Prompt", result["prompt"], height=360, key=f"{ui_id}_p")
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
                        height=min(220, 24 + 12 * row["text"].count("\n")),
                        key=f"{ui_id}_cite_{row['id']}",
                        label_visibility="collapsed",
                    )


def render_stage_trace(result: dict) -> None:
    stages = result.get("stage_trace", [])
    if not stages:
        return
    st.subheader("Query process trace")
    st.caption("Click each stage to inspect input/output.")
    for i, stg in enumerate(stages, start=1):
        title = stg.get("title") or stg.get("stage") or f"Stage {i}"
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
            retrieve_k = st.slider("FAISS top-K", 3, 30, 10, disabled=not ready)
            final_k = st.slider("Final passages to LLM", 1, 10, 3, disabled=not ready)
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
        retrieve_k = 10
        final_k = 3
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
            render_stage_trace(result)
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
        hist = (
            _history_to_text(st.session_state.get("query_history", []), max_turns=int(history_turns))
            if use_session_history
            else "None"
        )

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
            )
        if result.get("error"):
            st.error(result["error"])
            return
        st.session_state["last_rag"] = result
        st.session_state["last_question"] = question.strip()
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
