"""
Streamlit: sidebar nav (**Ingest** | **Query** | **Library**), MinIO-backed ingest, job browser.

Run::

    streamlit run demo/app.py
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, List, Optional

warnings.filterwarnings("ignore", message=r"Accessing `__path__`")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.models").setLevel(logging.ERROR)

import sys

import numpy as np
import streamlit as st

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
    GeminiGenerator,
    MockGenerator,
    OllamaGenerator,
    OpenAICompatibleGenerator,
    TextGenerator,
)
from src.prompts import PROMPT_TEMPLATES, format_rag_prompt
from src.rag_generation import passages_to_context
from src.reranker import Reranker, load_reranker
from src.retriever import FaissIndex, gather_texts_by_indices, search
from src.storage.minio_artifacts import MinioArtifactStore, load_minio_settings
from src.storage.redis_jobs import RedisJobStore


DEFAULT_EMBED = "BAAI/bge-base-en-v1.5"
DEFAULT_RERANK = "BAAI/bge-reranker-base"


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
) -> dict:
    k = min(retrieve_k, len(corpus_chunks))
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

    q = prepare_query(embedder.name, question)
    q_emb = embedder.encode([q])
    faiss_scores, idx = search(faiss_index, q_emb, top_k=k)
    idx_row = idx[0].tolist()
    score_row = faiss_scores[0].tolist()
    pool = gather_texts_by_indices(corpus_chunks, idx_row)

    retrieved = []
    rank = 0
    for i, chunk_idx in enumerate(idx_row):
        if chunk_idx < 0:
            continue
        rank += 1
        ci = int(chunk_idx)
        retrieved.append(
            {
                "faiss_rank": rank,
                "chunk_index": ci,
                "faiss_score": float(score_row[i]),
                "text": corpus_chunks[ci],
            }
        )

    rerank_rows: List[dict] = []
    final_passages: List[str] = []

    if use_rerank and reranker is not None and pool:
        rerank_rows = cross_encoder_rerank_trace(reranker, question, pool)
        final_passages = [r["text"] for r in rerank_rows[: min(final_k, len(rerank_rows))]]
    else:
        final_passages = pool[: min(final_k, len(pool))]

    raw_context = passages_to_context(final_passages)
    context = truncate_context(raw_context, max_context_chars, truncation)
    prompt = format_rag_prompt(prompt_template, context=context, question=question)
    answer = generator.generate(prompt)

    return {
        "error": None,
        "retrieved": retrieved,
        "rerank_rows": rerank_rows,
        "final_passages": final_passages,
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
        st.markdown(
            "Top-K from bi-encoder + FAISS (inner product on normalized vectors ≈ cosine similarity)."
        )
        for row in result["retrieved"]:
            with st.expander(
                f"Rank {row['faiss_rank']} · chunk #{row['chunk_index']} · {row['faiss_score']:.4f}"
            ):
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


def unload_index() -> None:
    st.session_state.corpus_chunks = []
    st.session_state.faiss_index = None
    st.session_state.source_label = None
    st.session_state.ingest_meta = {}
    st.session_state.loaded_job_id = None
    st.session_state.pop("last_rag", None)
    st.session_state.pop("last_question", None)


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

    with st.expander("Retrieval & generation settings", expanded=ready):
        retrieve_k = st.slider("FAISS top-K", 3, 30, 10, disabled=not ready)
        final_k = st.slider("Final passages to LLM", 1, 10, 3, disabled=not ready)
        use_rerank = st.checkbox("Cross-encoder rerank", value=True, disabled=not ready)
        template_key = st.selectbox(
            "Prompt template",
            list(PROMPT_TEMPLATES.keys()),
            disabled=not ready,
        )
        max_context_chars = st.number_input(
            "Max context chars", 500, 32000, 6000, step=500, disabled=not ready
        )
        truncation = st.selectbox("Truncation", ["head", "tail", "middle"], disabled=not ready)
        backend = st.selectbox(
            "LLM",
            ["mock", "gemini", "openai", "ollama"],
            disabled=not ready,
        )

    if not ready:
        st.info("Attach a job and click **Load** to enable questions.")
        return

    st.divider()
    col_q, col_a = st.columns([4, 1])
    with col_q:
        question = st.text_input("Question", placeholder="Ask about the document…")
    with col_a:
        ask = st.button("Ask", type="primary", use_container_width=True)

    if ask and question.strip():
        embedder = cached_embedder(DEFAULT_EMBED)
        reranker = cached_reranker(DEFAULT_RERANK) if use_rerank else None
        try:
            gen = make_generator(backend)
        except Exception as e:
            st.error(f"Generator: {e}")
            return
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
            )
        if result.get("error"):
            st.error(result["error"])
            return
        st.session_state["last_rag"] = result
        st.session_state["last_question"] = question.strip()
        st.session_state["last_rag_ui"] = st.session_state.get("last_rag_ui", 0) + 1
        st.session_state["last_final_k"] = final_k

    result = st.session_state.get("last_rag")
    if not result:
        st.info("Ask a question above.")
        return

    st.subheader("Answer")
    st.caption(f"Q: {st.session_state.get('last_question', '')}")
    st.write(result["answer"])
    ui_id = st.session_state.get("last_rag_ui", 0)
    fk = int(st.session_state.get("last_final_k", 3))
    render_rag_trace(result, ui_id=ui_id, final_k=fk)


if __name__ == "__main__":
    main()
