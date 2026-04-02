"""
Lightweight Streamlit demo for the RAG pipeline: upload a document, ask questions,
and inspect retrieval, reranking, and the final LLM prompt.

Run from repo root with the same environment as ``pip install -r requirements.txt``::

    .venv/bin/streamlit run demo/app.py

Using a different ``streamlit`` on ``PATH`` (e.g. conda base) may miss ``sentence_transformers``.

Project ``.streamlit/config.toml`` disables the file watcher so the terminal is not flooded by
Hugging Face ``transformers`` lazy-import messages; after code edits, refresh the app or restart
``streamlit run``.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import List, Optional

# Before importing sentence_transformers / transformers (via src.embedder).
warnings.filterwarnings("ignore", message=r"Accessing `__path__`")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.models").setLevel(logging.ERROR)

import io
import sys

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.context_truncation import TruncationStrategy, truncate_context
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
from src.rag_pipeline import build_corpus_chunks_from_documents, build_retrieval_index
from src.reranker import Reranker, load_reranker
from src.retriever import FaissIndex, gather_texts_by_indices, search


DEFAULT_EMBED = "BAAI/bge-base-en-v1.5"
DEFAULT_RERANK = "BAAI/bge-reranker-base"


@st.cache_resource
def cached_embedder(model_name: str) -> EmbeddingModel:
    return load_embedding_model(model_name, normalize=True)


@st.cache_resource
def cached_reranker(model_name: str) -> Reranker:
    return load_reranker(model_name)


def extract_text_from_upload(name: str, data: bytes) -> str:
    lower = name.lower()
    if lower.endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        parts: List[str] = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        return "\n\n".join(parts)
    return data.decode("utf-8", errors="replace")


def make_generator(backend: str) -> TextGenerator:
    if backend == "gemini":
        return GeminiGenerator()
    if backend == "openai":
        return OpenAICompatibleGenerator()
    if backend == "ollama":
        return OllamaGenerator()
    return MockGenerator()


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
            "error": "No chunks in index. Upload a non-empty document.",
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


def main() -> None:
    st.set_page_config(page_title="RAG Lab Demo", layout="wide")
    st.title("RAG pipeline demo")
    st.caption("Upload a PDF or text file, then ask questions. Expand sections below the answer to inspect retrieval and prompts.")

    if "corpus_chunks" not in st.session_state:
        st.session_state.corpus_chunks = []
    if "faiss_index" not in st.session_state:
        st.session_state.faiss_index = None
    if "source_label" not in st.session_state:
        st.session_state.source_label = None

    with st.sidebar:
        st.header("Document")
        uploaded = st.file_uploader(
            "PDF or text file",
            type=["pdf", "txt", "md"],
            help="PDFs are read with pypdf (text extraction). .txt/.md are UTF-8.",
        )
        if uploaded is not None:
            data = uploaded.getvalue()
            text = extract_text_from_upload(uploaded.name, data)
            if not text.strip():
                st.warning("No text extracted. Try another file or a .txt copy of the paper.")
            else:
                chunk_size = st.number_input("Chunk size", min_value=128, max_value=2048, value=512, step=64)
                chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=512, value=64, step=32)
                if st.button("Build index from upload", type="primary"):
                    with st.spinner("Chunking and embedding…"):
                        chunks = build_corpus_chunks_from_documents(
                            [text],
                            chunk_size=int(chunk_size),
                            chunk_overlap=int(chunk_overlap),
                        )
                        emb = cached_embedder(DEFAULT_EMBED)
                        st.session_state.corpus_chunks = chunks
                        st.session_state.faiss_index = build_retrieval_index(emb, chunks)
                        st.session_state.source_label = uploaded.name
                        st.session_state.pop("last_rag", None)
                        st.session_state.pop("last_question", None)
                    st.success(f"Indexed {len(chunks)} chunks from {uploaded.name}.")

        st.divider()
        st.header("Retrieval")
        retrieve_k = st.slider("FAISS top-K (pool)", min_value=3, max_value=30, value=10)
        final_k = st.slider("Final passages to LLM", min_value=1, max_value=10, value=3)
        use_rerank = st.checkbox("Use cross-encoder reranker", value=True)
        if use_rerank:
            st.caption(f"Model: `{DEFAULT_RERANK}` (cached after first load).")

        st.divider()
        st.header("Prompt & context")
        template_key = st.selectbox(
            "Prompt template",
            options=list(PROMPT_TEMPLATES.keys()),
            format_func=lambda k: f"{k}",
        )
        prompt_template = PROMPT_TEMPLATES[template_key]
        max_context_chars = st.number_input(
            "Max context chars (truncation)", min_value=500, max_value=32000, value=6000, step=500
        )
        truncation = st.selectbox(
            "Truncation strategy",
            options=["head", "tail", "middle"],
            format_func=lambda x: x,
        )

        st.divider()
        st.header("LLM")
        backend = st.selectbox(
            "Backend",
            options=["mock", "gemini", "openai", "ollama"],
            help="mock: no API. gemini / openai / ollama use .env keys like the experiments.",
        )

    col_q, col_go = st.columns([4, 1])
    with col_q:
        question = st.text_input(
            "Question",
            placeholder="Ask something answerable from your document…",
            label_visibility="collapsed",
        )
    with col_go:
        ask = st.button("Ask", type="primary", use_container_width=True)

    ready = (
        st.session_state.faiss_index is not None
        and len(st.session_state.corpus_chunks) > 0
    )
    if not ready:
        st.info("Upload a file and click **Build index from upload** in the sidebar to begin.")
        return

    st.caption(f"Indexed document: **{st.session_state.source_label}** — {len(st.session_state.corpus_chunks)} chunks.")

    if ask and question.strip():
        embedder = cached_embedder(DEFAULT_EMBED)
        reranker = cached_reranker(DEFAULT_RERANK) if use_rerank else None
        try:
            gen = make_generator(backend)
        except Exception as e:
            st.error(f"Could not initialize generator ({backend}): {e}")
            return

        with st.spinner("Retrieving and generating…"):
            result = run_pipeline(
                question.strip(),
                embedder=embedder,
                corpus_chunks=st.session_state.corpus_chunks,
                faiss_index=st.session_state.faiss_index,
                retrieve_k=retrieve_k,
                final_k=final_k,
                use_rerank=use_rerank,
                reranker=reranker,
                prompt_template=prompt_template,
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
        st.info("Type a question and click **Ask** to see retrieval traces and the generated answer.")
        return

    ui_id = st.session_state.get("last_rag_ui", 0)
    last_q = st.session_state.get("last_question", "")
    fk = int(st.session_state.get("last_final_k", final_k))

    st.subheader("Answer")
    if last_q:
        st.caption(f"Question: {last_q}")
    st.write(result["answer"])

    tab_chunks, tab_rerank, tab_prompt = st.tabs(
        ["Retrieved chunks (FAISS)", "Reranking (cross-encoder)", "Final prompt & answer"]
    )

    with tab_chunks:
        st.markdown(
            "Top-K passages from the bi-encoder + FAISS index (inner product on normalized embeddings ≈ cosine similarity)."
        )
        for row in result["retrieved"]:
            with st.expander(
                f"Rank {row['faiss_rank']} · chunk #{row['chunk_index']} · score {row['faiss_score']:.4f}"
            ):
                st.text_area(
                    "Chunk text",
                    row["text"],
                    height=min(240, 24 + 12 * row["text"].count("\n")),
                    key=f"{ui_id}_chunk_{row['faiss_rank']}_{row['chunk_index']}",
                    label_visibility="collapsed",
                )

    with tab_rerank:
        if not result["use_rerank"]:
            st.info("Cross-encoder reranking is **off**. Final passages are the first **Final passages to LLM** rows from the FAISS list (in FAISS order).")
        elif not result["rerank_rows"]:
            st.warning("No rerank rows (empty pool).")
        else:
            st.markdown(
                "**New rank** is after sorting by cross-encoder score. **FAISS rank** is the original bi-encoder order within the retrieved pool."
            )
            for row in result["rerank_rows"]:
                with st.expander(
                    f"New rank {row['new_rank']} (was FAISS #{row['faiss_rank']}) · CE score {row['cross_encoder_score']:.4f}"
                ):
                    st.text_area(
                        "Passage",
                        row["text"],
                        height=min(240, 24 + 12 * row["text"].count("\n")),
                        key=f"{ui_id}_rr_{row['new_rank']}",
                        label_visibility="collapsed",
                    )
            st.caption(
                f"Passages **{1}–{min(fk, len(result['rerank_rows']))}** above are merged into the context for the LLM (see next tab)."
            )

    with tab_prompt:
        raw_len = len(result.get("raw_context", ""))
        st.caption(
            f"Context length after merging passages: {raw_len} chars. "
            f"Truncation: **{result['truncation']}** to **{result['max_context_chars']}** chars before formatting the prompt."
        )
        st.text_area(
            "Prompt sent to the LLM",
            result["prompt"],
            height=360,
            key=f"{ui_id}_prompt_full",
        )
        st.subheader("Model reply (duplicate)")
        st.write(result["answer"])


if __name__ == "__main__":
    main()
