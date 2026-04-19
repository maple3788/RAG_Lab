"""
Persist FAISS indices + id alignments so repeated TREC/BEIR runs skip re-embedding.

Cache key: corpus file fingerprint (mtime + size + max_docs), embedding model id,
and for chunk mode: chunk_size + chunk_overlap.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import faiss

from src.retrieval.retriever import FaissIndex

CACHE_VERSION = 1


def _safe_model_id(model_name: str) -> str:
    return model_name.replace("/", "__").replace(":", "_")


def corpus_fingerprint(corpus_path: Path, max_docs: int | None) -> str:
    st = corpus_path.stat()
    md = "all" if max_docs is None else str(int(max_docs))
    return f"{st.st_mtime_ns}|{st.st_size}|{md}"


def _fp_slug(fingerprint: str) -> str:
    return hashlib.sha256(fingerprint.encode()).hexdigest()[:16]


def stem_doc_level(embedding_model: str, fingerprint: str) -> str:
    return f"v{CACHE_VERSION}_doc_{_safe_model_id(embedding_model)}_{_fp_slug(fingerprint)}"


def stem_chunked(
    embedding_model: str,
    fingerprint: str,
    chunk_size: int,
    chunk_overlap: int,
) -> str:
    return (
        f"v{CACHE_VERSION}_chunk_{_safe_model_id(embedding_model)}_"
        f"{chunk_size}_{chunk_overlap}_{_fp_slug(fingerprint)}"
    )


def _meta_path(cache_dir: Path, stem: str) -> Path:
    return cache_dir / f"{stem}.meta.json"


def _index_path(cache_dir: Path, stem: str) -> Path:
    return cache_dir / f"{stem}.faiss"


def _ids_path(cache_dir: Path, stem: str) -> Path:
    return cache_dir / f"{stem}.ids.json"


def save_doc_level_cache(
    cache_dir: Path,
    stem: str,
    *,
    faiss_index: FaissIndex,
    doc_ids: Sequence[str],
    embedding_model: str,
    corpus_fingerprint_str: str,
    max_docs: int | None,
    normalize_embeddings: bool,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {
        "version": CACHE_VERSION,
        "kind": "doc",
        "embedding_model": embedding_model,
        "corpus_fingerprint": corpus_fingerprint_str,
        "max_docs": max_docs,
        "dim": faiss_index.dim,
        "n_vectors": int(faiss_index.index.ntotal),
        "normalize_embeddings": normalize_embeddings,
    }
    _meta_path(cache_dir, stem).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _ids_path(cache_dir, stem).write_text(
        json.dumps(list(doc_ids), ensure_ascii=False), encoding="utf-8"
    )
    faiss.write_index(faiss_index.index, str(_index_path(cache_dir, stem)))


def try_load_doc_level_cache(
    cache_dir: Path,
    stem: str,
    *,
    embedding_model: str,
    corpus_fingerprint_str: str,
    doc_ids: Sequence[str],
    normalize_embeddings: bool,
) -> FaissIndex | None:
    mp, ip, yp = (
        _meta_path(cache_dir, stem),
        _index_path(cache_dir, stem),
        _ids_path(cache_dir, stem),
    )
    if not (mp.is_file() and ip.is_file() and yp.is_file()):
        return None
    try:
        meta = json.loads(mp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if meta.get("version") != CACHE_VERSION or meta.get("kind") != "doc":
        return None
    if (
        meta.get("embedding_model") != embedding_model
        or meta.get("corpus_fingerprint") != corpus_fingerprint_str
        or bool(meta.get("normalize_embeddings")) != normalize_embeddings
    ):
        return None
    try:
        cached_ids: List[str] = json.loads(yp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if list(doc_ids) != cached_ids:
        return None
    try:
        idx = faiss.read_index(str(ip))
    except RuntimeError:
        return None
    dim = int(idx.d)
    nvec = int(idx.ntotal)
    if (
        meta.get("dim") != dim
        or meta.get("n_vectors") != nvec
        or nvec != len(cached_ids)
    ):
        return None
    return FaissIndex(index=idx, dim=dim)


def save_chunk_cache(
    cache_dir: Path,
    stem: str,
    *,
    faiss_index: FaissIndex,
    chunk_parents: Sequence[str],
    embedding_model: str,
    corpus_fingerprint_str: str,
    max_docs: int | None,
    chunk_size: int,
    chunk_overlap: int,
    normalize_embeddings: bool,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "version": CACHE_VERSION,
        "kind": "chunk",
        "embedding_model": embedding_model,
        "corpus_fingerprint": corpus_fingerprint_str,
        "max_docs": max_docs,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "dim": faiss_index.dim,
        "n_vectors": int(faiss_index.index.ntotal),
        "normalize_embeddings": normalize_embeddings,
    }
    _meta_path(cache_dir, stem).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _ids_path(cache_dir, stem).write_text(
        json.dumps(list(chunk_parents), ensure_ascii=False), encoding="utf-8"
    )
    faiss.write_index(faiss_index.index, str(_index_path(cache_dir, stem)))


def try_load_chunk_cache(
    cache_dir: Path,
    stem: str,
    *,
    embedding_model: str,
    corpus_fingerprint_str: str,
    chunk_parents: Sequence[str],
    chunk_size: int,
    chunk_overlap: int,
    normalize_embeddings: bool,
) -> FaissIndex | None:
    mp, ip, yp = (
        _meta_path(cache_dir, stem),
        _index_path(cache_dir, stem),
        _ids_path(cache_dir, stem),
    )
    if not (mp.is_file() and ip.is_file() and yp.is_file()):
        return None
    try:
        meta = json.loads(mp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if meta.get("version") != CACHE_VERSION or meta.get("kind") != "chunk":
        return None
    if (
        meta.get("embedding_model") != embedding_model
        or meta.get("corpus_fingerprint") != corpus_fingerprint_str
        or int(meta.get("chunk_size", -1)) != chunk_size
        or int(meta.get("chunk_overlap", -1)) != chunk_overlap
        or bool(meta.get("normalize_embeddings")) != normalize_embeddings
    ):
        return None
    try:
        cached_parents: List[str] = json.loads(yp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if list(chunk_parents) != cached_parents:
        return None
    try:
        idx = faiss.read_index(str(ip))
    except RuntimeError:
        return None
    dim = int(idx.d)
    nvec = int(idx.ntotal)
    if (
        meta.get("dim") != dim
        or meta.get("n_vectors") != nvec
        or nvec != len(cached_parents)
    ):
        return None
    return FaissIndex(index=idx, dim=dim)
