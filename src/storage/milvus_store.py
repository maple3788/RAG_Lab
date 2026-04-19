from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List, Optional

from dotenv import load_dotenv

from src.embedder import EmbeddingModel, prepare_passage, prepare_query
from src.milvus_metadata import (
    DEFAULT_SECTION,
    FIELD_DOC_DATE,
    FIELD_SECTION,
    FIELD_SOURCE_TYPE,
    UNKNOWN_DOC_DATE,
    combine_job_filter_with_metadata,
)


@dataclass(frozen=True)
class MilvusSettings:
    uri: str
    token: str = ""
    collection: str = "rag_chunks"
    #: Seconds for MilvusClient RPC/connect (avoids hanging forever when Milvus is down)
    timeout: float = 10.0

    @staticmethod
    def from_env() -> "MilvusSettings":
        load_dotenv()
        raw_timeout = os.environ.get("MILVUS_CONNECT_TIMEOUT", "10")
        try:
            timeout = float(raw_timeout)
        except ValueError:
            timeout = 10.0
        return MilvusSettings(
            uri=os.environ.get("MILVUS_URI", "http://localhost:19530"),
            token=os.environ.get("MILVUS_TOKEN", ""),
            collection=os.environ.get("MILVUS_COLLECTION", "rag_chunks"),
            timeout=timeout,
        )


@dataclass(frozen=True)
class MilvusIndexConfig:
    """
    Milvus ANN index knobs used when creating vector index on ``vector`` field.
    """

    index_type: str = "AUTOINDEX"
    metric_type: str = "COSINE"
    ivf_nlist: int = 1024
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200


@dataclass(frozen=True)
class MilvusSearchConfig:
    """
    Per-query ANN search knobs.
    """

    metric_type: str = "COSINE"
    ivf_nprobe: int = 32
    hnsw_ef: int = 64


class MilvusChunkStore:
    """
    Thin wrapper over Milvus for chunk vectors.

    Data model:
    - id: int (deterministic from job_id + chunk_index)
    - job_id: str
    - chunk_index: int
    - text: str
    - vector: float[]
    - doc_date: str (``YYYY-MM-DD`` or ``unknown``)
    - section: str (e.g. ``document`` when not structure-aware)
    - source_type: str (e.g. ``pdf``, ``txt``)
    """

    def __init__(self, settings: Optional[MilvusSettings] = None):
        self.settings = settings or MilvusSettings.from_env()
        try:
            from pymilvus import MilvusClient
        except Exception as e:  # pragma: no cover - import guard
            raise RuntimeError(
                "pymilvus is required for Milvus backend. Install with `pip install pymilvus`."
            ) from e
        kwargs: dict = {
            "uri": self.settings.uri,
            "timeout": self.settings.timeout,
        }
        if self.settings.token:
            kwargs["token"] = self.settings.token
        self._client = MilvusClient(**kwargs)

    def _index_params_for(self, cfg: MilvusIndexConfig) -> dict[str, Any]:
        idx = (cfg.index_type or "AUTOINDEX").upper()
        out: dict[str, Any] = {
            "index_type": idx,
            "metric_type": (cfg.metric_type or "COSINE").upper(),
            "params": {},
        }
        if idx == "IVF_FLAT":
            out["params"] = {"nlist": int(cfg.ivf_nlist)}
        elif idx == "HNSW":
            out["params"] = {
                "M": int(cfg.hnsw_m),
                "efConstruction": int(cfg.hnsw_ef_construction),
            }
        return out

    def _collection_name(self, collection_name: Optional[str] = None) -> str:
        return (collection_name or self.settings.collection).strip()

    def _ensure_collection(
        self, dim: int, *, metric_type: str, collection_name: Optional[str] = None
    ) -> None:
        name = self._collection_name(collection_name)
        if self._client.has_collection(collection_name=name):
            return
        # pymilvus accepts id_type "int" | "string" (not the literal "int64"); "int" → INT64.
        self._client.create_collection(
            collection_name=name,
            dimension=dim,
            primary_field_name="id",
            id_type="int",
            vector_field_name="vector",
            metric_type=(metric_type or "COSINE").upper(),
            auto_id=False,
            consistency_level="Strong",
            enable_dynamic_field=True,
        )

    def _ensure_index(
        self, cfg: MilvusIndexConfig, *, collection_name: Optional[str] = None
    ) -> None:
        name = self._collection_name(collection_name)
        params = self._index_params_for(cfg)
        # Milvus allows one distinct index per vector field. If an index already
        # exists on this collection, re-creation would fail and only add noisy logs.
        try:
            existing = self._client.list_indexes(collection_name=name)
            if existing:
                return
        except Exception:
            # If index listing is unavailable on this client version, continue and
            # rely on create_index compatibility handling below.
            pass
        # Compatibility across pymilvus versions.
        try:
            if hasattr(self._client, "prepare_index_params"):
                ip = self._client.prepare_index_params()
                ip.add_index(
                    field_name="vector",
                    index_type=params["index_type"],
                    metric_type=params["metric_type"],
                    params=params.get("params", {}),
                )
                self._client.create_index(collection_name=name, index_params=ip)
            else:
                self._client.create_index(
                    collection_name=name,
                    field_name="vector",
                    index_type=params["index_type"],
                    metric_type=params["metric_type"],
                    params=params.get("params", {}),
                )
        except Exception:
            # Some configurations auto-index; keep ingest robust.
            return

    def describe_collection(
        self, *, collection_name: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Return live collection/index metadata for debugging UI.
        """
        name = self._collection_name(collection_name)
        out: dict[str, Any] = {"collection": name}
        try:
            if not self._client.has_collection(collection_name=name):
                out["exists"] = False
                return out
            out["exists"] = True
            try:
                out["stats"] = self._client.get_collection_stats(collection_name=name)
            except Exception:
                pass
            try:
                out["index"] = self._client.describe_index(
                    collection_name=name,
                    index_name="",
                )
            except Exception:
                pass
            try:
                out["indexes"] = self._client.list_indexes(collection_name=name)
            except Exception:
                pass
            return out
        except Exception as e:
            out["error"] = str(e)
            return out

    def list_collections(self) -> List[str]:
        try:
            return list(self._client.list_collections())
        except Exception:
            return []

    def load_collection(
        self, *, collection_name: Optional[str] = None, release_others: bool = False
    ) -> dict[str, Any]:
        """
        Load one collection for search. Optionally release all other loaded collections.
        """
        target = self._collection_name(collection_name)
        out: dict[str, Any] = {"target": target, "released": []}
        if not self._client.has_collection(collection_name=target):
            out["exists"] = False
            return out
        out["exists"] = True
        if release_others:
            for cname in self.list_collections():
                if cname == target:
                    continue
                try:
                    self._client.release_collection(collection_name=cname)
                    out["released"].append(cname)
                except Exception:
                    pass
        try:
            self._client.load_collection(collection_name=target)
            out["loaded"] = True
        except Exception as e:
            out["loaded"] = False
            out["error"] = str(e)
        return out

    @staticmethod
    def _stable_id(job_id: str, chunk_index: int) -> int:
        # Stable per (job, chunk), fits signed int64.
        raw = f"{job_id}:{chunk_index}"
        return abs(hash(raw)) % (2**63 - 1)

    def upsert_job_chunks(
        self,
        *,
        job_id: str,
        chunk_texts: List[str],
        embedder: EmbeddingModel,
        batch_size: int = 256,
        index_config: Optional[MilvusIndexConfig] = None,
        collection_name: Optional[str] = None,
        chunk_metadatas: Optional[List[dict[str, str]]] = None,
    ) -> int:
        if not chunk_texts:
            return 0
        passages = [prepare_passage(embedder.name, t) for t in chunk_texts]
        vecs = embedder.encode(passages)
        dim = int(vecs.shape[1])
        idx_cfg = index_config or MilvusIndexConfig()
        target = self._collection_name(collection_name)
        self._ensure_collection(
            dim, metric_type=idx_cfg.metric_type, collection_name=target
        )
        self._ensure_index(idx_cfg, collection_name=target)

        rows = []
        for i, text in enumerate(chunk_texts):
            meta: dict[str, str]
            if chunk_metadatas is not None and i < len(chunk_metadatas):
                m = chunk_metadatas[i]
                meta = {
                    FIELD_DOC_DATE: str(m.get(FIELD_DOC_DATE, UNKNOWN_DOC_DATE)),
                    FIELD_SECTION: str(m.get(FIELD_SECTION, DEFAULT_SECTION)),
                    FIELD_SOURCE_TYPE: str(m.get(FIELD_SOURCE_TYPE, "other")),
                }
            else:
                meta = {
                    FIELD_DOC_DATE: UNKNOWN_DOC_DATE,
                    FIELD_SECTION: DEFAULT_SECTION,
                    FIELD_SOURCE_TYPE: "other",
                }
            rows.append(
                {
                    "id": self._stable_id(job_id, i),
                    "job_id": job_id,
                    "chunk_index": i,
                    "text": text,
                    "vector": vecs[i].tolist(),
                    **meta,
                }
            )
        name = target
        for start in range(0, len(rows), max(1, int(batch_size))):
            self._client.upsert(
                collection_name=name,
                data=rows[start : start + batch_size],
            )
        self._client.flush(collection_name=name)
        return len(rows)

    def search_job_chunks(
        self,
        *,
        job_id: str,
        query: str,
        embedder: EmbeddingModel,
        top_k: int,
        search_config: Optional[MilvusSearchConfig] = None,
        index_type: str = "AUTOINDEX",
        collection_name: Optional[str] = None,
        metadata_filter_expr: Optional[str] = None,
    ) -> List[dict]:
        if top_k <= 0:
            return []
        s_cfg = search_config or MilvusSearchConfig()
        idx_type = (index_type or "AUTOINDEX").upper()
        target = self._collection_name(collection_name)
        q = prepare_query(embedder.name, query)
        q_vec = embedder.encode([q])[0].tolist()
        # Collection is created on first upsert; search before any sync had no collection.
        self._ensure_collection(
            len(q_vec), metric_type=s_cfg.metric_type, collection_name=target
        )
        search_params: dict[str, Any] = {
            "metric_type": (s_cfg.metric_type or "COSINE").upper(),
            "params": {},
        }
        if idx_type == "IVF_FLAT":
            search_params["params"] = {"nprobe": int(s_cfg.ivf_nprobe)}
        elif idx_type == "HNSW":
            search_params["params"] = {"ef": int(s_cfg.hnsw_ef)}
        flt = combine_job_filter_with_metadata(
            f'job_id == "{job_id}"', metadata_filter_expr
        )
        out = self._client.search(
            collection_name=target,
            data=[q_vec],
            filter=flt,
            limit=int(top_k),
            output_fields=[
                "job_id",
                "chunk_index",
                "text",
                FIELD_DOC_DATE,
                FIELD_SECTION,
                FIELD_SOURCE_TYPE,
            ],
            search_params=search_params,
        )
        hits = out[0] if out else []
        rows: List[dict] = []
        for i, h in enumerate(hits, start=1):
            ent = h.get("entity", {})
            rows.append(
                {
                    "query_used": query,
                    "faiss_rank": i,
                    "chunk_index": int(ent.get("chunk_index", -1)),
                    "faiss_score": float(h.get("distance", 0.0)),
                    "text": str(ent.get("text", "")),
                    "job_id": str(ent.get("job_id", job_id)),
                    FIELD_DOC_DATE: str(ent.get(FIELD_DOC_DATE, UNKNOWN_DOC_DATE)),
                    FIELD_SECTION: str(ent.get(FIELD_SECTION, DEFAULT_SECTION)),
                    FIELD_SOURCE_TYPE: str(ent.get(FIELD_SOURCE_TYPE, "other")),
                }
            )
        return rows

    @staticmethod
    def _filter_for_job_ids(job_ids: List[str]) -> str:
        cleaned = [j.strip() for j in job_ids if j and j.strip()]
        if not cleaned:
            raise ValueError("job_ids is empty")
        for j in cleaned:
            if '"' in j or "\\" in j:
                raise ValueError(f"unsupported job_id character in: {j!r}")
        parts = [f'job_id == "{j}"' for j in cleaned]
        if len(parts) == 1:
            return parts[0]
        return "(" + " || ".join(parts) + ")"

    def search_multi_job_chunks(
        self,
        *,
        job_ids: List[str],
        query: str,
        embedder: EmbeddingModel,
        top_k: int,
        search_config: Optional[MilvusSearchConfig] = None,
        index_type: str = "AUTOINDEX",
        collection_name: Optional[str] = None,
        metadata_filter_expr: Optional[str] = None,
    ) -> List[dict]:
        """
        Vector search restricted to rows whose ``job_id`` is one of ``job_ids`` (same Milvus collection).
        """
        if top_k <= 0 or not job_ids:
            return []
        if len(job_ids) == 1:
            return self.search_job_chunks(
                job_id=job_ids[0].strip(),
                query=query,
                embedder=embedder,
                top_k=top_k,
                search_config=search_config,
                index_type=index_type,
                collection_name=collection_name,
                metadata_filter_expr=metadata_filter_expr,
            )
        s_cfg = search_config or MilvusSearchConfig()
        idx_type = (index_type or "AUTOINDEX").upper()
        target = self._collection_name(collection_name)
        flt = combine_job_filter_with_metadata(
            self._filter_for_job_ids(job_ids), metadata_filter_expr
        )
        q = prepare_query(embedder.name, query)
        q_vec = embedder.encode([q])[0].tolist()
        self._ensure_collection(
            len(q_vec), metric_type=s_cfg.metric_type, collection_name=target
        )
        search_params: dict[str, Any] = {
            "metric_type": (s_cfg.metric_type or "COSINE").upper(),
            "params": {},
        }
        if idx_type == "IVF_FLAT":
            search_params["params"] = {"nprobe": int(s_cfg.ivf_nprobe)}
        elif idx_type == "HNSW":
            search_params["params"] = {"ef": int(s_cfg.hnsw_ef)}
        out = self._client.search(
            collection_name=target,
            data=[q_vec],
            filter=flt,
            limit=int(top_k),
            output_fields=[
                "job_id",
                "chunk_index",
                "text",
                FIELD_DOC_DATE,
                FIELD_SECTION,
                FIELD_SOURCE_TYPE,
            ],
            search_params=search_params,
        )
        hits = out[0] if out else []
        rows: List[dict] = []
        for i, h in enumerate(hits, start=1):
            ent = h.get("entity", {})
            rows.append(
                {
                    "query_used": query,
                    "faiss_rank": i,
                    "chunk_index": int(ent.get("chunk_index", -1)),
                    "faiss_score": float(h.get("distance", 0.0)),
                    "text": str(ent.get("text", "")),
                    "job_id": str(ent.get("job_id", "")),
                    FIELD_DOC_DATE: str(ent.get(FIELD_DOC_DATE, UNKNOWN_DOC_DATE)),
                    FIELD_SECTION: str(ent.get(FIELD_SECTION, DEFAULT_SECTION)),
                    FIELD_SOURCE_TYPE: str(ent.get(FIELD_SOURCE_TYPE, "other")),
                }
            )
        return rows
