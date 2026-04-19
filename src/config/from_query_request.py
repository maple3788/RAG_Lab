"""
Map API ``QueryRequest`` (Pydantic) → shared ``RAGPipelineConfig`` (YAML defaults + request fields).
"""

from __future__ import annotations

import os
from typing import Any, Literal, cast

from pydantic import BaseModel

from src.config.loader import load_rag_pipeline_config
from src.config.schema import RAGPipelineConfig


def rag_pipeline_config_from_query_request(req: BaseModel) -> RAGPipelineConfig:
    """
    Merge ``conf/rag_pipeline.yaml`` defaults with a resolved ``QueryRequest``-like model.

    API-only flags (``use_semantic_cache``, ``use_rerank``, …) override the matching
    nested sections; all other ``PipelineFeaturesConfig`` fields keep YAML defaults
    (query rewrite, decomposition, etc. are not exposed on the API).
    """
    base = load_rag_pipeline_config()
    d: dict[str, Any] = req.model_dump()
    metric = d.get("milvus_metric_type") or base.milvus.metric_type
    metric = str(metric).strip() if metric else base.milvus.metric_type

    trunc = d.get("truncation") or "head"
    if trunc not in ("head", "tail", "middle"):
        trunc = "head"
    truncation = cast(Literal["head", "tail", "middle"], trunc)

    mode = d.get("retrieval_mode") or "dense"
    if mode not in ("dense", "hybrid"):
        mode = "dense"
    mode = cast(Literal["dense", "hybrid"], mode)

    max_entries = int(
        os.environ.get(
            "RAG_SEMANTIC_CACHE_MAX_ENTRIES", str(base.semantic_cache.max_entries)
        )
    )

    return base.model_copy(
        deep=True,
        update={
            "retrieval": base.retrieval.model_copy(
                update={
                    "retrieve_k": int(d.get("retrieve_k", base.retrieval.retrieve_k)),
                    "final_k": int(d.get("final_k", base.retrieval.final_k)),
                    "mode": mode,
                    "fusion_list_k": d.get("fusion_list_k"),
                    "rrf_k": int(d.get("rrf_k", base.retrieval.rrf_k)),
                    "vdb_backend": "milvus",
                }
            ),
            "milvus": base.milvus.model_copy(
                update={
                    "job_id": d.get("job_id"),
                    "job_ids": d.get("job_ids"),
                    "collection": d.get("milvus_collection"),
                    "index_type": str(
                        d.get("milvus_index_type") or base.milvus.index_type
                    ).strip()
                    or "AUTOINDEX",
                    "metric_type": metric,
                    "ivf_nprobe": int(
                        d.get("milvus_ivf_nprobe", base.milvus.ivf_nprobe)
                    ),
                    "hnsw_ef": int(d.get("milvus_hnsw_ef", base.milvus.hnsw_ef)),
                }
            ),
            "metadata": base.metadata.model_copy(
                update={
                    "doc_date_min": d.get("filter_doc_date_min"),
                    "doc_date_max": d.get("filter_doc_date_max"),
                    "source_type": d.get("filter_source_type"),
                    "section": d.get("filter_section"),
                }
            ),
            "features": base.features.model_copy(
                update={
                    "use_rerank": bool(d.get("use_rerank", base.features.use_rerank)),
                }
            ),
            "prompt": base.prompt.model_copy(
                update={
                    "template_key": str(
                        d.get("prompt_template") or base.prompt.template_key
                    ),
                    "max_context_chars": int(
                        d.get("max_context_chars", base.prompt.max_context_chars)
                    ),
                    "truncation": truncation,
                }
            ),
            "semantic_cache": base.semantic_cache.model_copy(
                update={
                    "enabled": bool(
                        d.get("use_semantic_cache", base.semantic_cache.enabled)
                    ),
                    "threshold": float(
                        d.get(
                            "semantic_cache_threshold",
                            base.semantic_cache.threshold,
                        )
                    ),
                    "max_entries": max_entries,
                }
            ),
        },
    )
