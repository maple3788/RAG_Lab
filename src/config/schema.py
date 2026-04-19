"""
Nested Pydantic configs for RAG query and ingest. Defaults load from ``conf/*.yaml`` via OmegaConf.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from src.document_ingest_pipeline import IngestPipelineConfig
from src.milvus_metadata import DEFAULT_SECTION, metadata_filter_to_milvus_expr
from src.storage.milvus_store import MilvusSearchConfig


class MetadataFilterConfig(BaseModel):
    """Milvus scalar filters (aligned with API ``QueryRequest.filter_*``)."""

    model_config = {"extra": "forbid"}

    doc_date_min: Optional[str] = None
    doc_date_max: Optional[str] = None
    source_type: Optional[str] = None
    section: Optional[str] = None

    def milvus_expression(self) -> Optional[str]:
        return metadata_filter_to_milvus_expr(
            doc_date_min=self.doc_date_min,
            doc_date_max=self.doc_date_max,
            source_type=self.source_type,
            section=self.section,
        )


class MilvusRuntimeConfig(BaseModel):
    """Query-time Milvus client parameters (plus job routing)."""

    model_config = {"extra": "forbid"}

    index_type: str = "AUTOINDEX"
    metric_type: str = "COSINE"
    ivf_nprobe: int = Field(default=32, ge=1, le=256)
    hnsw_ef: int = Field(default=64, ge=8, le=512)
    job_id: Optional[str] = None
    job_ids: Optional[List[str]] = None
    collection: Optional[str] = None

    def search_config(self) -> MilvusSearchConfig:
        return MilvusSearchConfig(
            metric_type=self.metric_type.strip(),
            ivf_nprobe=int(self.ivf_nprobe),
            hnsw_ef=int(self.hnsw_ef),
        )

    def resolved_job_ids(self) -> List[str]:
        ids = [x.strip() for x in (self.job_ids or []) if x and str(x).strip()]
        if not ids and self.job_id and str(self.job_id).strip():
            ids = [str(self.job_id).strip()]
        return ids


class RetrievalConfig(BaseModel):
    model_config = {"extra": "forbid"}

    retrieve_k: int = Field(default=10, ge=1, le=128)
    final_k: int = Field(default=3, ge=1, le=32)
    mode: Literal["dense", "hybrid"] = "dense"
    fusion_list_k: Optional[int] = Field(default=30, ge=5, le=200)
    rrf_k: int = Field(default=60, ge=10, le=200)
    vdb_backend: Literal["faiss", "milvus"] = "milvus"


class PipelineFeaturesConfig(BaseModel):
    model_config = {"extra": "forbid"}

    use_query_rewrite: bool = False
    use_query_decomposition: bool = False
    max_subqueries: int = Field(default=3, ge=2, le=12)
    use_filter_generation: bool = False
    min_filter_keep: int = Field(default=3, ge=1, le=20)
    use_reflection_loops: bool = False
    max_reflection_loops: int = Field(default=2, ge=1, le=8)
    require_citations: bool = True
    use_rerank: bool = True


class PromptConfig(BaseModel):
    model_config = {"extra": "forbid"}

    template_key: str = "default"
    max_context_chars: int = Field(default=6000, ge=200, le=100_000)
    truncation: Literal["head", "tail", "middle"] = "head"


class SemanticCacheConfig(BaseModel):
    model_config = {"extra": "forbid"}

    enabled: bool = False
    threshold: float = Field(default=0.93, ge=0.5, le=0.999)
    max_entries: int = Field(default=512, ge=1, le=50_000)


class RAGPipelineConfig(BaseModel):
    """
    Single object for ``run_pipeline`` scalar/routing options. Runtime deps (embedder, stores)
    stay as separate arguments.
    """

    model_config = {"extra": "forbid"}

    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    milvus: MilvusRuntimeConfig = Field(default_factory=MilvusRuntimeConfig)
    metadata: MetadataFilterConfig = Field(default_factory=MetadataFilterConfig)
    features: PipelineFeaturesConfig = Field(default_factory=PipelineFeaturesConfig)
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    semantic_cache: SemanticCacheConfig = Field(default_factory=SemanticCacheConfig)


class IngestSettings(BaseModel):
    """Ingest options loadable from YAML; converts to frozen ``IngestPipelineConfig``."""

    model_config = {"extra": "forbid"}

    chunk_size: int = Field(default=512, ge=32, le=4096)
    chunk_overlap: int = Field(default=64, ge=0, le=2048)
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    extraction: Literal["shallow", "full"] = "shallow"
    summarization: Literal["single", "hierarchical", "iterative"] = "single"
    llm_min_interval_seconds: float = Field(default=0.75, ge=0.0, le=60.0)
    milvus_index_type: str = "AUTOINDEX"
    milvus_metric_type: str = "COSINE"
    milvus_ivf_nlist: int = 1024
    milvus_hnsw_m: int = 16
    milvus_hnsw_ef_construction: int = 200
    milvus_upsert_batch_size: int = Field(default=256, ge=1, le=4096)
    doc_date: Optional[str] = None
    doc_section: str = DEFAULT_SECTION

    @field_validator("doc_section", mode="before")
    @classmethod
    def _empty_section_to_default(cls, v: object) -> str:
        s = (str(v) if v is not None else "").strip()
        return s if s else DEFAULT_SECTION

    def to_pipeline_dataclass(self) -> IngestPipelineConfig:
        return IngestPipelineConfig(
            chunk_size=int(self.chunk_size),
            chunk_overlap=int(self.chunk_overlap),
            embedding_model=str(self.embedding_model),
            extraction=self.extraction,
            summarization=self.summarization,
            llm_min_interval_seconds=float(self.llm_min_interval_seconds),
            milvus_index_type=str(self.milvus_index_type),
            milvus_metric_type=str(self.milvus_metric_type),
            milvus_ivf_nlist=int(self.milvus_ivf_nlist),
            milvus_hnsw_m=int(self.milvus_hnsw_m),
            milvus_hnsw_ef_construction=int(self.milvus_hnsw_ef_construction),
            milvus_upsert_batch_size=int(self.milvus_upsert_batch_size),
            doc_date=(self.doc_date or "").strip() or None,
            doc_section=str(self.doc_section).strip() or DEFAULT_SECTION,
        )
