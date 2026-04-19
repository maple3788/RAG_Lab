"""
Load nested defaults from ``conf/*.yaml`` with OmegaConf, validate with Pydantic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

from omegaconf import OmegaConf

from src.config.schema import IngestSettings, RAGPipelineConfig


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    return OmegaConf.load(path)


def load_rag_pipeline_config(
    *,
    config_path: Optional[Union[str, Path]] = None,
    cli_overrides: Optional[list[str]] = None,
) -> RAGPipelineConfig:
    """
    Load ``RAGPipelineConfig`` from ``conf/rag_pipeline.yaml`` (or ``config_path``).

    Optional ``cli_overrides`` are OmegaConf dotlist strings, e.g.
    ``["retrieval.retrieve_k=20", "features.use_rerank=false"]``.
    """
    base = (
        Path(config_path)
        if config_path
        else _repo_root() / "conf" / "rag_pipeline.yaml"
    )
    cfg = _load_yaml(base)
    if cli_overrides:
        ov = OmegaConf.from_cli(cli_overrides)
        cfg = OmegaConf.merge(cfg, ov)
    raw = OmegaConf.to_container(cfg, resolve=True)
    return RAGPipelineConfig.model_validate(raw)


def load_ingest_config(
    *,
    config_path: Optional[Union[str, Path]] = None,
    cli_overrides: Optional[list[str]] = None,
) -> IngestSettings:
    """Load ``IngestSettings`` from ``conf/ingest.yaml``."""
    base = Path(config_path) if config_path else _repo_root() / "conf" / "ingest.yaml"
    cfg = _load_yaml(base)
    if cli_overrides:
        ov = OmegaConf.from_cli(cli_overrides)
        cfg = OmegaConf.merge(cfg, ov)
    raw = OmegaConf.to_container(cfg, resolve=True)
    return IngestSettings.model_validate(raw)


def merge_ingest_with_dict(
    base: IngestSettings, updates: dict[str, Any]
) -> IngestSettings:
    """Shallow merge: ``updates`` keys override top-level fields on a copy."""
    return base.model_copy(update=updates)
