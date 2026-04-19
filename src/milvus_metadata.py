"""
Per-chunk metadata for Milvus scalar filtering (aligned with ingest + API + Streamlit).

``doc_date`` is stored as ``YYYY-MM-DD`` or the literal ``unknown`` when not set.
``section`` defaults to ``document`` for flat chunking; ``source_type`` is a short
label derived from the upload filename (e.g. ``pdf``, ``txt``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional

# Stored on each Milvus row / MinIO chunk record
FIELD_DOC_DATE = "doc_date"
FIELD_SECTION = "section"
FIELD_SOURCE_TYPE = "source_type"

UNKNOWN_DOC_DATE = "unknown"
DEFAULT_SECTION = "document"


_iso_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def source_type_from_filename(filename: str) -> str:
    lower = (filename or "").lower().strip()
    for suf, label in (
        (".pdf", "pdf"),
        (".txt", "txt"),
        (".md", "md"),
        (".markdown", "md"),
        (".html", "html"),
        (".htm", "html"),
        (".docx", "docx"),
    ):
        if lower.endswith(suf):
            return label
    return "other"


def normalize_iso_date_or_unknown(raw: Optional[str]) -> str:
    s = (raw or "").strip()
    if not s:
        return UNKNOWN_DOC_DATE
    if _iso_re.match(s):
        return s
    return UNKNOWN_DOC_DATE


def assert_safe_milvus_string(s: str, *, field: str) -> str:
    if '"' in s or "\\" in s:
        raise ValueError(f"{field} must not contain quotes or backslashes: {s!r}")
    return s.strip()


def metadata_filter_to_milvus_expr(
    *,
    doc_date_min: Optional[str] = None,
    doc_date_max: Optional[str] = None,
    source_type: Optional[str] = None,
    section: Optional[str] = None,
) -> Optional[str]:
    """
    Build a Milvus boolean expression fragment (no job_id scope).
    Returns None when no metadata constraints are set.
    """
    parts: list[str] = []
    if doc_date_min:
        dm = assert_safe_milvus_string(doc_date_min, field="filter_doc_date_min")
        if not _iso_re.match(dm):
            raise ValueError("filter_doc_date_min must be YYYY-MM-DD")
        parts.append(
            f'({FIELD_DOC_DATE} != "{UNKNOWN_DOC_DATE}" && {FIELD_DOC_DATE} >= "{dm}")'
        )
    if doc_date_max:
        dm = assert_safe_milvus_string(doc_date_max, field="filter_doc_date_max")
        if not _iso_re.match(dm):
            raise ValueError("filter_doc_date_max must be YYYY-MM-DD")
        parts.append(
            f'({FIELD_DOC_DATE} != "{UNKNOWN_DOC_DATE}" && {FIELD_DOC_DATE} <= "{dm}")'
        )
    if source_type:
        st = assert_safe_milvus_string(source_type, field="filter_source_type")
        parts.append(f'{FIELD_SOURCE_TYPE} == "{st}"')
    if section:
        sec = assert_safe_milvus_string(section, field="filter_section")
        parts.append(f'{FIELD_SECTION} == "{sec}"')
    if not parts:
        return None
    return " && ".join(parts)


def combine_job_filter_with_metadata(
    job_expr: str, metadata_expr: Optional[str]
) -> str:
    if not metadata_expr:
        return job_expr
    return f"({job_expr}) && ({metadata_expr})"


@dataclass(frozen=True)
class ChunkMetadataFilter:
    """In-process filter for MinIO chunk records (hybrid BM25 path)."""

    doc_date_min: Optional[str] = None
    doc_date_max: Optional[str] = None
    source_type: Optional[str] = None
    section: Optional[str] = None

    @staticmethod
    def from_query_params(
        *,
        doc_date_min: Optional[str],
        doc_date_max: Optional[str],
        source_type: Optional[str],
        section: Optional[str],
    ) -> Optional["ChunkMetadataFilter"]:
        if not any(
            (
                (doc_date_min or "").strip(),
                (doc_date_max or "").strip(),
                (source_type or "").strip(),
                (section or "").strip(),
            )
        ):
            return None
        return ChunkMetadataFilter(
            doc_date_min=(doc_date_min or "").strip() or None,
            doc_date_max=(doc_date_max or "").strip() or None,
            source_type=(source_type or "").strip() or None,
            section=(section or "").strip() or None,
        )

    def matches_record(self, rec: Mapping[str, Any]) -> bool:
        dd = str(rec.get(FIELD_DOC_DATE) or UNKNOWN_DOC_DATE)
        sec = str(rec.get(FIELD_SECTION) or DEFAULT_SECTION)
        st = str(rec.get(FIELD_SOURCE_TYPE) or "other")
        if self.source_type is not None and st != self.source_type:
            return False
        if self.section is not None and sec != self.section:
            return False
        if self.doc_date_min or self.doc_date_max:
            if dd == UNKNOWN_DOC_DATE:
                return False
            if self.doc_date_min and dd < self.doc_date_min:
                return False
            if self.doc_date_max and dd > self.doc_date_max:
                return False
        return True


def chunk_record_defaults() -> dict[str, str]:
    return {
        FIELD_DOC_DATE: UNKNOWN_DOC_DATE,
        FIELD_SECTION: DEFAULT_SECTION,
        FIELD_SOURCE_TYPE: "other",
    }
