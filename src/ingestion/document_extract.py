"""Document extraction: PDF (pypdf) and plain text uploads."""

from __future__ import annotations

import io
import re
import logging
from typing import Any, Iterable, List, Literal, Optional

_log = logging.getLogger(__name__)


def pdf_ocr_dependency_error() -> Optional[str]:
    """
    If PDF rasterization / OCR deps are missing, return a short error string.
    Used to surface actionable install hints (especially on Python 3.13 where
    ``rapidocr-onnxruntime`` 1.4+ may not install).
    """
    try:
        import pypdfium2  # noqa: F401
        from rapidocr_onnxruntime import RapidOCR  # noqa: F401
    except ImportError as e:
        return str(e)
    return None


def try_pdf_metadata_iso_date(data: bytes) -> Optional[str]:
    """
    Best-effort ``YYYY-MM-DD`` from PDF ``/CreationDate`` or ``/ModDate`` metadata.
    Returns ``None`` if unavailable or unparseable.
    """
    try:
        from pypdf import PdfReader
    except Exception:
        return None
    try:
        import io

        reader = PdfReader(io.BytesIO(data))
        meta = reader.metadata
        if not meta:
            return None
        for key in ("/CreationDate", "/ModDate"):
            raw = meta.get(key)
            if raw:
                parsed = _parse_pdf_date_string(str(raw))
                if parsed:
                    return parsed
    except Exception:
        return None
    return None


def _parse_pdf_date_string(s: str) -> Optional[str]:
    """
    PDF dates look like ``D:20230102150400+01'00`` or ``D:20230102``.
    """
    m = re.search(r"D:(\d{4})(\d{2})(\d{2})", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


def extract_text_from_bytes(name: str, data: bytes) -> str:
    lower = name.lower()
    if lower.endswith(".pdf"):
        return extract_pdf_text(data, page_indices=None)
    return data.decode("utf-8", errors="replace")


def extract_pdf_for_ingest(
    data: bytes,
    *,
    page_indices: Optional[List[int]],
    extraction: Literal["shallow", "full"],
) -> str:
    """
    PDF text for ingest.

    ``full`` uses layout-aware extraction, cleanup, and OCR on nearly-empty pages.

    ``shallow`` uses fast pypdf text only; if that yields no usable text (common for
    scanned PDFs or some CJK layouts), this automatically retries with the same
    full pipeline so users do not need to switch extraction mode manually.
    """
    if extraction == "full":
        return extract_pdf_text_full(data, page_indices=page_indices)
    text = extract_pdf_text(data, page_indices=page_indices)
    if (text or "").strip():
        return text
    return extract_pdf_text_full(data, page_indices=page_indices)


def extract_pdf_text(data: bytes, *, page_indices: Optional[List[int]]) -> str:
    """
    ``page_indices``: 1-based page numbers to keep, or ``None`` for all pages.
    """
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    n = len(reader.pages)
    if n == 0:
        return ""
    if page_indices is None:
        indices = range(n)
    else:
        indices = []
        for p in page_indices:
            if isinstance(p, int) and 1 <= p <= n:
                indices.append(p - 1)
        if not indices:
            indices = range(n)
    parts: List[str] = []
    for i in indices:
        parts.append(reader.pages[i].extract_text() or "")
    return "\n\n".join(parts)


def _select_zero_based_indices(
    total_pages: int, page_indices: Optional[List[int]]
) -> List[int]:
    if total_pages <= 0:
        return []
    if page_indices is None:
        return list(range(total_pages))
    out: List[int] = []
    for p in page_indices:
        if isinstance(p, int) and 1 <= p <= total_pages:
            out.append(p - 1)
    return out if out else list(range(total_pages))


def _cleanup_page_text(text: str) -> str:
    """
    Normalize per-page PDF extraction for better downstream chunking:
    - join hyphenated line-break words (``retriev-\\nal`` -> ``retrieval``)
    - trim trailing spaces per line
    - collapse 3+ newlines to paragraph breaks
    """
    s = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"(?<=\w)-\n(?=\w)", "", s)
    s = "\n".join(line.rstrip() for line in s.split("\n"))
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _is_table_like_line(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    if "|" in s:
        return True
    # Heuristic: multiple "columns" separated by 2+ spaces.
    cols = [c for c in re.split(r"\s{2,}", s) if c.strip()]
    return len(cols) >= 3


def _table_line_to_pipe(line: str) -> str:
    s = (line or "").strip()
    if "|" in s:
        cols = [c.strip() for c in s.split("|") if c.strip()]
    else:
        cols = [c.strip() for c in re.split(r"\s{2,}", s) if c.strip()]
    return " | ".join(cols)


def _normalize_tables(text: str) -> str:
    """
    Convert table-like line runs into a normalized pipe-delimited block.
    This helps retrieval keep row/column relationships from PDF layout text.
    """
    lines = (text or "").split("\n")
    out: List[str] = []
    i = 0
    n = len(lines)
    while i < n:
        if _is_table_like_line(lines[i]):
            j = i
            block: List[str] = []
            while j < n and _is_table_like_line(lines[j]):
                block.append(lines[j])
                j += 1
            if len(block) >= 2:
                out.append("[TABLE]")
                out.extend(_table_line_to_pipe(ln) for ln in block)
                out.append("[/TABLE]")
            else:
                out.append(lines[i])
            i = j
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out).strip()


def _extract_page_text_layout(page: object) -> str:
    """
    Try pypdf layout-aware extraction first, then fallback to default text extraction.
    """
    try:
        txt = page.extract_text(extraction_mode="layout")  # type: ignore[attr-defined]
        if txt:
            return txt
    except TypeError:
        # Older pypdf without extraction_mode support.
        pass
    except Exception:
        pass
    try:
        return page.extract_text() or ""
    except Exception:
        return ""


def _join_rapidocr_texts(result: object) -> str:
    """
    Normalize RapidOCR outputs across versions:

    - New API: ``RapidOCROutput`` / similar with ``.txts``
    - Legacy: ``[[box, text, score], ...]`` or ``(result, elapsed)`` unpack
    """
    if result is None:
        return ""
    txts = getattr(result, "txts", None)
    if txts is not None:
        if isinstance(txts, (str, bytes)):
            return str(txts).strip()
        parts: List[str] = []
        for t in txts:
            s = str(t).strip()
            if s:
                parts.append(s)
        return "\n".join(parts)
    if isinstance(result, (list, tuple)) and result:
        parts: List[str] = []
        for item in result:
            if not item:
                continue
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                cell = item[1]
                if isinstance(cell, (list, tuple)) and cell:
                    cell = cell[0]
                s = str(cell).strip()
                if s:
                    parts.append(s)
            elif isinstance(item, dict):
                s = str(item.get("text") or item.get("txt") or "").strip()
                if s:
                    parts.append(s)
        return "\n".join(parts)
    return ""


def _call_rapidocr(ocr: Any, arr: Any, *, text_score: Optional[float]) -> str:
    """Invoke RapidOCR; supports optional ``text_score`` (newer packages)."""
    if text_score is not None:
        try:
            raw = ocr(arr, text_score=text_score)
        except TypeError:
            raw = ocr(arr)
    else:
        raw = ocr(arr)
    if isinstance(raw, tuple) and len(raw) >= 1:
        result = raw[0]
    else:
        result = raw
    return _join_rapidocr_texts(result).strip()


def _ocr_pages_optional(
    data: bytes,
    page_zero_indices: List[int],
    *,
    scale: float = 2.5,
    text_score: Optional[float] = None,
) -> dict[int, str]:
    """
    Optional OCR for PDF page images (rendered via pypdfium2).

    Requires ``pypdfium2`` and ``rapidocr-onnxruntime``. If imports fail or every
    page errors, returns an empty dict.
    """
    try:
        import numpy as np  # type: ignore
        import pypdfium2 as pdfium  # type: ignore
        from rapidocr_onnxruntime import RapidOCR  # type: ignore
    except ImportError as e:
        _log.warning(
            "PDF OCR skipped (install pypdfium2 + rapidocr-onnxruntime): %s", e
        )
        return {}

    try:
        ocr = RapidOCR()
    except Exception:
        return {}

    pdf = pdfium.PdfDocument(data)
    out: dict[int, str] = {}
    for i in page_zero_indices:
        try:
            page = pdf[i]
            pil = page.render(scale=scale).to_pil()
            if getattr(pil, "mode", None) != "RGB":
                pil = pil.convert("RGB")
            arr = np.array(pil)
            merged = _call_rapidocr(ocr, arr, text_score=text_score)
            if merged:
                out[i] = merged
        except Exception:
            continue
    return out


def _ocr_full_document_fallback(
    data: bytes, page_zero_indices: List[int]
) -> dict[int, str]:
    """
    When normal extraction + sparse-page OCR still yield nothing, retry OCR on
    every page with stronger rendering (higher scale) and looser confidence.
    """
    if not page_zero_indices:
        return {}
    for scale in (3.0, 4.5, 6.0):
        for ts in (None, 0.35, 0.15, 0.05):
            got = _ocr_pages_optional(
                data, page_zero_indices, scale=scale, text_score=ts
            )
            if any((t or "").strip() for t in got.values()):
                return got
    return {}


def extract_pdf_text_full(data: bytes, *, page_indices: Optional[List[int]]) -> str:
    """
    Richer PDF extraction than ``extract_pdf_text``.

    - Layout-aware pypdf text, optional per-page OCR when text is sparse (``< 40`` chars).
    - If the combined result is still empty, runs **whole-document OCR** (all pages,
      higher render scale, looser confidence) for scanned or image-only PDFs.
    - Adds ``[PAGE N]`` markers; OCR-sourced blocks are tagged ``[OCR]``.
    """
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    indices = _select_zero_based_indices(len(reader.pages), page_indices)
    if not indices:
        return ""

    blocks: List[str] = []
    ocr_candidates: List[int] = []
    raw_page_text: dict[int, str] = {}
    for i in indices:
        txt = _extract_page_text_layout(reader.pages[i])
        raw_page_text[i] = txt
        # Very short extraction likely indicates scanned page / extraction failure.
        if len((txt or "").strip()) < 40:
            ocr_candidates.append(i)

    ocr_text_by_page = _ocr_pages_optional(data, ocr_candidates)

    for i in indices:
        page_txt = raw_page_text.get(i, "") or ""
        used_ocr = False
        if len(page_txt.strip()) < 40 and i in ocr_text_by_page:
            page_txt = ocr_text_by_page[i]
            used_ocr = True

        cleaned = _cleanup_page_text(page_txt)
        cleaned = _normalize_tables(cleaned)
        if not cleaned:
            continue
        prefix = f"[PAGE {i + 1}]"
        if used_ocr:
            prefix += " [OCR]"
        blocks.append(f"{prefix}\n{cleaned}")

    merged_primary = "\n\n".join(blocks)
    if merged_primary.strip():
        return merged_primary

    # Whole-document OCR: image-only / CJK PDFs where pypdf is empty but OCR works
    fb = _ocr_full_document_fallback(data, indices)
    blocks_fb: List[str] = []
    for i in indices:
        page_txt = fb.get(i, "") or ""
        cleaned = _cleanup_page_text(page_txt)
        cleaned = _normalize_tables(cleaned)
        if not cleaned:
            continue
        blocks_fb.append(f"[PAGE {i + 1}] [OCR]\n{cleaned}")
    return "\n\n".join(blocks_fb)


def parse_page_list(spec: str) -> Optional[List[int]]:
    """
    Parse e.g. ``"1,3,5-7"`` or empty string → ``None`` (all pages).
    """
    spec = (spec or "").strip()
    if not spec:
        return None
    out: List[int] = []
    for part in re.split(r"[\s,]+", spec):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a.strip()), int(b.strip())
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    return sorted(set(out))
