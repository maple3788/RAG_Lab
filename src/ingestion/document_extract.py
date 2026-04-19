"""Document extraction: PDF (pypdf) and plain text uploads."""

from __future__ import annotations

import io
import re
from typing import Iterable, List, Optional


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


def _ocr_pages_optional(data: bytes, page_zero_indices: List[int]) -> dict[int, str]:
    """
    Optional OCR fallback for scanned pages.
    Requires both:
      - pypdfium2
      - rapidocr-onnxruntime
    If unavailable, returns empty dict.
    """
    try:
        import numpy as np  # type: ignore
        import pypdfium2 as pdfium  # type: ignore
        from rapidocr_onnxruntime import RapidOCR  # type: ignore
    except Exception:
        return {}

    ocr = RapidOCR()
    pdf = pdfium.PdfDocument(data)
    out: dict[int, str] = {}
    for i in page_zero_indices:
        try:
            page = pdf[i]
            bitmap = page.render(scale=2.0).to_pil()
            arr = np.array(bitmap)
            result, _ = ocr(arr)
            if not result:
                continue
            lines: List[str] = []
            for item in result:
                # RapidOCR result row shape: [box, text, score]
                if len(item) >= 2:
                    txt = str(item[1]).strip()
                    if txt:
                        lines.append(txt)
            merged = "\n".join(lines).strip()
            if merged:
                out[i] = merged
        except Exception:
            continue
    return out


def extract_pdf_text_full(data: bytes, *, page_indices: Optional[List[int]]) -> str:
    """
    Richer PDF extraction than ``extract_pdf_text`` for ingestion ``full`` mode.

    Current behavior:
    - adds explicit page boundaries (``[PAGE N]``) to preserve locality
    - applies line-break / whitespace cleanup for cleaner chunking

    This is intentionally model-agnostic (no OCR dependency); OCR/table extraction can
    be layered on top later without changing call sites.
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
    return "\n\n".join(blocks)


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
