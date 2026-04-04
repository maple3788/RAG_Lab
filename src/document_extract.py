"""Document extraction: PDF (pypdf) and plain text uploads."""

from __future__ import annotations

import io
import re
from typing import List, Optional


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
