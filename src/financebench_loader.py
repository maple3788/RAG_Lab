from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.loader import QAExample


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _contexts_from_evidence(row: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for ev in row.get("evidence") or []:
        if not isinstance(ev, dict):
            continue
        # Prefer full-page evidence if present; fallback to short evidence span.
        full = (ev.get("evidence_text_full_page") or "").strip()
        span = (ev.get("evidence_text") or "").strip()
        text = full or span
        if text:
            out.append(text)
    # De-duplicate while preserving order.
    seen = set()
    uniq: List[str] = []
    for t in out:
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(t)
    return uniq


def load_financebench_open_source(
    *,
    financebench_root: str | Path,
    max_examples: int | None = None,
) -> List[QAExample]:
    """
    Load FinanceBench open-source split (150 annotated examples) from cloned repo.

    Expected files under ``financebench_root``:
    - ``data/financebench_open_source.jsonl``
    - ``data/financebench_document_information.jsonl`` (optional metadata enrich)
    """
    root = Path(financebench_root)
    q_path = root / "data" / "financebench_open_source.jsonl"
    if not q_path.exists():
        raise FileNotFoundError(f"Missing file: {q_path}")

    rows = _load_jsonl(q_path)
    examples: List[QAExample] = []
    for i, row in enumerate(rows):
        if max_examples is not None and len(examples) >= max_examples:
            break
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if not question or not answer:
            continue
        contexts = _contexts_from_evidence(row)
        if not contexts:
            continue
        qid = str(row.get("financebench_id", row.get("id", i)))
        examples.append(
            QAExample(
                id=qid,
                question=question,
                answer=answer,
                contexts=contexts,
                source="financebench-open-source",
                answer_aliases=(answer,),
            )
        )
    return examples
