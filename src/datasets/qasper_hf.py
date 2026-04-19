"""
`allenai/qasper` — question answering over **full NLP papers** (long-document / “PDF-like” QA).

Loads the Parquet-backed revision (HF datasets ≥3 no longer runs legacy `qasper.py` scripts).

Gold labels are derived from annotator fields: non-placeholder **extractive spans**, **free-form**
answers, **yes/no**, and **highlighted_evidence** snippets (skipping `BIBREF*` / `TABREF*` only spans).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from src.datasets.loader import QAExample

# Resolves to Parquet files on the Hub (see allenai/qasper discussions).
DEFAULT_QASPER_REVISION = "refs/convert/parquet"

_PLACEHOLDER = re.compile(r"^(BIBREF\d+|TABREF\d+)$")


def _collect_aliases_from_answer(a: Dict[str, Any]) -> List[str]:
    if a.get("unanswerable"):
        return []
    out: List[str] = []
    ff = (a.get("free_form_answer") or "").strip()
    if ff:
        out.append(ff)
    for s in a.get("extractive_spans") or []:
        t = (s or "").strip()
        if t and not _PLACEHOLDER.match(t):
            out.append(t)
    yn = a.get("yes_no")
    if yn is True:
        out.append("yes")
    elif yn is False:
        out.append("no")
    for ev in a.get("highlighted_evidence") or []:
        t = (ev or "").strip()
        if len(t) > 12:
            out.append(t[:800])
    for ev in a.get("evidence") or []:
        t = (ev or "").strip()
        if len(t) > 40:
            out.append(t[:800])
    # Dedupe case-insensitive, keep order
    seen_lower: set[str] = set()
    deduped: List[str] = []
    for x in out:
        k = x.lower()
        if k not in seen_lower:
            seen_lower.add(k)
            deduped.append(x)
    return deduped


def _aliases_for_question(answers_block: Dict[str, Any]) -> List[str]:
    aliases: List[str] = []
    for a in answers_block.get("answer") or []:
        if isinstance(a, dict):
            aliases.extend(_collect_aliases_from_answer(a))
    seen: set[str] = set()
    out: List[str] = []
    for x in aliases:
        k = x.lower()
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out


def _document_text(row: Dict[str, Any]) -> str | None:
    ft = row.get("full_text") or {}
    paras: List[str] = []
    if isinstance(ft, dict):
        raw = ft.get("paragraphs")
        if isinstance(raw, list):
            for p in raw:
                if isinstance(p, str) and p.strip():
                    paras.append(p.strip())
                elif isinstance(p, list):
                    paras.extend(str(x).strip() for x in p if str(x).strip())
    body = "\n\n".join(paras)
    abstract = (row.get("abstract") or "").strip()
    if abstract and body:
        return abstract + "\n\n" + body
    if body:
        return body
    if abstract:
        return abstract
    return None


def load_qasper_hf(
    *,
    split: str = "validation",
    max_examples: int | None = None,
    dataset_id: str = "allenai/qasper",
    revision: str = DEFAULT_QASPER_REVISION,
) -> List[QAExample]:
    """
    Each row is one paper; expanded to **one QAExample per question** with the **whole paper**
    (abstract + paragraphs) as the retrieval pool — same ``per_example_retrieval`` pattern as TriviaQA RC.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split=split, revision=revision)
    examples: List[QAExample] = []
    for row in ds:
        if max_examples is not None and len(examples) >= max_examples:
            break
        row_d: Dict[str, Any] = dict(row)
        doc = _document_text(row_d)
        if not doc:
            continue
        qas = row_d.get("qas") or {}
        questions: List[str] = list(qas.get("question") or [])
        answer_blocks: List[Dict[str, Any]] = list(qas.get("answers") or [])
        paper_id = str(row_d.get("id", ""))
        for j, q in enumerate(questions):
            q = (q or "").strip()
            if not q:
                continue
            if j >= len(answer_blocks):
                continue
            aliases = _aliases_for_question(answer_blocks[j])
            if not aliases:
                continue
            qids = list(qas.get("question_id") or [])
            qid = str(qids[j]) if j < len(qids) else f"{paper_id}_{j}"
            examples.append(
                QAExample(
                    id=f"qasper:{qid}",
                    question=q,
                    answer=aliases[0],
                    contexts=(doc,),
                    source="qasper",
                    answer_aliases=tuple(aliases),
                )
            )
            if max_examples is not None and len(examples) >= max_examples:
                break
        else:
            continue
        break
    return examples
