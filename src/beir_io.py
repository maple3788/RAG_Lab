from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from ir_measures import Qrel


def _join_title_text(title: str | None, text: str | None) -> str:
    t = (title or "").strip()
    b = (text or "").strip()
    if t and b:
        return f"{t}\n{b}"
    return t or b


def load_beir_corpus_ordered(path: str | Path) -> List[Tuple[str, str]]:
    """Load BEIR ``corpus.jsonl``: each line has _id, title, text."""
    p = Path(path)
    rows: List[Tuple[str, str]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            did = str(o.get("_id", o.get("id", "")))
            if not did:
                continue
            body = _join_title_text(o.get("title"), o.get("text"))
            rows.append((did, body))
    return rows


def load_beir_queries_ordered(path: str | Path) -> List[Tuple[str, str]]:
    """Load BEIR ``queries.jsonl``: each line has _id, text."""
    p = Path(path)
    rows: List[Tuple[str, str]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            qid = str(o.get("_id", o.get("id", "")))
            if not qid:
                continue
            rows.append((qid, str(o.get("text", ""))))
    return rows


def load_beir_qrels(path: str | Path) -> List[Qrel]:
    """
    Load BEIR ``qrels/*.tsv``: query-id, corpus-id, score (tab-separated).

    Also accepts classic TREC qrels (4 columns): qid iter doc_id rel.
    """
    p = Path(path)
    qrels: List[Qrel] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) == 3:
                qid, doc_id, rel_s = parts
                if qid.lower() in ("query-id", "query_id"):
                    continue
                qrels.append(
                    Qrel(
                        query_id=str(qid),
                        doc_id=str(doc_id),
                        relevance=int(rel_s),
                        iteration="0",
                    )
                )
            elif len(parts) >= 4:
                qid, _iter, doc_id, rel_s = parts[0], parts[1], parts[2], parts[3]
                qrels.append(
                    Qrel(
                        query_id=str(qid),
                        doc_id=str(doc_id),
                        relevance=int(rel_s),
                        iteration=str(_iter),
                    )
                )
    return qrels


def corpus_list_to_dict(rows: Sequence[Tuple[str, str]]) -> Dict[str, str]:
    return {did: text for did, text in rows}


def ordered_qids_from_qrels(qrels: Sequence[Qrel]) -> List[str]:
    seen = set()
    out: List[str] = []
    for q in qrels:
        if q.query_id not in seen:
            seen.add(q.query_id)
            out.append(q.query_id)
    return out
