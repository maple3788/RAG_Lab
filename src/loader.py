from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class QAExample:
    id: str
    question: str
    answer: str
    contexts: Sequence[str]
    source: str | None = None
    #: If set (e.g. TriviaQA), EM/F1/gold_hit use the best score over any alias.
    answer_aliases: Tuple[str, ...] | None = None


def load_qa_jsonl(path: str | Path) -> List[QAExample]:
    p = Path(path)
    examples: List[QAExample] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            examples.append(
                QAExample(
                    id=str(obj.get("id", "")),
                    question=str(obj["question"]),
                    answer=str(obj["answer"]),
                    contexts=list(obj.get("contexts", [])),
                    source=obj.get("source"),
                )
            )
    return examples


def flatten_contexts(examples: Iterable[QAExample]) -> List[str]:
    docs: List[str] = []
    for ex in examples:
        docs.extend(list(ex.contexts))
    return docs


def load_beir_queries_as_qa_examples(
    queries_path: str | Path,
    *,
    qrels_path: str | Path | None = None,
    max_queries: int | None = None,
) -> List[QAExample]:
    """
    BEIR ``queries.jsonl`` for RAG generation eval: ``question`` = topic text;
    ``answer`` = short proxy from ``metadata.query`` (TREC topic keyword line), not an official NIST span.
    """
    from src.beir_io import load_beir_qrels, ordered_qids_from_qrels

    p = Path(queries_path)
    raw: List[QAExample] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = str(obj.get("_id", obj.get("id", "")))
            qtext = str(obj.get("text", ""))
            meta = obj.get("metadata") or {}
            short = (meta.get("query") or "").strip()
            if not short:
                short = qtext[:200].strip()
            raw.append(
                QAExample(
                    id=qid,
                    question=qtext,
                    answer=short,
                    contexts=(),
                    source="beir-queries",
                )
            )

    if qrels_path is not None:
        qrels = load_beir_qrels(qrels_path)
        order = ordered_qids_from_qrels(qrels)
        allowed = set(order)
        by_id = {e.id: e for e in raw}
        examples = [by_id[q] for q in order if q in by_id]
    else:
        examples = raw

    if max_queries is not None:
        examples = examples[: max_queries]
    return examples

