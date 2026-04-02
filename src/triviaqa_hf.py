from __future__ import annotations

from typing import Any, Dict, List, Sequence

from src.loader import QAExample


def _contexts_from_trivia_row(row: Dict[str, Any]) -> List[str]:
    """Reading-comprehension passages: prefer Wikipedia entity pages, else web search snippets."""
    out: List[str] = []
    ep = row.get("entity_pages")
    if ep is None:
        ep = {}
    if isinstance(ep, dict) and ep.get("wiki_context"):
        for c in ep["wiki_context"]:
            s = (c or "").strip()
            if s:
                out.append(s)
    if out:
        return out
    sr = row.get("search_results") or {}
    if isinstance(sr, dict) and sr.get("snippet"):
        for s in sr["snippet"]:
            t = (s or "").strip()
            if t:
                out.append(t)
    return out


def _aliases_from_answer(ans: Dict[str, Any]) -> List[str]:
    raw = list(ans.get("aliases") or [])
    val = (ans.get("value") or "").strip()
    seen: List[str] = []
    if val:
        seen.append(val)
    for a in raw:
        t = (a or "").strip()
        if t and t.lower() not in {x.lower() for x in seen}:
            seen.append(t)
    return seen


def load_triviaqa_rc_hf(
    *,
    split: str = "validation",
    max_examples: int | None = None,
    dataset_id: str = "mandarjoshi/trivia_qa",
    config: str = "rc",
) -> List[QAExample]:
    """
    Hugging Face `mandarjoshi/trivia_qa` / config `rc`.

    Each example uses **only that row's** passages as retrieval pool (see ``per_example_retrieval``).
    Gold labels use all **answer aliases** for EM/F1/gold_hit (max over aliases).
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_id, config, split=split)
    examples: List[QAExample] = []
    for i, row in enumerate(ds):
        if max_examples is not None and len(examples) >= max_examples:
            break
        row_d: Dict[str, Any] = dict(row)
        contexts = _contexts_from_trivia_row(row_d)
        if not contexts:
            continue
        ans = row_d.get("answer") or {}
        aliases = _aliases_from_answer(ans)
        if not aliases:
            continue
        qid = str(row_d.get("question_id") or row_d.get("id") or i)
        examples.append(
            QAExample(
                id=qid,
                question=str(row_d.get("question", "")),
                answer=aliases[0],
                contexts=contexts,
                source="triviaqa-rc",
                answer_aliases=tuple(aliases),
            )
        )
    return examples
