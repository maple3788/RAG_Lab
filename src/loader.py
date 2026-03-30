from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class QAExample:
    id: str
    question: str
    answer: str
    contexts: Sequence[str]
    source: str | None = None


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

