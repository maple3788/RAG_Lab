#!/usr/bin/env python3
"""
Build ``analysis/error_analysis_draft.md`` from a QASPER traces JSONL (from ``tools/export_qasper_traces.py``).

Picks:
  - 3 success rows: ``failure_bucket == none`` (prefer highest token_f1)
  - 1 retrieval, 1 ranking, 1 generation failure where possible

Usage::

    python scripts/build_error_analysis_draft.py --in analysis/qasper_traces.jsonl
    python scripts/build_error_analysis_draft.py --in analysis/qasper_traces.jsonl --out analysis/error_analysis_draft.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _fmt_block(title: str, r: dict) -> str:
    pred = (r.get("prediction") or "").strip()
    if len(pred) > 600:
        pred = pred[:600] + "…"
    aliases = r.get("gold_aliases") or []
    alias_s = "; ".join(str(a)[:120] for a in aliases[:5])
    if len(aliases) > 5:
        alias_s += f" … (+{len(aliases) - 5} more)"
    finals = r.get("final_previews") or []
    fp = (
        finals[0][:500] + "…"
        if finals and len(finals[0]) > 500
        else (finals[0] if finals else "")
    )
    return "\n".join(
        [
            f"#### {title}",
            "",
            f"- **ID:** `{r.get('question_id', '')}`",
            f"- **Question:** {r.get('question', '')}",
            f"- **Retrieval stage:** `{r.get('retrieval_stage', '')}` · **Failure bucket:** `{r.get('failure_bucket')}`",
            f"- **Gold aliases (sample):** {alias_s}",
            f"- **Prediction:** {pred or '*(empty)*'}",
            f"- **Metrics:** EM={r.get('exact_match')} · token_f1={r.get('token_f1')} · gold_hit={r.get('gold_hit')}",
            f"- **First final chunk preview:** {fp or '—'}",
            "",
        ]
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--in", dest="in_path", type=Path, default=Path("analysis/qasper_traces.jsonl")
    )
    p.add_argument("--out", type=Path, default=Path("analysis/error_analysis_draft.md"))
    args = p.parse_args()

    if not args.in_path.exists():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            _stub_no_input(args.in_path),
            encoding="utf-8",
        )
        print(f"No input file; wrote stub to {args.out}")
        return

    rows = [r for r in _load_rows(args.in_path) if "error" not in r]
    if not rows:
        args.out.write_text(
            "# Error analysis draft (QASPER)\n\n*No valid trace rows.*\n",
            encoding="utf-8",
        )
        return

    def success_sort_key(r: dict) -> tuple:
        b = r.get("failure_bucket")
        em = float(r.get("exact_match") or 0)
        gh = float(r.get("gold_hit") or 0)
        f1 = float(r.get("token_f1") or 0)
        strict = 2 if b == "none" else (1 if em >= 1.0 else 0)
        return (strict, em, gh, f1)

    rows_sorted = sorted(rows, key=success_sort_key, reverse=True)
    successes = rows_sorted[:3]
    note_soft = not any(r.get("failure_bucket") == "none" for r in successes)

    def pick_stage(stage: str) -> dict | None:
        cands = [r for r in rows if r.get("retrieval_stage") == stage]
        return cands[0] if cands else None

    def pick_generation() -> dict | None:
        cands = [
            r
            for r in rows
            if r.get("retrieval_stage") == "gold_in_final"
            and r.get("failure_bucket") == "generation"
        ]
        return cands[0] if cands else None

    fail_ret = pick_stage("retrieval")
    fail_rank = pick_stage("ranking")
    fail_gen = pick_generation()

    lines = [
        "# QASPER error analysis draft",
        "",
        f"*Auto-generated from `{args.in_path}` for review (Tech Lead). Regenerate after exporting fresh traces.*",
        "",
    ]
    if note_soft:
        lines.append(
            "*Note: No row had `failure_bucket: none` in this file — showing the **best-scoring** rows by (EM, gold_hit, F1). "
            "Re-export with a real LLM (`ollama` / `gemini`) for textbook “success” cases.*"
        )
        lines.append("")
    lines.extend(
        [
            "## Success cases (3)",
            "",
        ]
    )
    for i, r in enumerate(successes[:3], 1):
        lines.append(_fmt_block(f"Success {i}", r))

    lines.extend(["## Failure cases (3)", ""])

    if fail_ret:
        lines.append(
            _fmt_block("Failure — retrieval (no gold in retrieve_k pool)", fail_ret)
        )
    else:
        lines.append(
            "*(No retrieval-stage row in this export; increase `--max-examples` or check data.)*\n"
        )

    if fail_rank:
        lines.append(
            _fmt_block("Failure — ranking (gold in pool, not in final_k)", fail_rank)
        )
    else:
        lines.append("*(No ranking-stage row in this export.)*\n")

    if fail_gen:
        lines.append(
            _fmt_block(
                "Failure — generation (gold in final context, wrong answer)", fail_gen
            )
        )
    else:
        lines.append(
            "*(No generation bucket row; run export **with** LLM: omit `--skip-generation`.)*\n"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out}")


def _stub_no_input(expected: Path) -> str:
    return f"""# QASPER error analysis draft

**Input not found:** `{expected}`

## How to generate

1. Export traces (with LLM for failure buckets and predictions):

```bash
python tools/export_qasper_traces.py --out analysis/qasper_traces.jsonl --max-examples 60 --llm-backend ollama
```

2. Build this draft:

```bash
python scripts/build_error_analysis_draft.py --in analysis/qasper_traces.jsonl --out analysis/error_analysis_draft.md
```

3. Optional: retrieval-only labels (no API cost), then re-run step 2 for partial examples:

```bash
python tools/export_qasper_traces.py --out analysis/qasper_traces.jsonl --max-examples 80 --skip-generation
```
"""


if __name__ == "__main__":
    main()
