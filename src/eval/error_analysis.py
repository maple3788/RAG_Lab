from __future__ import annotations

from typing import Literal, Sequence

FailureBucket = Literal["retrieval", "ranking", "generation", "none"]


def text_contains_any_alias(text: str, aliases: Sequence[str]) -> bool:
    """Case-insensitive substring match for any non-empty alias."""
    tl = (text or "").lower()
    for a in aliases:
        s = (a or "").strip()
        if s and s.lower() in tl:
            return True
    return False


def gold_in_any_chunk(chunks: Sequence[str], aliases: Sequence[str]) -> bool:
    return any(text_contains_any_alias(c, aliases) for c in chunks)


def classify_failure_bucket(
    *,
    any_gold_in_pool: bool,
    any_gold_in_final: bool,
    exact_match_score: float,
) -> FailureBucket:
    """
    Operational taxonomy (see ``analysis/ERROR_ANALYSIS.md``).

    - ``none``: no failure for this taxonomy (answer matches exactly).
    - ``retrieval``: no chunk in the FAISS pool contains an alias.
    - ``ranking``: pool has gold but final top-k does not.
    - ``generation``: final has gold in chunks but prediction is not an exact match.
    """
    if exact_match_score >= 1.0:
        return "none"
    if not any_gold_in_pool:
        return "retrieval"
    if not any_gold_in_final:
        return "ranking"
    return "generation"


def gold_lost_to_truncation(
    raw_context: str,
    truncated_context: str,
    aliases: Sequence[str],
) -> bool:
    """True if some alias appears in raw but not in truncated prompt text."""
    raw_ok = text_contains_any_alias(raw_context, aliases)
    trunc_ok = text_contains_any_alias(truncated_context, aliases)
    return raw_ok and not trunc_ok
