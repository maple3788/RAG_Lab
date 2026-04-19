from __future__ import annotations

from typing import Literal

TruncationStrategy = Literal["head", "tail", "middle"]


def truncate_context(
    text: str, max_chars: int, strategy: TruncationStrategy = "head"
) -> str:
    """
    Truncate a single concatenated context string to fit a budget (char-level proxy for window).
    - head: keep the start
    - tail: keep the end (often where local details appear if chunks are ordered oddly)
    - middle: keep a window around the center
    """
    if max_chars <= 0:
        return ""
    s = text or ""
    if len(s) <= max_chars:
        return s
    if strategy == "head":
        return s[:max_chars]
    if strategy == "tail":
        return s[-max_chars:]
    # middle
    start = max(0, (len(s) - max_chars) // 2)
    return s[start : start + max_chars]
