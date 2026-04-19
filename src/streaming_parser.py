from __future__ import annotations

from dataclasses import dataclass, field


_HIDDEN_TAGS = {"think", "reasoning", "analysis"}


@dataclass
class HiddenReasoningStreamParser:
    """
    Incremental parser that removes hidden-reasoning XML-like blocks such as:
    <think> ... </think>, <reasoning> ... </reasoning>, <analysis> ... </analysis>.
    """

    _pending: str = ""
    _hidden_depth: int = 0
    _hidden_stack: list[str] = field(default_factory=list)

    def _parse_tag(self, raw: str) -> tuple[bool, bool, str]:
        s = raw.strip().lower()
        if not s:
            return False, False, ""
        is_close = s.startswith("/")
        name = s[1:] if is_close else s
        name = name.split()[0].strip("/")
        if name in _HIDDEN_TAGS:
            return True, is_close, name
        return False, is_close, name

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        self._pending += chunk
        out: list[str] = []
        i = 0
        n = len(self._pending)

        while i < n:
            ch = self._pending[i]
            if ch == "<":
                j = self._pending.find(">", i + 1)
                if j == -1:
                    break  # wait for next token to complete tag
                tag_body = self._pending[i + 1 : j]
                known, is_close, name = self._parse_tag(tag_body)
                if known:
                    if is_close:
                        if self._hidden_stack and self._hidden_stack[-1] == name:
                            self._hidden_stack.pop()
                        elif self._hidden_depth > 0:
                            self._hidden_depth -= 1
                        self._hidden_depth = len(self._hidden_stack)
                    else:
                        self._hidden_stack.append(name)
                        self._hidden_depth = len(self._hidden_stack)
                    i = j + 1
                    continue
            if self._hidden_depth == 0:
                out.append(ch)
            i += 1

        self._pending = self._pending[i:]
        return "".join(out)

    def flush(self) -> str:
        if not self._pending:
            return ""
        out = self._pending if self._hidden_depth == 0 else ""
        self._pending = ""
        return out


def strip_hidden_reasoning_text(text: str) -> str:
    parser = HiddenReasoningStreamParser()
    visible = parser.feed(text or "")
    visible += parser.flush()
    return visible.strip()
