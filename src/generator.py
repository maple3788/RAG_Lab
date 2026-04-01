from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from dotenv import load_dotenv


def _load_project_dotenv() -> None:
    """Load `<repo>/.env` so GEMINI_API_KEY need not be exported manually."""
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")


@runtime_checkable
class TextGenerator(Protocol):
    def generate(self, prompt: str) -> str: ...


@dataclass
class GeminiGenerator:
    """
    Google Gemini via the `google-genai` SDK (Generative Language API).
    Set GEMINI_API_KEY in the environment or in a `.env` file at the project root.
    """

    model: str = "gemini-2.0-flash"
    temperature: float = 0.0
    max_output_tokens: int = 256

    def __post_init__(self) -> None:
        from google import genai

        _load_project_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Export it or use MockGenerator for dry runs."
            )
        self._client = genai.Client(api_key=api_key)

    def generate(self, prompt: str) -> str:
        from google.genai import types

        resp = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            ),
        )
        text = getattr(resp, "text", None) or ""
        return text.strip()


@dataclass
class MockGenerator:
    """
    Deterministic stub: returns the first line of the prompt's context block for CI / wiring tests.
    Does not call any API. Metrics will be uninformative for research but prove the pipeline runs.
    """

    def generate(self, prompt: str) -> str:
        # Heuristic: take text after "Passages:" or "Context:" up to next blank line
        lower = prompt.lower()
        for marker in ("passages:", "context:"):
            idx = lower.find(marker)
            if idx >= 0:
                rest = prompt[idx + len(marker) :].lstrip()
                line = rest.split("\n")[0].strip()
                if line:
                    return line[:200]
        return "unknown"
