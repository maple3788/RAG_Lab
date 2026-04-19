from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable

from dotenv import load_dotenv


def _load_project_dotenv() -> None:
    """Load `<repo>/.env` so GEMINI_API_KEY need not be exported manually."""
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")


@runtime_checkable
class TextGenerator(Protocol):
    def generate(self, prompt: str) -> str: ...


@runtime_checkable
class StreamingTextGenerator(Protocol):
    def generate_stream(self, prompt: str) -> Iterator[str]: ...


@runtime_checkable
class ChatTextGenerator(Protocol):
    def generate_chat(self, *, system_prompt: str, user_prompt: str) -> str: ...


@runtime_checkable
class StreamingChatTextGenerator(Protocol):
    def generate_chat_stream(
        self, *, system_prompt: str, user_prompt: str
    ) -> Iterator[str]: ...


@dataclass
class GeminiGenerator:
    """
    Google Gemini via the `google-genai` SDK (Generative Language API).
    Set GEMINI_API_KEY in the environment or in a `.env` file at the project root.
    """

    model: str = "gemini-2.5-flash"
    temperature: float = 0.0
    max_output_tokens: int = 256

    def __post_init__(self) -> None:
        from google import genai

        _load_project_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Use --llm-backend ollama for local models, "
                "or MockGenerator for dry runs."
            )
        self._client = genai.Client(api_key=api_key)

    def generate(self, prompt: str) -> str:
        from google.genai import types

        try:
            resp = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens,
                ),
            )
        except Exception as e:
            msg = str(e).lower()
            if "location" in msg or "not supported for the api" in msg:
                raise RuntimeError(
                    "Gemini (Google AI Studio) rejected this request due to region / IP policy. "
                    "See docs/gemini-region-restriction.md. "
                    "Workaround without cloud LLM keys: `--llm-backend ollama` (local model). "
                    "Or use `--llm-backend openai` with OPENAI_API_KEY / OpenRouter."
                ) from e
            raise
        text = getattr(resp, "text", None) or ""
        return text.strip()

    def generate_stream(self, prompt: str) -> Iterator[str]:
        from google.genai import types

        stream = self._client.models.generate_content_stream(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            ),
        )
        for chunk in stream:
            text = getattr(chunk, "text", None) or ""
            if text:
                yield text


@dataclass
class OpenAICompatibleGenerator:
    """
    OpenAI Chat Completions API or any compatible server (OpenRouter, Azure, etc.).
    Set OPENAI_API_KEY; optional OPENAI_BASE_URL.
    """

    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 256

    def __post_init__(self) -> None:
        from openai import OpenAI

        _load_project_dotenv()
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set (required for --llm-backend openai). "
                "Or use --llm-backend ollama (local) or Gemini with GEMINI_API_KEY."
            )
        base_url = os.environ.get("OPENAI_BASE_URL")
        self._client = (
            OpenAI(api_key=api_key, base_url=base_url)
            if base_url
            else OpenAI(api_key=api_key)
        )

    def generate(self, prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        choice = resp.choices[0]
        content = choice.message.content
        return (content or "").strip()

    def generate_chat(self, *, system_prompt: str, user_prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        choice = resp.choices[0]
        content = choice.message.content
        return (content or "").strip()

    def generate_stream(self, prompt: str) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )
        for event in stream:
            if not event.choices:
                continue
            delta = event.choices[0].delta
            text = getattr(delta, "content", None) or ""
            if text:
                yield text

    def generate_chat_stream(
        self, *, system_prompt: str, user_prompt: str
    ) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )
        for event in stream:
            if not event.choices:
                continue
            delta = event.choices[0].delta
            text = getattr(delta, "content", None) or ""
            if text:
                yield text


@dataclass
class OllamaGenerator:
    """
    Local open-weight models via Ollama (OpenAI-compatible API). See https://ollama.com
    No Gemini/OpenAI cloud key required. Install Ollama, then e.g. ollama pull llama3.2
    Default API: http://127.0.0.1:11434/v1 — set OLLAMA_BASE_URL to override.
    """

    model: str = "llama3.2"
    temperature: float = 0.0
    max_tokens: int = 256

    def __post_init__(self) -> None:
        from openai import OpenAI

        _load_project_dotenv()
        base = (
            os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
            .strip()
            .rstrip("/")
        )
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        self._client = OpenAI(
            base_url=base,
            api_key="ollama",
            timeout=120.0,
        )

    def generate(self, prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        choice = resp.choices[0]
        content = choice.message.content
        return (content or "").strip()

    def generate_chat(self, *, system_prompt: str, user_prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        choice = resp.choices[0]
        content = choice.message.content
        return (content or "").strip()

    def generate_stream(self, prompt: str) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )
        for event in stream:
            if not event.choices:
                continue
            delta = event.choices[0].delta
            text = getattr(delta, "content", None) or ""
            if text:
                yield text

    def generate_chat_stream(
        self, *, system_prompt: str, user_prompt: str
    ) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )
        for event in stream:
            if not event.choices:
                continue
            delta = event.choices[0].delta
            text = getattr(delta, "content", None) or ""
            if text:
                yield text


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

    def generate_stream(self, prompt: str) -> Iterator[str]:
        yield self.generate(prompt)
