"""
Single-turn RAGAS scores for UI (Streamlit).

Primary path: ``ragas.evaluate()`` with LangChain LLM + embeddings (same stack as
``experiments/exp_ragas_financebench.py``), which is more reliable than collections
``metric.score()`` + Instructor judges for some backends.

UI mapping (legacy metrics):
  - Context relevance → ``context_precision`` if a reference exists, else
    ``LLMContextPrecisionWithoutReference``
  - Response groundedness → ``faithfulness``
  - Answer accuracy → ``answer_correctness`` (only when reference is set)

``evaluate()`` runs in a worker thread when Streamlit has an asyncio loop.

**JSON output:** Faithfulness relies on strict JSON from the chat model. For **Gemini**, set
``RAGAS_GEMINI_JSON_MODE=1`` (default) to use ``response_mime_type=application/json``. For
**Ollama**, set ``RAGAS_OLLAMA_JSON_MODE=1`` (default) to pass ``format=json`` via
``extra_body``. Disable either if your server/model misbehaves.
"""

from __future__ import annotations

import concurrent.futures
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

_log = logging.getLogger(__name__)


def _resolve_metric(m: Any, name: str) -> Any:
    """Normalize ragas metric imports across versions (same idea as exp_ragas_financebench)."""
    if m is None:
        raise ValueError(f"Metric {name} is None")
    if hasattr(m, "__dict__") and hasattr(m, "__name__") and hasattr(m, name):
        obj = getattr(m, name)
        return obj() if isinstance(obj, type) else obj
    if isinstance(m, type):
        return m()
    return m


def build_langchain_ragas_eval_stack(
    *,
    backend: str,
    model: str,
    embed_model: Optional[str] = None,
    use_embeddings: bool = True,
) -> Tuple[Any, Optional[Any]]:
    """
    Build ``LangchainLLMWrapper`` + optional ``LangchainEmbeddingsWrapper`` for ``ragas.evaluate``.

    Mirrors ``_make_ragas_eval_stack`` in ``experiments/exp_ragas_financebench.py``.
    """
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
    except Exception as e:
        raise ImportError(
            "RAGAS legacy evaluate needs langchain-openai. "
            "Install with: pip install langchain-openai\n"
            f"Import error: {e}"
        ) from e

    load_dotenv()
    if backend == "mock":
        raise ValueError("RAGAS evaluate requires a real LLM backend (not mock).")

    _log.info(
        "RAGAS: building LangChain eval stack (backend=%s, model=%s, use_embeddings=%s)",
        backend,
        model,
        use_embeddings,
    )

    if embed_model is None:
        if backend == "ollama":
            em = "nomic-embed-text"
        elif backend == "gemini":
            em = "gemini-embedding-001"
        else:
            em = "text-embedding-3-small"
    else:
        em = embed_model

    if backend == "ollama":
        base = (
            os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
            .strip()
            .rstrip("/")
        )
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        os.environ.setdefault("OPENAI_API_KEY", "ollama")
        os.environ.setdefault("OPENAI_BASE_URL", base)
        os.environ.setdefault("OPENAI_MODEL", model)
        os.environ.setdefault("OPENAI_EMBEDDING_MODEL", em)
        # RAGAS pydantic prompts use ``model_validate_json()`` on completions. Local models
        # often emit markdown or prose unless JSON is forced (same class of issue as Gemini).
        ollama_json = os.environ.get(
            "RAGAS_OLLAMA_JSON_MODE", "1"
        ).strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
        ollama_llm_kwargs: Dict[str, Any] = {
            "model": model,
            "base_url": base,
            "api_key": "ollama",
            "temperature": 0.0,
        }
        if ollama_json:
            # Ollama chat API accepts ``format: "json"`` so the model emits parseable JSON
            # (RAGAS faithfulness uses ``model_validate_json`` on the raw string).
            ollama_llm_kwargs["extra_body"] = {"format": "json"}
            _log.info(
                "RAGAS: Ollama evaluator using extra_body format=json "
                "(set RAGAS_OLLAMA_JSON_MODE=0 to disable)"
            )
        lc_llm = ChatOpenAI(**ollama_llm_kwargs)
        lc_emb = (
            OpenAIEmbeddings(
                model=em,
                base_url=base,
                api_key="ollama",
            )
            if use_embeddings
            else None
        )
    elif backend == "gemini":
        _log.info("RAGAS: importing langchain_google_genai for Gemini evaluator")
        try:
            from langchain_google_genai import (
                ChatGoogleGenerativeAI,
                GoogleGenerativeAIEmbeddings,
            )
        except Exception as e:
            _log.error(
                "RAGAS: langchain_google_genai import failed: %s. Run: pip install langchain-google-genai",
                e,
            )
            raise ImportError(
                "Gemini RAGAS evaluator requires langchain-google-genai.\n"
                "Install with: pip install langchain-google-genai\n"
                f"Import error: {e}"
            ) from e
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for gemini RAGAS evaluator.")
        json_mode = os.environ.get(
            "RAGAS_GEMINI_JSON_MODE", "1"
        ).strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
        gemini_llm_kwargs: Dict[str, Any] = {
            "model": model,
            "google_api_key": api_key,
            "temperature": 0.0,
            "include_thoughts": False,
        }
        if json_mode:
            gemini_llm_kwargs["response_mime_type"] = "application/json"
            _log.info(
                "RAGAS: Gemini evaluator using response_mime_type=application/json "
                "(set RAGAS_GEMINI_JSON_MODE=0 to disable)"
            )
        lc_llm = ChatGoogleGenerativeAI(**gemini_llm_kwargs)
        lc_emb = (
            GoogleGenerativeAIEmbeddings(
                model=em,
                google_api_key=api_key,
            )
            if use_embeddings
            else None
        )
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for openai RAGAS evaluator.")
        lc_llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0.0,
        )
        lc_emb = (
            OpenAIEmbeddings(
                model=em,
                api_key=api_key,
                base_url=base_url,
            )
            if use_embeddings
            else None
        )

    wrapped_emb = LangchainEmbeddingsWrapper(lc_emb) if lc_emb is not None else None
    _log.info(
        "RAGAS: LangChain eval stack ready (embeddings=%s)", wrapped_emb is not None
    )
    return LangchainLLMWrapper(lc_llm), wrapped_emb


def _evaluate_in_thread(**kwargs: Any) -> Any:
    """Run ``ragas.evaluate`` off the main thread when an asyncio loop is running (e.g. Streamlit)."""
    from ragas import evaluate

    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        _log.info("RAGAS: calling evaluate() on main thread (no asyncio loop)")
        return evaluate(**kwargs)

    def _call() -> Any:
        return evaluate(**kwargs)

    _log.info("RAGAS: calling evaluate() in worker thread (asyncio loop active)")
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_call).result(timeout=600.0)


def _eval_result_first_row(
    result: Any,
    col_candidates: List[str],
) -> Tuple[Optional[float], Optional[str]]:
    """Read first scalar from EvaluationResult for the first matching column name."""
    try:
        if hasattr(result, "to_pandas"):
            df = result.to_pandas()
        else:
            import pandas as pd

            df = pd.DataFrame([dict(result)])
    except Exception as e:
        return None, f"could not read scores: {e}"

    if df is None or df.empty:
        return None, "empty evaluation result"
    for col in col_candidates:
        if col in df.columns:
            v = df[col].iloc[0]
            try:
                x = float(v)
            except (TypeError, ValueError):
                return None, f"non-numeric {col}: {v!r}"
            if not math.isfinite(x):
                return (
                    None,
                    "non-finite score (often LLM parse failure — try another model or backend)",
                )
            return x, None
    return None, f"no column in {col_candidates!r} (got {list(df.columns)})"


def run_ragas_legacy_evaluate(
    *,
    backend: str,
    model: str,
    user_input: str,
    response: str,
    retrieved_contexts: List[str],
    reference: Optional[str] = None,
    embed_model: Optional[str] = None,
    use_embeddings: bool = True,
) -> Dict[str, Any]:
    """
    Single-row ``ragas.evaluate`` with FinanceBench-compatible columns.

    Returns keys ``context_relevance``, ``response_groundedness``, ``answer_accuracy``
    (and optional ``*_error`` / notes) aligned with the Streamlit table.
    """
    from datasets import Dataset

    try:
        from ragas.metrics import (
            LLMContextPrecisionWithoutReference,
            answer_correctness,
            context_precision,
            faithfulness,
        )
    except Exception as e:
        _log.error("RAGAS: ragas.metrics import failed: %s", e)
        return {"error": f"ragas metrics import failed: {e}"}

    if not retrieved_contexts:
        return {
            "error": "No retrieved contexts to score.",
            "context_relevance": None,
            "response_groundedness": None,
            "answer_accuracy": None,
        }

    ref = (reference or "").strip()
    has_ref = bool(ref)

    row: Dict[str, Any] = {
        "question": user_input,
        "answer": response,
        "contexts": list(retrieved_contexts),
        "ground_truth": ref if has_ref else "",
    }
    ds = Dataset.from_list([row])

    _log.info(
        "RAGAS: starting legacy evaluate (has_reference=%s, contexts=%d, q_len=%d, ans_len=%d)",
        has_ref,
        len(retrieved_contexts),
        len(user_input or ""),
        len(response or ""),
    )

    try:
        eval_llm, eval_embeddings = build_langchain_ragas_eval_stack(
            backend=backend,
            model=model,
            embed_model=embed_model,
            use_embeddings=use_embeddings,
        )
    except Exception as e:
        _log.error("RAGAS: build_langchain_ragas_eval_stack failed: %s", e)
        return {"error": str(e)}

    f_metric = _resolve_metric(faithfulness, "faithfulness")
    if has_ref:
        ctx_metric = _resolve_metric(context_precision, "context_precision")
        ctx_result_names = ["context_precision"]
    else:
        ctx_metric = _resolve_metric(
            LLMContextPrecisionWithoutReference,
            "LLMContextPrecisionWithoutReference",
        )
        ctx_result_names = [
            getattr(ctx_metric, "name", None)
            or "llm_context_precision_without_reference",
            "llm_context_precision_without_reference",
        ]

    metrics: List[Any] = [f_metric, ctx_metric]
    if has_ref:
        metrics.append(_resolve_metric(answer_correctness, "answer_correctness"))

    kwargs: Dict[str, Any] = {
        "dataset": ds,
        "metrics": metrics,
        "llm": eval_llm,
        "show_progress": False,
        "raise_exceptions": False,
    }
    if eval_embeddings is not None:
        kwargs["embeddings"] = eval_embeddings

    try:
        result = _evaluate_in_thread(**kwargs)
    except Exception as e:
        _log.exception("RAGAS: evaluate() raised")
        return {"error": str(e)}

    _log.info(
        "RAGAS: evaluate() finished (context_metric=%s)",
        ctx_result_names[0],
    )

    out: Dict[str, Any] = {
        "ragas_eval_mode": "legacy_evaluate",
        "context_metric": ctx_result_names[0],
    }

    rg_val, rg_err = _eval_result_first_row(result, ["faithfulness"])
    out["response_groundedness"] = rg_val
    if rg_err:
        out["response_groundedness_error"] = rg_err

    cr_val, cr_err = _eval_result_first_row(result, ctx_result_names)
    out["context_relevance"] = cr_val
    if cr_err:
        out["context_relevance_error"] = cr_err
    if not has_ref:
        out["context_relevance_note"] = (
            "LLM context precision (no reference; add reference for context_precision)"
        )

    if has_ref:
        aa_val, aa_err = _eval_result_first_row(result, ["answer_correctness"])
        out["answer_accuracy"] = aa_val
        if aa_err:
            out["answer_accuracy_error"] = aa_err
    else:
        out["answer_accuracy"] = None
        out["answer_accuracy_note"] = "skipped"

    _log.info(
        "RAGAS: scores faithfulness=%s context=%s answer_corr=%s",
        out.get("response_groundedness"),
        out.get("context_relevance"),
        out.get("answer_accuracy"),
    )

    return out


def _metric_value_to_float(mr: Any) -> Tuple[Optional[float], Optional[str]]:
    """Turn ``MetricResult`` into a finite float or ``(None, reason)`` (e.g. NaN from failed judges)."""
    v = getattr(mr, "value", None)
    if v is None:
        return None, "no value on MetricResult"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None, f"non-numeric value: {v!r}"
    if not math.isfinite(x):
        return (
            None,
            "non-finite score (often LLM/judge parse failure — try another model or backend)",
        )
    return x, None


def _metric_score_safe(metric: Any, **kwargs: Any) -> Any:
    """Call sync ``score`` in a worker thread if an asyncio loop is already running."""
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return metric.score(**kwargs)

    def _call() -> Any:
        return metric.score(**kwargs)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_call).result(timeout=120.0)


def make_ragas_instructor_llm(backend: str, model: str) -> Any:
    """
    Build ``InstructorBaseRagasLLM`` for RAGAS collections metrics (see ``ragas.llms.llm_factory``).
    """
    from openai import OpenAI
    from ragas.llms import llm_factory

    load_dotenv()
    if backend == "mock":
        raise ValueError("RAGAS requires a real LLM backend (not mock).")
    if backend == "ollama":
        base = (
            os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
            .strip()
            .rstrip("/")
        )
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        client = OpenAI(base_url=base, api_key="ollama")
        return llm_factory(model, provider="openai", client=client)
    if backend == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        base_url = os.environ.get("OPENAI_BASE_URL")
        client = (
            OpenAI(api_key=api_key, base_url=base_url)
            if base_url
            else OpenAI(api_key=api_key)
        )
        return llm_factory(model, provider="openai", client=client)
    if backend == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set.")
        client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        return llm_factory(model, provider="openai", client=client)
    raise ValueError(f"Unsupported LLM backend for RAGAS: {backend}")


def run_ragas_collections_metrics(
    *,
    llm: Any,
    user_input: str,
    response: str,
    retrieved_contexts: List[str],
    reference: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Returns dict with float scores 0–1 and optional per-metric ``*_error`` strings.
    Keys: ``context_relevance``, ``response_groundedness``, ``answer_accuracy`` (may be None).
    """
    from ragas.metrics.collections import (
        AnswerAccuracy,
        ContextRelevance,
        ResponseGroundedness,
    )

    if not retrieved_contexts:
        return {
            "error": "No retrieved contexts to score.",
            "context_relevance": None,
            "response_groundedness": None,
            "answer_accuracy": None,
        }

    out: Dict[str, Any] = {}

    try:
        cr = ContextRelevance(llm=llm)
        cr_val, cr_err = _metric_value_to_float(
            _metric_score_safe(
                cr, user_input=user_input, retrieved_contexts=retrieved_contexts
            )
        )
        out["context_relevance"] = cr_val
        if cr_err:
            out["context_relevance_error"] = cr_err
    except Exception as e:
        out["context_relevance"] = None
        out["context_relevance_error"] = str(e)

    try:
        rg = ResponseGroundedness(llm=llm)
        rg_val, rg_err = _metric_value_to_float(
            _metric_score_safe(
                rg, response=response, retrieved_contexts=retrieved_contexts
            )
        )
        out["response_groundedness"] = rg_val
        if rg_err:
            out["response_groundedness_error"] = rg_err
    except Exception as e:
        out["response_groundedness"] = None
        out["response_groundedness_error"] = str(e)

    ref = (reference or "").strip()
    if ref:
        try:
            aa = AnswerAccuracy(llm=llm)
            aa_val, aa_err = _metric_value_to_float(
                _metric_score_safe(
                    aa,
                    user_input=user_input,
                    response=response,
                    reference=ref,
                )
            )
            out["answer_accuracy"] = aa_val
            if aa_err:
                out["answer_accuracy_error"] = aa_err
        except Exception as e:
            out["answer_accuracy"] = None
            out["answer_accuracy_error"] = str(e)
    else:
        out["answer_accuracy"] = None
        out["answer_accuracy_note"] = "skipped"

    return out
