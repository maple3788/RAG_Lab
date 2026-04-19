from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ROOT / "results" / "experiment_db.sqlite"

FAILURE_FEEDBACK_LABELS = (
    "Retrieval Miss",
    "LLM Hallucination",
    "Context Truncation",
    "Ambiguous Question",
    "Other",
)


def get_db_path() -> Path:
    return Path(os.environ.get("RAG_EXPERIMENT_DB", str(DEFAULT_DB_PATH)))


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(path: Optional[Path] = None) -> Path:
    p = path or get_db_path()
    with _connect(p) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS experiment_runs (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              created_at TEXT NOT NULL,
              source TEXT NOT NULL,
              run_label TEXT,
              experiment_tag TEXT,
              embedding_model TEXT,
              chunk_size INTEGER,
              retrieve_k INTEGER,
              final_k INTEGER,
              use_hybrid INTEGER DEFAULT 0,
              use_rerank INTEGER DEFAULT 0,
              llm_backend TEXT,
              llm_model TEXT,
              job_id TEXT,
              latency_ms REAL,
              token_count INTEGER,
              metrics_json TEXT NOT NULL,
              config_json TEXT
            );

            CREATE TABLE IF NOT EXISTS experiment_queries (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              run_id INTEGER REFERENCES experiment_runs(id) ON DELETE SET NULL,
              created_at TEXT NOT NULL,
              source TEXT NOT NULL,
              question TEXT NOT NULL,
              reference_answer TEXT,
              llm_output TEXT,
              retrieved_chunks_json TEXT,
              latency_ms REAL,
              token_count INTEGER,
              gold_hit REAL,
              token_f1 REAL,
              exact_match REAL,
              ragas_faithfulness REAL,
              ragas_context_precision REAL,
              ragas_answer_accuracy REAL,
              human_feedback TEXT,
              model_config_json TEXT,
              stage_trace_json TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_runs_created ON experiment_runs(created_at);
            CREATE INDEX IF NOT EXISTS idx_queries_run ON experiment_queries(run_id);
            CREATE INDEX IF NOT EXISTS idx_queries_created ON experiment_queries(created_at);
            CREATE INDEX IF NOT EXISTS idx_queries_fail ON experiment_queries(gold_hit, token_f1);
            """
        )
        conn.commit()
    return p


def log_experiment_run(
    *,
    source: str,
    run_label: Optional[str],
    experiment_tag: Optional[str],
    embedding_model: Optional[str],
    model_config: Dict[str, Any],
    latency_ms: float,
    token_count: int,
    metrics: Dict[str, Any],
    llm_backend: Optional[str] = None,
    llm_model: Optional[str] = None,
    job_id: Optional[str] = None,
    path: Optional[Path] = None,
) -> int:
    init_db(path)
    p = path or get_db_path()
    cfg = json.dumps(model_config, ensure_ascii=False)
    met = json.dumps(metrics, ensure_ascii=False)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with _connect(p) as conn:
        cur = conn.execute(
            """
            INSERT INTO experiment_runs (
              created_at, source, run_label, experiment_tag, embedding_model,
              chunk_size, retrieve_k, final_k, use_hybrid, use_rerank,
              llm_backend, llm_model, job_id, latency_ms, token_count,
              metrics_json, config_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now,
                source,
                run_label,
                experiment_tag,
                embedding_model,
                int(model_config.get("chunk_size") or 0),
                int(model_config.get("retrieve_k") or 0),
                int(model_config.get("final_k") or 0),
                1 if model_config.get("use_hybrid") else 0,
                1 if model_config.get("use_rerank") else 0,
                llm_backend,
                llm_model,
                job_id,
                float(latency_ms),
                int(token_count),
                met,
                cfg,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def log_query_event(
    *,
    source: str,
    question: str,
    llm_output: str,
    retrieved_chunks: List[str],
    latency_ms: float,
    token_count: int,
    model_config: Dict[str, Any],
    reference_answer: Optional[str] = None,
    gold_hit: Optional[float] = None,
    token_f1: Optional[float] = None,
    exact_match: Optional[float] = None,
    ragas: Optional[Dict[str, Any]] = None,
    run_id: Optional[int] = None,
    stage_trace: Optional[List[Dict[str, Any]]] = None,
    path: Optional[Path] = None,
) -> int:
    init_db(path)
    p = path or get_db_path()
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rf = ragas or {}
    with _connect(p) as conn:
        cur = conn.execute(
            """
            INSERT INTO experiment_queries (
              run_id, created_at, source, question, reference_answer, llm_output,
              retrieved_chunks_json, latency_ms, token_count,
              gold_hit, token_f1, exact_match,
              ragas_faithfulness, ragas_context_precision, ragas_answer_accuracy,
              human_feedback, model_config_json, stage_trace_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                now,
                source,
                question,
                reference_answer,
                llm_output,
                json.dumps(retrieved_chunks, ensure_ascii=False),
                float(latency_ms),
                int(token_count),
                gold_hit,
                token_f1,
                exact_match,
                rf.get("response_groundedness"),
                rf.get("context_relevance"),
                rf.get("answer_accuracy"),
                None,
                json.dumps(model_config, ensure_ascii=False),
                json.dumps(stage_trace, ensure_ascii=False) if stage_trace else None,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def update_query_feedback(
    query_id: int, feedback: str, path: Optional[Path] = None
) -> None:
    p = path or get_db_path()
    with _connect(p) as conn:
        conn.execute(
            "UPDATE experiment_queries SET human_feedback = ? WHERE id = ?",
            (feedback, int(query_id)),
        )
        conn.commit()


def fetch_runs_dataframe(path: Optional[Path] = None):
    import pandas as pd

    init_db(path)
    p = path or get_db_path()
    with _connect(p) as conn:
        rows = conn.execute(
            "SELECT * FROM experiment_runs ORDER BY id DESC LIMIT 5000"
        ).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    return df


def fetch_queries_dataframe(path: Optional[Path] = None, *, failed_only: bool = False):
    import pandas as pd

    init_db(path)
    p = path or get_db_path()
    q = "SELECT * FROM experiment_queries ORDER BY id DESC LIMIT 5000"
    if failed_only:
        q = """
        SELECT * FROM experiment_queries
        WHERE (gold_hit IS NOT NULL AND gold_hit < 0.5)
           OR (token_f1 IS NOT NULL AND token_f1 < 0.2)
        ORDER BY id DESC
        LIMIT 2000
        """
    with _connect(p) as conn:
        rows = conn.execute(q).fetchall()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([dict(r) for r in rows])


def aggregate_token_totals(path: Optional[Path] = None) -> Dict[str, float]:
    init_db(path)
    p = path or get_db_path()
    with _connect(p) as conn:
        r_run = conn.execute(
            "SELECT COALESCE(SUM(token_count), 0) FROM experiment_runs"
        ).fetchone()
        r_q = conn.execute(
            "SELECT COALESCE(SUM(token_count), 0) FROM experiment_queries"
        ).fetchone()
    return {
        "run_table_tokens": float(r_run[0] or 0),
        "query_table_tokens": float(r_q[0] or 0),
    }


def log_evaluation_batch(
    *,
    source: str,
    experiment_name: str,
    setting_label: str,
    embedding_model: str,
    llm_backend: str,
    llm_model: str,
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    per_example: List[Dict[str, Any]],
    path: Optional[Path] = None,
) -> int:
    """
    One aggregate row in experiment_runs plus one experiment_queries row per example
    (for failure analysis in the dashboard).
    """
    total_tokens = sum(int(x.get("token_count") or 0) for x in per_example)
    run_id = log_experiment_run(
        source=source,
        run_label=f"{experiment_name}:{setting_label}",
        experiment_tag=experiment_name,
        embedding_model=embedding_model,
        model_config=config,
        latency_ms=float(metrics.get("latency_total_ms") or 0.0),
        token_count=max(total_tokens, int(metrics.get("approx_total_tokens") or 0)),
        metrics=metrics,
        llm_backend=llm_backend,
        llm_model=llm_model,
        path=path,
    )
    for row in per_example:
        log_query_event(
            source=source,
            question=str(row.get("question") or ""),
            reference_answer=row.get("reference_answer"),
            llm_output=str(row.get("prediction") or ""),
            retrieved_chunks=list(row.get("retrieved_passages") or []),
            latency_ms=float(row.get("latency_total_ms") or 0.0),
            token_count=int(row.get("token_count") or 0),
            model_config=config,
            gold_hit=(
                float(row["gold_hit"]) if row.get("gold_hit") is not None else None
            ),
            token_f1=(
                float(row["token_f1"]) if row.get("token_f1") is not None else None
            ),
            exact_match=(
                float(row["exact_match"])
                if row.get("exact_match") is not None
                else None
            ),
            run_id=run_id,
            path=path,
        )
    return run_id
