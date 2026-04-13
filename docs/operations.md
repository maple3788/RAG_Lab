# Operations, Traces, and Error Analysis

## Streamlit benchmark database

SQLite path:
- `results/experiment_db.sqlite`

Captured data:
- run config and model metadata
- latency and token estimates
- answer quality metrics
- per-query payloads for failure analysis
- human feedback tags in bad-case review

## Trace export tools

- `tools/export_triviaqa_traces.py`
- `tools/export_qasper_traces.py`
- `scripts/build_error_analysis_draft.py`

Typical flow:

```bash
python tools/export_qasper_traces.py --out analysis/qasper_traces.jsonl --max-examples 60 --llm-backend ollama
python scripts/build_error_analysis_draft.py --in analysis/qasper_traces.jsonl --out analysis/error_analysis_draft.md
```

## Failure taxonomy

Common categories:
- retrieval miss
- ranking miss
- generation hallucination
- context truncation loss

See:
- `analysis/ERROR_ANALYSIS.md`
- `analysis/TRACES_JSONL_FIELDS.md`

## Ingest job lifecycle

Redis key:
- `ingest:job:{job_id}`

Typical stages:
- `queued -> extracting -> chunking -> embedding -> storing -> completed`
- `failed` with error payload

## Maintenance tips

- Use Library "Remove from library" to delete MinIO artifacts and clear Redis status for old jobs.
- Keep `results/experiment_db.sqlite` local (ignored by git).
- If Grafana provisioning changes, run `docker compose restart grafana`.

