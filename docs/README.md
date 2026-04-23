# Documentation Index

Use this page as the entry point for project documentation.

## Core Guides

- [`quickstart.md`](quickstart.md): local setup, Streamlit workflow, and first-run checklist
- [`architecture.md`](architecture.md): retrieval/generation system design and component map
- [`experiments.md`](experiments.md): experiment scripts, metrics, and output locations
- [`results.md`](results.md): benchmark summary and interpretation notes
- [`operations.md`](operations.md): tracing, error analysis, benchmark DB, and maintenance

## Production and Serving

- [`api-deployment.md`](api-deployment.md): FastAPI service path (`Ollama + Milvus + Redis + Nginx`) and monitoring
- Includes multi-job querying (`job_ids`), session-bound default jobs, and dense/hybrid retrieval controls.

## Additional Notes

- [`gemini-region-restriction.md`](gemini-region-restriction.md): Gemini regional/API restrictions and workaround notes
- [`knowledge/advanced-rag-techniques-summary.md`](knowledge/advanced-rag-techniques-summary.md): concise retrieval/generation technique notes (query expansion, reranking, hybrid retrieval)
- [`knowledge/self-rag-summary.md`](knowledge/self-rag-summary.md): self-RAG feedback-loop concepts and practical rollout notes
- [`knowledge/late-chunking-summary.md`](knowledge/late-chunking-summary.md): late chunking concept and context-preserving retrieval notes
- [`knowledge/corrective-rag-crag-summary.md`](knowledge/corrective-rag-crag-summary.md): CRAG flow (retrieve, evaluate, correct, generate)
- [`knowledge/rag-vs-cag-summary.md`](knowledge/rag-vs-cag-summary.md): decision framework for RAG vs CAG vs hybrid routing
- [`knowledge/raft-summary.md`](knowledge/raft-summary.md): retrieval-aware fine-tuning (RAFT) summary and implications
- [`knowledge/nlu-summary.md`](knowledge/nlu-summary.md): NLU fundamentals and relevance to query understanding
- [`SESSION_SUMMARY.md`](SESSION_SUMMARY.md): historical session summary notes
- [`gpuhub-git-push.md`](gpuhub-git-push.md): environment-specific git push notes

