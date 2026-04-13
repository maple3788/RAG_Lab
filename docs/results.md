# Results Snapshot

This page summarizes representative findings from existing experiments.

## TREC-COVID (IR)

Setup:
- full corpus and qrels
- FAISS `IndexFlatIP` with normalized embeddings
- `retrieve_k=100`
- optional cross-encoder rerank over top-100

Observed trend:
- reranking consistently improves `P@10` and `nDCG@10`
- `R@100` stays unchanged when rerank only reorders the same candidate pool

## TriviaQA (RAG generation)

Observed trend (sampled runs):
- reranker usually improves EM/F1/gold_hit
- `final_k` too small hurts recall; moderate `final_k` performs best
- prompt style has measurable but smaller impact than retrieval quality

## QASPER (long-document QA)

Observed trend:
- hybrid retrieval (BM25 + dense + RRF) improves retrieval oracle hit rate
- reranker improves Token F1 and Gold Hit
- when context budget is tight, `head` truncation outperforms `tail`/`middle`

Interpretation:
- long-document QA is bottlenecked by retrieval precision and context packing
- EM is harsh for alias-rich answers; F1 and Gold Hit are usually more informative

## Where to find raw outputs

- `results/*.csv`
- `assets/*.png`
- `analysis/ERROR_ANALYSIS.md`

For reproducibility, run scripts listed in [`experiments.md`](experiments.md).

