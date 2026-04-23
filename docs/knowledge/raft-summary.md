# RAFT (RAG + Fine-Tuning) - Summary and Practical Notes

Source: [DataCamp - What is RAFT? Combining RAG and Fine-Tuning](https://www.datacamp.com/blog/what-is-raft-combining-rag-and-fine-tuning)

## 1) What RAFT Is

- **RAFT (Retrieval-Augmented Fine-Tuning)** combines:
  - RAG (retrieve external context at inference),
  - fine-tuning (adapt model behavior to a target domain).
- Goal: make LLMs stronger in specialized domains while staying robust to imperfect retrieval.

## 2) Why RAFT Exists

- RAG alone:
  - can access fresh external knowledge,
  - but model may not be trained to reliably use retrieved context.
- Fine-tuning alone:
  - can learn domain style and patterns,
  - but may not adapt well to dynamic external docs and can go stale.
- RAFT bridges this gap by training the model in a retrieval-like setting.

## 3) Core RAFT Training Design

Each training example includes:

- a question,
- a set of retrieved-like documents:
  - **oracle docs** (contain answer evidence),
  - **distractor docs** (irrelevant/noisy),
- a target answer often in chain-of-thought style.

This teaches the model to:
- identify relevant evidence,
- ignore irrelevant documents,
- produce domain-appropriate answers grounded in context.

## 4) Key Components Mentioned

- Mixture of question instances:
  - some with oracle + distractors,
  - some with only distractors (to improve robustness).
- Supervised fine-tuning objective:
  - learn to generate accurate answers from provided evidence.
- Inference still uses normal retrieval:
  - RAFT improves the answer model; retriever remains separate.

## 5) Reported Results (from paper summary in article)

- RAFT outperformed:
  - plain base model,
  - base model + RAG,
  - standard domain-specific fine-tuning (DSF),
  - DSF + RAG,
  - and in some settings GPT-3.5 + RAG.
- Benchmarks included:
  - general QA sets,
  - software/API documentation tasks,
  - biomedical/domain-specific QA.

## 6) Ablation Insights Mentioned

- Chain-of-thought style supervision generally improved performance.
- A high proportion of oracle-containing examples (around 80%) worked well across datasets.
- A balanced distractor setting (for example, 1 oracle + 4 distractors) was effective.

## 7) Benefits of RAFT

- Better domain accuracy.
- Better use of retrieved documents.
- Better robustness to retrieval noise.
- Stronger answer quality than naive "RAG only" or "fine-tune only" approaches.

## 8) Trade-offs / Practical Costs

- More complex data preparation than standard fine-tuning.
- Requires careful construction of oracle/distractor training samples.
- Depends on domain-quality supervision data.

## 9) How This Applies to `RAG_Lab`

Your current stack already has retrieval and reranking components, so RAFT-style evolution would be:

1. Build supervised QA training rows with:
   - retrieved contexts (good + distractors),
   - target answers.
2. Fine-tune a local model on this retrieval-aware format.
3. Keep existing retriever/reranker pipeline at inference.
4. Compare against current baseline on:
   - EM/F1/gold-hit,
   - robustness under noisy retrieval.

## 10) Bottom Line

- RAFT is best viewed as **retrieval-aware fine-tuning** rather than replacing RAG.
- It is useful when domain accuracy is critical and retrieval noise is a real problem.
- For production, RAFT can complement (not replace) strong retrieval engineering.
