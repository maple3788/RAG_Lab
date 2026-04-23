# Corrective RAG (CRAG) - Summary and Practical Notes

Source: [DataCamp - Corrective RAG (CRAG) Implementation With LangGraph](https://www.datacamp.com/tutorial/corrective-rag-crag)

## 1) What CRAG Is

- CRAG is an enhanced RAG strategy that **checks and corrects retrieval quality** before final generation.
- Instead of trusting retrieved documents by default, CRAG adds a **retrieval evaluator** step.
- Goal: reduce hallucinations and improve answer reliability.

## 2) Core CRAG Flow

1. Retrieve documents from a knowledge base.
2. Evaluate retrieved docs for relevance/confidence.
3. Take corrective action based on confidence:
   - **Correct**: refine retrieved knowledge and generate.
   - **Incorrect**: discard docs and use web search for fresh evidence.
   - **Ambiguous**: combine refined retrieval + web evidence.
4. Generate final answer from corrected evidence.

## 3) CRAG vs Traditional RAG

- Traditional RAG:
  - retrieve -> generate
  - weak guardrails when retrieval quality is poor
- CRAG:
  - retrieve -> evaluate -> correct -> generate
  - stronger safeguards against irrelevant/noisy context
  - better grounding from quality control before generation

## 4) Tutorial Implementation Components (LangGraph)

- **Retriever** over vector store (Chroma in tutorial).
- **RAG chain** for answer generation.
- **Retrieval evaluator** (binary relevant/not relevant).
- **Question rewriter** to improve search query.
- **Web search tool** (Tavily) when KB evidence is weak.
- **LangGraph workflow** to route logic dynamically.

## 5) LangGraph Decision Logic in Tutorial

- If enough retrieved docs are relevant -> generate directly.
- If relevance is weak -> rewrite query, run web search, then generate.
- This creates a conditional correction path rather than a fixed pipeline.

## 6) Benefits Highlighted

- Better factuality via retrieval quality checks.
- Less noise from irrelevant documents.
- Better fallback behavior when internal KB is insufficient.
- More robust answers on out-of-KB questions due to web augmentation.

## 7) Limitations Highlighted

- Heavy dependence on retrieval evaluator quality.
- Evaluator tuning/maintenance adds complexity and cost.
- Web search can introduce noisy, biased, or low-quality sources.
- Requires careful governance of external evidence quality.

## 8) Practical Guardrails to Use

- Add confidence thresholds and explicit fallback behavior.
- Track evaluator precision/recall, not just final answer quality.
- Keep source filtering and citation requirements for web-derived evidence.
- Add latency and cost budgets for correction paths.

## 9) How This Applies to `RAG_Lab`

Your project already has strong building blocks for CRAG-style behavior:

- retrieval + reranking,
- query rewriting capability,
- strict citation prompt option,
- API and experiment infrastructure.

Natural next step:

- add a retrieval quality gate before generation,
- trigger rewrite and/or external retrieval when confidence is low,
- then generate with stricter grounding constraints.

## 10) Bottom Line

- CRAG is a practical upgrade when retrieval quality is inconsistent.
- It shifts RAG from "retrieve once and hope" to "retrieve, assess, and correct."
- Most value comes from high-quality evaluator logic plus safe evidence handling.
