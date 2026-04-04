# QASPER RAG Pipeline: Error Analysis

As part of the evaluation process, we conducted a qualitative analysis of system performance. Below are representative success cases and failure cases categorized by pipeline stage.

## ✅ Success Cases

When the system works well, it correctly bridges the semantic gap between the user's question and the academic language used in the document.

**1. Semantic Entity Extraction**
*   **Question:** "What is the source of the news sentences?"
*   **Retrieval:** Successfully retrieved the chunk mentioning "news texts from ilur.am".
*   **Generation:** Accurately extracted `ilur.am` as the source.
*   **Why it worked:** The embedding model easily mapped "source of news" to "news texts from", demonstrating strong semantic alignment.

**2. Multi-Hop/Compound Fact Retrieval**
*   **Question:** "Which datasets did they experiment with?"
*   **Retrieval:** Retrieved chunks mentioning experimental setups.
*   **Generation:** Accurately extracted both `Europarl` and `MultiUN`.
*   **Why it worked:** The reranker properly elevated the chunk containing the dataset statistics Table over other chunks that merely mentioned "data".

**3. Concept Abstraction**
*   **Question:** "How do they match words before reordering them?"
*   **Retrieval:** Retrieved chunks discussing the CFILT-preorder system and generic rules for Indian languages.
*   **Generation:** Successfully summarized the pre-ordering system rules.
*   **Why it worked:** The pipeline successfully handled a conceptual question rather than a strict factual lookup, finding the specific methodology section.

---

## ❌ Failure Cases & Bottleneck Analysis

To improve the system, we categorize failures into three distinct bottlenecks: **Retrieval**, **Ranking**, and **Generation**.

### 1. Retrieval Failure (Gold chunk not in top-K pool)
*   **Question:** "What accuracy does the proposed system achieve?"
*   **Result:** The system failed to retrieve the table/text containing the F1 scores (85.99 on DL-PS, etc.).
*   **System Bottleneck:** **Dense vs. Sparse Mismatch.** Dense embeddings (E5/BGE) often struggle with specific metric lookups (mapping "accuracy" to "F1 score") and tabular data extraction. 
*   **Potential Fix:** Implement a Hybrid Search (BM25 + Dense) to catch exact keyword matches for metrics and table headers.

### 2. Ranking Failure (Gold chunk retrieved, but pushed out of final context)
*   **Question:** "Which multilingual approaches do they compare with?"
*   **Result:** The correct chunk mentioning "multilingual NMT (MNMT)" was in the initial pool of 100, but the Cross-Encoder reranker pushed it out of the `final_k` (top 5).
*   **System Bottleneck:** **Lexical Distraction.** The cross-encoder favored chunks heavily mentioning "languages" and "evaluation" over the specific "approaches" baseline section.
*   **Potential Fix:** Tune the reranker domain, or increase `final_k` context window if the LLM can handle a larger context without degrading generation quality.

### 3. Generation Failure (Gold chunk in final context, but LLM fails to answer)
*   **Question:** "What are the pivot-based baselines?"
*   **Result:** The gold chunk mentioning "related approaches of pivoting" was successfully fed to the LLM, but the LLM failed to extract it, getting distracted by statistics in the same chunk.
*   **System Bottleneck:** **"Lost in the Middle" / Distraction.** When context chunks are dense and packed with tables or statistics, smaller LLMs (or truncated contexts) can miss the specific answer hiding within the text.
*   **Potential Fix:** Implement strict prompt engineering (e.g., instructing the LLM to output "Not Found" rather than hallucinate, or using chain-of-thought to parse the context before answering) or utilize an LLM with better long-context reasoning.