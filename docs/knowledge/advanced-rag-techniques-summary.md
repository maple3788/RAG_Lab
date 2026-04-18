# Advanced RAG Techniques - Bullet-Point Summary

Source: [DataCamp - Advanced RAG Techniques](https://www.datacamp.com/blog/rag-advanced)

## 1) Limits of Basic RAG

- Hallucinations reduce trust when answers are not grounded in retrieved evidence.
- General pipelines can miss domain-specific nuance and return weak documents.
- Multi-step and multi-turn queries often lose context and coherence.

## 2) Advanced Retrieval Techniques

- **Dense retrieval** (for example, DPR) captures semantic meaning via embeddings.
- **Hybrid search** combines sparse and dense retrieval for better precision + recall.
- **Reranking** reorders retrieved docs so the best evidence reaches the generator first.
- **Query expansion** broadens recall with:
  - Synonym expansion.
  - Conceptual expansion.

## 3) Improving Relevance and Context Quality

- Retrieval alone is not enough; quality control of context is critical.
- **Advanced filtering** removes low-value results before generation:
  - Metadata-based filters (date, author, source, type).
  - Content-based filters (semantic similarity, required terms, relevance thresholds).
- **Context distillation** condenses long/noisy docs into the most useful evidence.

## 4) Generation Optimization

- Prompt quality strongly affects answer quality.
- Prompt engineering practices:
  - Provide explicit context and instructions.
  - Structure prompts clearly and reduce ambiguity.
  - Test multiple prompt formats iteratively.
- **Multi-step reasoning** helps with complex tasks:
  - Chain retrieval and generation.
  - Use intermediate reasoning steps.
  - Apply multi-hop QA across multiple documents.

## 5) Hallucination Mitigation

- Ground responses tightly in retrieved documents.
- Improve context conditioning so irrelevant material is excluded.
- Add feedback/verification loops that check generated output against sources.

## 6) Handling Complex Conversations and Ambiguity

- Multi-turn robustness requires memory and context management:
  - Conversation history tracking.
  - Dynamic context windowing.
  - Retrieval-based memory for long sessions.
- Ambiguous queries are handled with:
  - Clarification questions.
  - Query decomposition into smaller sub-problems.
  - Contextual clues from prior turns.
  - Advanced retrieval (including multi-hop retrieval) for hard questions.

## 7) Common RAG Challenges and Solutions

- **Bias**
  - Bias-aware retrieval for source diversity.
  - Fairer generation via curated fine-tuning.
  - Post-generation filtering for harmful or biased outputs.
- **Computational overhead**
  - Efficient indexing/search (for example, approximate nearest neighbors).
  - Model optimization (distillation, quantization, pruning).
- **Data limitations**
  - Data augmentation.
  - Domain adaptation.
  - Active learning to prioritize high-value annotation.

## 8) Implementation Stack and Workflow

- Common tools/frameworks:
  - LangChain.
  - Haystack.
  - OpenAI API (as the generation layer with retrieval frameworks).
- Suggested rollout sequence:
  1. Choose framework based on use case and scale.
  2. Build retrieval (dense or hybrid).
  3. Add reranking and filtering.
  4. Improve generation (prompting, distillation, multi-step reasoning).
  5. Evaluate with metrics and A/B tests.
  6. Optimize for scale and latency.
  7. Monitor continuously and refresh models/indexes.

## 9) Evaluation Metrics for Advanced RAG

- **Accuracy**: correctness of final answers.
- **Relevance**: ranking and answer usefulness (for example, MRR, Precision@K).
- **Latency**: response speed in retrieval + generation.
- **Coverage**: breadth of query types handled well.

## 10) Key Use Cases

- Complex question-answering systems.
- Domain-heavy assistants:
  - Healthcare (guidelines, papers, patient context).
  - Finance (reports, filings, forecasts).
- Personalized recommendations:
  - E-commerce.
  - Content/news platforms.

## 11) Future Directions Mentioned

- Better integration of diverse data sources (DBs, APIs, real-time feeds).
- Improved handling of ambiguous/incomplete queries.
- Stronger multi-step reasoning and synthesis across sources.
- More personalization from user history and preferences.
- Better neural retrievers and tighter retrieval-generation coupling.
- Research pointers cited in article: DPR, few-shot RAG adaptation, retrieval-enhanced generation (for example, RAVEN).
