# Late Chunking for RAG - Summary and Practical Notes

Source: [DataCamp - Late Chunking for RAG](https://www.datacamp.com/tutorial/late-chunking)

## 1) Why Late Chunking

- Standard (naive) chunking splits a document first, then embeds each chunk independently.
- This can lose long-range context (for example, pronouns or references separated across chunks).
- Late chunking flips the order:
  - embed the full document first (token-level contextualized embeddings),
  - split into chunks afterward,
  - pool token embeddings for each chunk span.

## 2) Main Idea

- **Naive chunking**: `split -> embed each chunk`
- **Late chunking**: `embed full doc -> split -> pool token vectors by span`
- Result: chunk embeddings keep more document-wide context.

## 3) Benefits Mentioned

- Better context retention across chunk boundaries.
- Better retrieval relevance for references spread across the document.
- Useful for long texts when using long-context embedding models.

## 4) Tutorial Implementation Flow

1. Create chunks and span annotations.
2. Tokenize full document and compute token embeddings.
3. For each chunk span, mean-pool token embeddings (late chunk vectors).
4. Compare retrieval similarity vs traditional chunk embeddings.

## 5) Key Takeaway from Example

- In the DataCamp demo, late chunking raised similarity scores on chunks that did not explicitly restate "Berlin" but were still about Berlin through context.
- This demonstrates late chunking preserving implicit context better than naive chunking.

## 6) Practical Trade-offs

- Pros:
  - richer chunk embeddings
  - improved retrieval for context-dependent queries
- Cons:
  - more complex pipeline
  - higher compute/memory than simple chunk-then-embed

## 7) How to Apply in `RAG_Lab`

- Keep current chunking pipeline as baseline.
- Add an experiment path for late chunking and compare:
  - `recall@k`
  - latency/cost
  - behavior on long or context-heavy questions
- Start with small benchmark sets before rolling into full ingest path.

## 8) Bottom Line

- Late chunking is a practical retrieval-quality upgrade when your queries depend on cross-chunk context.
- It is most valuable when context linkage matters more than lowest-latency indexing.
