# RAG error analysis — how to do it (retrieval / ranking / generation)

Average EM / F1 hides *where* the pipeline fails. This doc defines **three failure buckets** and a **repeatable workflow** so you can fill **success vs failure case studies** (e.g. 3 + 3) for README or interviews.

---

## 1. What “supporting evidence” means (TriviaQA RC)

In **this repo’s TriviaQA RC setup** (`per_example_retrieval=True`):

- Each question has a **fixed set of passages** (Wikipedia / search snippets).
- Text is **chunked**; retrieval returns **top‑`retrieve_k` chunks**, then optionally **rerank → top‑`final_k`** passed to the LLM (after truncation).

**Oracle check (automatic, cheap):**  
A chunk is **“gold‑supporting”** if **any official answer alias** appears in that chunk (case‑insensitive substring), same spirit as `gold_hit` on the model output.

> Limitation: some answers are **paraphrased** in text — substring check **misses** those; you may label a few borderline cases manually.

---

## 2. Three failure types (definitions)

Assume you have:

- `pool` = chunks after **FAISS**, size ≤ `retrieve_k`
- `final` = chunks fed to the LLM (after rerank / slice to `final_k`, before **prompt truncation** if you analyze “what the model saw” — ideally log **after** truncation too)

Let `has_gold(chunk)` = true iff some alias appears in `chunk`.

| Type | Definition (operational) |
|------|---------------------------|
| **Retrieval failure** | **No** chunk in `pool` has `has_gold` → bi‑encoder never surfaced text that literally contains an alias. |
| **Ranking failure** | **Some** chunk in `pool` has `has_gold`, but **no** chunk in `final` has `has_gold` → answer was in the **first‑stage** list but **dropped** by top‑k cut or **rerank** order. |
| **Generation failure** | **Some** chunk in `final` has `has_gold`, but prediction is still wrong (EM/F1 low) → **context was enough in principle** (by substring), but the **LLM** picked wrong wording / ignored evidence / followed prompt badly. |

**Decision order:** check retrieval → ranking → generation (first match wins).

**Success (for a case study):** e.g. EM=1 or high F1 **and** `has_gold` in `final` — optionally note clean retrieval + good answer.

---

## 3. How to get the data (implementation path)

You need **per‑question traces**, not only aggregates.

### Minimum fields to log (JSONL one row per question)

- `question_id`, `question`
- `gold_aliases` (list)
- `retrieve_k`, `final_k`, `use_rerank`, `prompt_template`, `truncation` (for reproducibility)
- `pool_chunk_texts` (or hashes) and order
- `final_chunk_texts` (what actually goes into the prompt **after** truncation — best for explaining generation errors)
- `prediction` (raw LLM string)
- `em`, `token_f1`, `gold_hit` (per example)
- **Derived:** `any_gold_in_pool`, `any_gold_in_final`, `failure_bucket` ∈ {`none`, `retrieval`, `ranking`, `generation`}

### Where to hook in code

- Reuse the same path as `evaluate_rag_answer_quality` in `src/rag_generation.py`: after `retrieve_passages_for_query`, you already have **passages**; before `truncate_context`, you can label **ranking** vs **retrieval** using **full** retrieved list vs **final** list.
- For **retrieval**, you need the **full** `retrieve_k` list **before** cutting to `final_k` — today `retrieve_passages_for_query` returns only `final_k` passages. To classify **ranking**, extend retrieval to return **(pool, final)** or call a lower‑level function that returns top `retrieve_k` then reranks.

**Concrete next step in repo:** add optional `--export-traces path.jsonl` to `exp_rag_generation_triviaqa.py` (or a small script under `tools/`) that:

1. Loops examples like the eval loop.
2. Gets **pool** = top `retrieve_k` texts (same as FAISS order).
3. Applies rerank if enabled → **final** = top `final_k`.
4. Computes `has_gold` flags and `failure_bucket`.
5. Appends one JSON object per line.

Then you **manually** pick 3 successes and 3 failures from the JSONL and paste short excerpts into README (no long dumps — redact if needed).

---

## 4. Filling the “3 + 3” case study (template)

For **each** case, keep it short:

| Field | What to write |
|-------|----------------|
| **id** | `question_id` |
| **question** | One line |
| **Outcome** | Success / Failure |
| **Bucket** | `none` \| `retrieval` \| `ranking` \| `generation` |
| **Evidence** | 1–2 sentences: does any alias appear in `pool` / `final`? |
| **Why** | Plain language: e.g. “query–passage vocabulary mismatch”, “reranker preferred wrong chunk”, “model answered with city instead of country” |
| **Snippet** | Optional: 1 short quote from context or prediction |

### Example failure one‑liners (illustrative)

- **Retrieval:** “Aliases only appear in a passage chunk that scored below top‑10.”
- **Ranking:** “Alias appears in chunk ranked 4th; only top‑3 sent to LLM.”
- **Generation:** “Alias string present in passage [2] but model output ‘unknown’ due to strict_cite.”

---

## 5. Pitfalls

- **Truncation:** If you only check `final` **before** `truncate_context`, you might mislabel **ranking** as **generation** — gold falls out of the **tail** of the packed context. Log whether gold was **lost to truncation** as a fourth tag if needed (`truncation_loss`).
- **Aliases vs paraphrase:** substring misses correct **semantic** answers → some “generation” labels are debatable; mention this in writeups.
- **TREC / proxy gold:** error taxonomy for `metadata.query` is different; prefer **TriviaQA** for clean failure analysis first.

---

## 6. Tool in this repo

**Field-by-field explanation (Chinese):** see **`TRACES_JSONL_FIELDS.md`** in this folder.

After `pip install -r requirements.txt`, run from the project root:

```bash
# Fast: no LLM — only retrieval_stage + gold flags (good for mining retrieval/ranking errors)
python tools/export_triviaqa_traces.py --out analysis/triviaqa_traces.jsonl --max-examples 100 --skip-generation

# Full: includes prediction + failure_bucket (retrieval | ranking | generation | none)
python tools/export_triviaqa_traces.py --out analysis/triviaqa_traces.jsonl --max-examples 50 --llm-backend ollama --llm-model llama3.2
```

`retrieve_passages_pool_and_final` in `src/rag_pipeline.py` exposes **pool** vs **final**; `evaluate_rag_answer_quality` is **unchanged**.

## 7. Suggested order of work

1. Use **`tools/export_triviaqa_traces.py`** to produce JSONL.
2. Run **50–200** examples; aggregate counts: `% retrieval / ranking / generation` among errors.
3. Pick **3 successes** (diverse: easy retrieval, rerank helped, hard question).
4. Pick **3 failures** (one per bucket if possible).
5. Copy summaries into **README** (`analysis/` 里保留详细版也可以).

This matches what strong ML / RAG engineers do: **metrics + stratified error analysis + concrete examples.**
