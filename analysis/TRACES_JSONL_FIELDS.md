# `triviaqa_traces.jsonl` 字段说明（中文）

`tools/export_triviaqa_traces.py` 输出的每一行是一个 **JSON 对象**，对应 **一道 TriviaQA RC 题** 的一次追踪记录。

---

## 一、`pool` 和 `final` 是什么？

对**当前这道题**，先把官方给的段落切成 **chunk**，再在**这些 chunk 上**建 FAISS 索引，用 **问题** 做向量检索：

| 名字 | 含义 |
|------|------|
| **pool（候选池）** | **向量检索**阶段，按相似度取回的 **前 `retrieve_k` 个 chunk 全文**（例如 10 条）。表示「双塔模型认为和问题最相关的一批段落」。 |
| **final（送进模型的段落）** | 在 pool 基础上：若开了 **rerank**，用交叉编码器重排后只保留 **前 `final_k` 条**；若**没开 rerank**，就取 pool 的 **前 `final_k` 条**。这就是后面拼进 **prompt 里 `Context` 的段落**（在截断之前）。 |

流程可以记成：

```text
全部 chunk → FAISS 按题检索 → pool（top retrieve_k）
         → [可选 rerank] → final（top final_k）→ 拼成 prompt → [可选 LLM]
```

`pool_previews` / `final_previews` 是 **截断后的预览**（默认每段约前 400 字符），方便肉眼看；完整文本在内存里更长。

---

## 二、每个字段什么意思？

### 标识与题目

| 字段 | 含义 |
|------|------|
| `question_id` | 题目 ID（数据集中自带）。 |
| `question` | 问题全文。 |
| `gold_aliases` | 官方参考答案的 **别名列表**（最多存 20 个），用于判断「金标是否出现在某段文字里」。 |

### 本次运行的配置（可复现）

| 字段 | 含义 |
|------|------|
| `config.retrieve_k` | FAISS 取回多少条进 **pool**。 |
| `config.final_k` | 最终给 LLM 多少条（**final** 的长度上限）。 |
| `config.use_rerank` | 是否在 pool 上做 **交叉编码器 rerank**。 |
| `config.prompt` | 用的提示模板名：`default` / `bullets` / `strict_cite`。 |
| `config.truncation` | 拼好 context 后的截断策略：`head` / `tail` / `middle`。 |
| `config.max_context_chars` | 拼好的 context **字符串**最多多少字符（再喂给模型）。 |

### 金标与检索阶段（**不依赖 LLM**）

| 字段 | 含义 |
|------|------|
| `any_gold_in_pool` | **pool 里**是否至少有一个 chunk，其文字里 **子串匹配**到任一 `gold_aliases`（忽略大小写）。`true` 说明「检索阶段已经捞到含答案字样的段落」。 |
| `any_gold_in_final_passages` | **final 里**是否至少有一个 chunk 含金标子串。`false` 但 `any_gold_in_pool` 为 `true` 时，典型是 **排序/截断问题**：金标在 pool 里较后的位置，没进 top `final_k`。 |
| `gold_lost_to_truncation` | 拼成 **完整** context（`raw`）时金标还在，但 **截断后**（`trunc_ctx`）金标子串不见了 → 模型看到的文字里可能根本没有答案词（容易误判成「生成错」）。 |

### 阶段标签

| 字段 | 含义 |
|------|------|
| `retrieval_stage` | 粗分三类：**`retrieval`** = pool 里就没有金标；**`ranking`** = pool 有、final 没有；**`gold_in_final`** = final 里已有含金标的段落（从「证据是否进 prompt」角度已够）。 |

### 与 LLM 相关的字段

| 字段 | 含义 |
|------|------|
| `llm_called` | 本次是否 **真的调用了** 大模型。`false` = 用了 `--skip-generation`。 |
| `failure_bucket` | **`llm_called == true` 时才有意义**：`none`（EM=1）、`retrieval`、`ranking`、`generation`。见 `ERROR_ANALYSIS.md`。为 `null` 时表示没跑 LLM，**不填**该分类。 |
| `prediction` | 模型输出的**原始字符串**（过长会截断写入，以脚本为准）。不调 LLM 时为 `""`。 |
| `exact_match` / `token_f1` / `gold_hit` | 与主实验相同的指标；**不调 LLM 时一般为 0**（因为没有 `prediction`）。 |

### 预览列表

| 字段 | 含义 |
|------|------|
| `pool_previews` | **pool** 里每条 chunk 的**短预览**（按检索顺序）。 |
| `final_previews` | **final** 里每条 chunk 的短预览（即实际用于拼 prompt 的段落，截断前）。 |

### 异常行

若某题切完 chunk 为空，可能出现：

| 字段 | 含义 |
|------|------|
| `error` | 例如 `"no_chunks_after_splitting"`，没有其它分析字段。 |

---

## 三、不调 LLM（`--skip-generation`）和调 LLM 的区别

| | **加 `--skip-generation`** | **不加（正常调 Ollama/Gemini 等）** |
|---|---------------------------|-------------------------------------|
| **速度 / 成本** | 只做向量检索 + 规则打标，**很快**，无 API/本地推理费用。 | 每题要 **生成一次**，慢、占 GPU/配额。 |
| **`pool` / `final` / `any_gold_*` / `retrieval_stage`** | **一样算**，与是否调 LLM **无关**。 | 同上。 |
| **`prediction`** | 空字符串。 | 有模型输出。 |
| **`exact_match` / `token_f1` / `gold_hit`** | 通常为 **0**（没有预测可比对）。 | 有真实分数。 |
| **`failure_bucket`** | **恒为 `null`**（脚本故意不算「生成失败」）。 | 有值：`none` / `retrieval` / `ranking` / `generation`。 |
| **`llm_called`** | `false`。 | `true`。 |
| **适用场景** | 只想做 **检索/排序错误** 统计、筛 case、不写答案。 | 要做完整 **端到端** 错误分析（含生成错）。 |

**总结：**  
- 只看 **检索有没有捞到金标、排序有没有把金标挤掉** → 用 **`--skip-generation` 就够了**。  
- 要区分 **「上下文里已有答案但模型答错」** → 必须 **调 LLM**，并看 `failure_bucket == "generation"` 等。

---

## 四、与主实验 CSV 的关系

- `exp_rag_generation_triviaqa.py` 跑的是 **平均分**，**不写**这个 jsonl。  
- `export_triviaqa_traces.py` 是 **单独脚本**，专门产出 **逐题** 记录，**不改变** benchmark 脚本行为。
