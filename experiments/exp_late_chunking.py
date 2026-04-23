from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.loader import QAExample, flatten_contexts, load_qa_jsonl


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class TextChunk:
    text: str
    start: int
    end: int


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if n <= 1e-12:
        return vec.astype(np.float32, copy=False)
    return (vec / n).astype(np.float32, copy=False)


def split_with_spans(text: str, *, max_chars: int = 300) -> List[TextChunk]:
    """
    Simple sentence-aware chunking that preserves char spans in original text.
    """
    text = text or ""
    if not text.strip():
        return []

    sentences = _SENT_SPLIT_RE.split(text)
    spans: List[TextChunk] = []
    cursor = 0
    for s in sentences:
        if not s.strip():
            continue
        idx = text.find(s, cursor)
        if idx < 0:
            idx = cursor
        start = idx
        end = idx + len(s)
        cursor = end
        spans.append(TextChunk(text=s, start=start, end=end))

    out: List[TextChunk] = []
    cur_text = ""
    cur_start = 0
    cur_end = 0
    for i, part in enumerate(spans):
        if not cur_text:
            cur_text = part.text
            cur_start = part.start
            cur_end = part.end
            continue
        trial = f"{cur_text} {part.text}"
        if len(trial) <= max_chars:
            cur_text = trial
            cur_end = part.end
        else:
            out.append(TextChunk(text=cur_text.strip(), start=cur_start, end=cur_end))
            cur_text = part.text
            cur_start = part.start
            cur_end = part.end
        if i == len(spans) - 1:
            out.append(TextChunk(text=cur_text.strip(), start=cur_start, end=cur_end))
    if spans and (not out or out[-1].end != cur_end):
        out.append(TextChunk(text=cur_text.strip(), start=cur_start, end=cur_end))
    return out


def _mean_pool_last_hidden(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> np.ndarray:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    masked = last_hidden * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    pooled = summed / counts
    return pooled[0].detach().cpu().numpy().astype(np.float32, copy=False)


def encode_text(model, tokenizer, text: str, device: str) -> np.ndarray:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192)
    inputs = {k: v.to(device) for k, v in inputs.items() if k != "offset_mapping"}
    with torch.no_grad():
        out = model(**inputs)
    vec = _mean_pool_last_hidden(out.last_hidden_state, inputs["attention_mask"])
    return _l2_normalize(vec)


def encode_late_chunks_for_doc(
    model,
    tokenizer,
    doc_text: str,
    chunks: Sequence[TextChunk],
    device: str,
) -> List[np.ndarray]:
    if not chunks:
        return []
    enc = tokenizer(
        doc_text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=8192,
    )
    offsets = enc["offset_mapping"][0].detach().cpu().tolist()
    model_inputs = {
        k: v.to(device)
        for k, v in enc.items()
        if k in ("input_ids", "attention_mask", "token_type_ids")
    }
    with torch.no_grad():
        out = model(**model_inputs)
    token_emb = out.last_hidden_state[0]  # [seq, dim]

    chunk_vecs: List[np.ndarray] = []
    for ch in chunks:
        token_ids: List[int] = []
        for i, (s, e) in enumerate(offsets):
            if e <= s:
                continue
            if s < ch.end and e > ch.start:
                token_ids.append(i)
        if not token_ids:
            continue
        pooled = token_emb[token_ids].mean(dim=0).detach().cpu().numpy().astype(np.float32, copy=False)
        chunk_vecs.append(_l2_normalize(pooled))
    return chunk_vecs


def _golds(ex: QAExample) -> List[str]:
    if ex.answer_aliases:
        return list(ex.answer_aliases)
    return [ex.answer]


def answer_hit(chunk_text: str, ex: QAExample) -> bool:
    low = (chunk_text or "").lower()
    for g in _golds(ex):
        gg = (g or "").strip().lower()
        if gg and gg in low:
            return True
    return False


def recall_at_k(
    *,
    examples: Sequence[QAExample],
    query_vecs: Sequence[np.ndarray],
    chunk_vecs: Sequence[np.ndarray],
    chunk_texts: Sequence[str],
    k: int,
) -> float:
    if not chunk_vecs:
        return 0.0
    mat = np.vstack(chunk_vecs).astype(np.float32, copy=False)
    hits = 0.0
    for ex, q in zip(examples, query_vecs):
        sims = mat @ q
        top = np.argsort(-sims)[: min(k, len(sims))]
        ok = any(answer_hit(chunk_texts[int(i)], ex) for i in top)
        hits += 1.0 if ok else 0.0
    return hits / max(1, len(examples))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simple late chunking vs naive chunking retrieval experiment."
    )
    parser.add_argument("--data-path", type=Path, default=ROOT / "datasets" / "qa_dataset.jsonl")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--embedding-model",
        default="jinaai/jina-embeddings-v2-base-en",
        help="Long-context embedding model for late chunking.",
    )
    parser.add_argument("--chunk-max-chars", type=int, default=300)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-examples", type=int, default=0)
    args = parser.parse_args()

    try:
        from transformers import AutoModel, AutoTokenizer
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "transformers is required for this experiment. Install with `pip install transformers`."
        ) from e

    examples = load_qa_jsonl(args.data_path)
    if args.max_examples > 0:
        examples = examples[: args.max_examples]
    docs = flatten_contexts(examples)
    if not examples or not docs:
        raise ValueError("Dataset is empty or contains no contexts.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model: {args.embedding_model} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.embedding_model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.embedding_model, trust_remote_code=True).to(device).eval()

    naive_chunk_texts: List[str] = []
    late_chunk_texts: List[str] = []
    naive_chunk_vecs: List[np.ndarray] = []
    late_chunk_vecs: List[np.ndarray] = []

    print("Building chunk embeddings (naive and late)...")
    for d in docs:
        chunks = split_with_spans(d, max_chars=args.chunk_max_chars)
        if not chunks:
            continue
        for ch in chunks:
            naive_chunk_texts.append(ch.text)
            naive_chunk_vecs.append(encode_text(model, tokenizer, ch.text, device))
            late_chunk_texts.append(ch.text)
        late_chunk_vecs.extend(encode_late_chunks_for_doc(model, tokenizer, d, chunks, device))

    if len(late_chunk_vecs) != len(late_chunk_texts):
        n = min(len(late_chunk_vecs), len(late_chunk_texts))
        late_chunk_vecs = late_chunk_vecs[:n]
        late_chunk_texts = late_chunk_texts[:n]

    print("Encoding queries...")
    query_vecs = [encode_text(model, tokenizer, ex.question, device) for ex in examples]

    r_naive = recall_at_k(
        examples=examples,
        query_vecs=query_vecs,
        chunk_vecs=naive_chunk_vecs,
        chunk_texts=naive_chunk_texts,
        k=args.top_k,
    )
    r_late = recall_at_k(
        examples=examples,
        query_vecs=query_vecs,
        chunk_vecs=late_chunk_vecs,
        chunk_texts=late_chunk_texts,
        k=args.top_k,
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "method": "naive_chunking",
            "model": args.embedding_model,
            "chunk_max_chars": args.chunk_max_chars,
            "top_k": args.top_k,
            "n_examples": len(examples),
            "n_chunks": len(naive_chunk_vecs),
            f"recall@{args.top_k}": r_naive,
        },
        {
            "method": "late_chunking",
            "model": args.embedding_model,
            "chunk_max_chars": args.chunk_max_chars,
            "top_k": args.top_k,
            "n_examples": len(examples),
            "n_chunks": len(late_chunk_vecs),
            f"recall@{args.top_k}": r_late,
        },
    ]
    df = pd.DataFrame(rows)
    out_path = out_dir / "late_chunking_results.csv"
    df.to_csv(out_path, index=False)

    print("\n=== Late Chunking Experiment ===")
    print(df.to_string(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
