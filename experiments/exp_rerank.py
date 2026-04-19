from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.embedder import load_embedding_model
from src.loader import load_qa_jsonl
from src.rag_pipeline import build_retrieval_corpus, evaluate_retrieval
from src.reranker import load_reranker


def main() -> None:
    root = ROOT
    data_path = root / "datasets" / "qa_dataset.jsonl"
    out_dir = root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = load_qa_jsonl(data_path)

    embed_model = "BAAI/bge-base-en-v1.5"
    rerank_model = "BAAI/bge-reranker-base"

    top_k = 10  # retrieve more, rerank down to 3
    rerank_top_k = 3
    chunk_size = 512
    chunk_overlap = 64

    corpus_chunks = build_retrieval_corpus(
        examples, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    embedder = load_embedding_model(embed_model, normalize=True)

    rows = []

    no_rerank = evaluate_retrieval(
        examples,
        embedder=embedder,
        corpus_chunks=corpus_chunks,
        top_k=rerank_top_k,
        reranker=None,
    )
    rows.append(
        {
            "setting": "no_rerank",
            f"recall@{rerank_top_k}": no_rerank[f"recall@{rerank_top_k}"],
        }
    )
    print("no_rerank", no_rerank)

    reranker = load_reranker(rerank_model)
    with_rerank = evaluate_retrieval(
        examples,
        embedder=embedder,
        corpus_chunks=corpus_chunks,
        top_k=top_k,
        reranker=reranker,
        rerank_top_k=rerank_top_k,
    )
    rows.append(
        {
            "setting": "bge_reranker",
            f"recall@{rerank_top_k}": with_rerank[f"recall@{rerank_top_k}"],
        }
    )
    print("with_rerank", with_rerank)

    df = pd.DataFrame(rows)
    csv_path = out_dir / "rerank_results.csv"
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(6, 4))
    plt.bar(df["setting"], df[f"recall@{rerank_top_k}"])
    plt.ylabel(f"recall@{rerank_top_k}")
    plt.title("Reranker impact")
    plt.tight_layout()
    plot_path = out_dir / "rerank_plot.png"
    plt.savefig(plot_path, dpi=200)

    print(f"Saved: {csv_path}")
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
