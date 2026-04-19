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


def main() -> None:
    root = ROOT
    data_path = root / "datasets" / "qa_dataset.jsonl"
    out_dir = root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = load_qa_jsonl(data_path)
    top_k = 3
    chunk_overlap = 64
    embed_model = "BAAI/bge-base-en-v1.5"
    embedder = load_embedding_model(embed_model, normalize=True)

    chunk_sizes = [256, 512, 1024]

    rows = []
    for cs in chunk_sizes:
        corpus_chunks = build_retrieval_corpus(
            examples, chunk_size=cs, chunk_overlap=min(chunk_overlap, cs // 4)
        )
        metrics = evaluate_retrieval(
            examples,
            embedder=embedder,
            corpus_chunks=corpus_chunks,
            top_k=top_k,
            reranker=None,
        )
        rows.append(
            {
                "chunk_size": cs,
                "model": embed_model,
                f"recall@{top_k}": metrics[f"recall@{top_k}"],
                "n_chunks": int(metrics["n_chunks"]),
            }
        )
        print(cs, metrics)

    df = pd.DataFrame(rows).sort_values(by="chunk_size", ascending=True)
    csv_path = out_dir / "chunk_size_results.csv"
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(6, 4))
    plt.plot(df["chunk_size"], df[f"recall@{top_k}"], marker="o")
    plt.xlabel("chunk_size")
    plt.ylabel(f"recall@{top_k}")
    plt.title("Chunk size impact")
    plt.tight_layout()
    plot_path = out_dir / "chunk_size_plot.png"
    plt.savefig(plot_path, dpi=200)

    print(f"Saved: {csv_path}")
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
