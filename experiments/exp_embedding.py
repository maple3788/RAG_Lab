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
    chunk_size = 512
    chunk_overlap = 64

    models = [
        "BAAI/bge-small-en-v1.5",
        "BAAI/bge-base-en-v1.5",
        "intfloat/e5-small-v2",
    ]

    corpus_chunks = build_retrieval_corpus(
        examples, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    rows = []
    for model_name in models:
        embedder = load_embedding_model(model_name, normalize=True)
        metrics = evaluate_retrieval(
            examples,
            embedder=embedder,
            corpus_chunks=corpus_chunks,
            top_k=top_k,
            reranker=None,
        )
        rows.append(
            {
                "model": model_name,
                f"recall@{top_k}": metrics[f"recall@{top_k}"],
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            }
        )
        print(model_name, metrics)

    df = pd.DataFrame(rows).sort_values(by=f"recall@{top_k}", ascending=False)
    csv_path = out_dir / "embedding_results.csv"
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(8, 4))
    plt.bar(df["model"], df[f"recall@{top_k}"])
    plt.ylabel(f"recall@{top_k}")
    plt.title("Embedding model comparison")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plot_path = out_dir / "embedding_plot.png"
    plt.savefig(plot_path, dpi=200)

    print(f"Saved: {csv_path}")
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
