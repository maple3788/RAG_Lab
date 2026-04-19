#!/usr/bin/env python3
"""
Build README figures from ``results/qasper_rag_generation_results.csv``.

Usage::

    python scripts/generate_charts.py
    python scripts/generate_charts.py --csv results/qasper_rag_generation_results.csv --out-dir assets
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Professional, colorblind-friendly palette
COL_F1 = "#0173B2"
COL_GH = "#DE8F05"


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 150,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.axisbelow": True,
        }
    )


def chart_rerank(df: pd.DataFrame, out: Path) -> None:
    sub = df[df["experiment"] == "rerank"].set_index("setting")
    metrics = ["token_f1", "gold_hit"]
    labels = ["No rerank", "With rerank"]
    x = range(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.2))
    f1_vals = [sub.loc["no_rerank", "token_f1"], sub.loc["with_rerank", "token_f1"]]
    gh_vals = [sub.loc["no_rerank", "gold_hit"], sub.loc["with_rerank", "gold_hit"]]
    ax.bar([i - width / 2 for i in x], f1_vals, width, label="Token F1", color=COL_F1)
    ax.bar([i + width / 2 for i in x], gh_vals, width, label="Gold hit", color=COL_GH)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0, max(max(f1_vals), max(gh_vals)) * 1.15)
    ax.set_title("QASPER: rerank impact (n=200)")
    ax.legend(loc="upper left", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def chart_topk(df: pd.DataFrame, out: Path) -> None:
    sub = df[df["experiment"] == "topk"].copy()
    sub["k"] = sub["setting"].str.replace("final_k=", "").astype(int)
    sub = sub.sort_values("k")
    ks = sub["k"].tolist()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(
        ks,
        sub["token_f1"],
        "o-",
        color=COL_F1,
        linewidth=2,
        markersize=8,
        label="Token F1",
    )
    ax.plot(
        ks,
        sub["gold_hit"],
        "s-",
        color=COL_GH,
        linewidth=2,
        markersize=8,
        label="Gold hit",
    )
    ax.set_xticks(ks)
    ax.set_xlabel("final_k (passages to LLM)")
    ax.set_ylabel("Score")
    ax.set_title("QASPER: top-k scaling (n=200)")
    ax.legend(loc="lower right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def chart_truncation(df: pd.DataFrame, out: Path) -> None:
    sub = df[df["experiment"] == "truncation"].set_index("setting")
    order = ["head", "tail", "middle"]
    labels = [s.capitalize() for s in order]
    vals = [sub.loc[s, "gold_hit"] for s in order]
    colors = ["#0072B2", "#D55E00", "#CC79A7"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_ylabel("Gold hit")
    ax.set_ylim(0, max(vals) * 1.2)
    ax.set_title("QASPER: truncation vs gold hit (tight char budget, n=200)")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate QASPER README charts")
    parser.add_argument(
        "--csv", type=Path, default=Path("results/qasper_rag_generation_results.csv")
    )
    parser.add_argument("--out-dir", type=Path, default=Path("assets"))
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _setup_style()

    chart_rerank(df, args.out_dir / "qasper_rerank.png")
    chart_topk(df, args.out_dir / "qasper_topk.png")
    chart_truncation(df, args.out_dir / "qasper_truncation.png")
    print(f"Wrote charts to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
