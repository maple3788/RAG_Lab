from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.exp_viina_rag_vs_kg import (
    HybridRetriever,
    Neo4jKG,
    build_graph,
    kg_answer,
    load_control_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask a single query on VIINA control data: hybrid vs KG."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("datasets/control_latest_2022.csv"),
    )
    parser.add_argument("--max-rows", type=int, default=200000)
    parser.add_argument("--query", required=True, help="Natural language query.")
    parser.add_argument(
        "--kg-backend", choices=["memory", "neo4j"], default="neo4j"
    )
    parser.add_argument(
        "--neo4j-uri", default=os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
    )
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD", ""))
    parser.add_argument("--neo4j-database", default=os.getenv("NEO4J_DATABASE", "neo4j"))
    args = parser.parse_args()

    df = load_control_csv(args.data_path, args.max_rows)
    hybrid = HybridRetriever(df)

    neo4j_kg: Neo4jKG | None = None
    try:
        if args.kg_backend == "neo4j":
            if not args.neo4j_password:
                raise ValueError(
                    "Neo4j password is required. Set --neo4j-password or NEO4J_PASSWORD."
                )
            neo4j_kg = Neo4jKG(
                uri=args.neo4j_uri,
                user=args.neo4j_user,
                password=args.neo4j_password,
                database=args.neo4j_database,
            )
            neo4j_kg.build_from_df(df)
            kg_pred = neo4j_kg.answer(args.query)
        else:
            graph = build_graph(df)
            kg_pred = kg_answer(args.query, graph)

        out = {
            "query": args.query,
            "hybrid_answer": hybrid.answer(args.query),
            "kg_answer": kg_pred,
            "dataset_rows_used": len(df),
            "kg_backend": args.kg_backend,
        }
        print(json.dumps(out, indent=2, ensure_ascii=True))
    finally:
        if neo4j_kg is not None:
            neo4j_kg.close()


if __name__ == "__main__":
    main()
