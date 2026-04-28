from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import pandas as pd
from rank_bm25 import BM25Okapi
try:
    from neo4j import GraphDatabase
except ImportError:  # pragma: no cover - runtime optional dependency
    GraphDatabase = None


@dataclass(frozen=True)
class QAExample:
    question: str
    answer: str
    kind: str
    geonameid: int
    date: int
    date_2: int | None = None


def normalize_tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-zA-Z0-9_]+", text.lower()) if t]


@dataclass(frozen=True)
class ControlRow:
    geonameid: int
    date: int
    status: str
    text: str


class HybridRetriever:
    """
    Lightweight hybrid retriever:
    - Sparse signal: BM25 over row text.
    - Structured signal: exact geoname/date boosts.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.rows: List[ControlRow] = []
        tokenized: List[List[str]] = []
        for row in df.itertuples(index=False):
            text = (
                f"geonameid={int(row.geonameid)} date={int(row.date)} "
                f"status={row.status} status_wiki={row.status_wiki} "
                f"status_boost={row.status_boost} status_dsm={row.status_dsm} "
                f"status_isw={row.status_isw}"
            )
            self.rows.append(
                ControlRow(
                    geonameid=int(row.geonameid),
                    date=int(row.date),
                    status=str(row.status),
                    text=text,
                )
            )
            toks = normalize_tokens(text)
            tokenized.append(toks if toks else ["empty"])
        self.bm25 = BM25Okapi(tokenized)

    def _extract_constraints(self, question: str) -> Tuple[int | None, List[int]]:
        q = question.lower()
        geo_match = re.search(r"geonameid\s+(\d+)", q)
        date_matches = [int(d) for d in re.findall(r"date\s+(\d{8})", q)]
        geonameid = int(geo_match.group(1)) if geo_match else None
        return geonameid, date_matches

    def _best_row(self, question: str) -> ControlRow | None:
        if not self.rows:
            return None
        query_tokens = normalize_tokens(question)
        if not query_tokens:
            query_tokens = ["query"]
        bm25_scores = self.bm25.get_scores(query_tokens)
        geonameid, dates = self._extract_constraints(question)
        dates_set = set(dates)

        best_idx = -1
        best_score = float("-inf")
        for i, row in enumerate(self.rows):
            s = float(bm25_scores[i])
            if geonameid is not None and row.geonameid == geonameid:
                s += 8.0
            if dates_set and row.date in dates_set:
                s += 8.0
            if s > best_score:
                best_score = s
                best_idx = i
        if best_idx < 0:
            return None
        return self.rows[best_idx]

    def answer(self, question: str) -> str:
        q_lower = question.lower()
        if "change status between date" in q_lower:
            geo_match = re.search(r"geonameid\s+(\d+)", q_lower)
            date_matches = re.findall(r"date\s+(\d{8})", q_lower)
            if not geo_match or len(date_matches) < 2:
                return "UNKNOWN"
            geonameid = int(geo_match.group(1))
            d1 = int(date_matches[0])
            d2 = int(date_matches[1])
            row1 = self._best_row(f"geonameid {geonameid} date {d1}")
            row2 = self._best_row(f"geonameid {geonameid} date {d2}")
            if row1 is None or row2 is None:
                return "UNKNOWN"
            return "YES" if row1.status != row2.status else "NO"

        row = self._best_row(question)
        if row is None:
            return "UNKNOWN"
        return row.status


def build_graph(df: pd.DataFrame) -> Dict[Tuple[int, int], str]:
    graph: Dict[Tuple[int, int], str] = {}
    for row in df.itertuples(index=False):
        graph[(int(row.geonameid), int(row.date))] = str(row.status)
    return graph


class Neo4jKG:
    def __init__(
        self, uri: str, user: str, password: str, database: str = "neo4j"
    ) -> None:
        if GraphDatabase is None:
            raise RuntimeError(
                "neo4j driver is not installed. Install it with: pip install neo4j"
            )
        self.database = database
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.driver.verify_connectivity()

    def close(self) -> None:
        self.driver.close()

    def build_from_df(self, df: pd.DataFrame, batch_size: int = 5000) -> None:
        rows = (
            df[["geonameid", "date", "status"]]
            .drop_duplicates(subset=["geonameid", "date"])
            .to_dict("records")
        )
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
            for i in range(0, len(rows), batch_size):
                chunk = rows[i : i + batch_size]
                session.run(
                    """
                    UNWIND $rows AS row
                    MERGE (l:Location {geonameid: toInteger(row.geonameid)})
                    MERGE (s:Snapshot {
                        geonameid: toInteger(row.geonameid),
                        date: toInteger(row.date)
                    })
                    SET s.status = row.status
                    MERGE (l)-[:HAS_STATUS]->(s)
                    """,
                    rows=chunk,
                )

    def get_status(self, geonameid: int, date: int) -> str:
        with self.driver.session(database=self.database) as session:
            record = session.run(
                """
                MATCH (s:Snapshot {geonameid: $geonameid, date: $date})
                RETURN s.status AS status
                LIMIT 1
                """,
                geonameid=geonameid,
                date=date,
            ).single()
            if not record:
                return "UNKNOWN"
            return str(record["status"])

    def answer(self, question: str) -> str:
        if "change status between date" in question.lower():
            geo_match = re.search(r"geonameid\s+(\d+)", question.lower())
            date_matches = re.findall(r"date\s+(\d{8})", question.lower())
            if not geo_match or len(date_matches) < 2:
                return "UNKNOWN"
            geonameid = int(geo_match.group(1))
            d1 = int(date_matches[0])
            d2 = int(date_matches[1])
            s1 = self.get_status(geonameid, d1)
            s2 = self.get_status(geonameid, d2)
            if s1 == "UNKNOWN" or s2 == "UNKNOWN":
                return "UNKNOWN"
            return "YES" if s1 != s2 else "NO"

        geo_match = re.search(r"geonameid\s+(\d+)", question.lower())
        date_match = re.search(r"date\s+(\d{8})", question.lower())
        if not geo_match or not date_match:
            return "UNKNOWN"
        return self.get_status(int(geo_match.group(1)), int(date_match.group(1)))


def kg_answer(question: str, graph: Dict[Tuple[int, int], str]) -> str:
    if "change status between date" in question.lower():
        geo_match = re.search(r"geonameid\s+(\d+)", question.lower())
        date_matches = re.findall(r"date\s+(\d{8})", question.lower())
        if not geo_match or len(date_matches) < 2:
            return "UNKNOWN"
        geonameid = int(geo_match.group(1))
        d1 = int(date_matches[0])
        d2 = int(date_matches[1])
        s1 = graph.get((geonameid, d1))
        s2 = graph.get((geonameid, d2))
        if s1 is None or s2 is None:
            return "UNKNOWN"
        return "YES" if s1 != s2 else "NO"

    geo_match = re.search(r"geonameid\s+(\d+)", question.lower())
    date_match = re.search(r"date\s+(\d{8})", question.lower())
    if not geo_match or not date_match:
        return "UNKNOWN"
    key = (int(geo_match.group(1)), int(date_match.group(1)))
    return graph.get(key, "UNKNOWN")


def make_questions(df: pd.DataFrame, n_questions: int, seed: int) -> List[QAExample]:
    random.seed(seed)
    lookup_n = max(n_questions // 2, 1)
    transition_n = max(n_questions - lookup_n, 1)
    unique_rows = df.drop_duplicates(subset=["geonameid", "date", "status"])
    if len(unique_rows) < lookup_n:
        lookup_n = len(unique_rows)
    sampled = unique_rows.sample(n=lookup_n, random_state=seed)
    out: List[QAExample] = []
    for row in sampled.itertuples(index=False):
        q = (
            f"What is the territorial control status for geonameid {int(row.geonameid)} "
            f"on date {int(row.date)}?"
        )
        out.append(
            QAExample(
                question=q,
                answer=str(row.status),
                kind="lookup_status",
                geonameid=int(row.geonameid),
                date=int(row.date),
            )
        )

    grouped = df.sort_values(["geonameid", "date"]).groupby("geonameid")
    transition_candidates: List[Tuple[int, int, int, str]] = []
    for geonameid, gdf in grouped:
        rows = list(gdf[["date", "status"]].itertuples(index=False, name=None))
        if len(rows) < 2:
            continue
        for i in range(len(rows) - 1):
            d1, s1 = rows[i]
            d2, s2 = rows[i + 1]
            transition_candidates.append(
                (int(geonameid), int(d1), int(d2), "YES" if s1 != s2 else "NO")
            )

    random.shuffle(transition_candidates)
    for geonameid, d1, d2, ans in transition_candidates[:transition_n]:
        q = (
            f"Did geonameid {geonameid} change status between date {d1} "
            f"and date {d2}? Answer YES or NO."
        )
        out.append(
            QAExample(
                question=q,
                answer=ans,
                kind="status_change",
                geonameid=geonameid,
                date=d1,
                date_2=d2,
            )
        )
    return out


def evaluate(
    examples: Sequence[QAExample],
    hybrid_predictor: Callable[[str], str],
    kg_predictor: Callable[[str], str],
) -> Dict[str, object]:
    hybrid_correct = 0
    kg_correct = 0
    per_example: List[Dict[str, object]] = []

    for ex in examples:
        hybrid_pred = hybrid_predictor(ex.question)
        kg_pred = kg_predictor(ex.question)
        hybrid_ok = hybrid_pred == ex.answer
        kg_ok = kg_pred == ex.answer
        hybrid_correct += int(hybrid_ok)
        kg_correct += int(kg_ok)
        per_example.append(
            {
                "question": ex.question,
                "answer": ex.answer,
                "hybrid_prediction": hybrid_pred,
                "kg_prediction": kg_pred,
                "hybrid_correct": hybrid_ok,
                "kg_correct": kg_ok,
                "geonameid": ex.geonameid,
                "date": ex.date,
            }
        )

    n = max(len(examples), 1)
    return {
        "n_questions": len(examples),
        "hybrid_accuracy": hybrid_correct / n,
        "kg_accuracy": kg_correct / n,
        "accuracy_lift": (kg_correct - hybrid_correct) / n,
        "per_example": per_example,
    }


def load_control_csv(path: Path, max_rows: int | None) -> pd.DataFrame:
    usecols = [
        "geonameid",
        "date",
        "status_wiki",
        "status_boost",
        "status_dsm",
        "status_isw",
        "status",
    ]
    df = pd.read_csv(path, usecols=usecols, nrows=max_rows)
    df["status"] = df["status"].astype(str)
    df = df[df["status"].isin(["UA", "RU", "CONTESTED"])].copy()
    df["geonameid"] = pd.to_numeric(df["geonameid"], errors="coerce")
    df["date"] = pd.to_numeric(df["date"], errors="coerce")
    df = df.dropna(subset=["geonameid", "date", "status"])
    df["geonameid"] = df["geonameid"].astype(int)
    df["date"] = df["date"].astype(int)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare hybrid retrieval vs KG on VIINA control data."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("datasets/control_latest_2022.csv"),
    )
    parser.add_argument("--max-rows", type=int, default=200000)
    parser.add_argument("--n-questions", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
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

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_control_csv(args.data_path, args.max_rows)
    examples = make_questions(df, n_questions=args.n_questions, seed=args.seed)
    hybrid_retriever = HybridRetriever(df)
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
            kg_predictor = neo4j_kg.answer
        else:
            graph = build_graph(df)
            kg_predictor = lambda q: kg_answer(q, graph)

        result = evaluate(examples, hybrid_retriever.answer, kg_predictor)
    finally:
        if neo4j_kg is not None:
            neo4j_kg.close()

    metrics_path = args.out_dir / "viina_rag_vs_kg_metrics.json"
    per_example_path = args.out_dir / "viina_rag_vs_kg_per_example.csv"

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {k: v for k, v in result.items() if k != "per_example"},
            f,
            indent=2,
            ensure_ascii=True,
        )

    pd.DataFrame(result["per_example"]).to_csv(per_example_path, index=False)

    print("Experiment complete")
    print(json.dumps({k: v for k, v in result.items() if k != "per_example"}, indent=2))
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved per-example predictions: {per_example_path}")


if __name__ == "__main__":
    main()
