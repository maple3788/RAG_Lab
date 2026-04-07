from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from dotenv import load_dotenv


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


@dataclass(frozen=True)
class SemanticCacheEntry:
    question: str
    answer: str
    embedding: list[float]


class RedisSemanticCache:
    """
    Redis-backed semantic cache.
    Stores recent entries in a Redis list and does cosine similarity scan.
    """

    def __init__(self, namespace: str, redis_url: Optional[str] = None):
        load_dotenv()
        import redis

        url = redis_url or os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        self._r = redis.from_url(url, decode_responses=True)
        self._key = f"rag:semantic-cache:{namespace}"

    def lookup(
        self,
        query_embedding: np.ndarray,
        *,
        threshold: float,
        max_scan: int = 200,
    ) -> Optional[str]:
        rows = self._r.lrange(self._key, 0, max(0, int(max_scan) - 1))
        best_score = -1.0
        best_answer: Optional[str] = None
        for raw in rows:
            try:
                d = json.loads(raw)
                emb = np.asarray(d.get("embedding", []), dtype=np.float32)
                if emb.size == 0:
                    continue
                sim = _cosine_similarity(query_embedding, emb)
                if sim > best_score:
                    best_score = sim
                    best_answer = str(d.get("answer", ""))
            except Exception:
                continue
        if best_answer is not None and best_score >= float(threshold):
            return best_answer
        return None

    def write(
        self,
        *,
        question: str,
        answer: str,
        query_embedding: np.ndarray,
        max_entries: int = 512,
        ttl_seconds: int = 86400,
    ) -> None:
        entry = SemanticCacheEntry(
            question=question,
            answer=answer,
            embedding=query_embedding.astype(np.float32, copy=False).tolist(),
        )
        payload = json.dumps(entry.__dict__, ensure_ascii=False)
        self._r.lpush(self._key, payload)
        self._r.ltrim(self._key, 0, max(0, int(max_entries) - 1))
        self._r.expire(self._key, int(ttl_seconds))
