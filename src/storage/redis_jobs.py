from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Literal, Optional

from dotenv import load_dotenv

JobStage = Literal[
    "queued",
    "extracting",
    "chunking",
    "embedding",
    "storing",
    "completed",
    "failed",
]


@dataclass
class JobStatus:
    job_id: str
    stage: str  # use JobStage values when possible
    message: str = ""
    error: Optional[str] = None
    updated_at: float = 0.0
    meta: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.updated_at == 0.0:
            self.updated_at = time.time()


class RedisJobStore:
    """Redis-backed status for ingest jobs (key: ``ingest:job:{id}``)."""

    def __init__(self, redis_url: Optional[str] = None):
        load_dotenv()
        import redis

        url = redis_url or os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        self._r = redis.from_url(url, decode_responses=True)

    def _key(self, job_id: str) -> str:
        return f"ingest:job:{job_id}"

    def set_status(self, status: JobStatus, *, ttl_seconds: int = 86400 * 7) -> None:
        payload = json.dumps(asdict(status), ensure_ascii=False)
        self._r.set(self._key(status.job_id), payload, ex=ttl_seconds)

    def get_status(self, job_id: str) -> Optional[JobStatus]:
        raw = self._r.get(self._key(job_id))
        if not raw:
            return None
        d = json.loads(raw)
        return JobStatus(
            job_id=d["job_id"],
            stage=d["stage"],
            message=d.get("message", ""),
            error=d.get("error"),
            updated_at=float(d.get("updated_at", 0)),
            meta=d.get("meta"),
        )
