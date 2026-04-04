from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from typing import Any, List, Optional

from dotenv import load_dotenv


@dataclass(frozen=True)
class MinioSettings:
    endpoint: str
    access_key: str
    secret_key: str
    bucket: str
    secure: bool
    #: Verify TLS certificates (set false for local https + self-signed, e.g. AiStor on localhost)
    cert_check: bool


def load_minio_settings() -> MinioSettings:
    load_dotenv()
    return MinioSettings(
        endpoint=os.environ.get("MINIO_ENDPOINT", "localhost:9000"),
        access_key=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
        bucket=os.environ.get("MINIO_BUCKET", "rag-lab-ingest"),
        secure=os.environ.get("MINIO_USE_SSL", "false").lower() in ("1", "true", "yes"),
        cert_check=os.environ.get("MINIO_CERT_CHECK", "true").lower()
        in ("1", "true", "yes"),
    )


def _build_minio_client(settings: MinioSettings):
    import urllib3
    from minio import Minio

    kwargs: dict = {
        "endpoint": settings.endpoint,
        "access_key": settings.access_key,
        "secret_key": settings.secret_key,
        "secure": settings.secure,
    }
    if not settings.cert_check:
        http_client = urllib3.PoolManager(cert_reqs="CERT_NONE")
        kwargs["http_client"] = http_client
    return Minio(**kwargs)


class MinioArtifactStore:
    """
    S3-compatible object storage for ingest artifacts (chunks JSON, serialized FAISS, summaries).

    MinIO is **object storage**, not a vector database: we store **files** (index bytes, JSON)
    and load them into FAISS in memory at query time.
    """

    def __init__(self, settings: Optional[MinioSettings] = None):
        self._settings = settings or load_minio_settings()
        self._client = _build_minio_client(self._settings)
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        name = self._settings.bucket
        if not self._client.bucket_exists(name):
            self._client.make_bucket(name)

    @property
    def bucket_name(self) -> str:
        return self._settings.bucket

    def put_json(self, object_name: str, obj: Any, content_type: str = "application/json") -> None:
        raw = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
        self.put_bytes(object_name, raw, content_type=content_type)

    def put_bytes(
        self,
        object_name: str,
        data: bytes,
        *,
        content_type: str = "application/octet-stream",
    ) -> None:
        self._client.put_object(
            self._settings.bucket,
            object_name,
            io.BytesIO(data),
            length=len(data),
            content_type=content_type,
        )

    def get_json(self, object_name: str) -> Any:
        return json.loads(self.get_bytes(object_name).decode("utf-8"))

    def get_bytes(self, object_name: str) -> bytes:
        r = self._client.get_object(self._settings.bucket, object_name)
        try:
            return r.read()
        finally:
            r.close()
            r.release_conn()

    def list_job_ids(self) -> List[str]:
        """Top-level prefixes in the bucket (each ingest job is ``{job_id}/…``)."""
        seen: set[str] = set()
        try:
            for obj in self._client.list_objects(self._settings.bucket, recursive=True):
                name = getattr(obj, "object_name", None) or ""
                if "/" in name:
                    seen.add(name.split("/", 1)[0])
        except Exception:
            pass
        return sorted(seen)

    def get_job_metadata(self, job_id: str) -> Optional[dict[str, Any]]:
        try:
            return self.get_json(f"{job_id}/metadata.json")
        except Exception:
            return None

    def list_ingest_jobs_table(self) -> List[dict[str, Any]]:
        """Rows for UI tables: job_id plus fields from ``metadata.json`` when present."""
        rows: List[dict[str, Any]] = []
        for jid in self.list_job_ids():
            meta = self.get_job_metadata(jid) or {}
            rows.append(
                {
                    "job_id": jid,
                    "filename": meta.get("filename", "—"),
                    "n_chunks": meta.get("n_chunks", "—"),
                    "embedding_model": meta.get("embedding_model", "—"),
                    "summarization": meta.get("summarization", "—"),
                    "extraction": meta.get("extraction", "—"),
                }
            )
        return rows
