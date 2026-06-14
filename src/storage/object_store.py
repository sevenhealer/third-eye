from __future__ import annotations

from datetime import UTC, datetime

from src.core.config import get_settings
from src.core.logging import get_logger

logger = get_logger(__name__)

try:
    import boto3
    from botocore.client import Config as _BotoConfig
    from botocore.exceptions import BotoCoreError, ClientError
    _BOTO_AVAILABLE = True
except ImportError:
    boto3 = None  # type: ignore[assignment]
    _BotoConfig = None  # type: ignore[assignment]
    BotoCoreError = ClientError = Exception  # type: ignore[assignment,misc]
    _BOTO_AVAILABLE = False


class StorageError(Exception):
    pass


def object_key(kind: str, camera_id: str, ext: str, when: datetime | None = None) -> str:
    """Date-partitioned key, e.g. snapshots/2026/06/14/cam0/1718360000000.jpg —
    partitioning keeps prefixes listable and lifecycle/retention simple.
    `kind` is a top-level folder: snapshots | clips | crops | evidence."""
    ts = when or datetime.now(UTC)
    stamp = int(ts.timestamp() * 1000)
    return f"{kind}/{ts:%Y/%m/%d}/{camera_id}/{stamp}.{ext.lstrip('.')}"


class ObjectStore:
    """
    S3-compatible object storage (MinIO self-hosted, or any S3) for surveillance
    artifacts: alert snapshots, event clips, face/person crops, evidence.

    Same code path works against MinIO and AWS S3 — only the endpoint differs,
    which keeps the system self-hostable AND cloud-portable. Disabled (no-op
    guard via is_enabled) when `s3_endpoint_url` is empty, so the pipeline runs
    without object storage in MVP/dev.
    """

    def __init__(
        self,
        endpoint_url: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        region: str = "us-east-1",
    ) -> None:
        self.bucket = bucket
        self._endpoint = endpoint_url
        self._client = None
        if not endpoint_url:
            return
        if not _BOTO_AVAILABLE:
            raise StorageError("boto3 not installed — run: pip install boto3")
        # path-style addressing is required for MinIO (no virtual-host buckets)
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
            config=_BotoConfig(signature_version="s3v4", s3={"addressing_style": "path"}),
        )

    @property
    def is_enabled(self) -> bool:
        return self._client is not None

    def ensure_bucket(self) -> None:
        """Create the bucket if missing (idempotent). Call once at startup."""
        if not self.is_enabled:
            return
        try:
            self._client.head_bucket(Bucket=self.bucket)
        except (ClientError, BotoCoreError):
            try:
                self._client.create_bucket(Bucket=self.bucket)
                logger.info("object_store_bucket_created", bucket=self.bucket)
            except (ClientError, BotoCoreError) as exc:
                raise StorageError(f"could not create bucket {self.bucket}: {exc}") from exc

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        """Store raw bytes; returns the key. No-op-safe: raises if disabled so
        callers can choose to guard on is_enabled."""
        if not self.is_enabled:
            raise StorageError("object storage disabled (empty s3_endpoint_url)")
        try:
            self._client.put_object(
                Bucket=self.bucket, Key=key, Body=data, ContentType=content_type
            )
        except (ClientError, BotoCoreError) as exc:
            raise StorageError(f"put_object failed for {key}: {exc}") from exc
        return key

    def put_file(self, key: str, path: str, content_type: str | None = None) -> str:
        if not self.is_enabled:
            raise StorageError("object storage disabled (empty s3_endpoint_url)")
        extra = {"ContentType": content_type} if content_type else None
        try:
            self._client.upload_file(path, self.bucket, key, ExtraArgs=extra)
        except (ClientError, BotoCoreError, OSError) as exc:
            raise StorageError(f"upload_file failed for {key}: {exc}") from exc
        return key

    def get_bytes(self, key: str) -> bytes:
        if not self.is_enabled:
            raise StorageError("object storage disabled (empty s3_endpoint_url)")
        try:
            return self._client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except (ClientError, BotoCoreError) as exc:
            raise StorageError(f"get_object failed for {key}: {exc}") from exc

    def presigned_url(self, key: str, expires_seconds: int = 3600) -> str:
        """Time-limited URL to view/download an artifact (e.g. in an alert)."""
        if not self.is_enabled:
            raise StorageError("object storage disabled (empty s3_endpoint_url)")
        try:
            return self._client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=expires_seconds,
            )
        except (ClientError, BotoCoreError) as exc:
            raise StorageError(f"presign failed for {key}: {exc}") from exc


_store: ObjectStore | None = None


def get_object_store() -> ObjectStore:
    """Process-wide singleton built from settings."""
    global _store
    if _store is None:
        s = get_settings()
        _store = ObjectStore(
            endpoint_url=s.s3_endpoint_url,
            access_key=s.s3_access_key,
            secret_key=s.s3_secret_key,
            bucket=s.s3_bucket,
            region=s.s3_region,
        )
    return _store
