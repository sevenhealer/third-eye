from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ───────────────────────────────────────────────────────────
    app_env: Literal["development", "production"] = "development"
    app_secret_key: str = Field(min_length=32)
    app_log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # ── JWT ───────────────────────────────────────────────────────────────────
    jwt_secret_key: str = Field(min_length=32)
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 15
    jwt_refresh_token_expire_days: int = 7

    # ── Databases ─────────────────────────────────────────────────────────────
    database_url: str
    db_echo: bool = False   # SQLAlchemy SQL logging — very noisy; opt-in only
    timescale_url: str
    redis_url: str = "redis://redis:6379/0"
    redis_password: str = ""
    neo4j_uri: str = "bolt://neo4j:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str

    # ── Kafka ─────────────────────────────────────────────────────────────────
    kafka_bootstrap_servers: str = "kafka:9092"
    kafka_consumer_group: str = "third-eye-main"

    # ── Object storage (MinIO / S3-compatible) — clips, snapshots, evidence ────
    s3_endpoint_url: str = "http://minio:9000"   # empty disables object storage
    s3_access_key: str = "thirdeye"
    s3_secret_key: str = ""
    s3_bucket: str = "thirdeye-evidence"
    s3_region: str = "us-east-1"

    # ── Models ────────────────────────────────────────────────────────────────
    models_dir: Path = Path("/app/models")
    use_tensorrt: bool = False
    device: str = "cuda:0"

    # ── Ollama ────────────────────────────────────────────────────────────────
    ollama_host: str = "http://ollama:11434"
    ollama_llm_model: str = "mistral:7b-instruct-v0.3-q4_K_M"
    ollama_vlm_model: str = "llava:7b-v1.5-q4_K_M"

    # ── Camera defaults ───────────────────────────────────────────────────────
    camera_reconnect_interval_max: int = 60
    camera_frame_buffer_size: int = 32

    # ── Face recognition ──────────────────────────────────────────────────────
    face_match_accept_threshold: float = 0.45
    face_match_reject_threshold: float = 0.35
    face_quality_min: float = 0.5
    enrollment_min_crops: int = 10
    enrollment_window_seconds: int = 30

    # ── Anti-spoofing ─────────────────────────────────────────────────────────
    antispoofing_confidence_min: float = 0.85
    antispoofing_temporal_frames: int = 10

    # ── Alerts ────────────────────────────────────────────────────────────────
    alert_webhook_url: str = ""
    alert_email_smtp_host: str = ""
    alert_email_smtp_port: int = 587
    alert_email_from: str = ""
    alert_email_to: str = ""
    alert_email_password: str = ""

    # ── ReID / Cross-camera tracking ─────────────────────────────────────────
    reid_match_threshold: float = 0.70    # cosine similarity threshold for cross-camera identity linkage
    reid_iou_weight: float = 0.5          # weight of IoU vs. appearance in StrongSORT cost matrix

    # ── Retention ─────────────────────────────────────────────────────────────
    event_retention_days: int = 90
    audit_log_retention_days: int = 365
    video_archive_retention_days: int = 30

    # ── Model registry ────────────────────────────────────────────────────────
    # The manifest path is resolved repo-relatively in src.core.model_registry
    # (DEFAULT_MANIFEST_PATH) so it works both natively and in the container —
    # no hardcoded /app setting here, which used to break native startup_verify.

    # ── Encryption ────────────────────────────────────────────────────────────
    embedding_encryption_key: str = ""

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        try:
            import torch
            if v.startswith("cuda") and not torch.cuda.is_available():
                return "cpu"
        except ImportError:
            return "cpu"
        return v

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def weights_dir(self) -> Path:
        return self.models_dir / "weights"

    @property
    def tensorrt_dir(self) -> Path:
        return self.models_dir / "tensorrt"

    @property
    def custom_dir(self) -> Path:
        return self.models_dir / "custom"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
