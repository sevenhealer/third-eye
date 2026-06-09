from __future__ import annotations


class ThirdEyeError(Exception):
    """Base exception for all Third-Eye errors."""


class CameraError(ThirdEyeError):
    """Camera connection or decode failure."""


class CameraTamperError(CameraError):
    """Camera tamper suspected."""


class SpoofingDetectedError(ThirdEyeError):
    """Anti-spoofing gate rejected the presentation."""


class LowConfidenceError(ThirdEyeError):
    """Model confidence below minimum threshold — default-deny applied."""


class EnrollmentError(ThirdEyeError):
    """Enrollment workflow failure."""


class GalleryError(ThirdEyeError):
    """Face gallery read/write failure."""


class ModelLoadError(ThirdEyeError):
    """ML model failed to load or checksum mismatch."""


class ModelChecksumError(ModelLoadError):
    """Loaded model weight SHA-256 does not match registered hash."""


class VRAMBudgetError(ThirdEyeError):
    """Requested model would exceed VRAM budget."""


class KafkaPublishError(ThirdEyeError):
    """Failed to publish event to Kafka."""


class DatabaseError(ThirdEyeError):
    """Database read/write failure."""


class AuditLogError(ThirdEyeError):
    """Audit log integrity violation detected."""


class AuthorizationError(ThirdEyeError):
    """Action requires authorization that was not granted."""


class PromptInjectionError(ThirdEyeError):
    """Potential prompt injection detected in user input."""


class QueryError(ThirdEyeError):
    """NL query parsing or execution failure."""


class RetentionError(ThirdEyeError):
    """Data retention policy violation."""
