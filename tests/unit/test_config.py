from unittest.mock import patch

import pytest


def test_settings_loads_defaults():
    with patch.dict(
        "os.environ",
        {
            "APP_SECRET_KEY": "a" * 32,
            "JWT_SECRET_KEY": "b" * 32,
            "DATABASE_URL": "postgresql+asyncpg://u:p@localhost/db",
            "TIMESCALE_URL": "postgresql+asyncpg://u:p@localhost/ts",
            "NEO4J_PASSWORD": "secret",
            "POSTGRES_PASSWORD": "secret",
        },
        clear=False,
    ):
        from src.core.config import Settings
        s = Settings()
        assert s.app_env == "development"
        assert s.face_match_accept_threshold == 0.45
        assert s.face_match_reject_threshold == 0.35
        assert s.antispoofing_confidence_min == 0.85
        assert s.enrollment_min_crops == 10


def test_production_flag():
    with patch.dict(
        "os.environ",
        {
            "APP_ENV": "production",
            "APP_SECRET_KEY": "a" * 32,
            "JWT_SECRET_KEY": "b" * 32,
            "DATABASE_URL": "postgresql+asyncpg://u:p@localhost/db",
            "TIMESCALE_URL": "postgresql+asyncpg://u:p@localhost/ts",
            "NEO4J_PASSWORD": "secret",
            "POSTGRES_PASSWORD": "secret",
        },
        clear=False,
    ):
        from src.core.config import Settings
        s = Settings()
        assert s.is_production is True
