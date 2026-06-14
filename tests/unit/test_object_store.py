from datetime import UTC, datetime

import pytest

from src.storage.object_store import ObjectStore, StorageError, object_key


def test_object_key_is_date_partitioned():
    when = datetime(2026, 6, 14, 10, 30, tzinfo=UTC)
    key = object_key("snapshots", "cam0", "jpg", when=when)
    assert key.startswith("snapshots/2026/06/14/cam0/")
    assert key.endswith(".jpg")


def test_object_key_strips_leading_dot_on_ext():
    key = object_key("clips", "cam1", ".mp4")
    assert key.endswith(".mp4")
    assert ".." not in key


def test_disabled_when_no_endpoint():
    store = ObjectStore(endpoint_url="", access_key="", secret_key="", bucket="b")
    assert store.is_enabled is False
    # ensure_bucket is a safe no-op when disabled
    store.ensure_bucket()


def test_disabled_store_raises_on_put():
    store = ObjectStore(endpoint_url="", access_key="", secret_key="", bucket="b")
    with pytest.raises(StorageError):
        store.put_bytes("k", b"data")
