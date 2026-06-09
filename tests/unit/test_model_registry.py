from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.core.exceptions import ModelChecksumError, ModelLoadError
from src.core.model_registry import ModelRegistry, compute_sha256, startup_verify


# ── compute_sha256 ────────────────────────────────────────────────────────────

def test_sha256_consistent(tmp_path):
    f = tmp_path / "weights.bin"
    f.write_bytes(b"\x00" * 1024)
    assert compute_sha256(f) == compute_sha256(f)


def test_sha256_differs_after_file_change(tmp_path):
    f = tmp_path / "w.bin"
    f.write_bytes(b"original content")
    digest_before = compute_sha256(f)
    f.write_bytes(b"tampered content")
    assert compute_sha256(f) != digest_before


def test_sha256_empty_file(tmp_path):
    f = tmp_path / "empty.bin"
    f.write_bytes(b"")
    # SHA-256 of empty string is well-known
    assert compute_sha256(f) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


# ── ModelRegistry.sign ────────────────────────────────────────────────────────

class TestSign:
    def test_sign_returns_entry(self, tmp_path):
        reg = ModelRegistry()
        f = tmp_path / "model.pt"
        f.write_bytes(b"fake weights")
        entry = reg.sign("test_model", f)
        assert entry.name == "test_model"
        assert len(entry.sha256) == 64
        assert entry.path == str(f)

    def test_sign_registers_entry(self, tmp_path):
        reg = ModelRegistry()
        f = tmp_path / "model.pt"
        f.write_bytes(b"data")
        reg.sign("model_a", f)
        assert "model_a" in reg

    def test_sign_missing_file_raises(self, tmp_path):
        reg = ModelRegistry()
        with pytest.raises(ModelLoadError, match="Cannot sign"):
            reg.sign("missing", tmp_path / "no_such_file.pt")

    def test_sign_overwrites_existing_entry(self, tmp_path):
        reg = ModelRegistry()
        f = tmp_path / "m.pt"
        f.write_bytes(b"v1")
        reg.sign("m", f)
        first_hash = reg.entries()[0].sha256
        f.write_bytes(b"v2")
        reg.sign("m", f)
        assert reg.entries()[0].sha256 != first_hash


# ── ModelRegistry.verify ──────────────────────────────────────────────────────

class TestVerify:
    def test_verify_passes_for_untampered_file(self, tmp_path):
        reg = ModelRegistry()
        f = tmp_path / "ok.pt"
        f.write_bytes(b"good weights")
        reg.sign("ok_model", f)
        assert reg.verify("ok_model") is True

    def test_verify_raises_on_tampered_file(self, tmp_path):
        reg = ModelRegistry()
        f = tmp_path / "model.pt"
        f.write_bytes(b"original")
        reg.sign("model", f)
        f.write_bytes(b"tampered!")
        with pytest.raises(ModelChecksumError, match="mismatch"):
            reg.verify("model")

    def test_verify_raises_on_missing_file(self, tmp_path):
        reg = ModelRegistry()
        f = tmp_path / "exists.pt"
        f.write_bytes(b"data")
        reg.sign("model", f)
        f.unlink()
        with pytest.raises(ModelChecksumError, match="missing"):
            reg.verify("model")

    def test_verify_raises_key_error_for_unknown_name(self):
        reg = ModelRegistry()
        with pytest.raises(KeyError):
            reg.verify("not_registered")


# ── ModelRegistry.verify_all ─────────────────────────────────────────────────

class TestVerifyAll:
    def test_verify_all_returns_dict_for_all_entries(self, tmp_path):
        reg = ModelRegistry()
        for name in ["a", "b", "c"]:
            f = tmp_path / f"{name}.pt"
            f.write_bytes(name.encode())
            reg.sign(name, f)
        results = reg.verify_all()
        assert set(results) == {"a", "b", "c"}
        assert all(results.values())

    def test_verify_all_raises_on_first_failure(self, tmp_path):
        reg = ModelRegistry()
        good = tmp_path / "good.pt"
        bad = tmp_path / "bad.pt"
        good.write_bytes(b"fine")
        bad.write_bytes(b"original")
        reg.sign("good", good)
        reg.sign("bad", bad)
        bad.write_bytes(b"corrupted")
        with pytest.raises(ModelChecksumError):
            reg.verify_all()


# ── Manifest persistence ──────────────────────────────────────────────────────

class TestManifest:
    def test_save_and_load_roundtrip(self, tmp_path):
        reg = ModelRegistry()
        f = tmp_path / "weights.pt"
        f.write_bytes(b"model_data")
        reg.sign("net", f)

        manifest = tmp_path / "manifest.json"
        reg.save_manifest(manifest)

        reg2 = ModelRegistry()
        reg2.load_manifest(manifest)
        assert "net" in reg2
        assert reg2.entries()[0].sha256 == reg.entries()[0].sha256

    def test_manifest_is_valid_json(self, tmp_path):
        reg = ModelRegistry()
        f = tmp_path / "m.pt"
        f.write_bytes(b"data")
        reg.sign("m", f)
        mpath = tmp_path / "manifest.json"
        reg.save_manifest(mpath)
        data = json.loads(mpath.read_text())
        assert "m" in data
        assert "sha256" in data["m"]

    def test_load_missing_manifest_raises(self, tmp_path):
        reg = ModelRegistry()
        with pytest.raises(ModelLoadError, match="not found"):
            reg.load_manifest(tmp_path / "missing.json")

    def test_load_malformed_manifest_raises(self, tmp_path):
        reg = ModelRegistry()
        mpath = tmp_path / "bad.json"
        mpath.write_text("{ not valid json }")
        with pytest.raises(ModelLoadError, match="Malformed"):
            reg.load_manifest(mpath)

    def test_load_merges_with_existing_entries(self, tmp_path):
        reg = ModelRegistry()
        f1 = tmp_path / "f1.pt"
        f2 = tmp_path / "f2.pt"
        f1.write_bytes(b"a")
        f2.write_bytes(b"b")
        reg.sign("f1", f1)
        mpath = tmp_path / "m.json"
        reg.save_manifest(mpath)

        reg2 = ModelRegistry()
        reg2.sign("f2", f2)
        reg2.load_manifest(mpath)
        assert "f1" in reg2
        assert "f2" in reg2


# ── startup_verify ────────────────────────────────────────────────────────────

class TestStartupVerify:
    def test_no_ops_when_manifest_absent(self, tmp_path):
        # Should not raise — first boot scenario
        startup_verify(tmp_path / "nonexistent.json")

    def test_verifies_clean_manifest(self, tmp_path):
        reg = ModelRegistry()
        f = tmp_path / "w.pt"
        f.write_bytes(b"clean")
        reg.sign("w", f)
        mpath = tmp_path / "manifest.json"
        reg.save_manifest(mpath)
        # Must not raise
        startup_verify(mpath)

    def test_raises_on_tampered_file(self, tmp_path):
        reg = ModelRegistry()
        f = tmp_path / "w.pt"
        f.write_bytes(b"original")
        reg.sign("w", f)
        mpath = tmp_path / "manifest.json"
        reg.save_manifest(mpath)
        f.write_bytes(b"tampered")
        with pytest.raises(ModelChecksumError):
            startup_verify(mpath)
