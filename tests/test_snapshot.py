import hashlib

import pytest
import yaml

from triage_lab import snapshot


@pytest.fixture
def fake_snapshot(tmp_path):
    path = tmp_path / "complaints.csv.zip"
    path.write_bytes(b"fake cfpb data" * 100)
    return path


@pytest.fixture
def manifest_path(tmp_path):
    return tmp_path / "SNAPSHOT_MANIFEST.yaml"


def test_sha256_file_matches_hashlib(fake_snapshot):
    expected = hashlib.sha256(fake_snapshot.read_bytes()).hexdigest()
    assert snapshot.sha256_file(fake_snapshot) == expected


def test_freeze_writes_expected_manifest(fake_snapshot, manifest_path):
    manifest = snapshot.freeze(
        fake_snapshot,
        manifest_path,
        url="https://example.com/complaints.csv.zip",
        download_date=__import__("datetime").date(2026, 1, 1),
    )
    assert manifest_path.exists()
    on_disk = yaml.safe_load(manifest_path.read_text())
    assert on_disk == manifest
    assert manifest["url"] == "https://example.com/complaints.csv.zip"
    assert manifest["filename"] == "complaints.csv.zip"
    assert manifest["download_date"] == "2026-01-01"
    assert manifest["sha256"] == hashlib.sha256(fake_snapshot.read_bytes()).hexdigest()
    assert manifest["size_bytes"] == fake_snapshot.stat().st_size


def test_freeze_refuses_when_manifest_exists(fake_snapshot, manifest_path):
    snapshot.freeze(fake_snapshot, manifest_path, url="https://example.com/x")
    with pytest.raises(RuntimeError):
        snapshot.freeze(fake_snapshot, manifest_path, url="https://example.com/x")


def test_verify_passes_on_match(fake_snapshot, manifest_path):
    snapshot.freeze(fake_snapshot, manifest_path, url="https://example.com/x")
    manifest = snapshot.verify(fake_snapshot, manifest_path)
    assert manifest["sha256"] == snapshot.sha256_file(fake_snapshot)


def test_verify_fails_on_content_tamper(fake_snapshot, manifest_path):
    snapshot.freeze(fake_snapshot, manifest_path, url="https://example.com/x")
    tampered = (b"tampered data!" * 100)[: len(fake_snapshot.read_bytes())]
    fake_snapshot.write_bytes(tampered)
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        snapshot.verify(fake_snapshot, manifest_path)


def test_verify_fails_on_size_mismatch(fake_snapshot, manifest_path):
    snapshot.freeze(fake_snapshot, manifest_path, url="https://example.com/x")
    fake_snapshot.write_bytes(b"short")
    with pytest.raises(RuntimeError, match="size mismatch"):
        snapshot.verify(fake_snapshot, manifest_path)


def test_verify_fails_when_snapshot_missing(fake_snapshot, manifest_path):
    snapshot.freeze(fake_snapshot, manifest_path, url="https://example.com/x")
    fake_snapshot.unlink()
    with pytest.raises(RuntimeError, match="not found"):
        snapshot.verify(fake_snapshot, manifest_path)
