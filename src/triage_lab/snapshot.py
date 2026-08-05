"""CFPB snapshot freeze/verify: hash and manifest the raw dataset download."""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import date
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "SNAPSHOT_MANIFEST.yaml"
DEFAULT_SNAPSHOT_PATH = REPO_ROOT / "data" / "raw" / "complaints.csv.zip"
DEFAULT_URL = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"

CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def freeze(
    snapshot_path: Path,
    manifest_path: Path = MANIFEST_PATH,
    url: str = DEFAULT_URL,
    download_date: date | None = None,
) -> dict:
    if manifest_path.exists():
        raise RuntimeError(
            f"manifest already exists at {manifest_path} — snapshot is frozen; "
            "delete it manually if you really intend to re-freeze"
        )
    snapshot_path = Path(snapshot_path)
    if download_date is None:
        download_date = date.today()  # noqa: DTZ011
    manifest = {
        "url": url,
        "filename": snapshot_path.name,
        "download_date": download_date.isoformat(),
        "sha256": sha256_file(snapshot_path),
        "size_bytes": snapshot_path.stat().st_size,
    }
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    return manifest


def verify(snapshot_path: Path, manifest_path: Path = MANIFEST_PATH) -> dict:
    snapshot_path = Path(snapshot_path)
    if not manifest_path.exists():
        raise RuntimeError(f"manifest not found at {manifest_path}")
    manifest = yaml.safe_load(manifest_path.read_text())
    if not snapshot_path.exists():
        raise RuntimeError(f"snapshot not found at {snapshot_path}")

    actual_size = snapshot_path.stat().st_size
    if actual_size != manifest["size_bytes"]:
        raise RuntimeError(
            f"size mismatch for {snapshot_path}: expected {manifest['size_bytes']} bytes, "
            f"got {actual_size} bytes"
        )

    actual_sha256 = sha256_file(snapshot_path)
    if actual_sha256 != manifest["sha256"]:
        raise RuntimeError(
            f"sha256 mismatch for {snapshot_path}: expected {manifest['sha256']}, "
            f"got {actual_sha256}"
        )

    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m triage_lab.snapshot")
    parser.add_argument("action", choices=["freeze", "verify"])
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--url", type=str, default=DEFAULT_URL)
    args = parser.parse_args(argv)

    try:
        if args.action == "freeze":
            manifest = freeze(args.snapshot, args.manifest, args.url)
        else:
            manifest = verify(args.snapshot, args.manifest)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(f"OK: sha256={manifest['sha256']} size_bytes={manifest['size_bytes']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
