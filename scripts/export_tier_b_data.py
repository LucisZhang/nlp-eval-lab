#!/usr/bin/env python
"""Export the Tier-B training kit: TRAIN + CAL parquets + a provenance manifest.

Phase 2 trains on a cloud GPU (Colab T4 / A10) that does NOT have this repo or the
frozen DuckDB splits. This script materializes the *exact* rows the cloud box needs
— nothing more — into a self-contained directory the human uploads:

    <out>/train.parquet   (complaint_id, narrative, class), ordered by complaint_id
    <out>/cal.parquet     (same columns)                    ordered by complaint_id
    <out>/manifest.json   sha256 of each exported file, row counts, column list,
                          the source split sha256s and the dataset input_sha256
                          (all copied from the frozen splits_stats.yaml).

Determinism: the exported parquets are byte-identical for a fixed DuckDB version
(single-threaded, preserved insertion order, snappy, fixed row-group size — the same
write settings splits.py froze the splits with). The manifest is sorted-key JSON.
`scripts/train_tier_b.py` re-hashes the two parquets against this manifest before
training, so a corrupted / wrong-version upload fails loud instead of silently
training on the wrong data.

Only the three columns the model consumes are exported (complaint_id for ordering,
narrative for the text, class for the label); the other CFPB columns are dropped to
keep the upload small. Provenance still survives because the manifest records the
*source* split sha256 (the full frozen parquet) alongside the exported-file sha256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLITS_DIR = REPO_ROOT / "data" / "splits"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "tier_b_kit"

# Mirror splits.py write settings so exported parquets are byte-reproducible.
ROW_GROUP_SIZE = 122880
EXPORT_COLUMNS = ("complaint_id", "narrative", "class")
EXPORT_SPLITS = ("train", "cal")

CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _write_export(src_path: Path, out_path: Path) -> int:
    """Copy the three model columns out of a frozen split parquet, ordered by id."""
    cols = ", ".join(f'"{c}"' for c in EXPORT_COLUMNS)
    con = duckdb.connect()
    try:
        con.execute("SET threads=1")
        con.execute("SET preserve_insertion_order=true")
        con.execute(
            f"""
            COPY (
                SELECT {cols}
                FROM read_parquet('{src_path}')
                ORDER BY complaint_id
            ) TO '{out_path}'
            (FORMAT PARQUET, COMPRESSION 'snappy', ROW_GROUP_SIZE {ROW_GROUP_SIZE})
            """
        )
        n = con.execute(f"SELECT count(*) FROM read_parquet('{out_path}')").fetchone()[0]
    finally:
        con.close()
    return int(n)


def export(splits_dir: Path, out_dir: Path) -> dict:
    splits_dir = Path(splits_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = yaml.safe_load((splits_dir / "splits_stats.yaml").read_text())

    manifest: dict = {
        "kind": "tier_b_data_kit",
        "input_sha256": stats["input_sha256"],
        "columns": list(EXPORT_COLUMNS),
        "row_group_size": ROW_GROUP_SIZE,
        "compression": "snappy",
        "files": {},
    }
    for split in EXPORT_SPLITS:
        src = splits_dir / f"{split}.parquet"
        out = out_dir / f"{split}.parquet"
        n = _write_export(src, out)
        manifest["files"][f"{split}.parquet"] = {
            "split": split,
            "n_rows": n,
            "sha256": sha256_file(out),
            "source_split_sha256": stats["splits"][split]["sha256"],
        }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="export_tier_b_data")
    p.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = p.parse_args(argv)

    manifest = export(args.splits_dir, args.out)
    print(f"exported tier-b kit -> {args.out}")
    for name, info in manifest["files"].items():
        print(f"  {name}: {info['n_rows']} rows  sha256={info['sha256'][:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
