"""CFPB ingest: CSV -> filtered, deterministic narratives.parquet via DuckDB."""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import duckdb
import yaml

from triage_lab.snapshot import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ZIP_PATH = REPO_ROOT / "data" / "raw" / "complaints.csv.zip"
DEFAULT_CSV_PATH = REPO_ROOT / "data" / "raw" / "complaints.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "ingest"

# Fixed for deterministic parquet output; DuckDB's default, pinned explicitly.
ROW_GROUP_SIZE = 122880


def ensure_extracted(zip_path: Path, csv_path: Path) -> Path:
    """Extract the single CSV from the zip to csv_path; skip if already present."""
    zip_path = Path(zip_path)
    csv_path = Path(csv_path)
    if csv_path.exists():
        return csv_path
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1:
            raise RuntimeError(f"expected exactly one .csv in {zip_path}, found {names}")
        with zf.open(names[0]) as src, open(csv_path, "wb") as dst:
            while chunk := src.read(1024 * 1024):
                dst.write(chunk)
    return csv_path


def run_ingest(csv_path: Path, output_dir: Path) -> dict:
    """Read the CFPB CSV, filter to non-empty narratives, write deterministic parquet.

    Returns a stats dict and writes both narratives.parquet and ingest_stats.yaml
    into output_dir.
    """
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "narratives.parquet"
    stats_path = output_dir / "ingest_stats.yaml"

    con = duckdb.connect()
    try:
        # Determinism knobs: single thread + preserved order so the COPY emits
        # rows in ORDER BY order with a stable, reproducible byte layout.
        con.execute("SET threads=1")
        con.execute("SET preserve_insertion_order=true")

        # Read every field as text (no dtype guessing), then cast explicitly.
        # nullstr='' maps unquoted empty fields to NULL; the narrative filter
        # below also catches quoted empty strings.
        #
        # A tiny number of source rows are genuinely malformed (bad in-narrative
        # quoting desyncs the RFC-4180 parser). ignore_errors=true skips rows
        # whose column count is wrong; TRY_CAST + the NOT NULL guards below drop
        # the handful of rows that still parse into 16 columns but with shifted
        # content (statutory text landing in the date field, etc). Both classes
        # are counted and reported so the data loss is visible, not silent.
        # Single-threaded + fixed input => the dropped set is deterministic.
        con.execute(
            """
            CREATE TABLE raw AS
            SELECT
                TRY_CAST("Complaint ID" AS BIGINT)  AS complaint_id,
                TRY_CAST("Date received" AS DATE)   AS date_received,
                "Product"                           AS product,
                "Sub-product"                       AS sub_product,
                "Issue"                             AS issue,
                "Company"                           AS company,
                "State"                             AS state,
                "Consumer complaint narrative"      AS narrative
            FROM read_csv(
                ?,
                header=true,
                all_varchar=true,
                quote='"',
                escape='"',
                nullstr='',
                ignore_errors=true,
                store_rejects=true
            )
            """,
            [str(csv_path)],
        )

        total_rows_parsed = con.execute("SELECT count(*) FROM raw").fetchone()[0]
        parser_rejected_lines = con.execute(
            "SELECT count(DISTINCT line) FROM reject_errors"
        ).fetchone()[0]
        malformed_key_rows_dropped = con.execute(
            "SELECT count(*) FROM raw WHERE complaint_id IS NULL OR date_received IS NULL"
        ).fetchone()[0]

        keep_filter = (
            "complaint_id IS NOT NULL AND date_received IS NOT NULL "
            "AND narrative IS NOT NULL AND length(trim(narrative)) > 0"
        )

        con.execute(
            f"""
            COPY (
                SELECT complaint_id, date_received, product, sub_product,
                       issue, company, state, narrative
                FROM raw
                WHERE {keep_filter}
                ORDER BY complaint_id
            ) TO '{parquet_path}'
            (FORMAT PARQUET, COMPRESSION 'snappy', ROW_GROUP_SIZE {ROW_GROUP_SIZE})
            """
        )

        narrative_rows, min_date, max_date = con.execute(
            f"""
            SELECT count(*), min(date_received), max(date_received)
            FROM raw WHERE {keep_filter}
            """
        ).fetchone()
    finally:
        con.close()

    output_sha256 = sha256_file(parquet_path)

    stats = {
        "total_rows_parsed": int(total_rows_parsed),
        "parser_rejected_lines": int(parser_rejected_lines),
        "malformed_key_rows_dropped": int(malformed_key_rows_dropped),
        "narrative_rows": int(narrative_rows),
        "min_date_received": min_date.isoformat() if min_date is not None else None,
        "max_date_received": max_date.isoformat() if max_date is not None else None,
        "output_sha256": output_sha256,
    }
    stats_path.write_text(yaml.safe_dump(stats, sort_keys=False))
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m triage_lab.ingest")
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP_PATH)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    csv_path = ensure_extracted(args.zip, args.csv)
    stats = run_ingest(csv_path, args.out_dir)

    print(f"total_rows_parsed:           {stats['total_rows_parsed']}")
    print(f"parser_rejected_lines:       {stats['parser_rejected_lines']}")
    print(f"malformed_key_rows_dropped:  {stats['malformed_key_rows_dropped']}")
    print(f"narrative_rows:              {stats['narrative_rows']}")
    print(f"min_date_received:           {stats['min_date_received']}")
    print(f"max_date_received:           {stats['max_date_received']}")
    print(f"output_sha256:               {stats['output_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
