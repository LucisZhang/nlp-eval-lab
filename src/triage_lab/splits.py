"""Temporal split materialization for the CFPB triage lab (UPGRADE_PLAN.md §5).

Consumes the deduped, un-labelled narratives parquet, applies the frozen taxonomy
map (taxonomy_map.yaml) to attach a harmonized `class`, and materializes the frozen
temporal splits into data/splits/. Rows whose product maps to a dropped product are
excluded before any split is formed.

Determinism is absolute: rerunning from the same deduped parquet reproduces every
output file byte-identically. That holds because:

- Split membership is a pure function of (complaint_id, date_received, product).
- Sub-sampled splits (TRAIN, TEST-DRIFT-*) use an order-and-take scheme, never an
  RNG: within each stratum, rows are ranked by a stable BLAKE2b hash of
  (SEED, complaint_id) with complaint_id as tie-break, and the top-k are taken.
  The hash is hashlib (stable across interpreters/versions forever), not Python's
  salted `hash()`.
- Per-stratum quotas come from largest-remainder apportionment of the exact
  proportional shares, so the per-stratum counts sum to exactly the target and the
  allocation is a pure function of the stratum counts (ties broken by canonical
  stratum key). See `stratum_quotas`.
- The DuckDB writes are single-threaded with preserved insertion order, a fixed
  row-group size, and an ORDER BY complaint_id, over inputs already sorted by
  complaint_id — matching ingest.py / dedup.py.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import yaml

from triage_lab.snapshot import sha256_file
from triage_lab.taxonomy import DEFAULT_TAXONOMY_PATH, Taxonomy, load_taxonomy

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = REPO_ROOT / "data" / "dedup" / "narratives_deduped.parquet"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "splits"

# Fixed for deterministic parquet output; DuckDB's default, pinned explicitly
# (identical to ingest.py / dedup.py so all Phase-0 parquets share a byte layout).
ROW_GROUP_SIZE = 122880

# Frozen sub-sampling seed. Order-and-take, not an RNG: this constant is only a
# hash salt, so it can never introduce interpreter- or platform-dependent behavior.
SEED = 20260805

# Contamination boundary for TEST-POSTCUTOFF. Recorded verbatim in the stats YAML.
POSTCUTOFF_RATIONALE = (
    "TEST-POSTCUTOFF = date_received >= 2026-02-01. Tier C uses "
    "claude-haiku-4-5-20251001 (training data cutoff Jul 2025) and claude-sonnet-5 "
    "(training data cutoff Jan 2026); the boundary is set strictly after the LATEST "
    "Tier C training cutoff so the slice is contamination-safe for both models. "
    "Verified 2026-08-05 from "
    "https://platform.claude.com/docs/en/about-claude/models/overview.md. "
    "Intentional overlap with TEST-DRIFT-2026H1 (different purpose). Months near the "
    "snapshot date (frozen 2026-08-05) are thinned by CFPB publication lag."
)

QUOTA_SCHEME = (
    "Largest-remainder (Hamilton) apportionment: per-stratum quota = "
    "floor(count * target / total), then the leftover (target - sum of floors) "
    "seats are handed one each to the strata with the largest fractional remainder, "
    "ties broken by canonical (ascending) stratum key. Within a stratum, rows are "
    "ranked by (blake2b_16(f'{SEED}:{complaint_id}'), complaint_id) and the top-quota "
    "rows are taken. SEED = 20260805."
)

_STRATA_CLASS = "class"
_STRATA_CLASS_YEAR = "class_year"
_STRATA_NONE = "none"


@dataclass(frozen=True)
class SplitSpec:
    name: str
    start: date  # inclusive lower bound on date_received
    end: date | None  # inclusive upper bound; None => open-ended
    target: int | None  # None => take every row in range; else sub-sample to target
    strata: str  # _STRATA_NONE | _STRATA_CLASS | _STRATA_CLASS_YEAR


SPECS: tuple[SplitSpec, ...] = (
    SplitSpec("train", date(2015, 7, 1), date(2021, 12, 31), 300_000, _STRATA_CLASS_YEAR),
    SplitSpec("cal", date(2022, 1, 1), date(2022, 6, 30), None, _STRATA_NONE),
    SplitSpec("test_iid", date(2022, 7, 1), date(2022, 12, 31), None, _STRATA_NONE),
    SplitSpec("test_drift_2023", date(2023, 1, 1), date(2023, 12, 31), 20_000, _STRATA_CLASS),
    SplitSpec("test_drift_2024", date(2024, 1, 1), date(2024, 12, 31), 20_000, _STRATA_CLASS),
    SplitSpec("test_drift_2025", date(2025, 1, 1), date(2025, 12, 31), 20_000, _STRATA_CLASS),
    SplitSpec("test_drift_2026h1", date(2026, 1, 1), date(2026, 6, 30), 20_000, _STRATA_CLASS),
    SplitSpec("test_postcutoff", date(2026, 2, 1), None, None, _STRATA_NONE),
)

# The single documented complaint_id overlap: the contamination-safe LLM slice and
# the 2026-H1 yearly drift slice share their Feb-Jun 2026 rows by design.
_ALLOWED_OVERLAP = frozenset({("test_drift_2026h1", "test_postcutoff")})

# Boundary used only for the "rows before TRAIN start" accounting line.
_TRAIN_START = SPECS[0].start


def _rank_key(complaint_id: int) -> bytes:
    """Stable per-row sub-sampling rank key: blake2b(SEED:complaint_id)."""
    return hashlib.blake2b(
        f"{SEED}:{complaint_id}".encode(), digest_size=16
    ).digest()


def _stratum_key(cls: str, d: date, kind: str) -> tuple:
    if kind == _STRATA_CLASS:
        return (cls,)
    if kind == _STRATA_CLASS_YEAR:
        return (cls, d.year)
    raise ValueError(f"unstratified split has no stratum key (kind={kind!r})")


def stratum_quotas(counts: dict[tuple, int], target: int) -> dict[tuple, int]:
    """Largest-remainder apportionment of `target` across strata by exact share.

    Pure function of `counts` and `target`; deterministic tie-break by stratum key.
    If target >= total available, every stratum keeps all its rows (take-all).
    """
    total = sum(counts.values())
    keys = sorted(counts)
    if target >= total:
        return {k: counts[k] for k in keys}

    exact = {k: counts[k] * target / total for k in keys}
    quota = {k: math.floor(exact[k]) for k in keys}
    remaining = target - sum(quota.values())
    # Hand leftover seats to the largest fractional remainders; ties -> canonical key.
    order = sorted(keys, key=lambda k: (-(exact[k] - quota[k]), k))
    for k in order[:remaining]:
        quota[k] += 1
    for k in keys:
        # floor(count*target/total) <= count-1 since target < total, so +1 never
        # exceeds availability; assert to fail loud if that reasoning ever breaks.
        if quota[k] > counts[k]:
            raise AssertionError(f"quota {quota[k]} exceeds available {counts[k]} for {k}")
    return quota


def select_stratified(
    rows: list[tuple[int, date, str]], target: int, kind: str
) -> set[int]:
    """Order-and-take stratified sub-sample -> set of selected complaint_ids."""
    strata: dict[tuple, list[int]] = defaultdict(list)
    for cid, d, cls in rows:
        strata[_stratum_key(cls, d, kind)].append(cid)
    quotas = stratum_quotas({k: len(v) for k, v in strata.items()}, target)
    selected: set[int] = set()
    for key in sorted(strata):
        ranked = sorted(strata[key], key=lambda c: (_rank_key(c), c))
        selected.update(ranked[: quotas[key]])
    return selected


def _load_mapped_rows(
    input_path: Path, taxonomy: Taxonomy
) -> tuple[list[tuple[int, date, str]], int, int]:
    """Load deduped rows, attach harmonized class, drop dropped-product rows.

    Returns (mapped_rows sorted by complaint_id, total_deduped_rows, dropped_rows).
    """
    con = duckdb.connect()
    try:
        con.execute("SET threads=1")
        con.execute("SET preserve_insertion_order=true")
        cur = con.execute(
            f"""
            SELECT complaint_id, date_received, product
            FROM read_parquet('{input_path}')
            ORDER BY complaint_id
            """
        )
        rows_all = cur.fetchall()
    finally:
        con.close()

    p2c = taxonomy.product_to_class
    mapped: list[tuple[int, date, str]] = []
    dropped = 0
    for cid, d, product in rows_all:
        cls = p2c.get(product)
        if cls is None:
            dropped += 1  # dropped_products (coverage validated upstream in taxonomy)
            continue
        mapped.append((int(cid), d, cls))
    return mapped, len(rows_all), dropped


def _assign_splits(
    mapped: list[tuple[int, date, str]],
) -> tuple[dict[str, set[int]], dict[str, dict]]:
    """Compute selected complaint_ids per split and per-split diagnostics."""
    selected: dict[str, set[int]] = {}
    diagnostics: dict[str, dict] = {}
    for spec in SPECS:
        in_range = [
            (cid, d, cls)
            for cid, d, cls in mapped
            if d >= spec.start and (spec.end is None or d <= spec.end)
        ]
        if spec.target is None:
            ids = {cid for cid, _, _ in in_range}
        else:
            ids = select_stratified(in_range, spec.target, spec.strata)
        selected[spec.name] = ids
        diagnostics[spec.name] = {
            "n_candidates": len(in_range),
            "n_selected": len(ids),
        }
    return selected, diagnostics


def _check_disjoint(selected: dict[str, set[int]]) -> None:
    """Splits must be pairwise disjoint on complaint_id except the documented overlap."""
    names = [spec.name for spec in SPECS]
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = selected[a] & selected[b]
            if overlap and (a, b) not in _ALLOWED_OVERLAP:
                raise AssertionError(
                    f"undocumented split overlap: {a} ∩ {b} = {len(overlap)} complaint_ids"
                )


def _class_year_matrix(
    ids: set[int], id_to_row: dict[int, tuple[date, str]]
) -> dict[str, dict[int, int]]:
    matrix: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for cid in ids:
        d, cls = id_to_row[cid]
        matrix[cls][d.year] += 1
    return {cls: dict(sorted(years.items())) for cls, years in sorted(matrix.items())}


def _write_split(
    input_path: Path,
    out_path: Path,
    ids: set[int],
    id_to_cls: dict[int, str],
) -> None:
    """Join selected ids back to the deduped parquet, attach class, write parquet."""
    id_arr = np.fromiter(sorted(ids), dtype=np.int64, count=len(ids))
    cls_arr = np.array([id_to_cls[int(c)] for c in id_arr], dtype=object)
    con = duckdb.connect()
    try:
        con.execute("SET threads=1")
        con.execute("SET preserve_insertion_order=true")
        con.register("sel_view", {"complaint_id": id_arr, "class": cls_arr})
        con.execute(
            f"""
            COPY (
                SELECT n.*, s."class"
                FROM read_parquet('{input_path}') n
                JOIN sel_view s ON n.complaint_id = s.complaint_id
                ORDER BY n.complaint_id
            ) TO '{out_path}'
            (FORMAT PARQUET, COMPRESSION 'snappy', ROW_GROUP_SIZE {ROW_GROUP_SIZE})
            """
        )
    finally:
        con.close()


def run_splits(
    input_path: Path,
    out_dir: Path,
    taxonomy_path: Path = DEFAULT_TAXONOMY_PATH,
    specs: tuple[SplitSpec, ...] = SPECS,
) -> dict:
    """Materialize every temporal split + splits_stats.yaml. Returns the stats dict."""
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / "splits_stats.yaml"

    taxonomy = load_taxonomy(taxonomy_path)
    mapped, total_deduped, dropped_rows = _load_mapped_rows(input_path, taxonomy)
    id_to_row = {cid: (d, cls) for cid, d, cls in mapped}
    id_to_cls = {cid: cls for cid, (_, cls) in id_to_row.items()}

    selected, diagnostics = _assign_splits(mapped)
    _check_disjoint(selected)

    split_stats: dict[str, dict] = {}
    for spec in specs:
        out_path = out_dir / f"{spec.name}.parquet"
        ids = selected[spec.name]
        _write_split(input_path, out_path, ids, id_to_cls)
        split_stats[spec.name] = {
            "start": spec.start.isoformat(),
            "end": spec.end.isoformat() if spec.end is not None else None,
            "strata": spec.strata,
            "target": spec.target,
            "n_candidates": diagnostics[spec.name]["n_candidates"],
            "n_selected": diagnostics[spec.name]["n_selected"],
            "class_year_counts": _class_year_matrix(ids, id_to_row),
            "sha256": sha256_file(out_path),
        }

    union_ids: set[int] = set().union(*selected.values()) if selected else set()
    rows_before_train_start = sum(1 for _, d, _ in mapped if d < _TRAIN_START)
    overlap_ids = selected["test_drift_2026h1"] & selected["test_postcutoff"]

    stats = {
        "seed": SEED,
        "quota_scheme": QUOTA_SCHEME,
        "postcutoff_rationale": POSTCUTOFF_RATIONALE,
        "overlap_note": (
            f"test_drift_2026h1 ∩ test_postcutoff = {len(overlap_ids)} complaint_ids by "
            "design (different purposes: yearly drift slice vs contamination-safe LLM "
            "slice). This is the only permitted cross-split complaint_id overlap; all "
            "other split pairs are disjoint."
        ),
        "input_sha256": sha256_file(input_path),
        "taxonomy_version": taxonomy.version,
        "rows_deduped_total": total_deduped,
        "rows_dropped_product": dropped_rows,
        "rows_mapped_total": len(mapped),
        "rows_before_train_start": rows_before_train_start,
        "rows_in_no_split": len(mapped) - len(union_ids),
        "splits": split_stats,
    }
    stats_path.write_text(yaml.safe_dump(stats, sort_keys=False))
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m triage_lab.splits")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY_PATH)
    args = parser.parse_args(argv)

    stats = run_splits(args.input, args.out_dir, args.taxonomy)

    print(f"rows_deduped_total:      {stats['rows_deduped_total']}")
    print(f"rows_dropped_product:    {stats['rows_dropped_product']}")
    print(f"rows_mapped_total:       {stats['rows_mapped_total']}")
    print(f"rows_before_train_start: {stats['rows_before_train_start']}")
    print(f"rows_in_no_split:        {stats['rows_in_no_split']}")
    print("per-split (n_selected / n_candidates):")
    for name, s in stats["splits"].items():
        print(f"  {name:20s} {s['n_selected']:>8d} / {s['n_candidates']:<8d} sha={s['sha256'][:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
