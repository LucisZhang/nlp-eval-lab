from datetime import date

import duckdb
import yaml

from triage_lab import splits

# One row per interesting temporal boundary, plus a dropped-product row and a
# handful of rows to exercise class strata. Schema matches narratives_deduped.
# (complaint_id, date_received, product)
_MORTGAGE = "Mortgage"
_DEBT = "Debt collection"
_DROPPED = "Other financial service"  # taxonomy_map.yaml dropped product

FIXTURE_ROWS = [
    (1, "2015-06-30", _MORTGAGE),   # before TRAIN start -> no split
    (2, "2015-07-01", _MORTGAGE),   # TRAIN start (inclusive)
    (3, "2018-05-05", _DEBT),       # TRAIN
    (4, "2021-12-31", _MORTGAGE),   # TRAIN end (inclusive)
    (5, "2022-01-01", _MORTGAGE),   # CAL start
    (6, "2022-06-30", _DEBT),       # CAL end
    (7, "2022-07-01", _MORTGAGE),   # TEST-IID start
    (8, "2022-12-31", _DEBT),       # TEST-IID end
    (9, "2023-01-01", _MORTGAGE),   # TEST-DRIFT-2023
    (10, "2024-08-08", _DEBT),      # TEST-DRIFT-2024
    (11, "2025-09-09", _MORTGAGE),  # TEST-DRIFT-2025
    (12, "2026-01-15", _DEBT),      # TEST-DRIFT-2026H1 only (pre-cutoff)
    (13, "2026-02-01", _MORTGAGE),  # DRIFT-2026H1 AND POSTCUTOFF (documented overlap)
    (14, "2026-07-15", _DEBT),      # POSTCUTOFF only (after DRIFT-2026H1 window)
    (15, "2019-03-03", _DROPPED),   # dropped product, in TRAIN range -> excluded
]


def _write_fixture(tmp_path):
    parquet_path = tmp_path / "narratives_deduped.parquet"
    con = duckdb.connect()
    con.execute("SET threads=1")
    con.execute("SET preserve_insertion_order=true")
    con.execute(
        """
        CREATE TABLE t (
            complaint_id BIGINT, date_received DATE, product VARCHAR,
            sub_product VARCHAR, issue VARCHAR, company VARCHAR,
            state VARCHAR, narrative VARCHAR
        )
        """
    )
    con.executemany(
        "INSERT INTO t VALUES (?, ?, ?, 'SubProd', 'Issue', 'Co', 'ST', ?)",
        [
            (cid, date.fromisoformat(d), prod, f"narrative number {cid} unique text here")
            for cid, d, prod in FIXTURE_ROWS
        ],
    )
    con.execute(
        f"""
        COPY (SELECT * FROM t ORDER BY complaint_id) TO '{parquet_path}'
        (FORMAT PARQUET, COMPRESSION 'snappy', ROW_GROUP_SIZE {splits.ROW_GROUP_SIZE})
        """
    )
    con.close()
    return parquet_path


def _ids(parquet_path):
    con = duckdb.connect()
    ids = [
        r[0]
        for r in con.execute(
            f"SELECT complaint_id FROM read_parquet('{parquet_path}') ORDER BY complaint_id"
        ).fetchall()
    ]
    con.close()
    return ids


# ---------------------------------------------------------------------------
# Pure quota math
# ---------------------------------------------------------------------------

def test_quota_sums_to_target_and_ties_break_canonically():
    counts = {("a",): 7, ("b",): 3}
    quota = splits.stratum_quotas(counts, 5)
    assert sum(quota.values()) == 5
    # exact shares a=3.5 b=1.5; equal .5 remainders -> canonical key ("a",) wins.
    assert quota == {("a",): 4, ("b",): 1}


def test_quota_take_all_when_target_at_least_total():
    counts = {("a",): 4, ("b",): 6}
    assert splits.stratum_quotas(counts, 10) == counts
    assert splits.stratum_quotas(counts, 999) == counts


def test_quota_never_exceeds_availability():
    counts = {("a",): 1, ("b",): 1, ("c",): 1000}
    quota = splits.stratum_quotas(counts, 500)
    for k, avail in counts.items():
        assert quota[k] <= avail
    assert sum(quota.values()) == 500


# ---------------------------------------------------------------------------
# Stratified order-and-take
# ---------------------------------------------------------------------------

def test_select_stratified_exact_size_and_proportional():
    # 60 class-A + 40 class-B rows, target 10 -> 6 A + 4 B (proportional).
    rows = [(i, date(2020, 1, 1), "A") for i in range(60)]
    rows += [(1000 + i, date(2020, 1, 1), "B") for i in range(40)]
    sel = splits.select_stratified(rows, 10, splits._STRATA_CLASS)
    assert len(sel) == 10
    a = sum(1 for c in sel if c < 1000)
    b = sum(1 for c in sel if c >= 1000)
    assert (a, b) == (6, 4)


def test_select_stratified_is_deterministic():
    rows = [(i, date(2020, 1, 1), "A") for i in range(50)]
    a = splits.select_stratified(rows, 7, splits._STRATA_CLASS)
    b = splits.select_stratified(rows, 7, splits._STRATA_CLASS)
    assert a == b


# ---------------------------------------------------------------------------
# End-to-end materialization on the fixture (real SPECS: targets exceed fixture
# size, so every in-range row is taken -> boundaries are directly checkable).
# ---------------------------------------------------------------------------

def test_boundaries_and_membership(tmp_path):
    parquet_path = _write_fixture(tmp_path)
    out_dir = tmp_path / "splits"
    splits.run_splits(parquet_path, out_dir)

    assert _ids(out_dir / "train.parquet") == [2, 3, 4]  # 1 (pre-start) and 15 (dropped) excluded
    assert _ids(out_dir / "cal.parquet") == [5, 6]
    assert _ids(out_dir / "test_iid.parquet") == [7, 8]
    assert _ids(out_dir / "test_drift_2023.parquet") == [9]
    assert _ids(out_dir / "test_drift_2024.parquet") == [10]
    assert _ids(out_dir / "test_drift_2025.parquet") == [11]
    assert _ids(out_dir / "test_drift_2026h1.parquet") == [12, 13]
    assert _ids(out_dir / "test_postcutoff.parquet") == [13, 14]  # 13 is the documented overlap


def test_dropped_product_excluded_everywhere(tmp_path):
    parquet_path = _write_fixture(tmp_path)
    out_dir = tmp_path / "splits"
    splits.run_splits(parquet_path, out_dir)
    for spec in splits.SPECS:
        assert 15 not in _ids(out_dir / f"{spec.name}.parquet")


def test_class_column_attached_and_correct(tmp_path):
    parquet_path = _write_fixture(tmp_path)
    out_dir = tmp_path / "splits"
    splits.run_splits(parquet_path, out_dir)

    con = duckdb.connect()
    cols = [
        r[0]
        for r in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{out_dir / 'train.parquet'}')"
        ).fetchall()
    ]
    labelled = dict(
        con.execute(
            f"SELECT complaint_id, \"class\" FROM read_parquet('{out_dir / 'train.parquet'}')"
        ).fetchall()
    )
    con.close()

    # full source schema + appended class
    assert cols == [
        "complaint_id", "date_received", "product", "sub_product",
        "issue", "company", "state", "narrative", "class",
    ]
    assert labelled[2] == "mortgage"
    assert labelled[3] == "debt_collection"


def test_stats_yaml_accounting(tmp_path):
    parquet_path = _write_fixture(tmp_path)
    out_dir = tmp_path / "splits"
    stats = splits.run_splits(parquet_path, out_dir)

    on_disk = yaml.safe_load((out_dir / "splits_stats.yaml").read_text())
    assert on_disk == stats

    assert stats["seed"] == splits.SEED
    assert stats["rows_deduped_total"] == 15
    assert stats["rows_dropped_product"] == 1        # the Other-financial-service row
    assert stats["rows_mapped_total"] == 14
    assert stats["rows_before_train_start"] == 1     # complaint 1 @ 2015-06-30
    # row 1 is the only mapped row in no split (all others land somewhere).
    assert stats["rows_in_no_split"] == 1
    assert "2026-02-01" in stats["postcutoff_rationale"] or "2026-02-01" not in stats
    assert "claude-haiku-4-5-20251001" in stats["postcutoff_rationale"]
    # documented overlap = complaint 13 only
    assert "1 complaint_ids" in stats["overlap_note"]


def test_double_run_byte_identical(tmp_path):
    parquet_path = _write_fixture(tmp_path)
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    splits.run_splits(parquet_path, out_a)
    splits.run_splits(parquet_path, out_b)
    for spec in splits.SPECS:
        name = f"{spec.name}.parquet"
        assert (out_a / name).read_bytes() == (out_b / name).read_bytes()
