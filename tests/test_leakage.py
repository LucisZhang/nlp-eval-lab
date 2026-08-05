"""Phase-0 accept: no near-dup pairs across split boundaries at the LSH threshold.

Builds an adversarial raw fixture that plants near-duplicate narratives in
different temporal ranges (which, without dedup, would land in different splits
and leak), runs the real dedup -> splits pipeline, then asserts that no pair of
surviving narratives drawn from two different materialized splits estimates a
Jaccard >= dedup.LSH_THRESHOLD. Signing reuses dedup's MinHash machinery
verbatim (no reimplementation).
"""

from datetime import date

import duckdb
import numpy as np

from triage_lab import dedup, splits

# Near-dup = identical after CFPB-mask normalization (XXXX vs XXXXXXXX collapse to
# one token). Each pair is planted in two different split ranges; dedup keeps the
# earliest-dated row, so the later-split twin must be removed before splitting.
_NEAR_A_TRAIN = "on XXXX the mortgage servicer failed to credit my escrow payment for three consecutive months"
_NEAR_A_IID = "on XXXXXXXX the mortgage servicer failed to credit my escrow payment for three consecutive months"
_NEAR_B_CAL = "the debt collector called my workplace XXXX times a day after i sent a written cease contact letter"
_NEAR_B_DRIFT = "the debt collector called my workplace XXXXXXXX times a day after i sent a written cease contact letter"

# (complaint_id, date_received, product, narrative)
FIXTURE_ROWS = [
    # Planted near-dup A: TRAIN (2018) vs TEST-IID (2022-H2). Rep = TRAIN (earlier).
    (100, "2018-04-04", "Mortgage", _NEAR_A_TRAIN),
    (101, "2022-09-09", "Mortgage", _NEAR_A_IID),
    # Planted near-dup B: CAL (2022-H1) vs TEST-DRIFT-2023. Rep = CAL (earlier).
    (110, "2022-02-02", "Debt collection", _NEAR_B_CAL),
    (111, "2023-05-05", "Debt collection", _NEAR_B_DRIFT),
    # Distinct singletons giving every split some content to cross-check.
    (200, "2016-06-06", "Mortgage", "my student loan servicer misapplied a lump sum prepayment to future interest only"),
    (201, "2019-07-07", "Credit reporting", "a credit bureau kept reporting a charged off account that was paid in full years ago"),
    (202, "2022-03-03", "Credit reporting", "the bank reversed a provisional credit weeks after resolving my fraud dispute in my favor"),
    (203, "2022-08-08", "Credit reporting", "a prepaid card issuer froze my balance and gave no timeline for releasing the held funds"),
    (204, "2023-10-10", "Mortgage", "the loan servicer force placed hazard insurance despite my active policy on file"),
    (205, "2024-04-04", "Debt collection", "a collector reported the same medical debt twice under two different account numbers"),
    (206, "2025-05-05", "Mortgage", "my payment was returned then a late fee was charged even though funds were available"),
    (207, "2026-01-20", "Debt collection", "a collection agency threatened arrest over an alleged payday balance i never borrowed"),
    (208, "2026-03-03", "Credit reporting", "the reseller mixed another consumers tradelines into my file after a recent address change"),
    (209, "2026-07-20", "Mortgage", "the servicer lost my loss mitigation packet twice and restarted the review clock each time"),
]


def _write_ingest_fixture(tmp_path):
    """Fixture in the INGEST/dedup input schema (dedup output feeds splits)."""
    parquet_path = tmp_path / "narratives.parquet"
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
        [(cid, date.fromisoformat(d), prod, nar) for cid, d, prod, nar in FIXTURE_ROWS],
    )
    con.execute(
        f"""
        COPY (SELECT * FROM t ORDER BY complaint_id) TO '{parquet_path}'
        (FORMAT PARQUET, COMPRESSION 'snappy', ROW_GROUP_SIZE {dedup.ROW_GROUP_SIZE})
        """
    )
    con.close()
    return parquet_path


def _estimated_jaccard(a: str, b: str) -> float:
    """Fraction of equal MinHash values — dedup's own estimator, no reimpl."""
    sig, _ = dedup.sign_texts([a, b])
    return float(np.mean(sig[0] == sig[1]))


def _read_split(parquet_path):
    con = duckdb.connect()
    rows = con.execute(
        f"SELECT complaint_id, narrative FROM read_parquet('{parquet_path}')"
    ).fetchall()
    con.close()
    return rows


def test_fixture_is_adversarial():
    # Sanity: without dedup, the planted pairs WOULD be a cross-split near-dup leak.
    assert _estimated_jaccard(_NEAR_A_TRAIN, _NEAR_A_IID) >= dedup.LSH_THRESHOLD
    assert _estimated_jaccard(_NEAR_B_CAL, _NEAR_B_DRIFT) >= dedup.LSH_THRESHOLD


def test_no_near_dup_pairs_across_split_boundaries(tmp_path):
    ingest_fixture = _write_ingest_fixture(tmp_path)

    dedup_dir = tmp_path / "dedup"
    dedup.run_dedup(ingest_fixture, dedup_dir, workers=1)
    deduped = dedup_dir / "narratives_deduped.parquet"

    splits_dir = tmp_path / "splits"
    splits.run_splits(deduped, splits_dir)

    # The later-dated twin of each planted pair must have been removed pre-split.
    surviving = _read_split(deduped)
    surviving_ids = {cid for cid, _ in surviving}
    assert 101 not in surviving_ids  # TEST-IID twin of A removed
    assert 111 not in surviving_ids  # TEST-DRIFT-2023 twin of B removed

    per_split = {
        spec.name: _read_split(splits_dir / f"{spec.name}.parquet") for spec in splits.SPECS
    }
    names = [spec.name for spec in splits.SPECS]
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            for cid_a, nar_a in per_split[a]:
                for cid_b, nar_b in per_split[b]:
                    if cid_a == cid_b:
                        continue  # documented POSTCUTOFF ∩ DRIFT-2026H1 same-row overlap
                    est = _estimated_jaccard(nar_a, nar_b)
                    assert est < dedup.LSH_THRESHOLD, (
                        f"cross-split near-dup: {a}:{cid_a} vs {b}:{cid_b} "
                        f"estimated Jaccard {est:.3f} >= {dedup.LSH_THRESHOLD}"
                    )
