from datetime import date

import duckdb
import yaml

from triage_lab import dedup

# A shared 30-word prefix with two disjoint 30-word suffixes: ~30% shingle
# Jaccard, comfortably below any band-collision probability at 8x16 bands.
_BORDER_PREFIX = " ".join(f"word{i}" for i in range(30))
_BORDER_A = _BORDER_PREFIX + " " + " ".join(f"alpha{i}" for i in range(30))
_BORDER_B = _BORDER_PREFIX + " " + " ".join(f"beta{i}" for i in range(30))

# (complaint_id, date_received, narrative). Groups, in order:
#   exact-dup pair (rep by earliest date), near-dup mask trio (rep by earliest
#   date), case/punct-only pair, same-date tie-break pair (rep by smallest id),
#   <5-token row, empty-after-normalization row, borderline non-merging pair,
#   distinct singletons.
FIXTURE_ROWS = [
    # exact-dup pair -> rep 11 (earlier date)
    (10, "2023-03-01", "the bank charged me an overdraft fee of thirty five dollars without any prior notice or warning"),
    (11, "2023-01-01", "the bank charged me an overdraft fee of thirty five dollars without any prior notice or warning"),
    # near-dup mask trio (masks differ only, all collapse to one token) -> rep 21
    (20, "2022-05-01", "on XXXX i contacted the company about an unauthorized charge on my credit card account that i never approved"),
    (21, "2022-04-01", "on XXXXXXXX i contacted the company about an unauthorized charge on my credit card account that i never approved"),
    (22, "2022-06-01", "on XX i contacted the company about an unauthorized charge on my credit card account that i never approved"),
    # case/punct-only pair -> rep 30 (earlier date)
    (30, "2021-01-01", "The Mortgage Servicer FAILED to apply my payment, correctly for several months; in a row!"),
    (31, "2021-06-01", "the mortgage servicer failed to apply my payment correctly for several months in a row"),
    # same-date tie-break pair -> rep 40 (smallest id)
    (40, "2020-07-07", "a debt collector called me repeatedly at work despite my written request to stop all contact"),
    (41, "2020-07-07", "a debt collector called me repeatedly at work despite my written request to stop all contact"),
    # <5-token row
    (50, "2019-02-02", "loan payment lost"),
    # empty after normalization (pure punctuation)
    (55, "2019-03-03", "!!! ???"),
    # borderline ~30%-overlap pair that must NOT merge
    (60, "2018-01-01", _BORDER_A),
    (61, "2018-01-02", _BORDER_B),
    # distinct singletons
    (70, "2017-01-01", "my student loan servicer transferred my account to a new company without notifying me first"),
    (71, "2017-02-01", "the credit bureau reported a bankruptcy that was discharged many years ago as still open"),
    (72, "2017-03-01", "a prepaid card provider froze my funds and refused to explain the reason for the hold"),
    (73, "2017-04-01", "my auto lender repossessed the vehicle after a single missed payment during a hardship period"),
    (74, "2017-05-01", "the payday lender kept withdrawing money from my checking account after the loan was fully repaid"),
    (75, "2017-06-01", "an insurance company denied my valid claim citing a policy exclusion that does not apply here"),
    (76, "2017-07-01", "the money transfer service lost my wire and support has been unresponsive for several weeks now"),
    (77, "2017-08-01", "my landlord reported inaccurate rental history to a tenant screening agency causing a denied application"),
]

# kept representatives per the rules above.
EXPECTED_KEPT = {11, 21, 30, 40, 50, 55, 60, 61, 70, 71, 72, 73, 74, 75, 76, 77}
# removed -> its cluster representative's complaint_id
EXPECTED_REMOVED_TO_CLUSTER = {10: 11, 20: 21, 22: 21, 31: 30, 41: 40}


def _write_fixture(tmp_path):
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
        "INSERT INTO t VALUES (?, ?, 'Prod', 'SubProd', 'Issue', 'Co', 'ST', ?)",
        [(cid, date.fromisoformat(d), nar) for cid, d, nar in FIXTURE_ROWS],
    )
    con.execute(
        f"""
        COPY (SELECT * FROM t ORDER BY complaint_id) TO '{parquet_path}'
        (FORMAT PARQUET, COMPRESSION 'snappy', ROW_GROUP_SIZE {dedup.ROW_GROUP_SIZE})
        """
    )
    con.close()
    return parquet_path


def test_normalize_collapses_masks_case_and_punct():
    assert dedup.normalize("XXXX") == ["x"]
    assert dedup.normalize("XXXXXXXX") == ["x"]
    assert dedup.normalize("Hello, WORLD!") == ["hello", "world"]
    assert dedup.normalize("thirty-five dollars") == ["thirty", "five", "dollars"]
    # A pure-x mask and a longer pure-x mask normalize identically.
    assert dedup.normalize("paid XXXX today") == dedup.normalize("paid XXXXXXXX today")


def test_short_and_empty_normalized_rows_do_not_crash(tmp_path):
    parquet_path = _write_fixture(tmp_path)
    stats = dedup.run_dedup(parquet_path, tmp_path / "out", workers=1)
    # The single pure-punctuation narrative (id 55) is the one empty-normalized row.
    assert stats["empty_normalized_rows"] == 1


def test_borderline_pair_not_merged():
    # Guard the fixture design: true shingle Jaccard of the borderline pair is
    # well below threshold, so LSH must keep them in separate clusters.
    sh_a = set(dedup._shingles(dedup.normalize(_BORDER_A))[0])
    sh_b = set(dedup._shingles(dedup.normalize(_BORDER_B))[0])
    jaccard = len(sh_a & sh_b) / len(sh_a | sh_b)
    assert jaccard < 0.85


def test_clusters_representatives_and_map(tmp_path):
    parquet_path = _write_fixture(tmp_path)
    out_dir = tmp_path / "out"
    stats = dedup.run_dedup(parquet_path, out_dir, workers=1)

    con = duckdb.connect()
    kept_ids = {
        r[0]
        for r in con.execute(
            f"SELECT complaint_id FROM read_parquet('{out_dir / 'narratives_deduped.parquet'}')"
        ).fetchall()
    }
    dedup_cols = [
        r[0]
        for r in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{out_dir / 'narratives_deduped.parquet'}')"
        ).fetchall()
    ]
    map_rows = con.execute(
        f"SELECT complaint_id, cluster_id, kept "
        f"FROM read_parquet('{out_dir / 'dedup_map.parquet'}') ORDER BY complaint_id"
    ).fetchall()
    con.close()

    assert kept_ids == EXPECTED_KEPT
    # deduped output preserves the full ingest schema.
    assert dedup_cols == [
        "complaint_id", "date_received", "product", "sub_product",
        "issue", "company", "state", "narrative",
    ]
    # dedup_map covers every input row.
    assert [r[0] for r in map_rows] == sorted(cid for cid, _, _ in FIXTURE_ROWS)

    by_id = {cid: (cluster_id, kept) for cid, cluster_id, kept in map_rows}
    for cid in EXPECTED_KEPT:
        cluster_id, kept = by_id[cid]
        assert kept is True
        assert cluster_id == cid  # kept rows are their own cluster representative
    for removed, rep in EXPECTED_REMOVED_TO_CLUSTER.items():
        cluster_id, kept = by_id[removed]
        assert kept is False
        assert cluster_id == rep

    # borderline pair: both kept, distinct clusters.
    assert by_id[60][1] is True and by_id[61][1] is True
    assert by_id[60][0] != by_id[61][0]

    assert stats["output_rows"] == len(EXPECTED_KEPT)
    assert stats["input_rows"] == len(FIXTURE_ROWS)


def test_stats_yaml_and_dedup_rate(tmp_path):
    parquet_path = _write_fixture(tmp_path)
    out_dir = tmp_path / "out"
    stats = dedup.run_dedup(parquet_path, out_dir, workers=1)

    on_disk = yaml.safe_load((out_dir / "dedup_stats.yaml").read_text())
    assert on_disk == stats

    n = stats["input_rows"]
    assert stats["removed_rows"] == n - stats["output_rows"]
    assert stats["dedup_rate"] == (n - stats["output_rows"]) / n
    assert stats["params"]["num_perm"] == dedup.NUM_PERM
    assert stats["params"]["hashfunc"] == "sha1_hash32"


def test_determinism_across_worker_counts(tmp_path):
    parquet_path = _write_fixture(tmp_path)
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"

    stats_a = dedup.run_dedup(parquet_path, out_a, workers=1)
    stats_b = dedup.run_dedup(parquet_path, out_b, workers=2)

    for name in ("narratives_deduped.parquet", "dedup_map.parquet"):
        assert (out_a / name).read_bytes() == (out_b / name).read_bytes()
    assert stats_a["output_sha256"] == stats_b["output_sha256"]
    assert stats_a["dedup_map_sha256"] == stats_b["dedup_map_sha256"]
