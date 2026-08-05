import yaml

from triage_lab import ingest

# A tiny CFPB-schema CSV: full header, multiline quoted narrative, an embedded
# quote, one empty-string narrative, one missing (unquoted-empty) narrative,
# and rows intentionally out of complaint_id order to exercise ORDER BY.
FIXTURE_CSV = (
    "Date received,Product,Sub-product,Issue,Sub-issue,Consumer complaint narrative,"
    "Company public response,Company,State,ZIP code,Tags,Submitted via,"
    "Date sent to company,Company response to consumer,Timely response?,Complaint ID\n"
    # complaint 30: multiline + embedded double-quote in narrative
    '2023-07-12,Checking or savings account,Checking account,Managing an account,'
    'Deposits,"Line one of the story.\n'
    'Line two says ""hello"" to the bank.",,Big Bank,TX,78665,,Web,'
    '2023-07-12,Closed with explanation,Yes,30\n'
    # complaint 10: normal single-line narrative
    '2023-10-11,Credit reporting,Credit reporting,Incorrect information,'
    'Belongs to someone else,"Simple narrative here.",,Equifax,NC,28056,,Web,'
    '2023-10-11,Closed,Yes,10\n'
    # complaint 20: empty-string (quoted) narrative -> filtered out
    '2023-05-01,Mortgage,Conventional,Trouble paying,,"",,Some Lender,CA,90001,,Web,'
    '2023-05-01,Closed,Yes,20\n'
    # complaint 40: missing (unquoted empty) narrative -> filtered out
    '2023-06-01,Debt collection,Credit card debt,Attempts to collect,,,,'
    'Collector Inc,NY,10001,,Web,2023-06-01,Closed,Yes,40\n'
    # complaint 5: whitespace-only narrative -> filtered out
    '2023-04-01,Student loan,Federal,Dealing with servicer,,"   ",,Servicer,FL,33101,,Web,'
    '2023-04-01,Closed,Yes,5\n'
)


def _write_fixture(tmp_path):
    csv_path = tmp_path / "complaints.csv"
    csv_path.write_text(FIXTURE_CSV)
    return csv_path


def test_filters_empty_narratives_and_snake_case_schema(tmp_path):
    import duckdb

    csv_path = _write_fixture(tmp_path)
    out_dir = tmp_path / "ingest"
    stats = ingest.run_ingest(csv_path, out_dir)

    parquet_path = out_dir / "narratives.parquet"
    con = duckdb.connect()
    cols = [
        r[0]
        for r in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{parquet_path}')"
        ).fetchall()
    ]
    rows = con.execute(
        f"SELECT complaint_id, date_received, product, sub_product, issue, "
        f"company, state, narrative FROM read_parquet('{parquet_path}') "
        f"ORDER BY complaint_id"
    ).fetchall()
    con.close()

    assert cols == [
        "complaint_id",
        "date_received",
        "product",
        "sub_product",
        "issue",
        "company",
        "state",
        "narrative",
    ]
    # 5 CSV rows in, 3 empty/missing/whitespace narratives dropped -> 2 kept.
    assert stats["total_rows_parsed"] == 5
    assert stats["parser_rejected_lines"] == 0
    assert stats["malformed_key_rows_dropped"] == 0
    assert stats["narrative_rows"] == 2
    assert [r[0] for r in rows] == [10, 30]  # ORDER BY complaint_id
    assert rows[1][7].startswith("Line one")
    assert '"hello"' in rows[1][7]  # embedded quote survived
    assert str(rows[0][1]) == "2023-10-11"  # date_received is a real DATE


def test_malformed_key_row_dropped(tmp_path):
    # A well-formed 16-column row whose "Date received" holds non-date text
    # (the shifted-column corruption class) must be dropped and counted.
    bad_row = (
        '"not a date",Mortgage,Conventional,Trouble paying,,'
        '"Valid narrative text.",,Lender,CA,90001,,Web,'
        '2023-05-01,Closed,Yes,99\n'
    )
    csv_path = tmp_path / "complaints.csv"
    csv_path.write_text(FIXTURE_CSV + bad_row)
    out_dir = tmp_path / "ingest"
    stats = ingest.run_ingest(csv_path, out_dir)

    assert stats["total_rows_parsed"] == 6
    assert stats["malformed_key_rows_dropped"] == 1
    assert stats["narrative_rows"] == 2  # bad-key row excluded despite valid narrative


def test_stats_yaml_written(tmp_path):
    csv_path = _write_fixture(tmp_path)
    out_dir = tmp_path / "ingest"
    stats = ingest.run_ingest(csv_path, out_dir)

    on_disk = yaml.safe_load((out_dir / "ingest_stats.yaml").read_text())
    assert on_disk == stats
    assert on_disk["min_date_received"] == "2023-07-12"
    assert on_disk["max_date_received"] == "2023-10-11"
    assert on_disk["output_sha256"] == stats["output_sha256"]


def test_deterministic_double_run_byte_identical(tmp_path):
    csv_path = _write_fixture(tmp_path)
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"

    stats_a = ingest.run_ingest(csv_path, out_a)
    stats_b = ingest.run_ingest(csv_path, out_b)

    bytes_a = (out_a / "narratives.parquet").read_bytes()
    bytes_b = (out_b / "narratives.parquet").read_bytes()
    assert bytes_a == bytes_b
    assert stats_a["output_sha256"] == stats_b["output_sha256"]


def test_ensure_extracted_from_zip(tmp_path):
    import zipfile

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    csv_path = _write_fixture(src_dir)
    zip_path = tmp_path / "complaints.csv.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(csv_path, arcname="complaints.csv")

    dest = tmp_path / "extracted" / "complaints.csv"
    out = ingest.ensure_extracted(zip_path, dest)
    assert out == dest
    assert dest.read_text() == FIXTURE_CSV
    # Second call is a no-op (skip if present); mtime unchanged.
    first_mtime = dest.stat().st_mtime_ns
    ingest.ensure_extracted(zip_path, dest)
    assert dest.stat().st_mtime_ns == first_mtime
