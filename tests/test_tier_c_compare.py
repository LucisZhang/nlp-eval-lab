"""Tests for the read-only paired zero-shot-vs-few-shot comparison (tier_c_compare).

No network, no real data required for the pure paths: receipt parsing (incl. a parse_failed
line -> fallback), the paired-comparison core on synthetic gold labels, and the id-set
mismatch failure. One end-to-end main() test builds a tiny synthetic split parquet so the
CLI's y_true loader is exercised too.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from triage_lab import tier_c_compare

LABELS = [
    "card",
    "credit_reporting",
    "debt_collection",
    "deposit_account",
    "money_service",
    "mortgage",
    "payday_personal_loan",
    "student_loan",
    "vehicle_loan",
]
FALLBACK = LABELS[0]


def _write_receipts(path, rows):
    """rows: list of (complaint_id, content, parse_failed)."""
    lines = []
    for cid, content, failed in rows:
        lines.append(json.dumps({
            "complaint_id": cid, "content": content, "parse_failed": failed,
            "slug": "anthropic/claude-haiku-4.5", "provider": "Anthropic",
        }))
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Receipt parsing incl. parse_failed -> fallback
# ---------------------------------------------------------------------------

def test_read_receipt_predictions_incl_parse_failed_fallback(tmp_path):
    p = tmp_path / "a.jsonl"
    _write_receipts(p, [
        (10, '{"label": "mortgage"}', False),
        (11, '{"label": "card"}', False),
        (12, "garbage not json", True),          # parse_failed -> fallback
        (13, '{"label": "not_a_class"}', False),  # unknown label -> fallback (defensive)
    ])
    preds = tier_c_compare.read_receipt_predictions(p, LABELS, FALLBACK)
    assert preds == {10: "mortgage", 11: "card", 12: FALLBACK, 13: FALLBACK}


def test_read_receipt_predictions_rejects_duplicate_ids(tmp_path):
    p = tmp_path / "dup.jsonl"
    _write_receipts(p, [(10, '{"label": "card"}', False), (10, '{"label": "mortgage"}', False)])
    with pytest.raises(ValueError, match="duplicate complaint_id"):
        tier_c_compare.read_receipt_predictions(p, LABELS, FALLBACK)


# ---------------------------------------------------------------------------
# Paired comparison core
# ---------------------------------------------------------------------------

def test_compare_produces_paired_deltas_and_mcnemar():
    ids = list(range(20))
    # Gold, few-shot (A), zero-shot (B): A perfect, B wrong on the odd ids.
    id_to_true = {i: LABELS[i % len(LABELS)] for i in ids}
    preds_a = dict(id_to_true)
    preds_b = {i: (id_to_true[i] if i % 2 == 0 else "card") for i in ids}

    report = tier_c_compare.compare(
        preds_a, preds_b, id_to_true, LABELS,
        receipts_a="a.jsonl", receipts_b="b.jsonl", split="cal",
    )
    assert report["n_examples"] == 20
    assert set(report["deltas"]) == {"accuracy", "macro_f1"}
    # A beats B -> positive accuracy delta (A - B).
    assert report["deltas"]["accuracy"]["delta"] > 0
    assert {"delta", "ci_lo", "ci_hi"} <= set(report["deltas"]["accuracy"])
    mc = report["mcnemar"]
    assert set(mc) == {"b", "c", "n_discordant", "p_value"}
    # A (perfect) is right & B wrong exactly where B's forced "card" != gold; B is never
    # right where A is wrong (A is perfect), so c == 0.
    expected_b = sum(1 for i in ids if i % 2 == 1 and id_to_true[i] != "card")
    assert mc["b"] == expected_b and mc["c"] == 0
    assert mc["n_discordant"] == expected_b
    assert report["arm_a_receipts"] == "a.jsonl" and report["arm_b_receipts"] == "b.jsonl"


def test_compare_fails_loud_on_mismatched_id_sets():
    id_to_true = {i: LABELS[i % len(LABELS)] for i in range(5)}
    preds_a = {i: "card" for i in range(5)}
    preds_b = {i: "card" for i in range(1, 6)}   # id 5 present, id 0 missing
    with pytest.raises(ValueError, match="id sets differ"):
        tier_c_compare.compare(
            preds_a, preds_b, id_to_true, LABELS,
            receipts_a="a", receipts_b="b", split="cal",
        )


def test_one_hot_shape_and_placement():
    oh = tier_c_compare._one_hot(["card", "mortgage"], LABELS)
    assert oh.shape == (2, len(LABELS))
    assert oh[0, LABELS.index("card")] == 1.0
    assert oh[1, LABELS.index("mortgage")] == 1.0
    assert oh.sum() == 2.0


# ---------------------------------------------------------------------------
# End-to-end CLI over synthetic receipts + synthetic split parquet
# ---------------------------------------------------------------------------

def _synthetic_cal_parquet(splits_dir, id_to_true):
    import duckdb

    splits_dir.mkdir(parents=True, exist_ok=True)
    ids = np.array(sorted(id_to_true), dtype=np.int64)
    narr = np.array([f"complaint {i}" for i in ids], dtype=object)
    cls = np.array([id_to_true[int(i)] for i in ids], dtype=object)
    con = duckdb.connect()
    try:
        con.execute("SET threads=1")
        con.register("t", {"complaint_id": ids, "narrative": narr, "class": cls})
        con.execute(
            f"COPY (SELECT * FROM t ORDER BY complaint_id) "
            f"TO '{splits_dir / 'cal.parquet'}' (FORMAT PARQUET, COMPRESSION 'snappy')"
        )
    finally:
        con.close()


def test_main_end_to_end_reads_split_and_prints_report(tmp_path, capsys):
    ids = list(range(100, 108))
    id_to_true = {i: LABELS[i % len(LABELS)] for i in ids}
    splits_dir = tmp_path / "splits"
    _synthetic_cal_parquet(splits_dir, id_to_true)

    a = tmp_path / "few.jsonl"
    b = tmp_path / "zero.jsonl"
    _write_receipts(a, [(i, json.dumps({"label": id_to_true[i]}), False) for i in ids])
    # zero-shot wrong on one id + one genuinely parse_failed row -> fallback.
    b_rows = []
    for i in ids:
        if i == 103:
            b_rows.append((i, "junk", True))                       # parse_failed -> fallback
        elif i == 105:
            b_rows.append((i, json.dumps({"label": "card"}), False))  # wrong
        else:
            b_rows.append((i, json.dumps({"label": id_to_true[i]}), False))
    _write_receipts(b, b_rows)

    before = sorted(p.name for p in tmp_path.rglob("*"))
    rc = tier_c_compare.main([str(a), str(b), "--split", "cal", "--splits-dir", str(splits_dir)])
    after = sorted(p.name for p in tmp_path.rglob("*"))
    assert rc == 0
    assert before == after                    # strictly read-only: nothing written

    report = json.loads(capsys.readouterr().out)
    assert report["n_examples"] == len(ids)
    assert report["split"] == "cal"
    assert report["deltas"]["accuracy"]["delta"] >= 0   # few-shot >= zero-shot here
    assert report["mcnemar"]["n_discordant"] >= 1


# ---------------------------------------------------------------------------
# The 2026-08-13 additions: --pair-on shared and the committed artifact (--out)
# ---------------------------------------------------------------------------
#
# Both are opt-in. The test above still asserts the default invocation writes NOTHING, so
# the "strictly read-only unless asked" property is pinned by that test, not by these.

def _ablation_fixture(tmp_path, extra_b_ids=()):
    """A tiny paired CAL ablation; `extra_b_ids` land only in arm B (unpaired rows)."""
    ids = list(range(100, 108))
    id_to_true = {i: LABELS[i % len(LABELS)] for i in [*ids, *extra_b_ids]}
    splits_dir = tmp_path / "splits"
    _synthetic_cal_parquet(splits_dir, id_to_true)
    a, b = tmp_path / "few.jsonl", tmp_path / "zero.jsonl"
    _write_receipts(a, [(i, json.dumps({"label": id_to_true[i]}), False) for i in ids])
    _write_receipts(b, [
        (i, json.dumps({"label": "card" if i == 105 else id_to_true[i]}), False)
        for i in [*ids, *extra_b_ids]])
    return splits_dir, a, b, ids


def test_pair_on_shared_intersects_where_exact_refuses(tmp_path, capsys):
    """Arm B carries rows arm A never scored: `exact` fails, `shared` drops them."""
    splits_dir, a, b, ids = _ablation_fixture(tmp_path, extra_b_ids=(200, 201))
    argv = [str(a), str(b), "--split", "cal", "--splits-dir", str(splits_dir)]

    with pytest.raises(ValueError, match="id sets differ"):
        tier_c_compare.main(argv)
    capsys.readouterr()

    assert tier_c_compare.main([*argv, "--pair-on", "shared"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["n_examples"] == len(ids)


def test_pair_on_shared_matches_hand_filtering_the_receipts(tmp_path, capsys):
    """The whole point: the flag reproduces the pre-2026-08-13 temp-file procedure.

    That procedure is what produced the numbers EXPERIMENT_LOG records, so if these two
    paths ever diverge the committed artifacts stop being the logged comparison.
    """
    splits_dir, a, b, ids = _ablation_fixture(tmp_path, extra_b_ids=(200, 201))
    argv = [str(a), str(b), "--split", "cal", "--splits-dir", str(splits_dir)]
    tier_c_compare.main([*argv, "--pair-on", "shared"])
    via_flag = json.loads(capsys.readouterr().out)

    filtered = tmp_path / "zero_filtered.jsonl"
    keep = {str(i) for i in ids}
    filtered.write_text("".join(
        line + "\n" for line in b.read_text().splitlines()
        if str(json.loads(line)["complaint_id"]) in keep))
    tier_c_compare.main([str(a), str(filtered), "--split", "cal",
                         "--splits-dir", str(splits_dir)])
    via_filtering = json.loads(capsys.readouterr().out)

    assert via_flag["deltas"] == via_filtering["deltas"]
    assert via_flag["mcnemar"] == via_filtering["mcnemar"]
    assert via_flag["n_examples"] == via_filtering["n_examples"]


def test_out_writes_a_deterministic_artifact_with_provenance(tmp_path, capsys):
    splits_dir, a, b, ids = _ablation_fixture(tmp_path)
    out = tmp_path / "artifacts" / "demo_key.json"
    argv = [str(a), str(b), "--split", "cal", "--splits-dir", str(splits_dir),
            "--role-a", "arm A", "--role-b", "arm B", "--out", str(out)]

    assert tier_c_compare.main(argv) == 0
    stdout_first = capsys.readouterr().out
    first = out.read_text(encoding="utf-8")
    artifact = json.loads(first)

    assert artifact["schema_version"] == tier_c_compare.SCHEMA_VERSION
    assert artifact["analysis"] == "tier_c_paired_compare"
    assert artifact["key"] == "demo_key"
    assert artifact["pairing"] == "exact"
    assert artifact["n_examples"] == len(ids)
    assert artifact["cost_usd"] == 0.0
    assert artifact["arm_a"]["role"] == "arm A" and artifact["arm_b"]["role"] == "arm B"
    # Receipts outside the results log resolve to a null run id — honest, not invented.
    assert artifact["arm_a"]["run_id"] is None
    assert artifact["bootstrap"] == {
        "method": "percentile", "n_resamples": 1000, "seed": 20260805,
        "ci_pct": [2.5, 97.5], "pairing": "shared_index_vectors_per_replicate"}
    for band in artifact["deltas"].values():
        assert band["excludes_zero"] == (band["ci_lo"] > 0 or band["ci_hi"] < 0)
    assert artifact["repro_command"].startswith(
        "uv run python -m triage_lab.tier_c_compare ")
    # Canonical serialization, same rule as every other committed JSON in the repo.
    assert first == json.dumps(artifact, sort_keys=True, indent=2,
                               ensure_ascii=False) + "\n"

    # Determinism: identical modulo the two stamps every results/ artifact carries.
    tier_c_compare.main(argv)
    assert capsys.readouterr().out == stdout_first, "--out must not change stdout"
    second = json.loads(out.read_text(encoding="utf-8"))
    assert {k: v for k, v in second.items() if k != "generated_at"} == \
        {k: v for k, v in artifact.items() if k != "generated_at"}
    assert not list(out.parent.glob("*.tmp")), "atomic write left a temp file"


def test_out_resolves_run_ids_from_the_committed_receipt_paths():
    """A run id in the artifact is matched against runs.jsonl, never transcribed."""
    from triage_lab import harness, predictions

    records = predictions.load_records(harness.DEFAULT_RESULTS_PATH)
    with_logs = [r for r in records if (r.get("extra") or {}).get("raw_log_path")]
    assert with_logs, "no Tier C run logs receipts"
    record = with_logs[0]
    resolved = tier_c_compare.resolve_run_id(
        harness.REPO_ROOT / record["extra"]["raw_log_path"], records)
    assert resolved["run_id"] == record["run_id"]
    assert resolved["model_slug"] == record["extra"]["model_slug"]
    assert tier_c_compare.resolve_run_id("/nowhere/calls.jsonl", records)["run_id"] is None


def test_committed_tier_c_compare_artifacts_are_the_logged_comparisons():
    """The three artifacts the case study cites, checked against their own provenance.

    Values are not re-derived here (that needs data/); what is pinned is that each file is
    the comparison it claims to be, on the slice it claims, between two runs that exist.
    """
    from triage_lab import harness, predictions

    known = {r["run_id"] for r in predictions.load_records(harness.DEFAULT_RESULTS_PATH)}
    expected = {
        "sonnet_minus_haiku__test_iid": ("test_iid", "shared", False),
        "sonnet_minus_haiku__test_postcutoff": ("test_postcutoff", "shared", True),
        "haiku_fewshot_minus_zeroshot__cal": ("cal", "exact", False),
    }
    for key, (split, pairing, excludes_zero) in expected.items():
        path = tier_c_compare.DEFAULT_OUT_DIR / f"{key}.json"
        assert path.is_file(), f"missing committed artifact {path}"
        artifact = json.loads(path.read_text(encoding="utf-8"))
        assert artifact["key"] == key
        assert artifact["split"] == split
        assert artifact["pairing"] == pairing
        assert artifact["n_examples"] == 1500
        assert artifact["deltas"]["macro_f1"]["excludes_zero"] is excludes_zero
        for arm in ("arm_a", "arm_b"):
            assert artifact[arm]["run_id"] in known, f"{key}.{arm} names an unknown run"
        assert artifact["cost_usd"] == 0.0
        assert "runs.jsonl is untouched" in artifact["cost_note"]
