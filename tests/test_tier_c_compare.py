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
