"""Risk-coverage tests: threshold table vs brute force, tie handling, degenerate one-hot,
AURC-from-table == metrics.aurc_from_codes, bootstrap determinism + harness equality."""

from __future__ import annotations

import numpy as np
import pytest

from triage_lab import harness, metrics, risk_coverage

# ---------------------------------------------------------------------------
# Threshold-domain table vs a brute-force reference at every distinct threshold
# ---------------------------------------------------------------------------

def _brute_force_table(p_max, correct):
    p_max = np.asarray(p_max, float)
    correct = np.asarray(correct, float)
    n = len(p_max)
    rows = []
    for tau in np.unique(p_max)[::-1]:
        mask = p_max >= tau
        nc = int(mask.sum())
        acc = correct[mask].mean()
        rows.append((float(tau), nc, nc / n, float(acc)))
    return rows


def test_threshold_table_matches_brute_force():
    rng = np.random.default_rng(0)
    p_max = rng.random(200)
    correct = (rng.random(200) < 0.7).astype(float)
    table = risk_coverage.threshold_table(p_max, correct)  # full resolution
    ref = _brute_force_table(p_max, correct)
    assert len(table) == len(ref)
    for row, (tau, nc, cov, acc) in zip(table, ref, strict=True):
        assert row["tau"] == tau
        assert row["n_covered"] == nc
        assert row["coverage"] == cov
        assert row["selective_accuracy"] == acc
        assert row["selective_risk"] == 1.0 - acc


def test_threshold_table_tie_handling():
    # Duplicate p_max values collapse to one threshold row covering the whole tie block.
    p_max = np.array([0.9, 0.9, 0.5, 0.5, 0.5])
    correct = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
    table = risk_coverage.threshold_table(p_max, correct)
    assert [r["tau"] for r in table] == [0.9, 0.5]
    assert [r["n_covered"] for r in table] == [2, 5]
    assert table[0]["selective_accuracy"] == 0.5   # 1 of 2 correct at tau=0.9
    assert table[1]["selective_accuracy"] == 0.6   # 3 of 5 correct at tau=0.5
    assert table[1]["coverage"] == 1.0


def test_threshold_table_downsample_keeps_real_endpoints():
    rng = np.random.default_rng(1)
    p_max = rng.random(1000)
    correct = (rng.random(1000) < 0.6).astype(float)
    full = risk_coverage.threshold_table(p_max, correct)
    ds = risk_coverage.threshold_table(p_max, correct, max_points=64)
    assert len(ds) <= 64 < len(full)
    # endpoints preserved and every downsampled tau is a real breakpoint
    real_taus = {r["tau"] for r in full}
    assert all(r["tau"] in real_taus for r in ds)
    assert ds[0]["tau"] == full[0]["tau"]
    assert ds[-1]["tau"] == full[-1]["tau"]


# ---------------------------------------------------------------------------
# Degenerate one-hot (Tier C): coverage can only be 0 or 1
# ---------------------------------------------------------------------------

def test_degenerate_one_hot_single_full_coverage_row():
    p_max = np.ones(6)  # every LLM decision has confidence 1.0
    correct = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    table = risk_coverage.threshold_table(p_max, correct)
    assert len(table) == 1
    assert table[0]["tau"] == 1.0
    assert table[0]["coverage"] == 1.0
    assert table[0]["selective_accuracy"] == correct.mean()


# ---------------------------------------------------------------------------
# AURC-from-table domain == metrics.aurc_from_codes (property test)
# ---------------------------------------------------------------------------

def _codes_and_probs(rng, n, k):
    probs = rng.dirichlet(np.ones(k), size=n)
    true_idx = rng.integers(0, k, size=n)
    return true_idx, probs


def test_aurc_equals_metrics_on_random_inputs():
    for seed in range(8):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(20, 300))
        k = int(rng.integers(2, 6))
        true_idx, probs = _codes_and_probs(rng, n, k)
        pred = probs.argmax(axis=1)
        p_max = probs.max(axis=1)
        correct = (pred == true_idx).astype(float)
        got = risk_coverage.aurc(p_max, correct)
        want = metrics.aurc_from_codes(true_idx, probs)
        assert got == want, seed


def test_accuracy_at_coverage_rejects_out_of_domain_requests():
    # c = 0 must raise, not silently clamp to the single most-confident example.
    p_max = np.array([0.9, 0.6, 0.3])
    correct = np.array([1.0, 0.0, 1.0])
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="0 < c <= 1"):
            risk_coverage.accuracy_at_coverage(p_max, correct, (bad,))
    # the boundary c = 1 is in-domain (full coverage)
    assert risk_coverage.accuracy_at_coverage(p_max, correct, (1.0,))["1.00"] == correct.mean()
    # and the guard propagates through the bootstrap entry point
    with pytest.raises(ValueError, match="0 < c <= 1"):
        risk_coverage.bootstrap_summary(p_max, correct, (0.0,), n_resamples=2)


def test_accuracy_at_coverage_equals_metrics():
    rng = np.random.default_rng(3)
    n, k = 137, 4
    true_idx, probs = _codes_and_probs(rng, n, k)
    pred = probs.argmax(axis=1)
    p_max = probs.max(axis=1)
    correct = (pred == true_idx).astype(float)
    got = risk_coverage.accuracy_at_coverage(p_max, correct)
    want = metrics.accuracy_at_coverage_from_codes(true_idx, probs)
    assert got == want


# ---------------------------------------------------------------------------
# Bootstrap determinism + byte-equality with harness bootstrap_ci
# ---------------------------------------------------------------------------

def test_bootstrap_summary_is_deterministic():
    rng = np.random.default_rng(4)
    p_max = rng.random(150)
    correct = (rng.random(150) < 0.65).astype(float)
    a = risk_coverage.bootstrap_summary(p_max, correct)
    b = risk_coverage.bootstrap_summary(p_max, correct)
    assert a == b


def test_bootstrap_summary_matches_harness_bands():
    # The (p_max, correct) bootstrap must reproduce harness.bootstrap_ci's aurc /
    # acc_at_cov::* bands byte-for-byte (same RNG sequence + metric conventions).
    rng = np.random.default_rng(9)
    n, k = 220, 4
    labels = [f"c{i}" for i in range(k)]
    probs = rng.dirichlet(np.ones(k), size=n)
    true_idx = rng.integers(0, k, size=n)
    pred_idx = probs.argmax(axis=1)
    y_true = np.array([labels[i] for i in true_idx], dtype=object)
    y_pred = np.array([labels[i] for i in pred_idx], dtype=object)

    harness_ci = harness.bootstrap_ci(y_true, y_pred, probs, labels)
    p_max = probs.max(axis=1)
    correct = (pred_idx == true_idx).astype(float)
    summary = risk_coverage.bootstrap_summary(p_max, correct)

    for key in ("aurc", "acc_at_cov::0.50", "acc_at_cov::0.80",
                "acc_at_cov::0.90", "acc_at_cov::0.95"):
        assert summary[key] == harness_ci[key], key


# ---------------------------------------------------------------------------
# build_table end-to-end from an artifact (round-trips through DuckDB)
# ---------------------------------------------------------------------------

def test_build_table_from_artifact(tmp_path):
    from triage_lab import predictions

    rng = np.random.default_rng(6)
    n, k = 80, 3
    labels = [f"c{i}" for i in range(k)]
    probs = rng.dirichlet(np.ones(k), size=n)
    pred_idx = probs.argmax(axis=1)
    flip = rng.random(n) < 0.25
    true_idx = np.where(flip, (pred_idx + 1) % k, pred_idx)
    ids = np.arange(n, dtype=np.int64)
    y_pred = np.array([labels[i] for i in pred_idx], dtype=object)
    y_true = np.array([labels[i] for i in true_idx], dtype=object)
    prov = predictions.ArtifactProvenance(
        run_id="rc" * 10, config_sha256="c", split="cal",
        split_sha256="s", class_labels=labels,
    )
    path = tmp_path / "a.parquet"
    predictions.write_artifact(
        path, ids=ids, y_true=y_true, y_pred=y_pred, probs=probs,
        class_labels=labels, provenance=prov,
    )
    art = predictions.read_artifact(path)
    obj = risk_coverage.build_table(art, max_points=32)
    assert obj["run_id"] == "rc" * 10
    assert obj["split"] == "cal"
    assert obj["n_examples"] == n
    assert obj["class_labels"] == labels
    assert set(obj["summary"]) == {
        "aurc", "acc_at_cov::0.50", "acc_at_cov::0.80",
        "acc_at_cov::0.90", "acc_at_cov::0.95",
    }
    # summary AURC agrees with metrics on the same data
    want = metrics.aurc_from_codes(true_idx, probs)
    assert obj["summary"]["aurc"]["point"] == round(want, risk_coverage.JSON_ROUND)
    # deterministic JSON: writing twice is byte-identical
    p1 = risk_coverage.write_table_json(obj, tmp_path / "out1.json")
    p2 = risk_coverage.write_table_json(risk_coverage.build_table(art, max_points=32),
                                        tmp_path / "out2.json")
    assert p1.read_text() == p2.read_text()
