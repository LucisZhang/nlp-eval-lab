"""Harness tests: bootstrap determinism, paired tests, append-only IO, end-to-end."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from triage_lab import harness
from triage_lab.snapshot import sha256_file

LABELS = ["a", "b", "c"]


def _fixture(n=90, seed=7):
    """A well-behaved multiclass fixture: mostly-correct preds + peaked probs."""
    rng = np.random.default_rng(seed)
    y_true_idx = rng.integers(0, 3, size=n)
    # 80% correct predictions
    flip = rng.random(n) < 0.2
    y_pred_idx = np.where(flip, (y_true_idx + 1) % 3, y_true_idx)
    probs = np.full((n, 3), 0.1)
    probs[np.arange(n), y_pred_idx] = 0.8
    probs = probs / probs.sum(axis=1, keepdims=True)
    y_true = [LABELS[i] for i in y_true_idx]
    y_pred = [LABELS[i] for i in y_pred_idx]
    return y_true, y_pred, probs


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def test_bootstrap_ci_is_deterministic():
    y_true, y_pred, probs = _fixture()
    a = harness.bootstrap_ci(y_true, y_pred, probs, LABELS)
    b = harness.bootstrap_ci(y_true, y_pred, probs, LABELS)
    assert a == b  # byte-identical floats across two calls


def test_bootstrap_ci_contains_point():
    y_true, y_pred, probs = _fixture()
    ci = harness.bootstrap_ci(y_true, y_pred, probs, LABELS)
    for name in ("macro_f1", "balanced_accuracy", "accuracy", "brier", "aurc"):
        m = ci[name]
        assert m["ci_lo"] <= m["point"] <= m["ci_hi"], name


def test_bootstrap_uses_configured_constants():
    assert harness.N_RESAMPLES == 1000
    assert harness.BOOTSTRAP_SEED == 20260805


# ---------------------------------------------------------------------------
# Paired comparisons
# ---------------------------------------------------------------------------

def test_paired_delta_identical_systems_is_zero():
    y_true, y_pred, probs = _fixture()
    d = harness.paired_bootstrap_delta(
        y_true, y_pred, y_pred, probs, probs, "macro_f1", LABELS
    )
    assert d["delta"] == 0.0
    assert d["ci_lo"] == 0.0
    assert d["ci_hi"] == 0.0


def test_paired_delta_sign_and_ci():
    # System A strictly better than B on accuracy -> positive delta.
    y_true = ["a", "a", "b", "b", "c", "c", "a", "b"]
    pred_a = list(y_true)  # perfect
    pred_b = ["b", "a", "b", "c", "c", "a", "a", "b"]  # some errors
    probs = np.tile([0.5, 0.3, 0.2], (len(y_true), 1))
    d = harness.paired_bootstrap_delta(
        y_true, pred_a, pred_b, probs, probs, "accuracy", LABELS
    )
    assert d["delta"] > 0
    assert d["ci_lo"] <= d["delta"] <= d["ci_hi"]


def test_mcnemar_hand_computed():
    # b=5 (A right, B wrong), c=1 (A wrong, B right), plus 2 concordant-correct.
    # n=6, k=1: p = 2*(C(6,0)+C(6,1))*0.5^6 = 2*7/64 = 0.21875
    y_true = ["x"] * 8
    pred_a = ["x", "x", "x", "x", "x", "y", "x", "x"]
    pred_b = ["y", "y", "y", "y", "y", "x", "x", "x"]
    res = harness.mcnemar(y_true, pred_a, pred_b)
    assert res["b"] == 5
    assert res["c"] == 1
    assert res["n_discordant"] == 6
    assert res["p_value"] == pytest.approx(0.21875)


def test_mcnemar_no_discordant_is_p1():
    y_true = ["x", "y", "x"]
    res = harness.mcnemar(y_true, y_true, y_true)
    assert res["n_discordant"] == 0
    assert res["p_value"] == 1.0


# ---------------------------------------------------------------------------
# Append-only JSONL
# ---------------------------------------------------------------------------

def test_append_record_is_append_only(tmp_path):
    path = tmp_path / "runs.jsonl"
    rec1 = {"run_id": "r1", "metrics": {"macro_f1": {"point": 0.5}}}
    rec2 = {"run_id": "r2", "supersedes_run_id": "r1"}
    harness.append_record(path, rec1)
    first_line_after_1 = path.read_text().splitlines()[0]
    harness.append_record(path, rec2)
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    # first line untouched by the second append
    assert lines[0] == first_line_after_1
    assert lines[0] == json.dumps(rec1, sort_keys=True, separators=(",", ":"))
    assert json.loads(lines[1])["run_id"] == "r2"


def test_records_have_sorted_compact_keys(tmp_path):
    path = tmp_path / "runs.jsonl"
    rec = {"z": 1, "a": 2, "m": {"y": 1, "b": 2}}
    harness.append_record(path, rec)
    line = path.read_text().strip()
    assert line == '{"a":2,"m":{"b":2,"y":1},"z":1}'


# ---------------------------------------------------------------------------
# Config hashing
# ---------------------------------------------------------------------------

def test_config_hash_stable_and_matches_file_hash(tmp_path):
    body = "model:\n  runner: dummy\n  name: t\ndata:\n  split: cal\n"
    p1 = tmp_path / "c1.yaml"
    p2 = tmp_path / "c2.yaml"
    p1.write_text(body)
    p2.write_text(body)
    assert harness.config_sha256(p1) == harness.config_sha256(p2)
    assert harness.config_sha256(p1) == sha256_file(p1)


def test_load_config_validates_required_keys(tmp_path):
    good = tmp_path / "good.yaml"
    good.write_text("model:\n  runner: dummy\ndata:\n  split: cal\n")
    cfg = harness.load_config(good)
    assert cfg["model"]["runner"] == "dummy"

    bad = tmp_path / "bad.yaml"
    bad.write_text("model:\n  name: t\ndata:\n  split: cal\n")
    with pytest.raises(ValueError, match="model.runner"):
        harness.load_config(bad)


# ---------------------------------------------------------------------------
# Record schema completeness
# ---------------------------------------------------------------------------

REQUIRED_RECORD_KEYS = {
    "run_id",
    "timestamp_utc",
    "git_sha",
    "config_path",
    "config_sha256",
    "dataset",
    "metrics",
    "bootstrap",
    "wall_clock_seconds",
    "cost_usd",
}


def test_build_record_schema_complete():
    metrics_ci = {"macro_f1": {"point": 0.5, "ci_lo": 0.4, "ci_hi": 0.6}}
    dataset = {"split": "cal", "split_sha256": "deadbeef", "input_sha256": "cafef00d"}
    rec = harness.build_record(
        "configs/x.yaml",
        "abc123",
        metrics_ci,
        dataset,
        1.23,
        None,
        git_sha="feed",
        timestamp_utc="2026-08-05T00:00:00+00:00",
    )
    assert REQUIRED_RECORD_KEYS <= set(rec)
    assert set(rec["dataset"]) == {"split", "split_sha256", "input_sha256"}
    assert rec["bootstrap"] == {
        "n_resamples": harness.N_RESAMPLES,
        "seed": harness.BOOTSTRAP_SEED,
        "method": "percentile",
    }
    # run_id is a deterministic function of (config hash, git sha, timestamp)
    rec2 = harness.build_record(
        "configs/x.yaml", "abc123", metrics_ci, dataset, 9.9, None,
        git_sha="feed", timestamp_utc="2026-08-05T00:00:00+00:00",
    )
    assert rec["run_id"] == rec2["run_id"]


def test_build_record_supersedes_optional():
    rec = harness.build_record(
        "c.yaml", "h", {}, {"split": "cal"}, 0.0, None,
        supersedes_run_id="old", git_sha="g", timestamp_utc="t",
    )
    assert rec["supersedes_run_id"] == "old"


def test_build_record_persists_nonempty_extra():
    # Runner provenance (Tier B checkpoint sha / fitted T / hardware) must survive.
    rec = harness.build_record(
        "c.yaml", "h", {}, {"split": "cal"}, 0.0, None,
        git_sha="g", timestamp_utc="t",
        extra={"temperature": 1.5, "checkpoint_sha256": "abc", "run_type": "smoke-dryrun"},
    )
    assert rec["extra"]["temperature"] == 1.5
    assert rec["extra"]["run_type"] == "smoke-dryrun"


def test_build_record_omits_empty_extra():
    # Tier A supplies no extra -> record schema stays unchanged (no `extra` key).
    rec_empty = harness.build_record(
        "c.yaml", "h", {}, {"split": "cal"}, 0.0, None, git_sha="g", timestamp_utc="t", extra={}
    )
    rec_none = harness.build_record(
        "c.yaml", "h", {}, {"split": "cal"}, 0.0, None, git_sha="g", timestamp_utc="t"
    )
    assert "extra" not in rec_empty
    assert "extra" not in rec_none


# ---------------------------------------------------------------------------
# End-to-end: register a dummy runner, run() through a tmp config.
# ---------------------------------------------------------------------------

@harness.register_runner("dummy_e2e")
def _dummy_runner(config: dict) -> harness.RunnerResult:
    y_true, y_pred, probs = _fixture(n=60, seed=1)
    return harness.RunnerResult(
        y_true=np.array(y_true),
        y_pred=np.array(y_pred),
        probs=probs,
        class_labels=LABELS,
        dataset={"split": config["data"]["split"], "split_sha256": "x", "input_sha256": "y"},
        cost_usd=None,
    )


@harness.register_runner("dummy_extra")
def _dummy_extra_runner(config: dict) -> harness.RunnerResult:
    y_true, y_pred, probs = _fixture(n=40, seed=3)
    return harness.RunnerResult(
        y_true=np.array(y_true), y_pred=np.array(y_pred), probs=probs,
        class_labels=LABELS,
        dataset={"split": config["data"]["split"], "split_sha256": "x", "input_sha256": "y"},
        cost_usd=None,
        extra={"temperature": 2.0, "run_type": "smoke-dryrun"},
    )


def test_run_persists_runner_extra(tmp_path):
    cfg = tmp_path / "run.yaml"
    cfg.write_text("model:\n  runner: dummy_extra\n  name: d\ndata:\n  split: cal\n")
    results = tmp_path / "runs.jsonl"
    record = harness.run(cfg, results)
    on_disk = json.loads(results.read_text().splitlines()[0])
    assert on_disk["extra"] == {"temperature": 2.0, "run_type": "smoke-dryrun"}
    assert record["extra"]["run_type"] == "smoke-dryrun"


def test_run_end_to_end(tmp_path):
    cfg = tmp_path / "run.yaml"
    cfg.write_text("model:\n  runner: dummy_e2e\n  name: dummy\ndata:\n  split: cal\n")
    results = tmp_path / "results" / "runs.jsonl"

    record = harness.run(cfg, results)

    lines = results.read_text().splitlines()
    assert len(lines) == 1
    on_disk = json.loads(lines[0])
    assert on_disk == json.loads(json.dumps(record))  # round-trips
    assert REQUIRED_RECORD_KEYS <= set(on_disk)
    assert on_disk["dataset"]["split"] == "cal"
    m = on_disk["metrics"]["macro_f1"]
    assert set(m) == {"point", "ci_lo", "ci_hi"}
    assert m["ci_lo"] <= m["point"] <= m["ci_hi"]
    assert on_disk["wall_clock_seconds"] >= 0.0
    assert on_disk["cost_usd"] is None


def test_mcnemar_survives_large_discordant_counts():
    """Regression: the exact tail overflows float64 well before TEST-slice sizes.

    At n = 20,000 discordant pairs the exact binomial tail has ~6,000 digits, so the
    naive `2.0 * tail * 0.5**n` raised OverflowError and the test could not be reported
    at all on a full TEST-IID model-vs-model comparison. The log-space fallback must
    return a real p-value, and small-n results must stay bit-identical.
    """
    b, c = 10_500, 9_500
    y_true = ["x"] * (b + c)
    pred_a = ["x"] * b + ["y"] * c
    pred_b = ["y"] * b + ["x"] * c
    res = harness.mcnemar(y_true, pred_a, pred_b)
    assert res["n_discordant"] == b + c
    assert 0.0 < res["p_value"] < 1.0
    assert math.isfinite(res["p_value"])
    # normal approximation sanity: z = (|b-c|-1)/sqrt(b+c) -> p well below 1e-6 here
    z = (abs(b - c) - 1) / math.sqrt(b + c)
    assert res["p_value"] < math.exp(-z * z / 2)

    # exact small-n path is untouched (same value as the textbook formula)
    small = harness.mcnemar(["x"] * 6, ["x"] * 5 + ["y"], ["y"] * 5 + ["x"])
    assert small["p_value"] == 0.21875


@pytest.mark.parametrize("n", [1073, 1074, 1075, 1076, 2000])
def test_mcnemar_extreme_imbalance_does_not_silently_underflow(n):
    """Regression: `0.5**n` is 0.0 for n > 1074 and raises nothing.

    With min(b, c) = 0 the tail is 1, so the float expression `2.0 * 1 * 0.5**n` silently
    produced p = 0.0 for every strongly imbalanced comparison on more than ~1k discordant
    pairs — a fabricated "infinitely significant" result on a slice this repo routinely
    evaluates. The value must be the true 2^-(n-1), reported down to the smallest
    representable double and only then flushing to zero.
    """
    y_true = ["x"] * n
    res = harness.mcnemar(y_true, ["x"] * n, ["y"] * n)
    assert res["b"] == n and res["c"] == 0
    expected = 2.0 ** (-(n - 1))          # exact p = 2 * C(n,0) * 2^-n
    if expected > 0.0:                    # representable as a (sub)normal double
        assert res["p_value"] == pytest.approx(expected, rel=1e-12)
        assert res["p_value"] > 0.0
    else:
        assert res["p_value"] == 0.0      # genuinely below 2^-1074


def test_mcnemar_matches_the_exact_float_formula_wherever_it_is_defined():
    """The fast path must not have moved: same values as `2*Σ C(n,i)*0.5**n` for small n."""
    for b in range(0, 40, 3):
        for c in range(0, 40, 3):
            n = b + c
            if n == 0:
                continue
            k = min(b, c)
            expected = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1))
                           * (0.5**n))
            got = harness.mcnemar(["x"] * n, ["x"] * b + ["y"] * c,
                                  ["y"] * b + ["x"] * c)["p_value"]
            assert got == expected, (b, c)
