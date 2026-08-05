"""Tier A runner tests on small synthetic parquet fixtures.

No real data or real TEST-* splits are touched: every fixture is a throwaway
parquet under tmp_path with an accompanying splits_stats.yaml carrying the
correct sha256, so the runner's integrity gate is exercised for real.
"""

from __future__ import annotations

import json
import subprocess
import sys

import duckdb
import numpy as np
import pytest
import yaml

from triage_lab import harness, tier_a
from triage_lab.snapshot import sha256_file

# Two linearly separable classes with disjoint vocabularies + a little noise.
_ALPHA_WORDS = "apple banana cherry apricot avocado almond"
_BETA_WORDS = "zebra yak walrus wombat vulture urchin"


def _make_rows(n_per_class: int, start_id: int, seed: int):
    rng = np.random.default_rng(seed)
    ids, texts, labels = [], [], []
    cid = start_id
    for cls, vocab in (("alpha", _ALPHA_WORDS), ("beta", _BETA_WORDS)):
        toks = vocab.split()
        for _ in range(n_per_class):
            k = int(rng.integers(4, 8))
            words = rng.choice(toks, size=k, replace=True)
            texts.append(" ".join(words))
            labels.append(cls)
            ids.append(cid)
            cid += 1
    return ids, texts, labels


def _write_parquet(path, ids, texts, labels):
    con = duckdb.connect()
    try:
        con.execute("SET threads=1")
        con.register(
            "t",
            {
                "complaint_id": np.array(ids, dtype=np.int64),
                "narrative": np.array(texts, dtype=object),
                "class": np.array(labels, dtype=object),
            },
        )
        con.execute(
            f"COPY (SELECT * FROM t ORDER BY complaint_id) TO '{path}' (FORMAT PARQUET)"
        )
    finally:
        con.close()


def _build_splits(tmp_path):
    """Materialize train/cal/test parquets + a matching splits_stats.yaml."""
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    specs = {"train": (30, 1000, 1), "cal": (15, 5000, 2), "test": (15, 9000, 3)}
    split_stats = {}
    for name, (n, start, seed) in specs.items():
        path = splits_dir / f"{name}.parquet"
        ids, texts, labels = _make_rows(n, start, seed)
        _write_parquet(path, ids, texts, labels)
        split_stats[name] = {"sha256": sha256_file(path)}
    stats = {"input_sha256": "synthetic-input", "splits": split_stats}
    (splits_dir / "splits_stats.yaml").write_text(yaml.safe_dump(stats))
    return splits_dir


def _features(char_enabled=True):
    return {
        "word": {
            "enabled": True,
            "ngram_range": [1, 2],
            "min_df": 1,
            "max_features": None,
            "sublinear_tf": True,
        },
        "char": {
            "enabled": char_enabled,
            "ngram_range": [3, 5],
            "min_df": 1,
            "max_features": None,
            "sublinear_tf": True,
        },
    }


def _config(splits_dir, *, split="test", family="logreg", calibration="none", char_enabled=True):
    if family == "logreg":
        params = {"C": 1.0, "max_iter": 2000, "tol": 1.0e-4}
    else:
        params = {"alpha": 0.3, "norm": False}
    return {
        "model": {"runner": "tier_a", "name": "t", "family": family, "params": params},
        "data": {
            "split": split,
            "train_split": "train",
            "cal_split": "cal",
            "splits_dir": str(splits_dir),
            "text_column": "narrative",
            "label_column": "class",
            "order_column": "complaint_id",
        },
        "features": _features(char_enabled),
        "calibration": calibration,
        "seed": 20260805,
    }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_runner_registered():
    assert "tier_a" in harness.RUNNERS
    assert harness.RUNNERS["tier_a"] is tier_a.tier_a_runner


# ---------------------------------------------------------------------------
# Feature switch actually changes the feature space
# ---------------------------------------------------------------------------

def test_char_block_changes_feature_space():
    corpus = ["apple banana", "zebra yak walrus", "cherry apricot avocado"]
    word_only = tier_a.build_features(_features(char_enabled=False))
    word_char = tier_a.build_features(_features(char_enabled=True))
    n_word = word_only.fit_transform(corpus).shape[1]
    n_wordchar = word_char.fit_transform(corpus).shape[1]
    assert n_wordchar > n_word  # char_wb adds a whole feature block


def test_disabling_all_blocks_raises():
    cfg = _features(char_enabled=False)
    cfg["word"]["enabled"] = False
    with pytest.raises(ValueError, match="at least one"):
        tier_a.build_features(cfg)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_determinism_same_config_same_predictions(tmp_path):
    splits_dir = _build_splits(tmp_path)
    cfg = _config(splits_dir)
    _, p1, pr1, labels1 = tier_a.fit_predict(cfg)
    _, p2, pr2, labels2 = tier_a.fit_predict(cfg)
    assert labels1 == labels2
    assert np.array_equal(p1, p2)
    assert np.array_equal(pr1, pr2)


# ---------------------------------------------------------------------------
# Calibration produces a valid probability simplex
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family", ["logreg", "complement_nb"])
def test_isotonic_probs_are_valid_simplex(tmp_path, family):
    splits_dir = _build_splits(tmp_path)
    cfg = _config(splits_dir, family=family, calibration="isotonic")
    _, _, probs, labels = tier_a.fit_predict(cfg)
    assert probs.shape[1] == len(labels)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-9)


def test_uncalibrated_probs_are_valid_simplex(tmp_path):
    splits_dir = _build_splits(tmp_path)
    cfg = _config(splits_dir, calibration="none")
    _, _, probs, _ = tier_a.fit_predict(cfg)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-9)


# ---------------------------------------------------------------------------
# Integrity gate
# ---------------------------------------------------------------------------

def test_integrity_gate_rejects_tampered_stats(tmp_path):
    splits_dir = _build_splits(tmp_path)
    stats_path = splits_dir / "splits_stats.yaml"
    stats = yaml.safe_load(stats_path.read_text())
    stats["splits"]["test"]["sha256"] = "0" * 64  # pretend the parquet drifted
    stats_path.write_text(yaml.safe_dump(stats))
    with pytest.raises(ValueError, match="integrity check failed"):
        tier_a.fit_predict(_config(splits_dir))


# ---------------------------------------------------------------------------
# End-to-end through harness.run with a tmp results file
# ---------------------------------------------------------------------------

def test_end_to_end_through_harness(tmp_path):
    splits_dir = _build_splits(tmp_path)
    cfg = _config(splits_dir, calibration="isotonic")
    cfg_path = tmp_path / "run.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    results = tmp_path / "results" / "runs.jsonl"

    record = harness.run(cfg_path, results)

    lines = results.read_text().splitlines()
    assert len(lines) == 1
    on_disk = json.loads(lines[0])
    assert on_disk["dataset"]["split"] == "test"
    assert on_disk["dataset"]["input_sha256"] == "synthetic-input"
    assert on_disk["cost_usd"] is None
    m = on_disk["metrics"]["macro_f1"]
    assert m["ci_lo"] <= m["point"] <= m["ci_hi"]
    assert record["run_id"] == on_disk["run_id"]


def test_cli_module_resolves_tier_a_runner(tmp_path):
    """Exercise the real `python -m triage_lab.harness` lazy-load path.

    This is the regression guard for the `python -m` double-module bug: the CLI
    must resolve the `tier_a` runner via `_load_optional_runners()` without any
    prior manual import of triage_lab.tier_a in the subprocess.
    """
    splits_dir = _build_splits(tmp_path)
    cfg = _config(splits_dir, calibration="none")
    cfg_path = tmp_path / "run.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    results = tmp_path / "cli_runs.jsonl"

    proc = subprocess.run(
        [sys.executable, "-m", "triage_lab.harness", str(cfg_path), "--results", str(results)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, f"CLI failed:\nstdout={proc.stdout}\nstderr={proc.stderr}"
    assert "unknown runner" not in proc.stderr
    lines = results.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["dataset"]["split"] == "test"
