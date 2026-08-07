"""Predictions-artifact tests: round-trip, provenance binding, tier_c receipt
reconstruction, the structural + aggregate verify gate, input-hash validation, and
harness auto-persist. All run in a light env (numpy + duckdb only; no pyarrow/openai)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import duckdb
import numpy as np
import pytest

from triage_lab import harness, predictions

# ---------------------------------------------------------------------------
# Artifact round-trip: write -> read -> identical arrays + provenance
# ---------------------------------------------------------------------------

def _sample(n=12, k=3, seed=5):
    rng = np.random.default_rng(seed)
    labels = [f"c{i}" for i in range(k)]
    ids = np.arange(1000, 1000 + n, dtype=np.int64)
    probs = rng.dirichlet(np.ones(k), size=n)
    y_pred_idx = probs.argmax(axis=1)
    flip = rng.random(n) < 0.3
    y_true_idx = np.where(flip, (y_pred_idx + 1) % k, y_pred_idx)
    y_pred = np.array([labels[i] for i in y_pred_idx], dtype=object)
    y_true = np.array([labels[i] for i in y_true_idx], dtype=object)
    return ids, y_true, y_pred, probs, labels


def _provenance(labels, split="cal"):
    return predictions.ArtifactProvenance(
        run_id="deadbeef" * 8,
        config_sha256="cfg123",
        split=split,
        split_sha256="split123",
        class_labels=labels,
        git_sha="git123",
        input_sha256="input123",
        prompt_bundle_sha256="bundle123",
    )


def _write_split(path, ids, labels, *, id_col="complaint_id", label_col="class") -> None:
    """Materialize a minimal stand-in for a frozen split parquet (id + label only)."""
    con = duckdb.connect()
    try:
        con.execute("SET threads=1")
        con.register("_split", {
            id_col: np.asarray(ids, dtype=np.int64),
            label_col: np.asarray([str(v) for v in labels], dtype=object),
        })
        con.execute(f"COPY (SELECT * FROM _split) TO '{path}' (FORMAT parquet)")
    finally:
        con.close()


def _config(tmp_path, runner="tier_a", split="cal") -> dict:
    return {"model": {"runner": runner}, "data": {"split": split, "splits_dir": str(tmp_path)}}


def _record(art, split="cal") -> dict:
    """A record whose logged points are exactly what the artifact recomputes."""
    points = harness.evaluate(art.y_true, art.y_pred, art.probs, art.class_labels)
    return {
        "run_id": "deadbeef" * 8,
        "dataset": {"split": split},
        "metrics": {m: {"point": points[m]} for m in predictions.VERIFY_METRICS},
    }


def test_artifact_round_trip(tmp_path):
    ids, y_true, y_pred, probs, labels = _sample()
    path = tmp_path / "a.parquet"
    predictions.write_artifact(
        path, ids=ids, y_true=y_true, y_pred=y_pred, probs=probs,
        class_labels=labels, provenance=_provenance(labels),
    )
    art = predictions.read_artifact(path)
    assert np.array_equal(art.complaint_id, ids)
    assert list(art.y_true) == list(y_true)
    assert list(art.y_pred) == list(y_pred)
    assert np.allclose(art.probs, probs)
    assert np.allclose(art.p_max, probs.max(axis=1))
    assert art.class_labels == labels
    # provenance bound into the file: code, config, data, and prompt identity
    assert art.provenance["run_id"] == "deadbeef" * 8
    assert art.provenance["config_sha256"] == "cfg123"
    assert art.provenance["git_sha"] == "git123"
    assert art.provenance["split"] == "cal"
    assert art.provenance["split_sha256"] == "split123"
    assert art.provenance["input_sha256"] == "input123"
    assert art.provenance["prompt_bundle_sha256"] == "bundle123"
    assert json.loads(art.provenance["class_labels"]) == labels
    assert art.provenance["schema_version"] == predictions.SCHEMA_VERSION


def test_provenance_defaults_are_empty_not_missing(tmp_path):
    # A tier without a prompt still writes the key (fixed-width metadata schema).
    ids, y_true, y_pred, probs, labels = _sample()
    prov = predictions.ArtifactProvenance(
        run_id="r", config_sha256="c", split="cal", split_sha256="s", class_labels=labels,
    )
    path = tmp_path / "a.parquet"
    predictions.write_artifact(
        path, ids=ids, y_true=y_true, y_pred=y_pred, probs=probs,
        class_labels=labels, provenance=prov,
    )
    art = predictions.read_artifact(path)
    for key in ("git_sha", "input_sha256", "prompt_bundle_sha256"):
        assert art.provenance[key] == ""


def test_artifact_columns_present(tmp_path):
    ids, y_true, y_pred, probs, labels = _sample()
    path = tmp_path / "a.parquet"
    predictions.write_artifact(
        path, ids=ids, y_true=y_true, y_pred=y_pred, probs=probs,
        class_labels=labels, provenance=_provenance(labels),
    )
    import duckdb

    cols = [
        c[0]
        for c in duckdb.connect().execute(
            f"DESCRIBE SELECT * FROM read_parquet('{path}')"
        ).fetchall()
    ]
    assert cols[:4] == ["complaint_id", "y_true", "y_pred", "p_max"]
    assert cols[4:] == [f"prob::{lbl}" for lbl in labels]


# ---------------------------------------------------------------------------
# Tier C reconstruction from receipts (no network)
# ---------------------------------------------------------------------------

LABELS_C = ["card", "credit_reporting", "debt_collection"]


def test_reconstruct_tier_c_labels_parses_and_falls_back():
    ids = [11, 22, 33, 44]
    receipts = {
        11: '{"label": "debt_collection"}',
        22: '{"label": "credit_reporting"}',
        33: "not json at all",             # parse failure -> fallback
        44: '{"label": "unknown_class"}',  # invalid enum -> fallback
    }
    fallback = LABELS_C[0]
    y_pred = predictions.reconstruct_tier_c_labels(ids, receipts, LABELS_C, fallback)
    assert y_pred == ["debt_collection", "credit_reporting", "card", "card"]


def test_reconstruct_tier_c_labels_missing_receipt_fails_loud():
    with pytest.raises(KeyError):
        predictions.reconstruct_tier_c_labels([1, 2], {1: '{"label": "card"}'}, LABELS_C, "card")


def test_onehot_is_degenerate_pmax_one():
    y_pred = ["card", "debt_collection", "card"]
    probs = predictions._onehot(y_pred, LABELS_C)
    assert probs.shape == (3, 3)
    assert np.array_equal(probs.max(axis=1), np.ones(3))
    assert probs[0, 0] == 1.0 and probs[1, 2] == 1.0


def test_load_receipts_by_id(tmp_path):
    raw = tmp_path / "calls.jsonl"
    raw.write_text(
        json.dumps({"complaint_id": 7, "content": '{"label": "card"}'}) + "\n"
        + json.dumps({"complaint_id": 9, "content": None}) + "\n"
    )
    by_id = predictions.load_receipts_by_id(raw)
    assert by_id[7] == '{"label": "card"}'
    assert by_id[9] is None


def test_load_receipts_rejects_duplicate_ids(tmp_path):
    # Two receipts for one id -> reconstruction would depend on line order. Hard error.
    raw = tmp_path / "calls.jsonl"
    raw.write_text(
        json.dumps({"complaint_id": 7, "content": '{"label": "card"}'}) + "\n"
        + json.dumps({"complaint_id": 8, "content": '{"label": "card"}'}) + "\n"
        + json.dumps({"complaint_id": 7, "content": '{"label": "debt_collection"}'}) + "\n"
    )
    with pytest.raises(ValueError, match=r"duplicate complaint_id.*\[7\]"):
        predictions.load_receipts_by_id(raw)


# ---------------------------------------------------------------------------
# Input-hash validation: never stamp a historical hash onto different inputs
# ---------------------------------------------------------------------------

def _write_config(tmp_path, text="model:\n  runner: tier_a\n  name: d\ndata:\n  split: cal\n"):
    cfg = tmp_path / "run.yaml"
    cfg.write_text(text)
    return cfg


def test_load_config_checked_accepts_unchanged_config(tmp_path):
    cfg = _write_config(tmp_path)
    record = {"run_id": "r" * 64, "config_path": str(cfg),
              "config_sha256": harness.config_sha256(cfg)}
    assert predictions.load_config_checked(record)["model"]["runner"] == "tier_a"


def test_load_config_checked_rejects_changed_config(tmp_path):
    cfg = _write_config(tmp_path)
    record = {"run_id": "r" * 64, "config_path": str(cfg),
              "config_sha256": harness.config_sha256(cfg)}
    cfg.write_text(cfg.read_text() + "seed: 1\n")  # config edited after the run
    actual = harness.config_sha256(cfg)
    with pytest.raises(ValueError) as exc:
        predictions.load_config_checked(record)
    msg = str(exc.value)
    assert actual in msg and record["config_sha256"] in msg  # both hashes named


def test_prompt_bundle_hash_mismatch_is_hard_failure():
    bundle = SimpleNamespace(bundle_sha256="abc123")
    record = {"run_id": "r" * 64, "extra": {"prompt_bundle_sha256": "def456"}}
    with pytest.raises(ValueError) as exc:
        predictions._check_prompt_bundle(bundle, record, "v1")
    assert "abc123" in str(exc.value) and "def456" in str(exc.value)
    # matching hash passes silently
    predictions._check_prompt_bundle(bundle, {"extra": {"prompt_bundle_sha256": "abc123"}}, "v1")


def test_missing_prompt_bundle_hash_is_hard_failure():
    bundle = SimpleNamespace(bundle_sha256="abc123")
    with pytest.raises(ValueError, match="prompt_bundle_sha256"):
        predictions._check_prompt_bundle(bundle, {"run_id": "r" * 64, "extra": {}}, "v1")


# ---------------------------------------------------------------------------
# Verification gate catches an injected mismatch
# ---------------------------------------------------------------------------

def _write_and_read(tmp_path, ids, y_true, y_pred, probs, labels, name="v.parquet"):
    path = tmp_path / name
    predictions.write_artifact(
        path, ids=ids, y_true=y_true, y_pred=y_pred, probs=probs,
        class_labels=labels, provenance=_provenance(labels),
    )
    return path, predictions.read_artifact(path)


def _gate(tmp_path, art, path, record, runner="tier_a"):
    rows = predictions.verify_artifact(art, record, _config(tmp_path, runner), art_path=path)
    return rows, {r["check"]: r for r in rows}


def test_verify_gate_passes_on_faithful_record(tmp_path):
    ids, y_true, y_pred, probs, labels = _sample(n=40, seed=2)
    path, art = _write_and_read(tmp_path, ids, y_true, y_pred, probs, labels)
    _write_split(tmp_path / "cal.parquet", ids, y_true)
    rows, by_check = _gate(tmp_path, art, path, _record(art))
    assert all(r["ok"] for r in rows)
    # every layer is represented, structural rows first
    assert [r["kind"] for r in rows[:5]] == ["structural"] * 5
    for name in ("ids_unique_nonnull", "ids_in_split", "y_true_matches_split",
                 "p_max_equals_probs_max", "y_pred_is_argmax"):
        assert by_check[name]["ok"] is True
    assert all(r["abs_delta"] <= predictions.VERIFY_TOL
               for r in rows if r["kind"] == "metric")


def test_verify_gate_catches_injected_mismatch(tmp_path):
    ids, y_true, y_pred, probs, labels = _sample(n=40, seed=2)
    path, art = _write_and_read(tmp_path, ids, y_true, y_pred, probs, labels)
    _write_split(tmp_path / "cal.parquet", ids, y_true)
    record = _record(art)
    # Corrupt one logged point estimate -> gate must flag that metric ✗.
    record["metrics"]["accuracy"]["point"] += 0.05
    _, by_check = _gate(tmp_path, art, path, record)
    assert by_check["accuracy"]["ok"] is False
    assert by_check["macro_f1"]["ok"] is True


# --- the fault the aggregate gate structurally cannot see -------------------

def test_verify_gate_rejects_wrong_id_row_mapping(tmp_path):
    # Same rows, ids rolled by one: every aggregate metric is bit-identical (the old gate
    # passed), but y_true no longer matches the split's label for that id.
    ids, y_true, y_pred, probs, labels = _sample(n=40, seed=2)
    _write_split(tmp_path / "cal.parquet", ids, y_true)
    wrong_ids = np.roll(ids, 1)
    path, art = _write_and_read(tmp_path, wrong_ids, y_true, y_pred, probs, labels)
    record = _record(art)  # points recomputed from the same (unpermuted) rows

    rows, by_check = _gate(tmp_path, art, path, record)
    assert all(r["ok"] for r in rows if r["kind"] == "metric")   # old gate: green
    assert by_check["ids_in_split"]["ok"] is True                # still the same id set
    assert by_check["y_true_matches_split"]["ok"] is False       # new gate: ✗
    assert "disagree" in by_check["y_true_matches_split"]["detail"]
    assert not all(r["ok"] for r in rows)


def test_verify_gate_rejects_ids_outside_split(tmp_path):
    ids, y_true, y_pred, probs, labels = _sample(n=40, seed=2)
    path, art = _write_and_read(tmp_path, ids, y_true, y_pred, probs, labels)
    _write_split(tmp_path / "cal.parquet", ids[:-3], y_true[:-3])  # 3 ids not in the split
    _, by_check = _gate(tmp_path, art, path, _record(art))
    assert by_check["ids_in_split"]["ok"] is False
    assert "3 id(s) absent" in by_check["ids_in_split"]["detail"]


def test_verify_gate_accepts_a_subsampled_slice(tmp_path):
    # eval_rows_cap runs cover part of the split; membership, not equality, is the rule.
    ids, y_true, y_pred, probs, labels = _sample(n=40, seed=2)
    path, art = _write_and_read(tmp_path, ids[:10], y_true[:10], y_pred[:10], probs[:10], labels)
    _write_split(tmp_path / "cal.parquet", ids, y_true)
    _, by_check = _gate(tmp_path, art, path, _record(art))
    assert by_check["ids_in_split"]["ok"] is True
    assert by_check["y_true_matches_split"]["ok"] is True


def test_verify_gate_fails_when_split_is_unavailable(tmp_path):
    # A check that cannot be performed reports ✗ — "not verified" is not "verified".
    ids, y_true, y_pred, probs, labels = _sample(n=8, seed=2)
    path, art = _write_and_read(tmp_path, ids, y_true, y_pred, probs, labels)
    _, by_check = _gate(tmp_path, art, path, _record(art))  # no cal.parquet written
    assert by_check["ids_in_split"]["ok"] is False
    assert by_check["y_true_matches_split"]["ok"] is False
    assert "not found" in by_check["ids_in_split"]["detail"]


def test_verify_gate_fails_when_declared_split_disagrees_with_record(tmp_path):
    ids, y_true, y_pred, probs, labels = _sample(n=8, seed=2)
    path, art = _write_and_read(tmp_path, ids, y_true, y_pred, probs, labels)
    _write_split(tmp_path / "cal.parquet", ids, y_true)
    record = _record(art, split="test_iid")  # artifact says cal
    _, by_check = _gate(tmp_path, art, path, record)
    assert by_check["ids_in_split"]["ok"] is False
    assert "test_iid" in by_check["ids_in_split"]["detail"]


# --- id-column integrity ---------------------------------------------------

def test_check_ids_flags_duplicates(tmp_path):
    ids, y_true, y_pred, probs, labels = _sample(n=6, seed=2)
    dup_ids = ids.copy()
    dup_ids[3] = dup_ids[0]
    path, _ = _write_and_read(tmp_path, dup_ids, y_true, y_pred, probs, labels)
    row = predictions.check_ids(path)
    assert row["ok"] is False
    assert "1 duplicate row(s)" in row["detail"] and str(ids[0]) in row["detail"]


def test_check_ids_flags_nulls(tmp_path):
    # NULL is only representable in the file (read_artifact casts to int64), so the
    # check runs against the parquet.
    path = tmp_path / "nullids.parquet"
    con = duckdb.connect()
    con.execute(
        f"COPY (SELECT * FROM (VALUES (1), (NULL), (3)) t(complaint_id)) "
        f"TO '{path}' (FORMAT parquet)"
    )
    con.close()
    row = predictions.check_ids(path)
    assert row["ok"] is False
    assert "1 NULL" in row["detail"]


# --- probability-column consistency ----------------------------------------

def _artifact(ids, y_true, y_pred, p_max, probs, labels):
    return predictions.PredictionsArtifact(
        complaint_id=np.asarray(ids, dtype=np.int64),
        y_true=np.asarray(y_true, dtype=object),
        y_pred=np.asarray(y_pred, dtype=object),
        p_max=np.asarray(p_max, dtype=np.float64),
        probs=np.asarray(probs, dtype=np.float64),
        class_labels=list(labels),
    )


def test_check_probs_flags_p_max_not_equal_to_row_max():
    labels = ["a", "b"]
    probs = np.array([[0.7, 0.3], [0.2, 0.8]])
    art = _artifact([1, 2], ["a", "b"], ["a", "b"], [0.7, 0.9], probs, labels)
    by_check = {r["check"]: r for r in predictions.check_probs(art, "tier_a")}
    assert by_check["p_max_equals_probs_max"]["ok"] is False
    assert "1/2" in by_check["p_max_equals_probs_max"]["detail"]


def test_check_probs_flags_y_pred_not_argmax():
    labels = ["a", "b"]
    probs = np.array([[0.7, 0.3], [0.2, 0.8]])
    art = _artifact([1, 2], ["a", "b"], ["a", "a"], probs.max(axis=1), probs, labels)
    by_check = {r["check"]: r for r in predictions.check_probs(art, "tier_a")}
    assert by_check["p_max_equals_probs_max"]["ok"] is True
    assert by_check["y_pred_is_argmax"]["ok"] is False


def test_check_probs_tolerates_exact_ties():
    # y_pred carries maximal probability but is not argmax()'s lowest-index winner.
    labels = ["a", "b"]
    probs = np.array([[0.5, 0.5]])
    art = _artifact([1], ["a"], ["b"], [0.5], probs, labels)
    by_check = {r["check"]: r for r in predictions.check_probs(art, "tier_a")}
    assert by_check["y_pred_is_argmax"]["ok"] is True
    assert "1 exact tie(s)" in by_check["y_pred_is_argmax"]["detail"]


def test_check_probs_flags_unknown_y_pred_label():
    labels = ["a", "b"]
    probs = np.array([[0.7, 0.3]])
    art = _artifact([1], ["a"], ["zzz"], [0.7], probs, labels)
    by_check = {r["check"]: r for r in predictions.check_probs(art, "tier_a")}
    assert by_check["y_pred_in_class_labels"]["ok"] is False


def test_check_probs_tier_c_one_hot():
    labels = LABELS_C
    y_pred = ["card", "debt_collection"]
    probs = predictions._onehot(y_pred, labels)
    art = _artifact([1, 2], ["card", "card"], y_pred, [1.0, 1.0], probs, labels)
    by_check = {r["check"]: r for r in predictions.check_probs(art, "tier_c")}
    assert by_check["p_max_is_one"]["ok"] is True
    assert by_check["y_pred_matches_onehot"]["ok"] is True

    # a non-degenerate row, and a hot column that is not y_pred
    soft = np.array([[0.6, 0.4, 0.0], [0.0, 0.0, 1.0]])
    bad = _artifact([1, 2], ["card", "card"], ["card", "card"], [0.6, 1.0], soft, labels)
    by_check = {r["check"]: r for r in predictions.check_probs(bad, "tier_c")}
    assert by_check["p_max_is_one"]["ok"] is False
    assert by_check["y_pred_matches_onehot"]["ok"] is False


# ---------------------------------------------------------------------------
# Harness auto-persist: runner with ids -> artifact + extra.predictions_path
# ---------------------------------------------------------------------------

_AUTOP_LABELS = ["a", "b", "c"]


@harness.register_runner("dummy_ids")
def _dummy_ids_runner(config):
    n = 30
    rng = np.random.default_rng(11)
    y_true_idx = rng.integers(0, 3, size=n)
    flip = rng.random(n) < 0.2
    y_pred_idx = np.where(flip, (y_true_idx + 1) % 3, y_true_idx)
    probs = np.full((n, 3), 0.1)
    probs[np.arange(n), y_pred_idx] = 0.8
    probs = probs / probs.sum(axis=1, keepdims=True)
    return harness.RunnerResult(
        y_true=np.array([_AUTOP_LABELS[i] for i in y_true_idx], dtype=object),
        y_pred=np.array([_AUTOP_LABELS[i] for i in y_pred_idx], dtype=object),
        probs=probs,
        class_labels=_AUTOP_LABELS,
        dataset={"split": config["data"]["split"], "split_sha256": "s", "input_sha256": "i"},
        cost_usd=None,
        # stands in for Tier C's extra block, so the auto-persist path's prompt binding
        # is exercised without needing the real prompt bundle
        extra={"prompt_bundle_sha256": "bundle-abc"},
        ids=np.arange(500, 500 + n, dtype=np.int64),
    )


def test_run_auto_persists_predictions_artifact(tmp_path):
    cfg = tmp_path / "run.yaml"
    cfg.write_text("model:\n  runner: dummy_ids\n  name: d\ndata:\n  split: cal\n")
    results = tmp_path / "runs.jsonl"
    preds_dir = tmp_path / "preds"

    record = harness.run(cfg, results, preds_dir=preds_dir)

    art_path = preds_dir / f"{record['run_id']}.parquet"
    assert art_path.exists()
    assert record["extra"]["predictions_path"].endswith(f"{record['run_id']}.parquet")

    art = predictions.read_artifact(art_path)
    assert len(art) == 30
    assert art.provenance["run_id"] == record["run_id"]
    assert art.provenance["split"] == "cal"
    # the auto-persist path binds the same inputs the record names
    assert art.provenance["config_sha256"] == record["config_sha256"]
    assert art.provenance["git_sha"] == record["git_sha"]
    assert art.provenance["input_sha256"] == "i"
    assert art.provenance["prompt_bundle_sha256"] == "bundle-abc"
    # artifact must reproduce the logged point metrics exactly (the split-join layer is
    # covered above; this dummy runner's rows come from an RNG, not a frozen split)
    rows = predictions.verify_metrics(art, record)
    assert all(r["ok"] for r in rows)
    # and be internally consistent
    assert all(r["ok"] for r in predictions.check_probs(art, "dummy_ids"))
    assert predictions.check_ids(art_path)["ok"] is True


# ---------------------------------------------------------------------------
# Backfill driver: hash-checked inputs, artifact written, full gate applied
# ---------------------------------------------------------------------------

def _backfill_fixture(tmp_path):
    """A config + frozen-split stand-in + record consistent with the dummy_ids runner."""
    cfg = tmp_path / "run.yaml"
    cfg.write_text(
        "model:\n  runner: dummy_ids\n  name: d\n"
        f"data:\n  split: cal\n  splits_dir: {tmp_path}\n"
    )
    config = harness.load_config(cfg)
    result = _dummy_ids_runner(config)
    _write_split(tmp_path / "cal.parquet", result.ids, result.y_true)
    points = harness.evaluate(result.y_true, result.y_pred, result.probs, result.class_labels)
    record = {
        "run_id": "ab" * 32,
        "config_path": str(cfg),
        "config_sha256": harness.config_sha256(cfg),
        "git_sha": "gitsha",
        "dataset": {"split": "cal", "split_sha256": "s", "input_sha256": "i"},
        "extra": {"prompt_bundle_sha256": "bundle-abc"},
        "metrics": {m: {"point": points[m]} for m in predictions.VERIFY_METRICS},
    }
    return cfg, record


def test_backfill_writes_and_verifies(tmp_path):
    _, record = _backfill_fixture(tmp_path)
    preds_dir = tmp_path / "preds"
    summary = predictions.backfill([record], preds_dir=preds_dir)
    assert summary["ok"] is True
    assert summary["results"][0]["status"] == "written"
    assert all(r["ok"] for r in summary["results"][0]["verify"])
    art = predictions.read_artifact(preds_dir / f"{record['run_id']}.parquet")
    assert art.provenance["git_sha"] == "gitsha"
    assert art.provenance["input_sha256"] == "i"
    assert art.provenance["prompt_bundle_sha256"] == "bundle-abc"


def test_backfill_hard_fails_on_config_hash_mismatch(tmp_path):
    cfg, record = _backfill_fixture(tmp_path)
    cfg.write_text(cfg.read_text() + "seed: 7\n")  # config edited after the run
    preds_dir = tmp_path / "preds"
    summary = predictions.backfill([record], preds_dir=preds_dir)
    assert summary["ok"] is False
    assert "config hash mismatch" in summary["results"][0]["status"]
    # nothing written: the mismatch is caught before reconstruction
    assert not preds_dir.exists() or not list(preds_dir.glob("*.parquet"))


@harness.register_runner("dummy_no_ids")
def _dummy_no_ids_runner(config):
    # Registered here rather than reusing test_harness.py's dummy_e2e so this module is
    # runnable on its own (`pytest tests/test_predictions.py`).
    # 20 rows (not 4) so no bootstrap replicate can draw a single-class resample and
    # trip metrics.py's empty-class divide warning.
    labels = ["a", "b"]
    rng = np.random.default_rng(3)
    true_idx = rng.integers(0, 2, size=20)
    pred_idx = np.where(rng.random(20) < 0.2, 1 - true_idx, true_idx)
    probs = np.full((20, 2), 0.25)
    probs[np.arange(20), pred_idx] = 0.75
    return harness.RunnerResult(
        y_true=np.array([labels[i] for i in true_idx], dtype=object),
        y_pred=np.array([labels[i] for i in pred_idx], dtype=object),
        probs=probs,
        class_labels=labels,
        dataset={"split": config["data"]["split"], "split_sha256": "s", "input_sha256": "i"},
        cost_usd=None,
    )


def test_run_without_ids_writes_no_artifact_and_keeps_schema(tmp_path):
    # A runner that supplies no ids must not gain a predictions_path or an artifact.
    cfg = tmp_path / "run.yaml"
    cfg.write_text("model:\n  runner: dummy_no_ids\n  name: d\ndata:\n  split: cal\n")
    results = tmp_path / "runs.jsonl"
    preds_dir = tmp_path / "preds"
    record = harness.run(cfg, results, preds_dir=preds_dir)
    assert "extra" not in record or "predictions_path" not in record.get("extra", {})
    assert not preds_dir.exists() or not list(preds_dir.glob("*.parquet"))
