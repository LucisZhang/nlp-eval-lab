"""Tier B tests that run WITHOUT a GPU or any model download.

Covered here (pure logic): temperature-scaling recovers a known T and lowers NLL;
softmax stays a valid simplex; config/checkpoint helpers validate and prefer the
training-meta max_seq_length; the training kit's manifest integrity gate accepts a
clean kit and rejects tamper/missing files; the data export is byte-deterministic.

Anything needing a real checkpoint (the tier_b runner end-to-end, ONNX export) is
proven by the smoke pipeline in the runbook, not here — those need a model download,
so they are intentionally out of the unit suite. `torch` (the `tierb` dep group) is
required; the whole module skips cleanly without it, matching the harness's tolerant
optional-runner loading.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from triage_lab import harness, tier_b

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_runner_registered():
    assert "tier_b" in harness.RUNNERS
    assert harness.RUNNERS["tier_b"] is tier_b.tier_b_runner


# ---------------------------------------------------------------------------
# Temperature scaling: recover a known T, lower NLL
# ---------------------------------------------------------------------------

def _nll(logits, labels, T):
    z = np.asarray(logits) / T
    z = z - z.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return float(-logp[np.arange(len(labels)), labels].mean())


def test_fit_temperature_recovers_known_scale():
    rng = np.random.default_rng(0)
    n, k = 6000, 5
    z = rng.normal(0.0, 3.0, size=(n, k))
    p = tier_b.softmax_np(z)
    cum = p.cumsum(axis=1)
    u = rng.random((n, 1))
    labels = (u < cum).argmax(axis=1)

    # Present logits over-confident by a factor c: the NLL-optimal T is ~= c.
    c = 2.5
    logits_in = z * c
    T = tier_b.fit_temperature(logits_in, labels)
    assert abs(T - c) < 0.3, T
    # Fitting must not increase NLL vs the uncalibrated T=1.
    assert _nll(logits_in, labels, T) < _nll(logits_in, labels, 1.0)


def test_fit_temperature_wellcalibrated_stays_near_one():
    rng = np.random.default_rng(1)
    n, k = 6000, 4
    z = rng.normal(0.0, 2.0, size=(n, k))
    p = tier_b.softmax_np(z)
    labels = (rng.random((n, 1)) < p.cumsum(axis=1)).argmax(axis=1)
    T = tier_b.fit_temperature(z, labels)  # logits already match the label dist
    assert abs(T - 1.0) < 0.25, T


def test_softmax_np_is_valid_simplex():
    rng = np.random.default_rng(2)
    probs = tier_b.softmax_np(rng.normal(size=(50, 7)))
    assert np.all(probs >= 0) and np.all(probs <= 1)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Config / checkpoint helpers
# ---------------------------------------------------------------------------

def test_resolve_max_seq_length_prefers_training_meta():
    cfg = {"training": {"max_seq_length": 128}}
    assert tier_b.resolve_max_seq_length(cfg, {"max_seq_length": 256}) == 256  # meta wins
    assert tier_b.resolve_max_seq_length(cfg, {}) == 128                       # config fallback
    assert tier_b.resolve_max_seq_length({}, {}) == tier_b._DEFAULT_MAX_SEQ_LEN


def test_subsample_eval_is_seeded_sized_and_noop():
    texts = [f"t{i}" for i in range(100)]
    labels = np.array([f"c{i % 3}" for i in range(100)], dtype=object)

    t1, l1 = tier_b.subsample_eval(texts, labels, 10, 20260805)
    t2, l2 = tier_b.subsample_eval(texts, labels, 10, 20260805)
    assert len(t1) == 10
    assert t1 == t2 and np.array_equal(l1, l2)          # deterministic for a fixed seed
    assert set(t1).issubset(set(texts))                  # a real subset of the split

    # cap None or >= n is a no-op returning the full split unchanged (real configs).
    tf, lf = tier_b.subsample_eval(texts, labels, None, 1)
    assert tf is texts and lf is labels
    assert len(tier_b.subsample_eval(texts, labels, 999, 1)[0]) == 100


def test_checkpoint_dir_requires_path():
    with pytest.raises(ValueError, match="model.checkpoint"):
        tier_b.checkpoint_dir({"model": {}})
    assert tier_b.checkpoint_dir({"model": {"checkpoint": "x/y"}}) == Path("x/y")


def test_checkpoint_content_hash_requires_weights(tmp_path):
    with pytest.raises(FileNotFoundError, match="weights file"):
        tier_b.checkpoint_content_hash(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"pretend-weights")
    h = tier_b.checkpoint_content_hash(tmp_path)
    assert len(h) == 64


def test_shipped_configs_are_wellformed():
    import yaml

    for name in (
        "tier_b1_modernbert_sa",
        "tier_b1_modernbert_sb",
        "tier_b1_modernbert_sc",
        "tier_b2_distilbert_s0",
        "tier_b_smoke",
    ):
        cfg = yaml.safe_load((REPO_ROOT / "configs" / f"{name}.yaml").read_text())
        assert cfg["model"]["runner"] == "tier_b"
        assert cfg["model"]["base"]
        assert cfg["model"]["checkpoint"]
        assert cfg["calibration"] in ("temperature", "none")
        assert cfg["training"]["max_seq_length"] > 0

    # Frozen Tier-B seed list must stay {a:20260805, b:20260806, c:20260807}.
    seeds = [
        yaml.safe_load((REPO_ROOT / "configs" / f"tier_b1_modernbert_s{s}.yaml").read_text())["seed"]
        for s in ("a", "b", "c")
    ]
    assert seeds == [20260805, 20260806, 20260807]


# ---------------------------------------------------------------------------
# Training-kit manifest integrity gate (in scripts/train_tier_b.py)
# ---------------------------------------------------------------------------

def test_verify_manifest_accepts_clean_and_rejects_tamper(tmp_path):
    train = _load_script("train_tier_b")
    (tmp_path / "train.parquet").write_bytes(b"AAA")
    (tmp_path / "cal.parquet").write_bytes(b"BBB")
    manifest = {
        "input_sha256": "deadbeef",
        "files": {
            "train.parquet": {"sha256": train.sha256_file(tmp_path / "train.parquet")},
            "cal.parquet": {"sha256": train.sha256_file(tmp_path / "cal.parquet")},
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert train.verify_manifest(tmp_path)["input_sha256"] == "deadbeef"

    # Tamper a file -> loud failure.
    (tmp_path / "train.parquet").write_bytes(b"CCC")
    with pytest.raises(ValueError, match="integrity check failed"):
        train.verify_manifest(tmp_path)

    # Missing file -> loud failure.
    (tmp_path / "train.parquet").unlink()
    with pytest.raises(FileNotFoundError):
        train.verify_manifest(tmp_path)


class _FakeTokenizer:
    """Deterministic, order-independent stand-in for an HF tokenizer __call__."""

    def __call__(self, texts, truncation=True, max_length=None, padding=False, **kw):
        ids = []
        for t in texts:
            body = [ord(c) % 1000 for c in t]
            if max_length is not None:
                body = body[: max_length - 2]
            ids.append([101, *body, 102])
        return {"input_ids": ids, "attention_mask": [[1] * len(r) for r in ids]}


def test_tokenize_chunked_matches_whole_pass():
    train = _load_script("train_tier_b")
    tok = _FakeTokenizer()
    texts = [f"complaint number {i} about a widget dispute" for i in range(53)]  # not a chunk multiple

    whole = train.tokenize_chunked(tok, texts, max_len=16, chunk_size=1000)   # single tokenizer call
    chunked = train.tokenize_chunked(tok, texts, max_len=16, chunk_size=5)     # 11 chunks
    # Byte-identical token arrays regardless of chunk boundary.
    assert train.arrays_sha256(*whole) == train.arrays_sha256(*chunked)

    ids_rows, _ = chunked
    assert all(r.dtype == np.int32 for r in ids_rows)          # stored as int32 (memory fix)
    assert max(len(r) for r in ids_rows) <= 16                 # truncation honored


def test_verify_manifest_requires_manifest(tmp_path):
    train = _load_script("train_tier_b")
    with pytest.raises(FileNotFoundError, match="manifest"):
        train.verify_manifest(tmp_path)


# ---------------------------------------------------------------------------
# Data export determinism (scripts/export_tier_b_data.py)
# ---------------------------------------------------------------------------

def _synthetic_splits(tmp_path):
    import duckdb
    import yaml

    from triage_lab.snapshot import sha256_file

    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    stats_splits = {}
    for split, start in (("train", 1000), ("cal", 5000)):
        ids = np.arange(start, start + 40, dtype=np.int64)
        narr = np.array([f"complaint {i} text body" for i in ids], dtype=object)
        extra = np.array([f"co{i}" for i in ids], dtype=object)
        cls = np.array(["card" if i % 2 else "mortgage" for i in ids], dtype=object)
        con = duckdb.connect()
        try:
            con.execute("SET threads=1")
            con.register("t", {"complaint_id": ids, "company": extra,
                               "narrative": narr, "class": cls})
            con.execute(
                f"COPY (SELECT * FROM t ORDER BY complaint_id) "
                f"TO '{splits_dir / f'{split}.parquet'}' "
                f"(FORMAT PARQUET, COMPRESSION 'snappy')"
            )
        finally:
            con.close()
        stats_splits[split] = {"sha256": sha256_file(splits_dir / f"{split}.parquet")}
    (splits_dir / "splits_stats.yaml").write_text(
        yaml.safe_dump({"input_sha256": "synthetic-input", "splits": stats_splits})
    )
    return splits_dir


def test_export_is_byte_deterministic(tmp_path):
    export = _load_script("export_tier_b_data")
    splits_dir = _synthetic_splits(tmp_path)
    out1, out2 = tmp_path / "k1", tmp_path / "k2"
    m1 = export.export(splits_dir, out1)
    m2 = export.export(splits_dir, out2)

    # Manifest carries provenance and matches across runs.
    assert m1 == m2
    assert m1["input_sha256"] == "synthetic-input"
    assert m1["columns"] == ["complaint_id", "narrative", "class"]
    # Exported parquets are byte-identical run-to-run, and their recorded sha matches.
    for fname in ("train.parquet", "cal.parquet"):
        assert (out1 / fname).read_bytes() == (out2 / fname).read_bytes()
        assert export.sha256_file(out1 / fname) == m1["files"][fname]["sha256"]
