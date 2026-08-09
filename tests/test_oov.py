"""OOV / covariate-tracking tests: analyzer parity with the real tier_a feature stack, both
OOV definitions against hand-computed mini corpora, the counts-matrix bootstrap against a
naive per-replicate oracle, determinism (weights, CIs, serialized bytes), and the
split-integrity failure path.

No real split is read: every fixture is a throwaway parquet under tmp_path with a matching
splits_stats.yaml, so the integrity gate is exercised for real. The one exception is the
provenance test, which reads the committed `configs/tier_a_*.yaml` files (read-only) because
the feature-block parity assertion is precisely a claim about those files.
"""

from __future__ import annotations

import itertools
import json

import duckdb
import numpy as np
import pytest
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer

from triage_lab import harness, oov, tier_a
from triage_lab.snapshot import sha256_file

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# min_df=2 makes "cc" a pruned-but-known token: in TRAIN, absent from the model vocabulary.
# That is the whole point of carrying two OOV definitions, so every hand-computed
# expectation below turns on it.
MINI_TRAIN = ["aa bb", "aa bb", "aa cc"]
MINI_SLICE = ["aa cc dd dd"]


def _cfg(word_min_df=2, char_enabled=False, char_min_df=1):
    return {
        "word": {"enabled": True, "ngram_range": [1, 2], "min_df": word_min_df,
                 "max_features": None, "sublinear_tf": True},
        "char": {"enabled": char_enabled, "ngram_range": [3, 5], "min_df": char_min_df,
                 "max_features": None, "sublinear_tf": True},
    }


@pytest.fixture
def mini_ref():
    return oov.build_reference(MINI_TRAIN, _cfg())


def _write_parquet(path, ids, texts):
    con = duckdb.connect()
    try:
        con.execute("SET threads=1")
        con.register("t", {
            "complaint_id": np.array(ids, dtype=np.int64),
            "narrative": np.array(texts, dtype=object),
            "class": np.array(["alpha"] * len(ids), dtype=object),
        })
        con.execute(f"COPY (SELECT * FROM t ORDER BY complaint_id) TO '{path}' (FORMAT PARQUET)")
    finally:
        con.close()


_SLICE_TEXTS = {
    "train": [
        "loan payment late fee charged", "loan payment late fee charged twice",
        "credit report error dispute", "credit report error dispute again",
        "mortgage escrow shortage notice", "mortgage escrow shortage notice repeated",
        "debt collector called me daily", "debt collector called me daily again",
    ],
    "test_iid": ["loan payment late fee", "credit report dispute error"],
    "test_drift_2023": ["mortgage escrow notice", "debt collector called"],
    "test_drift_2024": ["loan payment cryptocurrency wallet", "credit report error"],
    "test_drift_2025": ["cryptocurrency wallet seized", "buy now pay later installment"],
    "test_drift_2026h1": ["cryptocurrency wallet seized entirely", ""],
}


def _build_splits(tmp_path):
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir(exist_ok=True)
    split_stats = {}
    cid = 1000
    for name, texts in _SLICE_TEXTS.items():
        path = splits_dir / f"{name}.parquet"
        ids = list(range(cid, cid + len(texts)))
        cid += 1000
        _write_parquet(path, ids, texts)
        split_stats[name] = {"sha256": sha256_file(path)}
    (splits_dir / "splits_stats.yaml").write_text(
        yaml.safe_dump({"input_sha256": "synthetic-input", "splits": split_stats})
    )
    return splits_dir


def _feature_config(tmp_path, word_min_df=1):
    path = tmp_path / "features.yaml"
    path.write_text(yaml.safe_dump({
        "model": {"runner": "tier_a", "name": "fixture"},
        "data": {"split": "test_iid"},
        "features": _cfg(word_min_df=word_min_df, char_enabled=True),
    }))
    return path


# ---------------------------------------------------------------------------
# (a) analyzer / vectorizer parity with tier_a
# ---------------------------------------------------------------------------

def test_reference_uses_the_real_tier_a_feature_union_params():
    """The vocabulary measured against must be the one tier_a fits, param for param."""
    ref = oov.build_reference(MINI_TRAIN, _cfg(word_min_df=1, char_enabled=True))
    word = oov._named_transformer(ref.features, "word")
    char = oov._named_transformer(ref.features, "char")
    assert isinstance(word, TfidfVectorizer) and isinstance(char, TfidfVectorizer)
    assert (word.analyzer, word.ngram_range, word.sublinear_tf, word.lowercase) == (
        "word", (1, 2), True, True)
    assert (char.analyzer, char.ngram_range, char.sublinear_tf, char.lowercase) == (
        "char_wb", (3, 5), True, True)
    assert ref.model_vocab is word.vocabulary_


def test_shipped_tier_a_config_is_the_150k_min_df_5_stack():
    """Guards the headline claim's provenance: the committed Tier A feature block."""
    prov = oov.resolve_feature_config()
    word = prov["features"]["word"]
    char = prov["features"]["char"]
    assert word == {"enabled": True, "ngram_range": [1, 2], "min_df": 5,
                    "max_features": 150000, "sublinear_tf": True}
    assert char == {"enabled": True, "ngram_range": [3, 5], "min_df": 5,
                    "max_features": 150000, "sublinear_tf": True}
    # and the four yearly Tier A runs + the CAL rung really do agree (the parity gate)
    assert len(prov["parity_configs"]) == 5
    assert prov["config_sha256"] == harness.config_sha256(oov.DEFAULT_FEATURE_CONFIG)


def test_feature_parity_gate_rejects_a_divergent_config(tmp_path):
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text(yaml.safe_dump({"model": {"runner": "tier_a"}, "data": {"split": "cal"},
                                 "features": _cfg(word_min_df=5)}))
    b.write_text(yaml.safe_dump({"model": {"runner": "tier_a"}, "data": {"split": "cal"},
                                 "features": _cfg(word_min_df=7)}))
    with pytest.raises(ValueError, match="feature-block parity failed"):
        oov.resolve_feature_config(a, parity_configs=("b.yaml",))
    # a missing parity file is skipped, not fabricated into a failure
    assert oov.resolve_feature_config(a, parity_configs=("nope.yaml",))["features"] == \
        _cfg(word_min_df=5)


def test_unigram_analyzer_is_the_prefix_of_the_fitted_12_analyzer(mini_ref):
    """(1, 2) output = the (1, 1) tokens in order, then the bigrams. Pinned, not assumed:
    the module derives token-level counts from the (1, 1) clone and n-gram-level counts
    from the fitted analyzer, and those two have to describe the same tokenization."""
    for text in ("Aa BB-cc dd!", "loan payment late fee charged", "", "  "):
        uni = mini_ref.uni_analyzer(text)
        grams = mini_ref.ngram_analyzer(text)
        assert grams[: len(uni)] == uni
        assert grams[len(uni):] == [f"{a} {b}" for a, b in itertools.pairwise(uni)]
        assert len(grams) == (2 * len(uni) - 1 if uni else 0)


def test_unigram_analyzer_inherits_lowercasing_from_the_fitted_vectorizer(mini_ref):
    assert mini_ref.uni_analyzer("AA Bb") == ["aa", "bb"]


def test_word_block_is_required():
    with pytest.raises(ValueError, match="no 'word' block"):
        oov.build_reference(MINI_TRAIN, {"word": {"enabled": False},
                                         "char": {"enabled": True, "min_df": 1,
                                                  "ngram_range": [3, 5],
                                                  "max_features": None,
                                                  "sublinear_tf": True}})


# ---------------------------------------------------------------------------
# (b) the two OOV definitions, hand-computed
# ---------------------------------------------------------------------------

def test_reference_separates_the_pruned_vocabulary_from_the_full_type_set(mini_ref):
    # min_df=2 over ["aa bb", "aa bb", "aa cc"]: aa df=3, bb df=2, cc df=1 (pruned);
    # bigrams "aa bb" df=2 (kept), "aa cc" df=1 (pruned).
    assert sorted(mini_ref.model_vocab) == ["aa", "aa bb", "bb"]
    assert mini_ref.train_types == {"aa", "bb", "cc"}
    # 3 vocabulary features but only 2 of the 3 TRAIN unigram types survive: the counts are
    # not interchangeable, which is why both are reported.
    assert mini_ref.n_train_types_in_vocab == 2


def test_token_and_type_level_oov_on_a_hand_computed_slice(mini_ref):
    stats = oov.token_stats(MINI_SLICE, mini_ref)
    # "aa cc dd dd": 4 unigram occurrences.
    assert stats.n_tokens.tolist() == [4]
    # model_vocab OOV = cc (pruned) + dd + dd = 3 of 4
    assert stats.n_oov_model.tolist() == [3]
    # corpus_novelty OOV = dd + dd = 2 of 4 (cc IS in TRAIN, just pruned)
    assert stats.n_oov_novel.tolist() == [2]
    # types {aa, cc, dd}: model-vocab OOV {cc, dd} = 2/3, novelty OOV {dd} = 1/3
    assert (stats.n_types, stats.n_oov_model_types, stats.n_oov_novel_types) == (3, 2, 1)
    # (1, 2)-grams: [aa, cc, dd, dd, "aa cc", "cc dd", "dd dd"] = 7; only "aa" is in vocab
    assert stats.n_ngrams.tolist() == [7]
    assert stats.n_oov_ngram.tolist() == [6]


def test_novelty_oov_is_structurally_zero_on_train(mini_ref):
    stats = oov.token_stats(MINI_TRAIN, mini_ref)
    assert stats.n_oov_novel.sum() == 0
    assert stats.n_oov_novel_types == 0
    # ...while model-vocab OOV is NOT zero on TRAIN: "cc" was pruned away.
    assert stats.n_oov_model.sum() == 1


def test_token_level_counts_occurrences_not_types(mini_ref):
    stats = oov.token_stats(["dd dd dd dd aa"], mini_ref)
    assert stats.n_tokens.tolist() == [5] and stats.n_oov_model.tolist() == [4]
    assert (stats.n_types, stats.n_oov_model_types) == (2, 1)


def test_empty_documents_contribute_nothing(mini_ref):
    stats = oov.token_stats(["", "aa"], mini_ref)
    assert stats.n_tokens.tolist() == [0, 1]
    assert stats.n_oov_model.tolist() == [0, 0]


def test_ratio_of_an_empty_slice_is_zero_not_nan():
    assert oov._ratio(0, 0) == 0.0
    assert oov._ratio(3, 4) == 0.75


# ---------------------------------------------------------------------------
# (c) bootstrap: determinism, stream contract, oracle agreement
# ---------------------------------------------------------------------------

def test_bootstrap_counts_are_deterministic_and_stream_defined():
    a = oov.bootstrap_counts(50, n_resamples=8, seed=harness.BOOTSTRAP_SEED)
    b = oov.bootstrap_counts(50, n_resamples=8, seed=harness.BOOTSTRAP_SEED)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, oov.bootstrap_counts(50, n_resamples=8, seed=1))
    # every replicate is a size-n resample
    np.testing.assert_array_equal(a.sum(axis=0), np.full(8, 50))
    # ...drawn as one rng.integers(0, n, n) per replicate, in order, from one stream
    rng = np.random.default_rng(harness.BOOTSTRAP_SEED)
    for col in range(8):
        expected = np.bincount(rng.integers(0, 50, size=50), minlength=50)
        np.testing.assert_array_equal(a[:, col], expected)


def test_weighted_ratio_replicates_matches_a_naive_oracle_and_ignores_block_size():
    rng = np.random.default_rng(3)
    n = 40
    denom = rng.integers(1, 30, size=n).astype(np.int64)
    numer = rng.integers(0, 1, size=n) + (denom // 3)
    counts = oov.bootstrap_counts(n, n_resamples=16, seed=7)
    got = oov.weighted_ratio_replicates(numer, denom, counts, block=5)
    naive = np.array([
        numer[np.repeat(np.arange(n), counts[:, b])].sum()
        / denom[np.repeat(np.arange(n), counts[:, b])].sum()
        for b in range(16)
    ])
    np.testing.assert_allclose(got, naive, rtol=1e-12)
    np.testing.assert_allclose(
        got, oov.weighted_ratio_replicates(numer, denom, counts, block=16), rtol=0, atol=0
    )


def test_weighted_ratio_survives_an_all_empty_replicate():
    counts = np.array([[1], [0]], dtype=np.uint16)
    got = oov.weighted_ratio_replicates(
        np.array([0, 5]), np.array([0, 9]), counts, block=1
    )
    assert got.tolist() == [0.0]  # 0/0 -> 0, never NaN in a committed JSON


def test_same_seed_gives_the_same_ci(mini_ref):
    stats = oov.token_stats(["aa cc dd", "dd ee aa", "aa aa bb"], mini_ref)
    def ci():
        counts = oov.bootstrap_counts(3, n_resamples=32, seed=harness.BOOTSTRAP_SEED)
        reps = oov.weighted_ratio_replicates(stats.n_oov_model, stats.n_tokens, counts)
        return oov.ci_block(0.5, reps)
    assert ci() == ci()
    assert ci()["ci_kind"] == "bootstrap_percentile"


# ---------------------------------------------------------------------------
# (d) centroid distance
# ---------------------------------------------------------------------------

def test_cosine_distance_conventions():
    a = np.array([1.0, 0.0, 0.0])
    assert oov.cosine_distance(a, a) == pytest.approx(0.0, abs=1e-15)
    assert oov.cosine_distance(a, np.array([0.0, 1.0, 0.0])) == pytest.approx(1.0)
    assert oov.cosine_distance(a, 7.0 * a) == pytest.approx(0.0, abs=1e-15)  # scale-free
    assert oov.cosine_distance(a, np.zeros(3)) == 0.0                       # origin -> 0


def test_self_distance_is_exactly_zero_never_a_negative_float():
    """Regression: on the real 300k TRAIN centroid the unclipped formula returns -2.2e-16,
    i.e. a NEGATIVE distance in a JSON that calls it a structural zero. Clipping the cosine
    into [-1, 1] is the fix; this pins it on a vector big enough to trip the rounding."""
    rng = np.random.default_rng(11)
    v = rng.random(200_000)
    d = oov.cosine_distance(v, v.copy())
    assert d == 0.0 and not np.signbit(d)
    assert oov.cosine_distance(v, 3.0 * v) == 0.0


def test_train_self_distance_is_exactly_zero_and_slice_distance_is_the_oracle():
    ref = oov.build_reference(MINI_TRAIN, _cfg(word_min_df=1, char_enabled=True))
    counts_t = oov.bootstrap_counts(len(MINI_TRAIN), n_resamples=8, seed=5)
    train_res = oov.centroid_pass(MINI_TRAIN, ref, counts_t, None, chunk_docs=2)
    assert train_res.distance == 0.0
    ref.centroid = train_res.centroid

    texts = ["aa cc dd dd", "aa bb", "zz zz zz"]
    counts = oov.bootstrap_counts(len(texts), n_resamples=8, seed=5)
    res = oov.centroid_pass(texts, ref, counts, ref.centroid, chunk_docs=2)

    # oracle: densify, unit-normalise rows, mean, cosine -- written the long way
    from sklearn.preprocessing import normalize as _normalize
    rows = _normalize(ref.features.transform(texts).tocsr(), norm="l2", axis=1).toarray()
    assert res.distance == pytest.approx(
        oov.cosine_distance(rows.mean(axis=0), ref.centroid), rel=1e-12
    )
    naive = [
        oov.cosine_distance(rows[np.repeat(np.arange(len(texts)), counts[:, b])].mean(axis=0),
                            ref.centroid)
        for b in range(counts.shape[1])
    ]
    np.testing.assert_allclose(res.replicates, naive, rtol=1e-10, atol=1e-12)


def test_centroid_is_chunk_size_invariant_up_to_float_noise():
    ref = oov.build_reference(MINI_TRAIN, _cfg(word_min_df=1, char_enabled=True))
    ref.centroid = oov.centroid_pass(MINI_TRAIN, ref, None, None, chunk_docs=3).centroid
    texts = ["aa cc dd dd", "aa bb", "zz zz zz", "bb cc", "dd aa"]
    counts = oov.bootstrap_counts(len(texts), n_resamples=4, seed=2)
    one = oov.centroid_pass(texts, ref, counts, ref.centroid, chunk_docs=len(texts))
    many = oov.centroid_pass(texts, ref, counts, ref.centroid, chunk_docs=2)
    assert one.distance == pytest.approx(many.distance, rel=1e-12)
    np.testing.assert_allclose(one.replicates, many.replicates, rtol=1e-10, atol=1e-12)


def test_documents_with_no_representable_feature_are_counted():
    ref = oov.build_reference(MINI_TRAIN, _cfg(word_min_df=2, char_enabled=False))
    ref.centroid = oov.centroid_pass(MINI_TRAIN, ref, None, None).centroid
    res = oov.centroid_pass(["", "zzz", "aa"], ref, None, ref.centroid)
    assert res.n_zero_rows == 2  # "" and "zzz" have no in-vocabulary word feature


# ---------------------------------------------------------------------------
# (e) split integrity
# ---------------------------------------------------------------------------

def test_split_integrity_gate_passes_on_a_matching_fixture(tmp_path):
    splits_dir = _build_splits(tmp_path)
    info = oov.verify_split_integrity("test_iid", splits_dir, splits_dir / "splits_stats.yaml")
    assert info["split"] == "test_iid" and len(info["split_sha256"]) == 64


def test_split_integrity_gate_fails_loud_on_a_drifted_parquet(tmp_path):
    splits_dir = _build_splits(tmp_path)
    stats_path = splits_dir / "splits_stats.yaml"
    stats = yaml.safe_load(stats_path.read_text())
    stats["splits"]["test_drift_2025"]["sha256"] = "0" * 64
    stats_path.write_text(yaml.safe_dump(stats))
    with pytest.raises(ValueError, match="integrity check failed for split 'test_drift_2025'"):
        oov.verify_split_integrity("test_drift_2025", splits_dir, stats_path)


def test_build_slice_refuses_to_run_before_the_reference_centroid_exists(tmp_path):
    splits_dir = _build_splits(tmp_path)
    ref = oov.build_reference(_SLICE_TEXTS["train"], _cfg(word_min_df=1, char_enabled=True))
    assert ref.centroid is None
    with pytest.raises(ValueError, match="needs the TRAIN centroid"):
        oov.build_slice(
            "test_iid", ref, splits_dir=splits_dir,
            splits_stats_path=splits_dir / "splits_stats.yaml", n_resamples=4,
        )


# ---------------------------------------------------------------------------
# (f) serialization + end-to-end determinism
# ---------------------------------------------------------------------------

def test_round_tree_makes_non_finite_values_json_safe():
    out = oov._round_tree({"a": float("inf"), "b": float("nan"), "c": np.float64(1 / 3),
                           "d": True, "e": None, "f": [np.int64(2), 0.123456789012345]})
    assert out == {"a": None, "b": None, "c": round(1 / 3, oov.JSON_ROUND), "d": True,
                   "e": None, "f": [2, round(0.123456789012345, 10)]}
    json.dumps(out)


def test_point_only_blocks_carry_no_ci_and_say_why():
    block = oov.point_block(0.25, "type_level_under_document_bootstrap")
    assert block["ci_lo"] is None and block["ci_hi"] is None and block["ci_kind"] is None
    assert block["no_ci_reason"]


def test_slice_selection_is_ordered_and_filtered():
    assert oov.select_slices(None) == list(oov.SLICE_ORDER)
    assert oov.select_slices(["test_drift_2025", "train"]) == ["train", "test_drift_2025"]
    assert oov.output_name("test_drift_2026h1") == "test_drift_2026h1.json"


def _strip_generated_at(text: str) -> str:
    obj = json.loads(text)
    obj.pop("generated_at", None)
    return json.dumps(obj, sort_keys=True)


def _run(tmp_path, out_dir, **kw):
    splits_dir = _build_splits(tmp_path)
    return oov.run(
        list(oov.SLICE_ORDER),
        splits_dir=splits_dir,
        splits_stats_path=splits_dir / "splits_stats.yaml",
        out_dir=out_dir,
        feature_config=_feature_config(tmp_path),
        write_summary=True,
        n_resamples=16,
        chunk_docs=3,
        log=lambda *a, **k: None,
        **kw,
    )


def test_end_to_end_run_writes_one_json_per_slice_plus_a_summary(tmp_path):
    out = tmp_path / "out"
    objs = _run(tmp_path, out)
    assert len(objs) == len(oov.SLICE_ORDER)
    for name in oov.SLICE_ORDER:
        assert (out / f"{name}.json").exists()
    summary = json.loads((out / "summary.json").read_text())
    assert summary["evidence_class"] == "measured"
    assert len(summary["rows"]) == len(oov.SLICE_ORDER) * (
        len(oov.BOOTSTRAPPED_METRICS) + len(oov.POINT_ONLY_METRICS)
    )

    train = json.loads((out / "train.json").read_text())
    assert train["is_reference_slice"] is True
    assert train["evidence_class"] == "measured"
    assert train["metrics"][oov.METRIC_NOVEL_TOKEN]["point"] == 0.0
    assert train["metrics"][oov.METRIC_CENTROID]["point"] == 0.0
    assert oov.METRIC_CENTROID in train["structural_values"]
    assert train["centroid"]["dense_encoder_note"].startswith("TF-IDF-space centroid")
    assert train["metrics"][oov.METRIC_MODEL_TYPE]["ci_kind"] is None
    assert "type" in train["methods_notes"]["token_vs_type"].lower()
    assert train["dataset"]["split_sha256"]

    # the drift years introduce vocabulary TRAIN never saw; 2023 (in-vocabulary) does not
    novel = {n: json.loads((out / f"{n}.json").read_text())
             ["metrics"][oov.METRIC_NOVEL_TOKEN]["point"] for n in oov.SLICE_ORDER}
    assert novel["test_drift_2023"] == 0.0
    assert novel["test_drift_2025"] > novel["test_drift_2024"] > 0.0
    dist = {n: json.loads((out / f"{n}.json").read_text())
            ["metrics"][oov.METRIC_CENTROID]["point"] for n in oov.SLICE_ORDER}
    assert dist["test_drift_2025"] > dist["test_iid"]


def test_rerunning_is_byte_identical_modulo_generated_at(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    _run(tmp_path, first)
    _run(tmp_path, second)
    names = [f"{n}.json" for n in oov.SLICE_ORDER] + ["summary.json"]
    for name in names:
        a, b = (first / name).read_text(), (second / name).read_text()
        assert _strip_generated_at(a) == _strip_generated_at(b), name
    # generated_at is the ONLY field allowed to move
    assert json.loads((first / "train.json").read_text())["generated_at"]


def test_partial_selection_still_measures_against_train(tmp_path):
    splits_dir = _build_splits(tmp_path)
    out = tmp_path / "out"
    objs = oov.run(
        ["test_drift_2026h1"],
        splits_dir=splits_dir,
        splits_stats_path=splits_dir / "splits_stats.yaml",
        out_dir=out,
        feature_config=_feature_config(tmp_path),
        write_summary=False,
        n_resamples=8,
        chunk_docs=3,
        log=lambda *a, **k: None,
    )
    assert [o["slice"] for o in objs] == ["test_drift_2026h1"]
    assert not (out / "train.json").exists() and not (out / "summary.json").exists()
    assert objs[0]["metrics"][oov.METRIC_CENTROID]["point"] > 0.0
    assert objs[0]["repro_command"].endswith("--slice test_drift_2026h1")


def test_summary_rows_are_flat_and_complete(tmp_path):
    out = tmp_path / "out"
    objs = _run(tmp_path, out)
    rows = oov.summary_rows(objs[0])
    assert {r["metric"] for r in rows} == set(oov.BOOTSTRAPPED_METRICS) | set(
        oov.POINT_ONLY_METRICS)
    for row in rows:
        assert row["slice"] == objs[0]["slice"]
        assert row["n_docs"] == objs[0]["counts"]["n_docs"]
        if row["metric"] in oov.POINT_ONLY_METRICS:
            assert row["ci_lo"] is None
        else:
            assert row["ci_lo"] is not None and row["ci_hi"] is not None


def test_cli_rejects_an_empty_selection():
    with pytest.raises(SystemExit):
        oov.main([])


def test_module_defaults_match_the_frozen_harness_protocol():
    """The bootstrap protocol is the repo's, not a local re-invention."""
    assert oov.bootstrap_counts.__defaults__[1] == harness.BOOTSTRAP_SEED == 20260805
    assert oov.bootstrap_counts.__defaults__[0] == harness.N_RESAMPLES == 1000
    assert oov.REFERENCE_SLICE == "train"
    assert oov.SLICE_ORDER[0] == "train"
    assert tier_a.DEFAULT_SPLITS_DIR == oov.DEFAULT_SPLITS_DIR
