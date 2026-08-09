"""Perturbation tests: determinism and order-independence of the per-document RNG keying,
the rate=0 identity, rate monotonicity, per-family scoping, the tier_a/harness integration
(the block reaches the record, TRAIN is never perturbed), and the perturb_report join +
paired-delta logic.

No real split, no real run record and no committed results file is touched: the harness
integration test builds a throwaway train/cal/test_iid parquet trio under tmp_path with a
matching splits_stats.yaml (so the sha256 integrity gate runs for real) and appends to a
tmp results log, and the report test writes two synthetic prediction artifacts and calls
the join/delta core directly.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import duckdb
import numpy as np
import pytest
import yaml

from triage_lab import harness, perturb, perturb_report, predictions, tier_a, tier_c
from triage_lab.snapshot import sha256_file

# A long-ish document with letters, digits, punctuation and every OCR-table character, so
# each family has plenty of eligible sites.
LONG_DOC = (
    "The Bank of SomeState closed my account on 10/15/2021 without notice. "
    "I called 5 times about the {$1,250.00} balance and the modern collector "
    "claimed a lien; nobody could explain the escrow shortage or the late fee. "
) * 4

SEED = perturb.DEFAULT_SEED


def _n_changed(a: str, b: str) -> int:
    """Positional character disagreements (only meaningful for length-preserving families)."""
    return sum(1 for x, y in zip(a, b, strict=False) if x != y) + abs(len(a) - len(b))


# ---------------------------------------------------------------------------
# (a) determinism / purity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family", perturb.FAMILIES)
def test_same_inputs_give_byte_identical_output(family):
    a = perturb.perturb_text(LONG_DOC, family, 0.2, SEED, 4242)
    b = perturb.perturb_text(LONG_DOC, family, 0.2, SEED, 4242)
    assert a == b
    assert isinstance(a, str)


@pytest.mark.parametrize("family", perturb.FAMILIES)
def test_output_is_independent_of_processing_order_and_batch_size(family):
    """The whole point of hash-keying per document: no stream state crosses documents."""
    texts = [LONG_DOC[: 60 + 7 * i] for i in range(12)]
    ids = list(range(9000, 9012))

    forward = perturb.perturb_texts(texts, ids, family, 0.15, SEED)
    reverse = perturb.perturb_texts(texts[::-1], ids[::-1], family, 0.15, SEED)
    assert reverse[::-1] == forward
    # ...and chunking the frame changes nothing either
    chunked = (
        perturb.perturb_texts(texts[:5], ids[:5], family, 0.15, SEED)
        + perturb.perturb_texts(texts[5:], ids[5:], family, 0.15, SEED)
    )
    assert chunked == forward
    # ...and one document alone reproduces its row in the frame
    assert perturb.perturb_text(texts[7], family, 0.15, SEED, ids[7]) == forward[7]


@pytest.mark.parametrize("family", perturb.FAMILIES)
def test_the_key_separates_documents_seeds_families_and_rates(family):
    base = perturb.perturb_text(LONG_DOC, family, 0.3, SEED, 1)
    assert base != perturb.perturb_text(LONG_DOC, family, 0.3, SEED, 2)      # doc_key
    assert base != perturb.perturb_text(LONG_DOC, family, 0.3, SEED + 1, 1)  # seed
    assert base != perturb.perturb_text(LONG_DOC, family, 0.31, SEED, 1)     # rate


def test_doc_key_token_is_type_stable():
    """A complaint_id must key identically whether it arrives as int, np.int64 or str."""
    assert perturb._key_token(12345) == perturb._key_token(np.int64(12345)) == "12345"
    assert perturb.perturb_text(LONG_DOC, "typo", 0.2, SEED, 12345) == perturb.perturb_text(
        LONG_DOC, "typo", 0.2, SEED, np.int64(12345)
    )
    with pytest.raises(TypeError):
        perturb._key_token(True)


def test_rate_token_is_write_style_invariant():
    """0.05 and 5.0e-2 are the same float, so they must be the same run."""
    assert perturb._rate_token(0.05) == perturb._rate_token(5.0e-2) == "0.05"
    assert perturb._rate_token(1) == "1.0"


# ---------------------------------------------------------------------------
# (b) rate semantics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family", perturb.FAMILIES)
def test_rate_zero_is_the_exact_identity(family):
    assert perturb.perturb_text(LONG_DOC, family, 0.0, SEED, 7) == LONG_DOC
    assert perturb.perturb_texts([LONG_DOC, ""], [1, 2], family, 0.0, SEED) == [LONG_DOC, ""]


@pytest.mark.parametrize("family", perturb.FAMILIES)
def test_higher_rate_changes_more_of_a_long_document(family):
    """Sanity, not an exact law: the arms are independent draws (rate is in the RNG key),
    so this is a statistical statement, checked across many documents to make it a safe one."""
    ids = list(range(5000, 5060))
    texts = [LONG_DOC] * len(ids)
    changed = {}
    for rate in (0.01, 0.05, 0.10, 0.30):
        out = perturb.perturb_texts(texts, ids, family, rate, SEED)
        changed[rate] = sum(_n_changed(a, b) for a, b in zip(texts, out, strict=True))
    assert changed[0.01] < changed[0.05] < changed[0.10] < changed[0.30]


@pytest.mark.parametrize("family", perturb.FAMILIES)
def test_rate_one_perturbs_every_eligible_site(family):
    """rate=1 is the fully-determined limit: `rng.random() < 1` is true at every site."""
    out = perturb.perturb_text(LONG_DOC, family, 1.0, SEED, 3)
    assert out != LONG_DOC
    if family == "case":
        assert out == LONG_DOC.swapcase()
    if family == "ocr":
        # rebuild the document by mapping every enumerated site: that IS the rate=1 output
        rebuilt, cursor = [], 0
        for start, key in perturb.ocr_sites(LONG_DOC):
            rebuilt.append(LONG_DOC[cursor:start])
            rebuilt.append(
                perturb.OCR_BIGRAM_CONFUSIONS[key] if len(key) == 2
                else perturb.OCR_UNIGRAM_CONFUSIONS[key]
            )
            cursor = start + len(key)
        rebuilt.append(LONG_DOC[cursor:])
        assert out == "".join(rebuilt)


def test_invalid_family_or_rate_fails_loud():
    with pytest.raises(ValueError, match="unknown perturbation family"):
        perturb.perturb_text("x", "sarcasm", 0.1, SEED, 1)
    with pytest.raises(ValueError, match="outside"):
        perturb.perturb_text("x", "typo", 1.5, SEED, 1)
    with pytest.raises(ValueError, match="outside"):
        perturb.perturb_text("x", "typo", -0.01, SEED, 1)
    with pytest.raises(ValueError, match="doc keys"):
        perturb.perturb_texts(["a", "b"], [1], "typo", 0.1, SEED)


# ---------------------------------------------------------------------------
# (c) family scoping
# ---------------------------------------------------------------------------

def test_case_family_only_flips_case():
    out = perturb.perturb_text(LONG_DOC, "case", 0.5, SEED, 11)
    assert len(out) == len(LONG_DOC)
    assert out != LONG_DOC
    assert out.lower() == LONG_DOC.lower()          # nothing but case moved
    for a, b in zip(LONG_DOC, out, strict=True):
        assert a == b or (a.isalpha() and a.swapcase() == b)


def test_case_family_is_a_structural_zero_for_a_lowercasing_vectorizer():
    """The claim the `case` configs are run to demonstrate: Tier A cannot see this family."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    docs = [LONG_DOC[:200], LONG_DOC[200:400], LONG_DOC[400:600]]
    mangled = perturb.perturb_texts(docs, [1, 2, 3], "case", 0.5, SEED)
    assert mangled != docs
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True).fit(docs)
    np.testing.assert_array_equal(
        vec.transform(docs).toarray(), vec.transform(mangled).toarray()
    )


def test_ocr_family_only_touches_confusion_table_sites():
    # no table character anywhere -> exact identity at any rate
    inert = "why not go paying fun. 476 ~ zzz"
    assert not any(ch in perturb.OCR_UNIGRAM_CONFUSIONS for ch in inert)
    assert perturb.perturb_text(inert, "ocr", 1.0, SEED, 1) == inert

    out = perturb.perturb_text(LONG_DOC, "ocr", 0.4, SEED, 2)
    assert out != LONG_DOC
    # every character the OCR family can EMIT is either an original character or a
    # confusion-table value; nothing else can appear
    emitted = set(out) - set(LONG_DOC)
    allowed = set("".join(perturb.OCR_BIGRAM_CONFUSIONS.values())) | set(
        "".join(perturb.OCR_UNIGRAM_CONFUSIONS.values())
    )
    assert emitted <= allowed


def test_ocr_sites_are_greedy_longest_match_and_non_overlapping():
    # "cl" is one 2-char site, not c-then-l; lowercase "o" is not in the table, "d" is.
    assert perturb.ocr_sites("clod") == [(0, "cl"), (3, "d")]
    # the 2-char key wins over the characters inside it
    assert perturb.ocr_sites("burn") == [(2, "rn")]
    # ...and a site is never re-entered: "cll" is (cl) then a bare "l"
    assert perturb.ocr_sites("cll") == [(0, "cl"), (2, "l")]
    # at rate 1 the mapping is exactly the table's
    assert perturb.perturb_text("burn", "ocr", 1.0, SEED, 1) == "bum"
    assert perturb.perturb_text("cl", "ocr", 1.0, SEED, 1) == "d"
    assert perturb.perturb_text("SOB18", "ocr", 1.0, SEED, 1) == "508lB"


def test_ocr_is_ascii_only():
    table = {**perturb.OCR_BIGRAM_CONFUSIONS, **perturb.OCR_UNIGRAM_CONFUSIONS}
    for key, value in table.items():
        assert key.isascii() and value.isascii()


def test_typo_length_stays_within_the_one_char_per_site_bound():
    """Each firing site changes length by at most 1, so the total drift is bounded by the
    number of eligible (non-whitespace) characters."""
    for rate in (0.05, 0.10, 0.5, 1.0):
        out = perturb.perturb_text(LONG_DOC, "typo", rate, SEED, 21)
        assert abs(len(out) - len(LONG_DOC)) <= len(perturb.typo_sites(LONG_DOC))
        # whitespace is never a site, so the output cannot LOSE all its word boundaries
        assert " " in out


def test_typo_uses_qwerty_neighbours_and_covers_the_op_mix():
    """At rate 1 on a repeated single letter, every emitted non-'g' character must be a
    QWERTY neighbour of 'g', and all four ops must show up over a long enough run."""
    text = "g" * 400
    out = perturb.perturb_text(text, "typo", 1.0, SEED, 5)
    assert set(out) <= {"g"} | set(perturb.QWERTY_ADJACENT["g"])
    assert set("fhtyvb") == set(perturb.QWERTY_ADJACENT["g"])
    # deletes shorten, inserts lengthen; with 400 sites both must have fired
    assert perturb.TYPO_OPS == ("substitute", "delete", "transpose", "insert")
    assert perturb.TYPO_OP_CUM[-1] == 1.0


def test_typo_preserves_the_case_of_a_substituted_key():
    out = perturb.perturb_text("G" * 200, "typo", 1.0, SEED, 6)
    assert set(out) <= {"G"} | set(perturb.QWERTY_ADJACENT["g"].upper())


def test_qwerty_adjacency_is_symmetric_and_plausible():
    for key, neighbours in perturb.QWERTY_ADJACENT.items():
        assert neighbours == "".join(sorted(neighbours))
        for other in neighbours:
            assert key in perturb.QWERTY_ADJACENT[other], (key, other)
    assert set(perturb.QWERTY_ADJACENT["a"]) == set("qwsz")


def test_empty_and_whitespace_documents_are_returned_unchanged():
    for family in perturb.FAMILIES:
        assert perturb.perturb_text("", family, 1.0, SEED, 1) == ""
        assert perturb.perturb_text("   \n\t", family, 1.0, SEED, 1) == "   \n\t"
        assert perturb.perturb_text(None, family, 1.0, SEED, 1) == ""


# ---------------------------------------------------------------------------
# (d) config plumbing
# ---------------------------------------------------------------------------

def test_spec_from_config_parses_validates_and_defaults_the_seed():
    assert perturb.spec_from_config({"data": {"split": "test_iid"}}) is None
    spec = perturb.spec_from_config(
        {"data": {"split": "test_iid", "perturbation": {"family": "ocr", "rate": 0.1}}}
    )
    assert spec == perturb.PerturbationSpec("ocr", 0.1, perturb.DEFAULT_SEED)
    assert spec.as_dict() == {"family": "ocr", "rate": 0.1, "seed": 20260805}

    with pytest.raises(ValueError, match="unknown key"):
        perturb.spec_from_config({"data": {"perturbation": {"familly": "ocr", "rate": 0.1}}})
    with pytest.raises(ValueError, match="missing required key"):
        perturb.spec_from_config({"data": {"perturbation": {"rate": 0.1}}})
    with pytest.raises(ValueError, match="unknown perturbation family"):
        perturb.spec_from_config({"data": {"perturbation": {"family": "nope", "rate": 0.1}}})
    with pytest.raises(TypeError, match="must be a mapping"):
        perturb.spec_from_config({"data": {"perturbation": "typo"}})


def test_apply_spec_with_no_spec_is_the_identity():
    assert perturb.apply_spec(["a", "b"], [1, 2], None) == ["a", "b"]


def test_every_shipped_perturbation_config_parses_to_its_declared_arm():
    """The 18 committed perturbed configs (15 Tier A + 3 Tier C) must each name a real
    family/rate, and must not perturb anything but a TEST-IID eval."""
    paths = sorted((harness.REPO_ROOT / "configs").glob("*_perturb_*.yaml"))
    assert len(paths) == 18
    for path in paths:
        config = harness.load_config(path)
        spec = perturb.spec_from_config(config)
        assert spec is not None, path.name
        assert config["data"]["split"] == "test_iid"
        assert spec.seed == perturb.DEFAULT_SEED
        tag = "05" if spec.rate == 0.05 else "10"
        assert path.stem.endswith(f"_perturb_{spec.family}_{tag}")
        if config["model"]["runner"] == "tier_a":
            # the model is fitted on CLEAN train and calibrated on CLEAN cal
            assert config["data"]["train_split"] == "train"
            assert config["data"]["cal_split"] == "cal"
        else:
            assert config["model"]["runner"] == "tier_c"
            assert "train_split" not in config["data"]


# ---------------------------------------------------------------------------
# (e) harness / tier_a integration on a synthetic fixture
# ---------------------------------------------------------------------------

_TRAIN_TEXTS = [
    "loan payment late fee charged on my mortgage account",
    "loan payment late fee charged twice on the mortgage",
    "credit report error dispute with the bureau again",
    "credit report error dispute never resolved by the bureau",
    "debt collector called me daily about an old balance",
    "debt collector called me daily and left voicemails",
    "mortgage escrow shortage notice arrived in the mail",
    "mortgage escrow shortage notice repeated every month",
]
_TRAIN_LABELS = ["loan", "loan", "credit", "credit", "debt", "debt", "loan", "credit"]
_EVAL_TEXTS = [
    "loan payment late fee charged again",
    "credit report error dispute pending",
    "debt collector called me daily today",
    "mortgage escrow shortage notice mailed",
]
_EVAL_LABELS = ["loan", "credit", "debt", "loan"]


def _write_parquet(path, ids, texts, labels):
    con = duckdb.connect()
    try:
        con.execute("SET threads=1")
        con.register("t", {
            "complaint_id": np.array(ids, dtype=np.int64),
            "narrative": np.array(texts, dtype=object),
            "class": np.array(labels, dtype=object),
        })
        con.execute(f"COPY (SELECT * FROM t ORDER BY complaint_id) TO '{path}' (FORMAT PARQUET)")
    finally:
        con.close()


@pytest.fixture
def mini_splits(tmp_path):
    """train / cal / test_iid parquets + a matching splits_stats.yaml (integrity gate is live)."""
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    stats = {}
    frames = {
        "train": (list(range(1000, 1008)), _TRAIN_TEXTS, _TRAIN_LABELS),
        "cal": (list(range(2000, 2008)), _TRAIN_TEXTS, _TRAIN_LABELS),
        "test_iid": (list(range(3000, 3004)), _EVAL_TEXTS, _EVAL_LABELS),
    }
    for name, (ids, texts, labels) in frames.items():
        path = splits_dir / f"{name}.parquet"
        _write_parquet(path, ids, texts, labels)
        stats[name] = {"sha256": sha256_file(path)}
    (splits_dir / "splits_stats.yaml").write_text(
        yaml.safe_dump({"input_sha256": "synthetic-input", "splits": stats})
    )
    return splits_dir


def _mini_config(splits_dir, perturbation=None) -> dict:
    config = {
        "model": {"runner": "tier_a", "name": "fixture", "family": "complement_nb",
                  "params": {"alpha": 0.3, "norm": False}},
        "data": {"split": "test_iid", "train_split": "train", "cal_split": "cal",
                 "text_column": "narrative", "label_column": "class",
                 "order_column": "complaint_id", "splits_dir": str(splits_dir)},
        "features": {
            "word": {"enabled": True, "ngram_range": [1, 2], "min_df": 1,
                     "max_features": None, "sublinear_tf": True},
            "char": {"enabled": False, "ngram_range": [3, 5], "min_df": 1,
                     "max_features": None, "sublinear_tf": True},
        },
        "calibration": "none",
        "seed": 20260805,
    }
    if perturbation is not None:
        config["data"]["perturbation"] = perturbation
    return config


def test_tier_a_never_perturbs_the_training_text(mini_splits, monkeypatch):
    """The load-bearing guarantee: a perturbed run is the CLEAN model on noisy input."""
    captured: dict = {}
    real_build = tier_a.build_pipeline

    def spy(config):
        pipe = real_build(config)
        real_fit = pipe.fit

        def fit(x, y):
            captured["train_texts"] = list(x)
            return real_fit(x, y)

        pipe.fit = fit
        return pipe

    monkeypatch.setattr(tier_a, "build_pipeline", spy)
    tier_a.fit_predict(
        _mini_config(mini_splits, {"family": "typo", "rate": 1.0, "seed": 20260805})
    )
    assert captured["train_texts"] == _TRAIN_TEXTS


def test_tier_a_perturbs_the_eval_text_exactly_as_the_pure_function_does(mini_splits):
    """The eval frame tier_a scores must be perturb_texts() of the frozen split, keyed by
    complaint_id — checked by scoring a hand-perturbed frame through the same model."""
    ids = list(range(3000, 3004))
    expected = perturb.perturb_texts(_EVAL_TEXTS, ids, "ocr", 1.0, 20260805)
    assert expected != _EVAL_TEXTS

    clean_cfg = _mini_config(mini_splits)
    pert_cfg = _mini_config(mini_splits, {"family": "ocr", "rate": 1.0, "seed": 20260805})
    _, _, clean_probs, labels = tier_a.fit_predict(clean_cfg)
    _, _, pert_probs, _ = tier_a.fit_predict(pert_cfg)

    # oracle: fit the same clean model, then score the hand-perturbed texts
    pipe = tier_a.build_pipeline(clean_cfg)
    pipe.fit(_TRAIN_TEXTS, np.array(_TRAIN_LABELS, dtype=object))
    np.testing.assert_allclose(pert_probs, pipe.predict_proba(expected))
    assert list(pipe.classes_) == list(labels)
    assert not np.allclose(pert_probs, clean_probs)


def test_case_perturbation_is_a_no_op_end_to_end_for_tier_a(mini_splits):
    """Same prediction, bit for bit — the structural-zero claim the case configs carry."""
    clean = tier_a.fit_predict(_mini_config(mini_splits))
    cased = tier_a.fit_predict(
        _mini_config(mini_splits, {"family": "case", "rate": 1.0, "seed": 20260805})
    )
    np.testing.assert_array_equal(clean[1], cased[1])
    np.testing.assert_array_equal(clean[2], cased[2])


def test_perturbation_block_flows_into_the_results_record(tmp_path, mini_splits):
    cfg_path = tmp_path / "run.yaml"
    cfg_path.write_text(yaml.safe_dump(
        _mini_config(mini_splits, {"family": "typo", "rate": 0.1, "seed": 20260805})
    ))
    results = tmp_path / "results" / "runs.jsonl"

    record = harness.run(cfg_path, results)

    on_disk = json.loads(results.read_text().splitlines()[0])
    assert on_disk["extra"]["perturbation"] == {
        "family": "typo", "rate": 0.1, "seed": 20260805,
    }
    assert record["dataset"]["split"] == "test_iid"
    # the artifact went beside the redirected results log, not into the repo's data/preds
    assert (results.parent / "preds" / f"{record['run_id']}.parquet").exists()
    assert not (harness.DEFAULT_PREDS_DIR / f"{record['run_id']}.parquet").exists()


def test_a_clean_config_leaves_the_record_schema_untouched(tmp_path, mini_splits):
    cfg_path = tmp_path / "clean.yaml"
    cfg_path.write_text(yaml.safe_dump(_mini_config(mini_splits)))
    record = harness.run(cfg_path, tmp_path / "runs.jsonl")
    assert "perturbation" not in record.get("extra", {})


@harness.register_runner("dummy_ignores_perturbation")
def _ignoring_runner(config: dict) -> harness.RunnerResult:
    """A runner that does NOT implement data.perturbation — the failure the gate exists for."""
    return harness.RunnerResult(
        y_true=np.array(["a", "b"], dtype=object),
        y_pred=np.array(["a", "b"], dtype=object),
        probs=np.array([[0.9, 0.1], [0.2, 0.8]]),
        class_labels=["a", "b"],
        dataset={"split": "test_iid", "split_sha256": "x", "input_sha256": "y"},
    )


def test_harness_refuses_a_perturbed_config_a_runner_ignored(tmp_path):
    cfg = tmp_path / "ignored.yaml"
    cfg.write_text(
        "model:\n  runner: dummy_ignores_perturbation\n  name: d\n"
        "data:\n  split: test_iid\n  perturbation:\n    family: typo\n    rate: 0.1\n"
    )
    with pytest.raises(ValueError, match="does not support eval-text perturbation"):
        harness.run(cfg, tmp_path / "runs.jsonl")
    assert not (tmp_path / "runs.jsonl").exists()


def test_harness_rejects_a_malformed_block_before_running_anything(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(
        "model:\n  runner: dummy_ignores_perturbation\n  name: d\n"
        "data:\n  split: test_iid\n  perturbation:\n    family: smudge\n    rate: 0.1\n"
    )
    with pytest.raises(ValueError, match="unknown perturbation family"):
        harness.run(cfg, tmp_path / "runs.jsonl")


# ---------------------------------------------------------------------------
# (f) perturb_report: the join + paired delta, on synthetic artifacts
# ---------------------------------------------------------------------------

LABELS = ["credit", "debt", "loan"]


def _synthetic_artifact(path, ids, y_true, y_pred, run_id="r"):
    """Write a real prediction artifact (one-hot probs) and read it back."""
    idx = {lbl: i for i, lbl in enumerate(LABELS)}
    probs = np.full((len(ids), len(LABELS)), 0.05)
    for i, lbl in enumerate(y_pred):
        probs[i, idx[lbl]] = 0.90
    predictions.write_artifact(
        path,
        ids=np.asarray(ids, dtype=np.int64),
        y_true=np.asarray(y_true, dtype=object),
        y_pred=np.asarray(y_pred, dtype=object),
        probs=probs,
        class_labels=LABELS,
        provenance=predictions.ArtifactProvenance(
            run_id=run_id, config_sha256="c", split="test_iid", split_sha256="s",
            class_labels=LABELS,
        ),
    )
    return predictions.read_artifact(path)


@pytest.fixture
def synthetic_pair(tmp_path):
    """120 rows; the perturbed system gets 18 of them wrong that the clean one got right."""
    rng = np.random.default_rng(0)
    ids = np.arange(7000, 7120)
    y_true = np.array([LABELS[i % 3] for i in range(120)], dtype=object)
    clean_pred = y_true.copy()
    pert_pred = y_true.copy()
    for i in rng.choice(120, size=18, replace=False):
        pert_pred[i] = LABELS[(LABELS.index(y_true[i]) + 1) % 3]
    # ids are deliberately written in DIFFERENT orders: the join must sort, not zip
    order = rng.permutation(120)
    clean = _synthetic_artifact(tmp_path / "clean.parquet", ids, y_true, clean_pred, "clean")
    pert = _synthetic_artifact(
        tmp_path / "pert.parquet", ids[order], y_true[order], pert_pred[order], "pert"
    )
    return clean, pert


def test_align_pair_sorts_both_sides_and_keeps_rows_together(synthetic_pair):
    clean, pert = synthetic_pair
    pair = perturb_report.align_pair(clean, pert)
    assert np.array_equal(pair.ids, np.sort(pair.ids))
    assert len(pair.ids) == 120
    # y_true is the same array on both sides after alignment (asserted inside align_pair)
    assert list(pair.class_labels) == LABELS
    # the clean system is perfect by construction; the perturbed one is wrong 18 times
    assert np.array_equal(pair.clean_pred, pair.y_true)
    assert int(np.count_nonzero(pair.perturbed_pred != pair.y_true)) == 18


def test_align_pair_refuses_a_row_set_mismatch(tmp_path, synthetic_pair):
    clean, _ = synthetic_pair
    short = _synthetic_artifact(
        tmp_path / "short.parquet",
        clean.complaint_id[:100], clean.y_true[:100], clean.y_pred[:100], "short",
    )
    with pytest.raises(ValueError, match="identical rows"):
        perturb_report.align_pair(clean, short)


def test_align_pair_refuses_a_relabelled_row(tmp_path, synthetic_pair):
    clean, _ = synthetic_pair
    bad_true = clean.y_true.copy()
    bad_true[3] = LABELS[(LABELS.index(bad_true[3]) + 1) % 3]
    bad = _synthetic_artifact(
        tmp_path / "bad.parquet", clean.complaint_id, bad_true, clean.y_pred, "bad"
    )
    with pytest.raises(ValueError, match="must never touch labels"):
        perturb_report.align_pair(clean, bad)


def test_pair_deltas_are_negative_and_exclude_zero(synthetic_pair):
    pair = perturb_report.align_pair(*synthetic_pair)
    out = perturb_report.pair_deltas(pair, n_resamples=200)
    f1 = out["macro_f1"]
    assert f1["clean"] == pytest.approx(1.0)
    assert f1["perturbed"] < f1["clean"]
    assert f1["delta"] == pytest.approx(f1["perturbed"] - f1["clean"])
    assert f1["ci_lo"] <= f1["delta"] <= f1["ci_hi"]
    assert f1["ci_hi"] < 0.0 and f1["ci_excludes_zero"] is True
    acc = out["accuracy"]
    assert acc["delta"] == pytest.approx(-18 / 120)


def test_an_identical_system_gives_an_exact_zero_delta_with_a_zero_ci(synthetic_pair):
    """The case-arm prediction, at the level of the report's own arithmetic."""
    clean, _ = synthetic_pair
    pair = perturb_report.align_pair(clean, clean)
    out = perturb_report.pair_deltas(pair, n_resamples=50)
    for name in perturb_report.DELTA_METRICS:
        assert out[name]["delta"] == 0.0
        assert (out[name]["ci_lo"], out[name]["ci_hi"]) == (0.0, 0.0)
        assert out[name]["ci_excludes_zero"] is False


def test_build_row_carries_the_full_provenance_of_the_cell(synthetic_pair):
    pair = perturb_report.align_pair(*synthetic_pair)
    arm = perturb_report.ARMS_BY_KEY["logreg_wordchar"]
    row = perturb_report.build_row(
        arm, "typo", "10",
        clean_record={"run_id": "aa" * 32, "config_path": "configs/clean.yaml"},
        perturbed_record={"run_id": "bb" * 32, "config_path": "configs/pert.yaml",
                          "extra": {"perturbation": {"family": "typo", "rate": 0.1,
                                                     "seed": 20260805}}},
        pair=pair, n_resamples=50,
    )
    assert (row["arm"], row["family"], row["rate"], row["n_rows"]) == (
        "logreg_wordchar", "typo", 0.10, 120)
    assert row["clean_run_id"].startswith("aa") and row["perturbed_run_id"].startswith("bb")
    assert row["recorded_perturbation"]["family"] == "typo"
    assert row["structural_expectation"] is None
    assert set(row["metrics"]) == set(perturb_report.DELTA_METRICS)


def test_case_rows_are_labelled_as_a_structural_zero(synthetic_pair):
    pair = perturb_report.align_pair(*synthetic_pair)
    row = perturb_report.build_row(
        perturb_report.ARMS_BY_KEY["cnb_wordchar"], "case", "05",
        clean_record={"run_id": "a", "config_path": "c.yaml"},
        perturbed_record={"run_id": "b", "config_path": "p.yaml", "extra": {}},
        pair=pair, n_resamples=20,
    )
    assert "STRUCTURAL ZERO" in row["structural_expectation"]


def test_recorded_perturbation_gate_catches_a_misfiled_run():
    record = {"run_id": "x" * 64, "config_path": "configs/p.yaml",
              "extra": {"perturbation": {"family": "ocr", "rate": 0.05, "seed": 20260805}}}
    assert perturb_report.check_recorded_perturbation(record, "ocr", 0.05)
    with pytest.raises(ValueError, match="config name and the record disagree"):
        perturb_report.check_recorded_perturbation(record, "ocr", 0.10)
    with pytest.raises(ValueError, match="carries no"):
        perturb_report.check_recorded_perturbation(
            {"run_id": "y" * 64, "config_path": "c.yaml"}, "ocr", 0.05
        )


def test_job_grid_matches_the_shipped_configs():
    """Every report cell has a committed config and vice versa (15 Tier A + 3 Tier C)."""
    cells = perturb_report.jobs()
    assert len(cells) == 18
    assert [c[0].key for c in cells[:6]] == ["logreg_wordchar"] * 6
    names = {arm.perturbed_config(fam, tag) for arm, fam, tag in cells}
    on_disk = {p.stem for p in (harness.REPO_ROOT / "configs").glob("*_perturb_*.yaml")}
    assert names == on_disk
    for arm in perturb_report.ARMS:
        assert (harness.REPO_ROOT / "configs" / f"{arm.clean_config}.yaml").exists()


def test_restricted_job_selection_is_ordered_and_filtered():
    cells = perturb_report.select_jobs(["logreg_word_only"], ["ocr"])
    assert [(c[0].key, c[1], c[2]) for c in cells] == [("logreg_word_only", "ocr", "10")]


def test_report_output_is_json_safe_and_deterministic(synthetic_pair):
    pair = perturb_report.align_pair(*synthetic_pair)
    row = perturb_report.build_row(
        perturb_report.ARMS[0], "typo", "05",
        clean_record={"run_id": "a", "config_path": "c.yaml"},
        perturbed_record={"run_id": "b", "config_path": "p.yaml", "extra": {}},
        pair=pair, n_resamples=20,
    )
    summary = {"evidence_class": perturb_report.EVIDENCE_CLASS,
               "protocol": perturb_report.protocol_block(),
               "rows": [row], "missing": []}
    text = json.dumps(perturb_report._round_tree(summary), sort_keys=True)
    assert json.loads(text)["evidence_class"] == "measured"
    assert json.loads(text)["protocol"]["default_seed"] == 20260805
    assert perturb_report.format_table({"rows": [row], "missing": []}).count("\n") >= 3


def test_cli_rejects_an_empty_selection():
    with pytest.raises(SystemExit):
        perturb_report.main([])


def test_module_defaults_match_the_frozen_protocol():
    assert perturb.DEFAULT_SEED == harness.BOOTSTRAP_SEED == 20260805
    assert perturb_report.DEFAULT_PREDS_DIR == harness.DEFAULT_PREDS_DIR
    assert sum(perturb.TYPO_OP_WEIGHTS) == pytest.approx(1.0)
    assert perturb_report.RATE_TAGS == {"05": 0.05, "10": 0.10}


# ---------------------------------------------------------------------------
# (g) Tier C: subsample-then-perturb ordering, and the echoed block
# ---------------------------------------------------------------------------

def test_subsample_selection_cannot_be_moved_by_perturbation():
    """The ordering guarantee, at the level of tier_c's own selector.

    `subsample_eval` draws a permutation of the row COUNT and never reads the text, so the
    selected ids are identical whether it is handed clean or perturbed narratives. This is
    what makes a perturbed Tier C run pair with its clean counterpart row for row -- and it
    is why the runner perturbs AFTER selecting, not before.
    """
    ids = list(range(4000, 4200))
    texts = [LONG_DOC[: 40 + i] for i in range(200)]
    labels = np.array(["card"] * 200, dtype=object)
    noisy = perturb.perturb_texts(texts, ids, "typo", 0.3, SEED)
    assert noisy != texts

    clean_ids, clean_texts, _ = tier_c.subsample_eval(ids, texts, labels, 50, 20260806)
    noisy_ids, noisy_texts, _ = tier_c.subsample_eval(ids, noisy, labels, 50, 20260806)
    assert clean_ids == noisy_ids
    # ...and perturbing the SELECTED rows is the same thing as selecting from the perturbed
    # frame, so the runner's order costs nothing but guarantees the id set
    assert perturb.perturb_texts(clean_texts, clean_ids, "typo", 0.3, SEED) == noisy_texts


def test_the_1500_row_cap_is_a_subset_of_the_5000_row_cap_at_the_same_seed():
    """The pairing claim behind the tier_c arm: same cap_seed, prefix of one permutation."""
    n = 104_443  # TEST-IID row count; the property is a fact about subsample_eval, not data
    ids = list(range(n))
    texts = [""] * n
    labels = np.zeros(n, dtype=object)
    big, _, _ = tier_c.subsample_eval(ids, texts, labels, 5000, 20260806)
    small, _, _ = tier_c.subsample_eval(ids, texts, labels, 1500, 20260806)
    assert len(small) == 1500 and len(big) == 5000
    assert set(small) <= set(big)
    # a different cap_seed breaks the pairing, which is why the configs must not touch it
    other, _, _ = tier_c.subsample_eval(ids, texts, labels, 1500, 20260807)
    assert not set(other) <= set(big)


def test_shipped_tier_c_perturb_configs_pair_with_the_sonnet_subset():
    sonnet = harness.load_config(
        harness.REPO_ROOT / "configs" / "tier_c_sonnet_zeroshot_test_iid.yaml"
    )["data"]
    clean = harness.load_config(
        harness.REPO_ROOT / "configs" / "tier_c_haiku_zeroshot_test_iid.yaml"
    )
    paths = sorted(
        (harness.REPO_ROOT / "configs").glob("tier_c_haiku_zeroshot_test_iid_perturb_*.yaml")
    )
    assert len(paths) == 3
    for path in paths:
        config = harness.load_config(path)
        data = config["data"]
        # every field that determines WHICH rows are evaluated matches the Sonnet arm
        for key in ("split", "eval_rows_cap", "cap_seed", "order_column", "text_column",
                    "label_column"):
            assert data[key] == sonnet[key], (path.name, key)
        # ...and everything that determines WHAT is asked matches the clean Haiku run
        assert config["model"]["slug"] == clean["model"]["slug"]
        assert config["prompt"] == clean["prompt"]
        assert config["model"]["params"] == clean["model"]["params"]
        assert config["seed"] == clean["seed"]
        spec = perturb.spec_from_config(config)
        assert spec.rate == 0.10 and spec.seed == perturb.DEFAULT_SEED
        assert path.stem.endswith(f"_perturb_{spec.family}_10")


_TIER_C_TEXTS = [
    "the collector called me daily about a closed card balance",
    "my mortgage escrow shortage notice was never explained",
    "credit report error the bureau refuses to correct",
    "student loan servicer misapplied my payment for months",
    "the bank closed my checking account without any notice",
    "a debt i never owed was reported to the bureaus",
]


class _RecordingCompletions:
    """Fake OpenRouter completions endpoint: records every request, returns a fixed label."""

    def __init__(self, label):
        self.label = label
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        usage = SimpleNamespace(prompt_tokens=120, completion_tokens=8, total_tokens=128,
                                cost=1.0e-4)
        choice = SimpleNamespace(
            message=SimpleNamespace(content=json.dumps({"label": self.label})),
            finish_reason="stop",
        )
        return SimpleNamespace(choices=[choice], usage=usage, provider="Anthropic")


@pytest.fixture
def tier_c_env(tmp_path, monkeypatch):
    """A network-free Tier C environment: synthetic split, fake client, tmp receipt root."""
    pytest.importorskip("openai")
    from triage_lab.tier_c_prompt import load_prompt_bundle

    bundle = load_prompt_bundle("v1")
    labels = list(bundle.labels)

    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    ids = list(range(6000, 6006))
    y = [labels[i % len(labels)] for i in range(len(_TIER_C_TEXTS))]
    path = splits_dir / "test_iid.parquet"
    _write_parquet(path, ids, _TIER_C_TEXTS, y)
    (splits_dir / "splits_stats.yaml").write_text(
        yaml.safe_dump({"input_sha256": "synthetic-input",
                        "splits": {"test_iid": {"sha256": sha256_file(path)}}})
    )

    completions = _RecordingCompletions(labels[0])
    monkeypatch.setattr(tier_c, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(tier_c, "RAW_LOG_ROOT", tmp_path / "tier_c_raw")
    monkeypatch.setattr(tier_c, "load_openrouter_key", lambda *a, **k: "test-key")
    monkeypatch.setattr(
        tier_c, "fetch_models_payload",
        lambda *a, **k: {"data": [{"id": "anthropic/claude-haiku-4.5",
                                   "pricing": {"prompt": "1e-6", "completion": "5e-6"}}]},
    )
    monkeypatch.setattr(
        tier_c, "build_client",
        lambda *a, **k: SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    return SimpleNamespace(splits_dir=splits_dir, ids=ids, labels=labels,
                           completions=completions, bundle=bundle, tmp_path=tmp_path)


def _tier_c_config(env, perturbation=None, cap=None) -> dict:
    config = {
        "model": {"runner": "tier_c", "name": "fixture_haiku",
                  "slug": "anthropic/claude-haiku-4.5",
                  "params": {"temperature": 0.0, "max_tokens": 64, "max_retries": 0,
                             "max_concurrency": 1}},
        "prompt": {"version": "v1", "num_exemplars": 0},
        "data": {"split": "test_iid", "text_column": "narrative", "label_column": "class",
                 "order_column": "complaint_id", "splits_dir": str(env.splits_dir),
                 "cap_seed": 20260806},
        "seed": 20260806,
    }
    if cap is not None:
        config["data"]["eval_rows_cap"] = cap
    if perturbation is not None:
        config["data"]["perturbation"] = perturbation
    return config


def test_tier_c_perturbs_the_narrative_it_sends_and_echoes_the_block(tier_c_env):
    spec = {"family": "ocr", "rate": 1.0, "seed": 20260805}
    result = tier_c.tier_c_runner(_tier_c_config(tier_c_env, spec))

    assert result.extra["perturbation"] == {"family": "ocr", "rate": 1.0, "seed": 20260805}
    expected = perturb.perturb_texts(_TIER_C_TEXTS, tier_c_env.ids, "ocr", 1.0, 20260805)
    assert expected != _TIER_C_TEXTS
    sent = [req["messages"][-1]["content"] for req in tier_c_env.completions.requests]
    assert len(sent) == len(expected)
    for narrative, message in zip(expected, sent, strict=True):
        assert narrative in message
    # ids, labels and the frozen prompt bundle are untouched by the perturbation
    np.testing.assert_array_equal(result.ids, np.array(tier_c_env.ids, dtype=np.int64))
    assert result.extra["prompt_bundle_sha256"] == tier_c_env.bundle.bundle_sha256
    assert result.extra["num_exemplars"] == 0


def test_tier_c_selects_rows_before_perturbing_them(tier_c_env):
    """Same cap + cap_seed -> the same complaint_ids, clean or perturbed."""
    clean = tier_c.tier_c_runner(_tier_c_config(tier_c_env, cap=3))
    clean_ids = list(clean.ids)
    tier_c_env.completions.requests.clear()

    perturbed = tier_c.tier_c_runner(
        _tier_c_config(tier_c_env, {"family": "typo", "rate": 1.0, "seed": 20260805}, cap=3)
    )
    assert list(perturbed.ids) == clean_ids
    np.testing.assert_array_equal(perturbed.y_true, clean.y_true)
    # and the sent text is the perturbation OF THE SELECTED ROWS
    selected = [_TIER_C_TEXTS[tier_c_env.ids.index(i)] for i in clean_ids]
    expected = perturb.perturb_texts(selected, clean_ids, "typo", 1.0, 20260805)
    sent = [req["messages"][-1]["content"] for req in tier_c_env.completions.requests]
    for narrative, message in zip(expected, sent, strict=True):
        assert narrative in message


def test_tier_c_clean_run_records_no_perturbation_key(tier_c_env):
    result = tier_c.tier_c_runner(_tier_c_config(tier_c_env))
    assert "perturbation" not in result.extra
    sent = [req["messages"][-1]["content"] for req in tier_c_env.completions.requests]
    for narrative, message in zip(_TIER_C_TEXTS, sent, strict=True):
        assert narrative in message


def test_tier_c_passes_the_harness_perturbation_gate(tier_c_env, tmp_path):
    """End-to-end: the block reaches the record and harness._check_perturbation_applied
    accepts it (a tier that ignored the block would fail here)."""
    cfg_path = tmp_path / "tier_c_run.yaml"
    cfg_path.write_text(yaml.safe_dump(
        _tier_c_config(tier_c_env, {"family": "case", "rate": 0.1, "seed": 20260805})
    ))
    results = tmp_path / "out" / "runs.jsonl"
    record = harness.run(cfg_path, results)
    on_disk = json.loads(results.read_text().splitlines()[0])
    assert on_disk["extra"]["perturbation"] == {"family": "case", "rate": 0.1,
                                                "seed": 20260805}
    assert on_disk["cost_usd"] == pytest.approx(6 * (120 * 1e-6 + 8 * 5e-6))
    assert record["extra"]["n_examples"] == 6


# ---------------------------------------------------------------------------
# (h) perturb_report: the tier_c subset join + optional-arm behaviour
# ---------------------------------------------------------------------------

@pytest.fixture
def subset_pair(tmp_path):
    """A 300-row clean artifact and a 100-row perturbed artifact drawn from its ids."""
    rng = np.random.default_rng(5)
    ids = np.arange(8000, 8300)
    y_true = np.array([LABELS[i % 3] for i in range(300)], dtype=object)
    clean_pred = y_true.copy()
    clean = _synthetic_artifact(tmp_path / "c5k.parquet", ids, y_true, clean_pred, "clean")

    take = np.sort(rng.permutation(300)[:100])
    pert_pred = y_true[take].copy()
    for j in range(0, 100, 5):          # 20 disagreements inside the subset
        pert_pred[j] = LABELS[(LABELS.index(pert_pred[j]) + 1) % 3]
    perturbed = _synthetic_artifact(
        tmp_path / "p1500.parquet", ids[take], y_true[take], pert_pred, "pert"
    )
    return clean, perturbed, ids[take]


def test_subset_join_restricts_the_clean_side_to_the_perturbed_ids(subset_pair):
    clean, perturbed, take_ids = subset_pair
    pair = perturb_report.align_pair(
        clean, perturbed, join=perturb_report.JOIN_SUBSET, expected_n=100
    )
    assert len(pair.ids) == 100
    np.testing.assert_array_equal(pair.ids, np.sort(take_ids))
    # the clean rows carried over are the ones belonging to those ids, not the first 100
    assert np.array_equal(pair.clean_pred, pair.y_true)
    assert int(np.count_nonzero(pair.perturbed_pred != pair.y_true)) == 20
    out = perturb_report.pair_deltas(pair, n_resamples=200)
    assert out["accuracy"]["delta"] == pytest.approx(-20 / 100)
    assert out["macro_f1"]["ci_hi"] < 0.0


def test_subset_join_refuses_a_perturbed_id_the_clean_run_never_scored(tmp_path, subset_pair):
    clean, perturbed, _ = subset_pair
    stray = perturbed.complaint_id.copy()
    stray[0] = 999_999
    bad = _synthetic_artifact(
        tmp_path / "stray.parquet", stray, perturbed.y_true, perturbed.y_pred, "stray"
    )
    with pytest.raises(ValueError, match="not a subset of the clean one"):
        perturb_report.align_pair(clean, bad, join=perturb_report.JOIN_SUBSET)


def test_subset_join_enforces_the_frozen_subsample_size(subset_pair):
    clean, perturbed, _ = subset_pair
    with pytest.raises(ValueError, match="expected 1500 paired rows"):
        perturb_report.align_pair(
            clean, perturbed, join=perturb_report.JOIN_SUBSET, expected_n=1500
        )


def test_identical_join_still_refuses_a_subset(subset_pair):
    """The Tier A arms must NOT silently accept a partial row set."""
    clean, perturbed, _ = subset_pair
    with pytest.raises(ValueError, match="identical rows"):
        perturb_report.align_pair(clean, perturbed)


def test_tier_c_arm_is_declared_as_the_optional_subset_arm():
    arm = perturb_report.ARMS_BY_KEY["tier_c_haiku"]
    assert arm.join == perturb_report.JOIN_SUBSET
    assert arm.expected_n == 1500
    assert arm.optional is True
    assert arm.rate_tags == ("10",)
    assert arm.clean_config == "tier_c_haiku_zeroshot_test_iid"
    # case is NOT a structural zero for an LLM: its tokenizer is case-sensitive
    assert arm.structural_zero_families == ()
    for tier_a_arm in perturb_report.ARMS:
        if tier_a_arm.key != "tier_c_haiku":
            assert tier_a_arm.structural_zero_families == ("case",)
            assert tier_a_arm.join == perturb_report.JOIN_IDENTICAL
            assert tier_a_arm.optional is False


def test_tier_c_case_row_is_not_labelled_structural(subset_pair):
    clean, perturbed, _ = subset_pair
    pair = perturb_report.align_pair(clean, perturbed, join=perturb_report.JOIN_SUBSET)
    row = perturb_report.build_row(
        perturb_report.ARMS_BY_KEY["tier_c_haiku"], "case", "10",
        clean_record={"run_id": "70a1b0c4", "config_path": "configs/clean.yaml"},
        perturbed_record={"run_id": "p", "config_path": "configs/p.yaml",
                          "extra": {"perturbation": {"family": "case", "rate": 0.1,
                                                     "seed": 20260805}}},
        pair=pair, n_resamples=20,
    )
    assert row["structural_expectation"] is None
    assert row["join"] == perturb_report.JOIN_SUBSET
    assert row["n_rows"] == 100


def test_all_grid_covers_tier_c_and_the_shipped_configs():
    cells = perturb_report.jobs()
    assert len(cells) == 18                        # 15 tier A + 3 tier C
    tier_c_cells = [(c[1], c[2]) for c in cells if c[0].key == "tier_c_haiku"]
    assert tier_c_cells == [("typo", "10"), ("ocr", "10"), ("case", "10")]
    names = {
        arm.perturbed_config(fam, tag) for arm, fam, tag in cells if arm.key == "tier_c_haiku"
    }
    on_disk = {
        p.stem for p in
        (harness.REPO_ROOT / "configs").glob("tier_c_haiku_zeroshot_test_iid_perturb_*.yaml")
    }
    assert names == on_disk


def test_unrun_tier_c_cells_are_skipped_not_reported_as_holes(tmp_path):
    """`--all` must not turn "the LLM arm has not been paid for yet" into a report hole."""
    results = tmp_path / "runs.jsonl"
    results.write_text("")
    lines: list[str] = []
    summary = perturb_report.run(
        perturb_report.select_jobs(["tier_c_haiku"], None),
        results_path=results, preds_dir=tmp_path / "preds", out_dir=tmp_path / "out",
        write_summary=False, log=lines.append,
    )
    assert summary["rows"] == [] and summary["missing"] == []
    assert [s["cell"] for s in summary["skipped"]] == [
        "tier_c_haiku/typo/10", "tier_c_haiku/ocr/10", "tier_c_haiku/case/10"
    ]
    assert all("not run yet" in line for line in lines)


def test_unrun_tier_a_cells_are_still_reported_as_holes(tmp_path):
    results = tmp_path / "runs.jsonl"
    results.write_text("")
    summary = perturb_report.run(
        perturb_report.select_jobs(["logreg_word_only"], ["ocr"]),
        results_path=results, preds_dir=tmp_path / "preds", out_dir=tmp_path / "out",
        write_summary=False, log=lambda *a: None,
    )
    assert summary["skipped"] == []
    assert [m["cell"] for m in summary["missing"]] == ["logreg_word_only/ocr/10"]
