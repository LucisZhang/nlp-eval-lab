"""Cost-model tests: hand-computed per-example arithmetic (API cost is incurred spend,
charged even on human-deferred rows), the human-queue assumption, shared-index bootstrap
determinism, the Tier C receipt gate (tokens, slug, price re-derivation, duplicates,
cost-sum), the artifact provenance gate, and CLI atomicity/determinism."""

from __future__ import annotations

import json

import numpy as np
import pytest

from triage_lab import cost_model, harness, predictions

C_MISROUTE = 6.00
C_HUMAN = 2.50

SLUG = "anthropic/claude-haiku-4.5"
PROMPT_RATE = 1e-6
COMPLETION_RATE = 5e-6
PRICING = {
    "slug": SLUG,
    "prompt_usd_per_token": PROMPT_RATE,
    "completion_usd_per_token": COMPLETION_RATE,
    "source": "https://openrouter.ai/api/v1/models",
}


# ---------------------------------------------------------------------------
# Inline fixtures (no conftest, no pyarrow: artifacts round-trip through DuckDB)
# ---------------------------------------------------------------------------

def _cost_yaml(*, version="v1", c_misroute=C_MISROUTE, c_human=C_HUMAN,
               tier_a_per_example="0.0"):
    per_example = (
        f"    per_example_usd: {tier_a_per_example}\n" if tier_a_per_example is not None
        else ""
    )
    return (
        f"version: {version}\n"
        "params:\n"
        f"  c_misroute_usd: {c_misroute}\n"
        f"  c_human_usd: {c_human}\n"
        "api_cost:\n"
        "  tier_a:\n"
        "    mode: amortized_zero\n"
        f"{per_example}"
        "    evidence_class: estimated\n"
        "    note: amortized CPU inference\n"
        "  tier_c:\n"
        "    mode: measured_receipts\n"
        "    evidence_class: measured\n"
        "    note: joined from committed per-call receipts\n"
        "evidence_class:\n"
        "  params.c_misroute_usd: estimated\n"
        "  params.c_human_usd: estimated\n"
    )


def _write_cost_config(tmp_path, *, name="cost.yaml", **kwargs):
    path = tmp_path / name
    path.write_text(_cost_yaml(**kwargs))
    return path


def _write_artifact(tmp_path, run_id, ids, y_true, y_pred, labels, *, split="cal",
                    split_sha256="splithash", config_sha256="cfghash",
                    prompt_bundle_sha256=""):
    index = {lbl: i for i, lbl in enumerate(labels)}
    probs = np.zeros((len(ids), len(labels)), dtype=np.float64)
    for i, pred in enumerate(y_pred):
        probs[i, index[pred]] = 1.0
    prov = predictions.ArtifactProvenance(
        run_id=run_id, config_sha256=config_sha256, split=split,
        split_sha256=split_sha256, class_labels=labels,
        prompt_bundle_sha256=prompt_bundle_sha256,
    )
    path = tmp_path / f"{run_id}.parquet"
    predictions.write_artifact(
        path, ids=np.asarray(ids, dtype=np.int64), y_true=y_true, y_pred=y_pred,
        probs=probs, class_labels=labels, provenance=prov,
    )
    return path


_UNSET = object()  # so a test can force computed_cost_usd to literal null


def _receipt(cid, *, prompt=995, completion=1, cost=_UNSET, total=None, slug=SLUG):
    """One receipt line. Defaults are internally consistent: cost follows from tokens."""
    return {
        "complaint_id": cid,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion if total is None else total,
        "computed_cost_usd": (prompt * PROMPT_RATE + completion * COMPLETION_RATE
                              if cost is _UNSET else cost),
        "slug": slug,
        "content": '{"label": "a"}',
        "provider": "Amazon Bedrock",
    }


def _write_receipts(path, receipts):
    path.write_text("".join(json.dumps(r) + "\n" for r in receipts), encoding="utf-8")
    return path


def _record(run_id, config_name, *, cost_usd=None, raw_log_path=None, split="cal",
            split_sha256="splithash", config_sha256="cfghash", model_slug=SLUG,
            pricing=PRICING, prompt_bundle_sha256=None):
    rec = {
        "run_id": run_id,
        "config_path": f"configs/{config_name}.yaml",
        "config_sha256": config_sha256,
        "dataset": {"split": split, "split_sha256": split_sha256},
    }
    if cost_usd is not None:
        rec["cost_usd"] = cost_usd
    extra = {}
    if raw_log_path is not None:
        extra["raw_log_path"] = str(raw_log_path)
        if model_slug is not None:
            extra["model_slug"] = model_slug
        if pricing is not None:
            extra["pricing_snapshot"] = pricing
    if prompt_bundle_sha256 is not None:
        extra["prompt_bundle_sha256"] = prompt_bundle_sha256
    if extra:
        rec["extra"] = extra
    return rec


def _tier_c_fixture(tmp_path, run_id, ids, receipts, *, y_true=None, y_pred=None,
                    labels=("a", "b")):
    """Artifact + receipts + record whose cost_usd is the receipts' true sum."""
    labels = list(labels)
    y_true = list(y_true) if y_true is not None else [labels[0]] * len(ids)
    y_pred = list(y_pred) if y_pred is not None else [labels[0]] * len(ids)
    _write_artifact(tmp_path, run_id, ids, y_true, y_pred, labels)
    log = _write_receipts(tmp_path / f"calls_{run_id[:4]}.jsonl", receipts)
    total = sum(r["computed_cost_usd"] for r in receipts
                if isinstance(r["computed_cost_usd"], (int, float)))
    return _record(run_id, "tier_c_toy", cost_usd=total, raw_log_path=log)


# ---------------------------------------------------------------------------
# Core arithmetic: hand-computed
# ---------------------------------------------------------------------------

def test_hand_computed_five_example_case():
    # 5 examples: two answered-and-wrong, one human-deferred (index 4), nonzero API costs.
    correct = np.array([True, False, True, False, True])
    api = np.array([0.001, 0.002, 0.003, 0.004, 0.005])
    to_human = np.array([False, False, False, False, True])

    comps = cost_model.cost_components(correct, api, to_human,
                                       c_misroute=C_MISROUTE, c_human=C_HUMAN)
    assert comps["misroute"].tolist() == [0.0, 6.0, 0.0, 6.0, 0.0]
    # index 4 was deferred to a human but its call was already paid for: still charged.
    assert comps["api"] == pytest.approx([0.001, 0.002, 0.003, 0.004, 0.005])
    assert comps["human"].tolist() == [0.0, 0.0, 0.0, 0.0, 2.5]

    per_example = cost_model.per_example_cost(correct, api, to_human,
                                              c_misroute=C_MISROUTE, c_human=C_HUMAN)
    assert per_example == pytest.approx([0.001, 6.002, 0.003, 6.004, 2.505])

    got = cost_model.expected_cost_per_1k(correct, api, to_human,
                                          c_misroute=C_MISROUTE, c_human=C_HUMAN)
    # misroute: 2 wrong-and-answered x $6 = $12 over 5 -> $2.40/example -> $2400/1k
    # api:      0.001+0.002+0.003+0.004+0.005 = $0.015 over 5 -> $0.003/ex -> $3.00/1k
    # human:    1 x $2.50 over 5 -> $0.50/example -> $500/1k
    assert got["misroute"] == pytest.approx(2400.0)
    assert got["api"] == pytest.approx(3.0)
    assert got["human"] == pytest.approx(500.0)
    assert got["total"] == pytest.approx(2903.0)


def test_human_deferred_row_pays_human_plus_incurred_api():
    # A paid call followed by a human hand-off: c_human AND the API spend, no misroute.
    comps = cost_model.cost_components(
        [False], [0.5], [True], c_misroute=C_MISROUTE, c_human=C_HUMAN,
    )
    assert comps["misroute"].tolist() == [0.0]
    assert comps["api"].tolist() == [0.5]
    assert comps["human"].tolist() == [C_HUMAN]
    got = cost_model.expected_cost_per_1k([False], [0.5], [True],
                                          c_misroute=C_MISROUTE, c_human=C_HUMAN)
    assert got["total"] == pytest.approx((0.5 + 2.5) * 1000)
    # ...and the same example answered instead pays misroute + api, never c_human.
    answered = cost_model.expected_cost_per_1k([False], [0.5], [False],
                                               c_misroute=C_MISROUTE, c_human=C_HUMAN)
    assert answered["total"] == pytest.approx((6.0 + 0.5) * 1000)
    assert answered["human"] == 0.0


def test_deferring_before_any_paid_call_costs_c_human_only():
    # The caller expresses "abstained before spending anything" as api_cost_usd = 0.0.
    got = cost_model.expected_cost_per_1k([False], [0.0], [True],
                                          c_misroute=C_MISROUTE, c_human=C_HUMAN)
    assert got["total"] == pytest.approx(2500.0)
    assert got["api"] == 0.0
    assert got["misroute"] == 0.0


def test_misaligned_and_empty_inputs_rejected():
    with pytest.raises(ValueError, match="misaligned"):
        cost_model.cost_components([True, False], [0.1], [False, False],
                                   c_misroute=C_MISROUTE, c_human=C_HUMAN)
    with pytest.raises(ValueError, match="empty policy"):
        cost_model.cost_components([], [], [], c_misroute=C_MISROUTE, c_human=C_HUMAN)
    with pytest.raises(ValueError, match="non-negative"):
        cost_model.cost_components([True], [-0.1], [False],
                                   c_misroute=C_MISROUTE, c_human=C_HUMAN)


# ---------------------------------------------------------------------------
# Bootstrap: frozen constants, shared indices, determinism, bracketing
# ---------------------------------------------------------------------------

def _random_policy(seed, n=200):
    rng = np.random.default_rng(seed)
    correct = rng.random(n) < 0.7
    api = rng.random(n) * 0.004
    to_human = rng.random(n) < 0.15
    return correct, api, to_human


def test_bootstrap_is_deterministic_and_brackets_the_point():
    correct, api, to_human = _random_policy(11)
    a = cost_model.bootstrap_cost(correct, api, to_human,
                                  c_misroute=C_MISROUTE, c_human=C_HUMAN)
    b = cost_model.bootstrap_cost(correct, api, to_human,
                                  c_misroute=C_MISROUTE, c_human=C_HUMAN)
    assert a == b
    for key in ("total", "misroute", "api", "human"):
        band = a[key]
        assert band["ci_lo"] <= band["point"] <= band["ci_hi"], key
        assert band["ci_lo"] < band["ci_hi"], key


def test_bootstrap_uses_the_frozen_harness_constants():
    # The cost bands must ride the same frozen contract as every other CI in the lab.
    correct, api, to_human = _random_policy(12, n=50)
    default = cost_model.bootstrap_cost(correct, api, to_human,
                                        c_misroute=C_MISROUTE, c_human=C_HUMAN)
    explicit = cost_model.bootstrap_cost(
        correct, api, to_human, c_misroute=C_MISROUTE, c_human=C_HUMAN,
        n_resamples=harness.N_RESAMPLES, seed=harness.BOOTSTRAP_SEED,
    )
    assert default == explicit
    assert (harness.N_RESAMPLES, harness.BOOTSTRAP_SEED) == (1000, 20260805)
    assert (harness.CI_LOWER_PCT, harness.CI_UPPER_PCT) == (2.5, 97.5)


def test_resample_uses_one_shared_index_draw_per_replicate():
    # Hand-computed: 4 examples, 3 replicates. Every replicate's component means must sum
    # to that replicate's total mean, which is only true if the SAME index vector is used
    # for all four arrays. Verified additionally against a hand-rolled reference that
    # replays default_rng(seed) itself.
    correct = np.array([True, False, False, True])
    api = np.array([0.001, 0.002, 0.003, 0.004])
    to_human = np.array([False, False, True, False])
    comps = cost_model.cost_components(correct, api, to_human,
                                       c_misroute=C_MISROUTE, c_human=C_HUMAN)
    total = comps["misroute"] + comps["api"] + comps["human"]
    arrays = {"total": total, **comps}

    reps = cost_model.resample_means(arrays, scale=cost_model.PER_N_COMPLAINTS,
                                     n_resamples=3, seed=42)
    summed = reps["misroute"] + reps["api"] + reps["human"]
    assert summed == pytest.approx(reps["total"], abs=1e-9)

    rng = np.random.default_rng(42)
    for i in range(3):
        idx = rng.integers(0, 4, size=4)
        for key, arr in arrays.items():
            assert reps[key][i] == pytest.approx(arr[idx].mean() * 1000, abs=1e-12), key

    with pytest.raises(ValueError, match="id-aligned"):
        cost_model.resample_means({"a": np.zeros(3), "b": np.zeros(4)},
                                  n_resamples=2, seed=1)


def test_component_points_sum_to_total():
    correct, api, to_human = _random_policy(13)
    bands = cost_model.bootstrap_cost(correct, api, to_human,
                                      c_misroute=C_MISROUTE, c_human=C_HUMAN)
    parts = sum(bands[k]["point"] for k in cost_model.COMPONENT_KEYS)
    assert parts == pytest.approx(bands["total"]["point"], abs=1e-9)
    spans = sum(bands[k]["ci_hi"] - bands[k]["ci_lo"] for k in cost_model.COMPONENT_KEYS)
    assert bands["total"]["ci_hi"] - bands["total"]["ci_lo"] <= spans + 1e-9


# ---------------------------------------------------------------------------
# Cost config: parsing, validation, hash binding
# ---------------------------------------------------------------------------

def test_shipped_cost_config_defaults():
    cfg = cost_model.load_cost_config()
    assert cfg.version == "v1"
    assert cfg.c_misroute_usd == 6.00
    assert cfg.c_human_usd == 2.50
    tier_a = cfg.api_policy("tier_a")
    assert tier_a["mode"] == "amortized_zero"
    assert tier_a["evidence_class"] == "estimated"
    assert tier_a["per_example_usd"] == 0.0  # stated explicitly, never defaulted in code
    assert cfg.api_policy("tier_c")["mode"] == "measured_receipts"
    assert cfg.api_policy("tier_c")["evidence_class"] == "measured"
    assert cfg.sha256 == harness.config_sha256(cost_model.DEFAULT_COST_CONFIG)


def test_unpriced_tier_hard_fails_rather_than_charging_zero():
    cfg = cost_model.load_cost_config()
    with pytest.raises(ValueError, match="does not name a tier priced"):
        cost_model.tier_of_config_name("tier_b1_modernbert_sa", cfg)
    with pytest.raises(ValueError, match="no api_cost policy for tier"):
        cfg.api_policy("tier_b")


def test_non_finite_or_negative_prices_are_rejected(tmp_path):
    for bad in (".nan", ".inf", "-1.0"):
        path = _write_cost_config(tmp_path, c_misroute=bad, name=f"bad_{bad}.yaml")
        with pytest.raises(ValueError, match="finite non-negative"):
            cost_model.load_cost_config(path)
    for bad in (".nan", "-2.0"):
        path = _write_cost_config(tmp_path, c_human=bad, name=f"badh_{bad}.yaml")
        with pytest.raises(ValueError, match="finite non-negative"):
            cost_model.load_cost_config(path)


def test_amortized_zero_requires_an_explicit_per_example_usd(tmp_path):
    run_id = "a0" * 32
    _write_artifact(tmp_path, run_id, [1, 2], ["a", "b"], ["a", "b"], ["a", "b"])
    record = _record(run_id, "tier_a_toy")
    cfg = cost_model.load_cost_config(
        _write_cost_config(tmp_path, tier_a_per_example=None, name="nopx.yaml"))
    with pytest.raises(ValueError, match="declares no per_example_usd"):
        cost_model.score_run(record, cfg, preds_dir=tmp_path)
    bad = cost_model.load_cost_config(
        _write_cost_config(tmp_path, tier_a_per_example="-0.5", name="negpx.yaml"))
    with pytest.raises(ValueError, match="finite and non-negative"):
        cost_model.score_run(record, bad, preds_dir=tmp_path)


def test_config_sha256_is_bound_into_output_and_tracks_edits(tmp_path):
    run_id = "a1" * 32
    _write_artifact(tmp_path, run_id, [1, 2, 3], ["a", "b", "a"], ["a", "b", "b"],
                    ["a", "b"])
    record = _record(run_id, "tier_a_toy")

    cfg = cost_model.load_cost_config(_write_cost_config(tmp_path))
    obj = cost_model.score_run(record, cfg, preds_dir=tmp_path)
    assert obj["cost_config"]["sha256"] == cfg.sha256
    assert obj["cost_config"]["version"] == "v1"
    assert obj["cost_config"]["params"] == {"c_misroute_usd": 6.0, "c_human_usd": 2.5}
    assert obj["schema_version"] == cost_model.SCHEMA_VERSION

    # Editing the YAML changes the recorded hash (and the numbers it produced).
    cfg2 = cost_model.load_cost_config(
        _write_cost_config(tmp_path, c_misroute=9.0, name="cost2.yaml"))
    obj2 = cost_model.score_run(record, cfg2, preds_dir=tmp_path)
    assert cfg2.sha256 != cfg.sha256
    assert obj2["cost_config"]["sha256"] == cfg2.sha256
    assert obj2["expected_cost_per_1k"]["misroute"]["point"] > \
        obj["expected_cost_per_1k"]["misroute"]["point"]


# ---------------------------------------------------------------------------
# Provenance gate
# ---------------------------------------------------------------------------

def test_provenance_mismatch_is_a_hard_failure(tmp_path):
    cfg = cost_model.load_cost_config(_write_cost_config(tmp_path))
    ids, y_true, y_pred, labels = [1, 2], ["a", "b"], ["a", "b"], ["a", "b"]

    # Artifact says split_sha256 'wrong'; the record names the real one.
    run_id = "b0" * 32
    _write_artifact(tmp_path, run_id, ids, y_true, y_pred, labels, split_sha256="wrong")
    with pytest.raises(ValueError, match="provenance mismatch.*split_sha256"):
        cost_model.score_run(_record(run_id, "tier_a_toy"), cfg, preds_dir=tmp_path)

    # Config hash disagreement (the artifact came from different config bytes).
    run_id = "b1" * 32
    _write_artifact(tmp_path, run_id, ids, y_true, y_pred, labels, config_sha256="other")
    with pytest.raises(ValueError, match="provenance mismatch.*config_sha256"):
        cost_model.score_run(_record(run_id, "tier_a_toy"), cfg, preds_dir=tmp_path)

    # Split name disagreement.
    run_id = "b2" * 32
    _write_artifact(tmp_path, run_id, ids, y_true, y_pred, labels, split="test_iid")
    with pytest.raises(ValueError, match="provenance mismatch.*split"):
        cost_model.score_run(_record(run_id, "tier_a_toy"), cfg, preds_dir=tmp_path)

    # Prompt hash present on the artifact but absent from the record: "not verified" is
    # not "verified", so the asymmetry fails too.
    run_id = "b3" * 32
    _write_artifact(tmp_path, run_id, ids, y_true, y_pred, labels,
                    prompt_bundle_sha256="ph")
    with pytest.raises(ValueError, match="provenance mismatch.*prompt_bundle_sha256"):
        cost_model.score_run(_record(run_id, "tier_a_toy"), cfg, preds_dir=tmp_path)


def test_matching_prompt_bundle_hash_passes(tmp_path):
    run_id = "b4" * 32
    ids = [10, 11]
    labels = ["a", "b"]
    _write_artifact(tmp_path, run_id, ids, ["a", "b"], ["a", "b"], labels,
                    prompt_bundle_sha256="ph")
    log = _write_receipts(tmp_path / "calls.jsonl",
                          [_receipt(10), _receipt(11, prompt=1995)])
    total = 995 * PROMPT_RATE + 1995 * PROMPT_RATE + 2 * COMPLETION_RATE
    record = _record(run_id, "tier_c_toy", cost_usd=total, raw_log_path=log,
                     prompt_bundle_sha256="ph")
    cfg = cost_model.load_cost_config(_write_cost_config(tmp_path))
    obj = cost_model.score_run(record, cfg, preds_dir=tmp_path)
    assert obj["cost_sum_check"]["ok"] is True


# ---------------------------------------------------------------------------
# Single-tier policy builders: Tier A amortized zero, Tier C measured receipts
# ---------------------------------------------------------------------------

def test_tier_a_policy_is_amortized_zero_and_all_answered(tmp_path):
    run_id = "c0" * 32
    _write_artifact(tmp_path, run_id, [1, 2, 3, 4], ["a", "a", "b", "b"],
                    ["a", "b", "b", "b"], ["a", "b"])
    cfg = cost_model.load_cost_config(_write_cost_config(tmp_path))
    obj = cost_model.score_run(_record(run_id, "tier_a_toy"), cfg, preds_dir=tmp_path)

    assert obj["tier"] == "tier_a"
    assert obj["policy"] == "single_tier_all_answered"
    assert obj["n_to_human"] == 0
    assert obj["accuracy"] == pytest.approx(0.75)
    assert obj["api_cost"]["mode"] == "amortized_zero"
    assert obj["api_cost"]["evidence_class"] == "estimated"
    assert obj["api_cost"]["total_usd"] == 0.0
    assert obj["cost_sum_check"] is None
    # one wrong of four, no API, no human -> $6 * 0.25 * 1000
    assert obj["expected_cost_per_1k"]["total"]["point"] == pytest.approx(1500.0)
    assert obj["expected_cost_per_1k"]["api"]["point"] == 0.0
    assert obj["expected_cost_per_1k"]["human"]["point"] == 0.0
    assert obj["bootstrap"]["seed"] == harness.BOOTSTRAP_SEED
    assert "incurred spend" in obj["human_assumption"].lower()


def test_tier_c_policy_joins_receipts_and_passes_cost_sum_gate(tmp_path):
    run_id = "c1" * 32
    ids = [10, 11, 12]
    receipts = [  # written in completion order, not id order
        _receipt(12, prompt=2995), _receipt(10, prompt=995), _receipt(11, prompt=1995),
    ]
    record = _tier_c_fixture(tmp_path, run_id, ids, receipts,
                             y_true=["a", "b", "a"], y_pred=["a", "b", "b"])
    cfg = cost_model.load_cost_config(_write_cost_config(tmp_path))
    obj = cost_model.score_run(record, cfg, preds_dir=tmp_path)

    assert obj["tier"] == "tier_c"
    assert obj["api_cost"]["mode"] == "measured_receipts"
    assert obj["api_cost"]["evidence_class"] == "measured"
    assert obj["api_cost"]["total_usd"] == pytest.approx(0.006)
    assert obj["cost_sum_check"]["ok"] is True
    assert obj["cost_sum_check"]["abs_delta"] <= cost_model.COST_SUM_TOL
    # api/1k = mean(0.001, 0.002, 0.003) * 1000 = $2.00; misroute = $6 * 1/3 * 1000
    assert obj["expected_cost_per_1k"]["api"]["point"] == pytest.approx(2.0)
    assert obj["expected_cost_per_1k"]["misroute"]["point"] == pytest.approx(2000.0)
    assert obj["expected_cost_per_1k"]["total"]["point"] == pytest.approx(2002.0)


def test_tier_c_join_is_keyed_on_complaint_id_not_line_order(tmp_path):
    # Artifact id order and receipt line order deliberately disagree, and each id has a
    # distinct cost: a positional zip would assign every example the wrong price while
    # leaving the cost SUM (and therefore the sum gate) perfectly intact.
    run_id = "c2" * 32
    ids = [12, 10, 11]
    receipts = [_receipt(10, prompt=995), _receipt(11, prompt=1995),
                _receipt(12, prompt=2995)]
    record = _tier_c_fixture(tmp_path, run_id, ids, receipts)
    cfg = cost_model.load_cost_config(_write_cost_config(tmp_path))
    art = predictions.read_artifact(tmp_path / f"{run_id}.parquet")
    policy = cost_model.build_single_tier_policy(record, art, cfg)

    by_id = {r["complaint_id"]: r["computed_cost_usd"] for r in receipts}
    assert policy.api_cost_usd == pytest.approx([by_id[i] for i in ids])
    assert policy.api_cost_usd[0] == pytest.approx(0.003)  # id 12 first in preds order


def test_tier_c_missing_receipt_is_a_hard_failure(tmp_path):
    run_id = "d0" * 32
    record = _tier_c_fixture(tmp_path, run_id, [10, 11, 12],
                             [_receipt(10), _receipt(11)])
    cfg = cost_model.load_cost_config(_write_cost_config(tmp_path))
    with pytest.raises(KeyError, match="no receipt"):
        cost_model.score_run(record, cfg, preds_dir=tmp_path)


def test_tier_c_duplicate_receipt_is_a_hard_failure(tmp_path):
    run_id = "d1" * 32
    record = _tier_c_fixture(tmp_path, run_id, [10, 11],
                             [_receipt(10), _receipt(11), _receipt(10, prompt=9995)])
    cfg = cost_model.load_cost_config(_write_cost_config(tmp_path))
    with pytest.raises(ValueError, match="duplicate complaint_id"):
        cost_model.score_run(record, cfg, preds_dir=tmp_path)


@pytest.mark.parametrize(("bad_receipt", "pattern"), [
    ({"prompt": 0}, "non-positive/non-integer token count"),
    ({"completion": 0}, "non-positive/non-integer token count"),
    ({"prompt": 995.0}, "non-positive/non-integer token count"),
    ({"total": 12345}, "total_tokens"),
    ({"slug": "anthropic/claude-sonnet-5"}, "slug is not the run's model"),
    ({"cost": 0.009}, "does not follow from tokens"),
    ({"cost": None}, "null/negative/non-finite"),
    ({"cost": -0.001}, "null/negative/non-finite"),
])
def test_tier_c_receipt_verification_hard_fails(tmp_path, bad_receipt, pattern):
    run_id = "e0" * 32
    receipts = [_receipt(10), _receipt(11, **bad_receipt)]
    record = _tier_c_fixture(tmp_path, run_id, [10, 11], receipts)
    cfg = cost_model.load_cost_config(_write_cost_config(tmp_path))
    with pytest.raises(ValueError, match=pattern):
        cost_model.score_run(record, cfg, preds_dir=tmp_path)


def test_tier_c_cost_sum_mismatch_beyond_tolerance_is_a_hard_failure(tmp_path):
    run_id = "e1" * 32
    ids = [10, 11]
    receipts = [_receipt(10), _receipt(11, prompt=1995)]
    _write_artifact(tmp_path, run_id, ids, ["a", "b"], ["a", "b"], ["a", "b"])
    log = _write_receipts(tmp_path / "calls.jsonl", receipts)
    true_total = sum(r["computed_cost_usd"] for r in receipts)
    cfg = cost_model.load_cost_config(_write_cost_config(tmp_path))

    # 1e-7 off: inside tolerance, scores fine and reports the delta.
    ok = cost_model.score_run(
        _record(run_id, "tier_c_toy", cost_usd=true_total + 1e-7, raw_log_path=log),
        cfg, preds_dir=tmp_path,
    )
    assert ok["cost_sum_check"]["ok"] is True

    # 1e-5 off: outside tolerance -> refuse, naming both totals.
    with pytest.raises(ValueError, match="cost_sum_check FAILED"):
        cost_model.score_run(
            _record(run_id, "tier_c_toy", cost_usd=true_total + 1e-5, raw_log_path=log),
            cfg, preds_dir=tmp_path,
        )


def test_tier_c_missing_record_fields_are_hard_failures(tmp_path):
    run_id = "e2" * 32
    _write_artifact(tmp_path, run_id, [10], ["a"], ["a"], ["a", "b"])
    log = _write_receipts(tmp_path / "calls.jsonl", [_receipt(10)])
    cfg = cost_model.load_cost_config(_write_cost_config(tmp_path))
    cost = 995 * PROMPT_RATE + COMPLETION_RATE

    with pytest.raises(ValueError, match="no extra.raw_log_path"):
        cost_model.score_run(_record(run_id, "tier_c_toy", cost_usd=cost), cfg,
                             preds_dir=tmp_path)
    with pytest.raises(ValueError, match="logged no cost_usd"):
        cost_model.score_run(_record(run_id, "tier_c_toy", raw_log_path=log), cfg,
                             preds_dir=tmp_path)
    with pytest.raises(ValueError, match="no extra.pricing_snapshot"):
        cost_model.score_run(
            _record(run_id, "tier_c_toy", cost_usd=cost, raw_log_path=log, pricing=None),
            cfg, preds_dir=tmp_path)
    with pytest.raises(ValueError, match="no extra.model_slug"):
        cost_model.score_run(
            _record(run_id, "tier_c_toy", cost_usd=cost, raw_log_path=log,
                    model_slug=None),
            cfg, preds_dir=tmp_path)


def test_pricing_snapshot_for_a_different_model_is_rejected(tmp_path):
    run_id = "e3" * 32
    _write_artifact(tmp_path, run_id, [10], ["a"], ["a"], ["a", "b"])
    log = _write_receipts(tmp_path / "calls.jsonl", [_receipt(10)])
    cfg = cost_model.load_cost_config(_write_cost_config(tmp_path))
    other = dict(PRICING, slug="anthropic/claude-sonnet-5")
    record = _record(run_id, "tier_c_toy", cost_usd=0.001, raw_log_path=log,
                     pricing=other)
    with pytest.raises(ValueError, match="prices do not belong to this run's model"):
        cost_model.score_run(record, cfg, preds_dir=tmp_path)


# ---------------------------------------------------------------------------
# Output determinism + CLI
# ---------------------------------------------------------------------------

def test_result_json_is_deterministic(tmp_path):
    run_id = "f0" * 32
    _write_artifact(tmp_path, run_id, [1, 2, 3], ["a", "b", "a"], ["a", "b", "b"],
                    ["a", "b"])
    cfg = cost_model.load_cost_config(_write_cost_config(tmp_path))
    record = _record(run_id, "tier_a_toy")
    p1 = cost_model.write_result_json(
        cost_model.score_run(record, cfg, preds_dir=tmp_path), tmp_path / "o1.json")
    p2 = cost_model.write_result_json(
        cost_model.score_run(record, cfg, preds_dir=tmp_path), tmp_path / "o2.json")
    assert p1.read_text() == p2.read_text()
    obj = json.loads(p1.read_text())
    assert obj["run_id"] == run_id
    assert obj["split"] == "cal"
    assert obj["split_sha256"] == "splithash"
    assert obj["config_sha256"] == "cfghash"
    assert obj["n_examples"] == 3
    assert not list(tmp_path.glob("*.tmp"))  # atomic write leaves no debris


def _two_run_cli_fixture(tmp_path):
    """A tier_a run and a tier_c run, plus the runs.jsonl they need."""
    a_id, c_id = "1a" * 32, "1c" * 32
    _write_artifact(tmp_path, a_id, [1, 2], ["a", "b"], ["a", "a"], ["a", "b"])
    a_rec = _record(a_id, "tier_a_toy")
    c_rec = _tier_c_fixture(tmp_path, c_id, [10, 11],
                            [_receipt(10), _receipt(11, prompt=1995)])
    results = tmp_path / "runs.jsonl"
    results.write_text("".join(json.dumps(r) + "\n" for r in (a_rec, c_rec)))
    return a_id, c_id, results


def test_cli_scores_selected_runs(tmp_path):
    a_id, _, results = _two_run_cli_fixture(tmp_path)
    out_dir = tmp_path / "out"
    rc = cost_model.main([
        a_id[:8], "--preds-dir", str(tmp_path), "--out-dir", str(out_dir),
        "--results", str(results), "--cost-config", str(_write_cost_config(tmp_path)),
    ])
    assert rc == 0
    obj = json.loads((out_dir / f"{a_id}.json").read_text())
    assert obj["expected_cost_per_1k"]["total"]["point"] == pytest.approx(3000.0)


def test_cli_is_byte_deterministic_over_multiple_artifacts(tmp_path):
    a_id, c_id, results = _two_run_cli_fixture(tmp_path)
    cost_cfg = _write_cost_config(tmp_path)
    outs = []
    for name in ("out1", "out2"):
        out_dir = tmp_path / name
        assert cost_model.main([
            "--all", "--preds-dir", str(tmp_path), "--out-dir", str(out_dir),
            "--results", str(results), "--cost-config", str(cost_cfg),
        ]) == 0
        outs.append(out_dir)
    for run_id in (a_id, c_id):
        assert (outs[0] / f"{run_id}.json").read_bytes() == \
            (outs[1] / f"{run_id}.json").read_bytes()


def test_cli_all_over_an_empty_preds_dir_fails(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    results = tmp_path / "runs.jsonl"
    results.write_text("")
    rc = cost_model.main([
        "--all", "--preds-dir", str(empty), "--out-dir", str(tmp_path / "out"),
        "--results", str(results), "--cost-config", str(_write_cost_config(tmp_path)),
    ])
    assert rc == 1


def test_cli_writes_nothing_when_any_selected_run_fails(tmp_path):
    # Batch atomicity: the tier_a run alone would score fine, but the tier_c run's
    # receipts are corrupt, so the whole batch must abort with an empty out-dir.
    a_id = "2a" * 32
    c_id = "2c" * 32
    _write_artifact(tmp_path, a_id, [1, 2], ["a", "b"], ["a", "a"], ["a", "b"])
    bad = _tier_c_fixture(tmp_path, c_id, [10, 11],
                          [_receipt(10), _receipt(11, slug="other/model")])
    results = tmp_path / "runs.jsonl"
    results.write_text("".join(
        json.dumps(r) + "\n" for r in (_record(a_id, "tier_a_toy"), bad)))
    out_dir = tmp_path / "out"
    with pytest.raises(ValueError, match="receipt verification failed"):
        cost_model.main([
            "--all", "--preds-dir", str(tmp_path), "--out-dir", str(out_dir),
            "--results", str(results), "--cost-config", str(_write_cost_config(tmp_path)),
        ])
    assert not out_dir.exists() or not list(out_dir.glob("*.json"))
