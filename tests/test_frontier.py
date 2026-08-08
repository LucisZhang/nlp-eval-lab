"""Frontier-claim tests: hand-computed paired cost/percent-reduction math (the ratio CI
must be bootstrapped per replicate, not derived from two marginal intervals), paired
macro-F1 resampling, the non-inferiority gate branches, answered-vs-system macro-F1
handling, Tier B placeholders, determinism, and a replay regression over the shipped
frontier file."""

from __future__ import annotations

import json

import numpy as np
import pytest

from triage_lab import cost_model, frontier, harness, metrics, router_sim

C_MISROUTE = 6.00
C_HUMAN = 2.50

_REAL_FRONTIER = sorted(frontier.DEFAULT_FRONTIER_DIR.glob("frontier__*__cost-*.json"))
_HAS_REAL = bool(_REAL_FRONTIER) and router_sim.DEFAULT_PREDS_DIR.exists()
_needs_real = pytest.mark.skipif(not _HAS_REAL, reason="real frontier outputs not present")


def _cost_config(tmp_path, *, c_misroute=C_MISROUTE, c_human=C_HUMAN):
    path = tmp_path / "cost.yaml"
    path.write_text(
        f"version: v1\nparams:\n  c_misroute_usd: {c_misroute}\n"
        f"  c_human_usd: {c_human}\napi_cost:\n  tier_a:\n    mode: amortized_zero\n"
        "    per_example_usd: 0.0\n    evidence_class: estimated\n    note: amortized\n"
        "  tier_c:\n    mode: measured_receipts\n    evidence_class: measured\n"
        "    note: receipts\nevidence_class:\n  params.c_misroute_usd: estimated\n"
    )
    return cost_model.load_cost_config(path)


def _policy(name, y_true, y_pred, *, to_human=None, api=None, tau=None, coverage=None):
    n = len(y_true)
    return router_sim.RouterPolicy(
        name=name,
        evaluation_set="fixture",
        ids=np.arange(n, dtype=np.int64),
        y_true=np.asarray(y_true, dtype=object),
        y_pred=np.asarray(y_pred, dtype=object),
        to_human=(np.zeros(n, dtype=bool) if to_human is None
                  else np.asarray(to_human, dtype=bool)),
        api_cost_usd=(np.zeros(n, dtype=np.float64) if api is None
                      else np.asarray(api, dtype=np.float64)),
        gate={"kind": "tier_a_threshold", "gate_model": "toy", "tau": tau,
              "transfer_mode": "threshold_transfer", "tau_source": {},
              "coverage_a": coverage if coverage is not None else 1.0},
    )


# ---------------------------------------------------------------------------
# Hand-computed cost claim
# ---------------------------------------------------------------------------

def test_hand_computed_cost_delta_and_percent_reduction(tmp_path):
    cfg = _cost_config(tmp_path)
    #   router  : 2 of 4 wrong -> $12 over 4 -> $3.00/example  -> $3000/1k
    #   baseline: 3 of 4 wrong -> $18 over 4 -> $4.50/example  -> $4500/1k
    #   delta = -1500/1k ; reduction = 100 * (4500 - 3000) / 4500 = 33.333...%
    router = _policy("router", ["a", "b", "a", "b"], ["a", "b", "b", "a"])
    baseline = _policy("baseline", ["a", "b", "a", "b"], ["a", "a", "b", "a"])
    got = frontier.paired_cost_claim(router, baseline, cfg, n_resamples=200)

    assert got["cost_per_1k_router"] == pytest.approx(3000.0)
    assert got["cost_per_1k_baseline"] == pytest.approx(4500.0)
    assert got["delta_cost_per_1k"]["point"] == pytest.approx(-1500.0)
    assert got["pct_cost_reduction"]["point"] == pytest.approx(100 * 1500 / 4500)


def test_percent_reduction_ci_is_bootstrapped_per_replicate(tmp_path):
    """The ratio CI must come from per-resample ratios, not from two marginal intervals.

    A ratio of means is not the mean of ratios; dividing the endpoints of two independent
    CIs would give a different — and wrong — interval. This recomputes the expected band
    from the same shared index draws and demands an exact match, then shows the naive
    construction disagrees.
    """
    cfg = _cost_config(tmp_path)
    rng = np.random.default_rng(3)
    n = 120
    labels = np.array(["a", "b", "c"], dtype=object)
    y_true = labels[rng.integers(0, 3, size=n)]
    router = _policy("router", y_true,
                     np.where(rng.random(n) < 0.8, y_true, labels[0]))
    baseline = _policy("baseline", y_true,
                       np.where(rng.random(n) < 0.6, y_true, labels[1]))
    got = frontier.paired_cost_claim(router, baseline, cfg, n_resamples=300)

    per_r = cost_model.per_example_cost(
        router.correct_for_cost, router.api_cost_usd, router.to_human,
        c_misroute=cfg.c_misroute_usd, c_human=cfg.c_human_usd)
    per_b = cost_model.per_example_cost(
        baseline.correct_for_cost, baseline.api_cost_usd, baseline.to_human,
        c_misroute=cfg.c_misroute_usd, c_human=cfg.c_human_usd)
    reps = cost_model.resample_means({"r": per_r, "b": per_b},
                                     scale=cost_model.PER_N_COMPLAINTS, n_resamples=300)
    pct = 100.0 * (reps["b"] - reps["r"]) / reps["b"]
    lo, hi = np.percentile(pct, [harness.CI_LOWER_PCT, harness.CI_UPPER_PCT])
    assert got["pct_cost_reduction"]["ci_lo"] == pytest.approx(lo, abs=1e-9)
    assert got["pct_cost_reduction"]["ci_hi"] == pytest.approx(hi, abs=1e-9)

    # the naive "divide the two marginal CIs" construction is a different interval
    naive_lo = 100.0 * (1 - np.percentile(reps["r"], harness.CI_UPPER_PCT)
                        / np.percentile(reps["b"], harness.CI_LOWER_PCT))
    assert naive_lo != pytest.approx(lo, abs=1e-6)


def test_zero_cost_baseline_is_refused(tmp_path):
    cfg = _cost_config(tmp_path)
    perfect = _policy("baseline", ["a", "b"], ["a", "b"])   # no errors, no api -> $0
    router = _policy("router", ["a", "b"], ["a", "a"])
    with pytest.raises(ValueError, match="percent cost reduction against a free baseline"):
        frontier.paired_cost_claim(router, perfect, cfg, n_resamples=20)


def test_degenerate_resamples_are_refused_not_silently_dropped(tmp_path):
    # 1 error in 8 rows: many resamples miss it entirely, leaving a zero-cost baseline.
    # Dropping those replicates would silently narrow the interval, so this must fail.
    cfg = _cost_config(tmp_path)
    y_true = ["a", "b"] * 4
    baseline = _policy("baseline", y_true, list(y_true[:7]) + ["a"])
    router = _policy("router", y_true, ["b"] * 8)
    with pytest.raises(ValueError, match="zero-cost baseline"):
        frontier.paired_cost_claim(router, baseline, cfg, n_resamples=100)


def test_cost_and_accuracy_deltas_share_the_index_draws():
    # Same seed, same n -> the accuracy bootstrap must see the same resamples as the cost
    # bootstrap, so the two deltas describe one set of hypothetical datasets.
    router = _policy("router", ["a", "b", "a", "b"], ["a", "b", "b", "a"])
    baseline = _policy("baseline", ["a", "b", "a", "b"], ["a", "a", "b", "a"])
    acc = frontier.paired_accuracy_delta(router, baseline, n_resamples=64)
    reps = cost_model.resample_means(
        {"a": router.correct_for_cost.astype(float),
         "b": baseline.correct_for_cost.astype(float)}, scale=1.0, n_resamples=64)
    lo, hi = np.percentile(reps["a"] - reps["b"],
                           [harness.CI_LOWER_PCT, harness.CI_UPPER_PCT])
    assert acc["ci_lo"] == pytest.approx(lo, abs=1e-9)
    assert acc["ci_hi"] == pytest.approx(hi, abs=1e-9)
    assert acc["point"] == pytest.approx(0.5 - 0.25)


# ---------------------------------------------------------------------------
# Paired macro-F1
# ---------------------------------------------------------------------------

def test_paired_macro_f1_delta_matches_the_harness_bootstrap():
    rng = np.random.default_rng(11)
    n, labels = 90, ["a", "b", "c"]
    arr = np.array(labels, dtype=object)
    y_true = arr[rng.integers(0, 3, size=n)]
    pred_a = np.where(rng.random(n) < 0.75, y_true, arr[0])
    pred_b = np.where(rng.random(n) < 0.55, y_true, arr[1])

    got = frontier.paired_macro_f1_delta(y_true, pred_a, pred_b, labels, n_resamples=100)
    want = harness.paired_bootstrap_delta(
        y_true, pred_a, pred_b, np.zeros((n, 3)), np.zeros((n, 3)),
        "macro_f1", labels, n_resamples=100)
    assert got["point"] == pytest.approx(want["delta"])
    assert got["ci_lo"] == pytest.approx(want["ci_lo"])
    assert got["ci_hi"] == pytest.approx(want["ci_hi"])
    # the point estimate really is the difference of the two macro-F1s
    assert got["point"] == pytest.approx(
        metrics.macro_f1(y_true, pred_a, labels) - metrics.macro_f1(y_true, pred_b, labels),
        abs=1e-9)


def test_paired_macro_f1_of_identical_systems_is_exactly_zero():
    labels = ["a", "b"]
    y_true = np.array(["a", "b", "a", "b"], dtype=object)
    pred = np.array(["a", "b", "b", "b"], dtype=object)
    got = frontier.paired_macro_f1_delta(y_true, pred, pred, labels, n_resamples=32)
    assert got == {"point": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "excludes_zero": False}


# ---------------------------------------------------------------------------
# Gates + claim assembly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("direction", "band", "expected"), [
    # higher-is-better (accuracy, macro-F1, percent cost reduction)
    (frontier.HIGHER_IS_BETTER, {"ci_lo": 0.01, "ci_hi": 0.09}, frontier.CLASS_FAVORABLE),
    (frontier.HIGHER_IS_BETTER, {"ci_lo": -0.01, "ci_hi": 0.09},
     frontier.CLASS_NOT_ESTABLISHED),
    (frontier.HIGHER_IS_BETTER, {"ci_lo": -0.09, "ci_hi": -0.01}, frontier.CLASS_ADVERSE),
    # exactly touching zero is NOT a clearance
    (frontier.HIGHER_IS_BETTER, {"ci_lo": 0.0, "ci_hi": 0.09},
     frontier.CLASS_NOT_ESTABLISHED),
    (frontier.HIGHER_IS_BETTER, {"ci_lo": -0.09, "ci_hi": 0.0},
     frontier.CLASS_NOT_ESTABLISHED),
    # a system against itself: degenerate [0, 0] must never certify
    (frontier.HIGHER_IS_BETTER, {"ci_lo": 0.0, "ci_hi": 0.0},
     frontier.CLASS_NOT_ESTABLISHED),
    # lower-is-better (cost delta)
    (frontier.LOWER_IS_BETTER, {"ci_lo": -60.0, "ci_hi": -10.0}, frontier.CLASS_FAVORABLE),
    (frontier.LOWER_IS_BETTER, {"ci_lo": -60.0, "ci_hi": 10.0},
     frontier.CLASS_NOT_ESTABLISHED),
    (frontier.LOWER_IS_BETTER, {"ci_lo": 10.0, "ci_hi": 60.0}, frontier.CLASS_ADVERSE),
    (frontier.LOWER_IS_BETTER, {"ci_lo": 0.0, "ci_hi": 0.0},
     frontier.CLASS_NOT_ESTABLISHED),
])
def test_classification_uses_the_ci_bound_never_the_point(direction, band, expected):
    # The point estimate is deliberately set on the WRONG side of the classification, so a
    # gate that read it would give the opposite answer.
    band = {**band, "point": -999.0 if expected == frontier.CLASS_FAVORABLE else 999.0}
    assert frontier.classify(band, direction) == expected


def test_classify_rejects_an_unknown_direction():
    with pytest.raises(ValueError, match="unknown direction"):
        frontier.classify({"ci_lo": 0.0, "ci_hi": 1.0, "point": 0.5}, "sideways")


@pytest.mark.parametrize("claim", [frontier.CLAIM_VS_ALL_LLM,
                                   frontier.CLAIM_VS_ALL_LINEAR])
def test_claim_certifies_only_when_every_gated_metric_is_favorable(claim):
    fav = {"point": 1.0, "ci_lo": 0.5, "ci_hi": 1.5}
    fav_low = {"point": -1.0, "ci_lo": -1.5, "ci_hi": -0.5}
    spans = {"point": 1.0, "ci_lo": -0.5, "ci_hi": 1.5}
    names = [n for n, _ in frontier.CLAIM_GATED_METRICS[claim]]
    directions = dict(frontier.CLAIM_GATED_METRICS[claim])

    def band_for(name, kind):
        if kind == "spans":
            return spans
        return fav if directions[name] == frontier.HIGHER_IS_BETTER else fav_low

    all_fav = {n: band_for(n, "fav") for n in names}
    assert frontier.build_gate(claim, all_fav)["certified"] is True
    for spoiled in names:
        bands = {n: band_for(n, "spans" if n == spoiled else "fav") for n in names}
        gate = frontier.build_gate(claim, bands)
        assert gate["certified"] is False
        assert gate["metrics"][spoiled]["classification"] == \
            frontier.CLASS_NOT_ESTABLISHED


def test_claim_reports_paired_answered_f1_only_when_rows_match(tmp_path):
    cfg = _cost_config(tmp_path)
    labels = ["a", "b"]
    y_true = ["a", "b", "a", "b"]
    # same answered rows (neither defers) -> paired answered macro-F1 available
    router = _policy("router", y_true, ["a", "b", "b", "a"])
    baseline = _policy("baseline", y_true, ["a", "a", "b", "a"])
    claim = frontier.build_claim(frontier.CLAIM_VS_ALL_LLM, router, baseline, cfg, labels,
                                 evaluation_set="fixture", n_resamples=50)
    assert claim["macro_f1_answered"]["paired"] is True
    assert "point" in claim["macro_f1_answered"]

    # different answered rows -> no paired CI, both unpaired points reported
    gated = _policy("router", y_true, ["a", "b", "b", "a"],
                    to_human=[False, False, True, True], coverage=0.5)
    claim2 = frontier.build_claim(frontier.CLAIM_VS_ALL_LINEAR, gated, baseline, cfg,
                                  labels, evaluation_set="fixture", n_resamples=50)
    answered = claim2["macro_f1_answered"]
    assert answered["paired"] is False
    assert "point" not in answered
    assert answered["router_macro_f1_answered"] is not None
    assert answered["baseline_macro_f1_answered"] is not None
    assert claim2["human_credit"]["router_n_to_human"] == 2
    assert claim2["human_credit"]["baseline_n_to_human"] == 0


def test_claim_refuses_misaligned_policies(tmp_path):
    cfg = _cost_config(tmp_path)
    a = _policy("router", ["a", "b"], ["a", "b"])
    b = _policy("baseline", ["a", "b", "a"], ["a", "b", "a"])
    with pytest.raises(ValueError, match="different numbers of rows"):
        frontier.build_claim(frontier.CLAIM_VS_ALL_LLM, a, b, cfg, ["a", "b"],
                             evaluation_set="fixture", n_resamples=10)


def test_verdict_sentence_switches_to_honest_diagnosis(tmp_path):
    cfg = _cost_config(tmp_path)
    labels = ["a", "b"]
    # Large enough that no resample can miss every baseline error (40 of them), so the
    # percentage is well defined and the ACCURACY gate is what fails.
    y_true = ["a", "b"] * 100
    better_pred = [t if i % 5 else ("b" if t == "a" else "a")
                   for i, t in enumerate(y_true)]          # baseline right on 160/200
    worse_pred = [t if i % 2 else ("b" if t == "a" else "a")
                  for i, t in enumerate(y_true)]           # router right on 100/200
    worse = _policy("router", y_true, worse_pred)
    better = _policy("baseline", y_true, better_pred)
    claim = frontier.build_claim(frontier.CLAIM_VS_ALL_LLM, worse, better, cfg, labels,
                                 evaluation_set="fixture", n_resamples=100)
    assert claim["gate"]["certified"] is False
    assert claim["gate"]["any_adverse"] is True
    assert claim["gate"]["metrics"]["delta_accuracy_system"]["classification"] == \
        frontier.CLASS_ADVERSE
    # an adverse result must produce an ADVERSE sentence, never a "cuts cost by -X%" one
    assert claim["verdict"].startswith("ADVERSE")
    assert "significantly WORSE" in claim["verdict"]
    assert "cuts cost" not in claim["verdict"]
    assert "raises" not in claim["verdict"]


def test_adverse_cost_direction_for_claim_2(tmp_path):
    # Router significantly MORE expensive: claim 2's cost gate must classify adverse and
    # the sentence must say so rather than narrating a negative "reduction".
    cfg = _cost_config(tmp_path)
    labels = ["a", "b"]
    y_true = ["a", "b"] * 100
    good = [t if i % 5 else ("b" if t == "a" else "a") for i, t in enumerate(y_true)]
    bad = [t if i % 2 else ("b" if t == "a" else "a") for i, t in enumerate(y_true)]
    claim = frontier.build_claim(
        frontier.CLAIM_VS_ALL_LINEAR, _policy("router", y_true, bad),
        _policy("baseline", y_true, good), cfg, labels,
        evaluation_set="fixture", n_resamples=200)
    assert claim["gate"]["metrics"]["delta_cost_per_1k"]["classification"] == \
        frontier.CLASS_ADVERSE
    assert claim["gate"]["certified"] is False
    assert claim["verdict"].startswith("ADVERSE")
    assert "cost per 1,000 complaints" in claim["verdict"]
    assert "raises system macro-F1" not in claim["verdict"]


def test_not_established_verdict_is_directional_only(tmp_path):
    cfg = _cost_config(tmp_path)
    labels = ["a", "b"]
    rng = np.random.default_rng(5)
    n = 60
    arr = np.array(labels, dtype=object)
    y_true = arr[rng.integers(0, 2, size=n)]
    router = _policy("router", y_true, np.where(rng.random(n) < 0.72, y_true, arr[0]))
    baseline = _policy("baseline", y_true, np.where(rng.random(n) < 0.70, y_true, arr[1]))
    claim = frontier.build_claim(frontier.CLAIM_VS_ALL_LLM, router, baseline, cfg, labels,
                                 evaluation_set="fixture", n_resamples=200)
    assert frontier.CLASS_NOT_ESTABLISHED in {
        m["classification"] for m in claim["gate"]["metrics"].values()}
    assert claim["gate"]["certified"] is False
    assert claim["gate"]["any_adverse"] is False
    assert claim["verdict"].startswith("NOT ESTABLISHED")
    assert "directional only" in claim["verdict"]


# ---------------------------------------------------------------------------
# Shipped output
# ---------------------------------------------------------------------------

@_needs_real
def test_shipped_frontier_declares_every_pending_tier_b_slot():
    obj = json.loads(_REAL_FRONTIER[0].read_text())
    points = {p["point"] for p in obj["pending"]}
    assert points == {slot["point"] for slot in frontier.PENDING_TIER_B}
    assert all(p["status"] == frontier.PENDING_STATUS for p in obj["pending"])
    assert obj["evidence_classes"]["c_misroute_usd"].startswith("estimated")
    assert obj["evidence_classes"]["tier_c_api_cost"].startswith("measured")


@_needs_real
def test_shipped_frontier_claims_replay_exactly():
    """Every published claim is recomputed from the artifacts it names."""
    cfg = cost_model.load_cost_config()
    obj = json.loads(_REAL_FRONTIER[0].read_text())
    op_version = obj["operating_point_version"]
    inputs = router_sim.load_test_inputs()
    cal = router_sim.load_cal_thresholds(cost_sha256=cfg.sha256,
                                        derivation=op_version, cost_config=cfg)
    builders = {router_sim.EVAL_FULL: router_sim.build_full_policies,
                router_sim.EVAL_PAIRED: router_sim.build_paired_policies}

    for claim in obj["claims"]:
        if claim.get("status") == "not_applicable":
            continue
        policies = {p.name: p for p in builders[claim["evaluation_set"]](inputs, cal)}
        router, baseline = policies[claim["router"]], policies[claim["baseline"]]
        cost = frontier.paired_cost_claim(router, baseline, cfg)
        assert cost["cost_per_1k_router"] == pytest.approx(
            claim["cost_per_1k_router"], abs=1e-9)
        assert cost["delta_cost_per_1k"] == claim["delta_cost_per_1k"]
        assert cost["pct_cost_reduction"] == claim["pct_cost_reduction"]
        acc = frontier.paired_accuracy_delta(router, baseline)
        assert acc == claim["delta_accuracy_system"]
        assert claim["gate"]["certified"] in (True, False)
        # the published verdict direction must follow the published classifications
        classes = {m["classification"] for m in claim["gate"]["metrics"].values()}
        if frontier.CLASS_ADVERSE in classes:
            assert claim["verdict"].startswith("ADVERSE")
        elif classes == {frontier.CLASS_FAVORABLE}:
            assert claim["gate"]["certified"] is True
            assert not claim["verdict"].startswith(("ADVERSE", "NOT ESTABLISHED"))
        else:
            assert claim["verdict"].startswith("NOT ESTABLISHED")
        # macro-F1 recomputes too
        f1 = frontier.paired_macro_f1_delta(
            router.y_true, router.y_pred_system, baseline.y_pred_system,
            list(inputs.art_a.class_labels))
        assert f1 == claim["delta_macro_f1_system"]


@_needs_real
def test_shipped_frontier_is_deterministic(tmp_path):
    cfg = cost_model.load_cost_config()
    obj = json.loads(_REAL_FRONTIER[0].read_text())
    rebuilt = frontier.build_frontier(cfg, op_version=obj["operating_point_version"])
    assert json.dumps(rebuilt, sort_keys=True) == json.dumps(obj, sort_keys=True)


def test_paired_macro_f1_handles_classes_missing_from_a_replicate():
    """A bootstrap replicate can omit a rare class entirely; the metric must stay finite.

    macro-F1 averages per-class F1, and a class with no true and no predicted rows has an
    undefined F1. `metrics.macro_f1_from_codes` scores it 0 via its `denom > 0` guard, so
    a replicate that drops the rare class scores lower rather than producing NaN. This
    pins that behaviour with an explicit oracle that replays the same index draws.
    """
    labels = ["a", "b", "rare"]
    n = 40
    y_true = np.array(["a", "b"] * 19 + ["rare", "rare"], dtype=object)
    pred_a = np.array(["a", "b"] * 19 + ["rare", "a"], dtype=object)
    pred_b = np.array(["a", "a"] * 19 + ["a", "a"], dtype=object)

    got = frontier.paired_macro_f1_delta(y_true, pred_a, pred_b, labels, n_resamples=64)
    assert np.isfinite(got["point"]) and np.isfinite(got["ci_lo"])

    true_idx = metrics.encode_labels(y_true, labels)
    a_idx = metrics.encode_labels(pred_a, labels)
    b_idx = metrics.encode_labels(pred_b, labels)
    rng = np.random.default_rng(harness.BOOTSTRAP_SEED)
    deltas, n_missing = [], 0
    for _ in range(64):
        idx = rng.integers(0, n, size=n)
        if 2 not in set(true_idx[idx]) | set(a_idx[idx]) | set(b_idx[idx]):
            n_missing += 1
        deltas.append(metrics.macro_f1_from_codes(true_idx[idx], a_idx[idx], 3)
                      - metrics.macro_f1_from_codes(true_idx[idx], b_idx[idx], 3))
    assert n_missing > 0, "fixture never drops the rare class; it proves nothing"
    lo, hi = np.percentile(deltas, [harness.CI_LOWER_PCT, harness.CI_UPPER_PCT])
    assert got["ci_lo"] == pytest.approx(lo, abs=1e-9)
    assert got["ci_hi"] == pytest.approx(hi, abs=1e-9)
    assert all(np.isfinite(d) for d in deltas)


@_needs_real
def test_frontier_cli_output_is_byte_deterministic(tmp_path):
    """Byte-level, not normalized-JSON: the file on disk must be reproducible verbatim."""
    outs = []
    for name in ("out1", "out2"):
        out_dir = tmp_path / name
        assert frontier.main(["--out-dir", str(out_dir), "--op-version", "v2-isocal"]) == 0
        outs.append(out_dir)
    files = sorted(p.name for p in outs[0].glob("*.json"))
    assert files
    for name in files:
        assert (outs[0] / name).read_bytes() == (outs[1] / name).read_bytes()
    # and identical to the committed artifact
    for name in files:
        committed = frontier.DEFAULT_FRONTIER_DIR / name
        if committed.exists():
            assert (outs[0] / name).read_bytes() == committed.read_bytes()
