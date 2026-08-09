"""Prior-shift decomposition tests: the (pi, C) cell vs a weighted-count brute force and
vs metrics.py, exact additivity of every path (point AND per replicate), Shapley = mean of
the two paths, the exact accuracy identity, zero-support conventions, the share gate,
one-at-a-time renormalisation, and bootstrap determinism / pairing."""

from __future__ import annotations

import json

import numpy as np
import pytest

from triage_lab import harness, metrics, prior_shift

K = 4


def _codes(n, k=K, seed=0, skew=None):
    rng = np.random.default_rng(seed)
    p = np.asarray(skew, dtype=float) if skew is not None else np.full(k, 1.0 / k)
    true_codes = rng.choice(k, size=n, p=p / p.sum())
    # predictions agree with truth ~65% of the time, else a uniform other class
    flip = rng.random(n) < 0.35
    pred_codes = np.where(flip, rng.integers(0, k, size=n), true_codes)
    return true_codes.astype(np.int64), pred_codes.astype(np.int64)


# ---------------------------------------------------------------------------
# (a) the cell reproduces metrics.py, and equals the weighted-count form
# ---------------------------------------------------------------------------

def _brute_force_weighted_macro_f1(true_codes, pred_codes, k, weights):
    """Oracle: build the weighted confusion explicitly, then 2TP/(2TP+FP+FN) per class.

    Weights are applied by TRUE class (row scaling) -- the convention the whole
    decomposition rests on. Deliberately written the long way, not via prior_shift.
    """
    cm = np.zeros((k, k), dtype=np.float64)
    np.add.at(cm, (true_codes, pred_codes), 1.0)
    cm = cm * np.asarray(weights, dtype=np.float64)[:, None]
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2.0 * tp + fp + fn
    return float(np.where(denom > 0, 2.0 * tp / np.where(denom > 0, denom, 1.0), 0.0).mean())


def test_cell_equals_metrics_macro_f1_unweighted():
    true_codes, pred_codes = _codes(500, seed=1)
    pi, cond, _ = prior_shift.prior_and_conditional(true_codes, pred_codes, K)
    assert prior_shift.macro_f1_cell(pi, cond) == pytest.approx(
        metrics.macro_f1_from_codes(true_codes, pred_codes, K), abs=1e-12
    )
    np.testing.assert_allclose(
        prior_shift.per_class_f1_cell(pi, cond),
        metrics.per_class_f1_from_codes(true_codes, pred_codes, K),
        atol=1e-12,
    )


@pytest.mark.parametrize("seed", [2, 3, 4])
def test_cell_equals_weighted_count_oracle(seed):
    """macro_f1_cell(pi_target, C_data) == macro-F1 of counts row-scaled by pi_target/pi_data."""
    true_codes, pred_codes = _codes(400, seed=seed, skew=[0.6, 0.2, 0.15, 0.05])
    pi_data, cond, _ = prior_shift.prior_and_conditional(true_codes, pred_codes, K)
    rng = np.random.default_rng(seed + 100)
    pi_target = rng.dirichlet(np.ones(K))
    weights = pi_target / pi_data
    assert prior_shift.macro_f1_cell(pi_target, cond) == pytest.approx(
        _brute_force_weighted_macro_f1(true_codes, pred_codes, K, weights), abs=1e-12
    )


def test_balanced_accuracy_cell_matches_metrics():
    true_codes, pred_codes = _codes(300, seed=5)
    _, cond, support = prior_shift.prior_and_conditional(true_codes, pred_codes, K)
    assert prior_shift.balanced_accuracy_cell(cond, support) == pytest.approx(
        metrics.balanced_accuracy_from_codes(true_codes, pred_codes, K), abs=1e-12
    )


# ---------------------------------------------------------------------------
# (b) + (c) additivity of every path; Shapley is the mean of P and Q
# ---------------------------------------------------------------------------

def test_all_paths_sum_to_total_exactly():
    comp = prior_shift.components_from_cells(0.7579, 0.7159, 0.6693, 0.6656)
    total = comp["total"]
    assert comp["prior::path_p"] + comp["within::path_p"] == pytest.approx(total, abs=1e-15)
    assert comp["prior::path_q"] + comp["within::path_q"] == pytest.approx(total, abs=1e-15)
    assert comp["prior::shapley"] + comp["within::shapley"] == pytest.approx(total, abs=1e-15)
    assert (
        comp["prior_main::anova"] + comp["within_main::anova"] + comp["interaction"]
        == pytest.approx(total, abs=1e-15)
    )


def test_shapley_is_the_average_of_the_two_paths():
    comp = prior_shift.components_from_cells(0.80, 0.72, 0.66, 0.61)
    assert comp["prior::shapley"] == pytest.approx(
        0.5 * (comp["prior::path_p"] + comp["prior::path_q"])
    )
    assert comp["within::shapley"] == pytest.approx(
        0.5 * (comp["within::path_p"] + comp["within::path_q"])
    )


def test_interaction_is_the_path_spread():
    """I = B + C - A - D = (C-D) - (A-B): the two paths' prior terms differ by exactly I,
    which is why `prior_bracket`'s width is the interaction magnitude."""
    comp = prior_shift.components_from_cells(0.80, 0.72, 0.66, 0.61)
    assert comp["prior::path_q"] - comp["prior::path_p"] == pytest.approx(comp["interaction"])
    assert comp["within::path_p"] - comp["within::path_q"] == pytest.approx(comp["interaction"])
    bracket = abs(comp["prior::path_p"] - comp["prior::path_q"])
    assert bracket == pytest.approx(abs(comp["interaction"]))


def test_additivity_checked_per_bootstrap_replicate():
    """Every replicate reconstructs its own total; the driver raises if one does not."""
    t_ref, p_ref = _codes(300, seed=6, skew=[0.5, 0.2, 0.2, 0.1])
    t_year, p_year = _codes(250, seed=7, skew=[0.1, 0.4, 0.3, 0.2])
    out = prior_shift.bootstrap_components(
        t_ref, p_ref, t_year, p_year, K, oat_index=1, n_resamples=50
    )
    reps = out["replicates"]
    np.testing.assert_allclose(
        reps["prior::path_p"] + reps["within::path_p"], reps["total"], atol=1e-12
    )
    np.testing.assert_allclose(
        reps["prior::path_q"] + reps["within::path_q"], reps["total"], atol=1e-12
    )
    np.testing.assert_allclose(
        reps["prior_main::anova"] + reps["within_main::anova"] + reps["interaction"],
        reps["total"], atol=1e-12,
    )


def test_additivity_violation_is_a_hard_error():
    bad = prior_shift.components_from_cells(0.8, 0.7, 0.6, 0.5)
    bad["within::path_p"] += 1e-6
    with pytest.raises(ValueError, match="do not sum to the total"):
        prior_shift._check_additivity(bad, "unit test")


def test_identical_systems_give_zero_everywhere():
    t, p = _codes(200, seed=8)
    out, cells = prior_shift.component_vector(
        *prior_shift.prior_and_conditional(t, p, K),
        *prior_shift.prior_and_conditional(t, p, K),
        oat_index=0,
    )
    assert out["total"] == pytest.approx(0.0, abs=1e-15)
    assert out["prior::path_p"] == pytest.approx(0.0, abs=1e-15)
    assert out["within::path_p"] == pytest.approx(0.0, abs=1e-15)
    assert out["interaction"] == pytest.approx(0.0, abs=1e-15)
    assert cells["A_ref_mix_ref_behavior"] == pytest.approx(cells["D_year_mix_year_behavior"])


# ---------------------------------------------------------------------------
# (d) exact accuracy decomposition
# ---------------------------------------------------------------------------

def test_accuracy_decomposition_is_an_exact_identity():
    t_ref, p_ref = _codes(400, seed=9, skew=[0.55, 0.2, 0.15, 0.1])
    t_year, p_year = _codes(350, seed=10, skew=[0.05, 0.45, 0.3, 0.2])
    pi_r, c_r, _ = prior_shift.prior_and_conditional(t_ref, p_ref, K)
    pi_y, c_y, _ = prior_shift.prior_and_conditional(t_year, p_year, K)
    labels = [f"c{i}" for i in range(K)]
    dec = prior_shift.accuracy_decomposition(pi_r, c_r, pi_y, c_y, labels)

    # endpoints are the ordinary accuracies
    assert dec["acc_ref"] == pytest.approx(metrics.accuracy_from_codes(t_ref, p_ref), abs=1e-12)
    assert dec["acc_year"] == pytest.approx(
        metrics.accuracy_from_codes(t_year, p_year), abs=1e-12
    )
    # three terms reconstruct the total exactly
    assert dec["prior"] + dec["within"] + dec["interaction"] == pytest.approx(
        dec["total"], abs=1e-12
    )
    # and the per-class terms sum to the aggregate terms
    for key in ("prior", "within", "interaction"):
        assert sum(dec["per_class"][lbl][key] for lbl in labels) == pytest.approx(
            dec[key], abs=1e-12
        )


def test_accuracy_prior_term_vanishes_under_a_uniform_mix():
    """The algebraic statement that balanced accuracy is prior-free."""
    t_ref, p_ref = _codes(300, seed=11)
    t_year, p_year = _codes(300, seed=12)
    _, c_r, _ = prior_shift.prior_and_conditional(t_ref, p_ref, K)
    _, c_y, _ = prior_shift.prior_and_conditional(t_year, p_year, K)
    uniform = np.full(K, 1.0 / K)
    dec = prior_shift.accuracy_decomposition(
        uniform, c_r, uniform, c_y, [f"c{i}" for i in range(K)]
    )
    assert dec["prior"] == pytest.approx(0.0, abs=1e-15)
    assert dec["interaction"] == pytest.approx(0.0, abs=1e-15)
    assert dec["total"] == pytest.approx(dec["within"], abs=1e-15)


# ---------------------------------------------------------------------------
# (e) zero-support conventions
# ---------------------------------------------------------------------------

def test_class_absent_from_truth_but_predicted():
    # class 2 is never true, but the model predicts it twice.
    true_codes = np.array([0, 0, 1, 1, 1, 0], dtype=np.int64)
    pred_codes = np.array([0, 2, 1, 2, 0, 0], dtype=np.int64)
    pi, cond, support = prior_shift.prior_and_conditional(true_codes, pred_codes, 3)
    assert support[2] == 0
    np.testing.assert_array_equal(cond[2], np.zeros(3))  # undefined row -> zeros
    f1 = prior_shift.per_class_f1_cell(pi, cond)
    assert f1[2] == 0.0
    # matches metrics.py exactly, including the macro average over ALL classes
    np.testing.assert_allclose(
        f1, metrics.per_class_f1_from_codes(true_codes, pred_codes, 3), atol=1e-12
    )
    assert prior_shift.macro_f1_cell(pi, cond) == pytest.approx(
        metrics.macro_f1_from_codes(true_codes, pred_codes, 3), abs=1e-12
    )


def test_class_absent_from_truth_and_predictions():
    true_codes = np.array([0, 0, 1, 1], dtype=np.int64)
    pred_codes = np.array([0, 1, 1, 0], dtype=np.int64)
    pi, cond, support = prior_shift.prior_and_conditional(true_codes, pred_codes, 3)
    f1 = prior_shift.per_class_f1_cell(pi, cond)
    assert support[2] == 0 and pi[2] == 0.0
    assert f1[2] == 0.0  # denominator 0 -> 0, and it still counts in the macro mean
    assert prior_shift.macro_f1_cell(pi, cond) == pytest.approx(float(f1.mean()))
    assert prior_shift.macro_f1_cell(pi, cond) == pytest.approx(
        metrics.macro_f1_from_codes(true_codes, pred_codes, 3), abs=1e-12
    )


def test_weight_block_reports_degeneracy_instead_of_capping():
    pi_target = np.array([0.5, 0.3, 0.2])
    pi_data = np.array([0.5, 0.5, 0.0])  # class 2 never observed -> infinite weight
    block = prior_shift.weight_block(pi_target, pi_data, n_data=100)
    assert block["degenerate_zero_share_class"] is True
    assert block["n_eff"] == 0.0
    assert block["max"] is None
    assert "cap" not in block  # no capping anywhere in the numerics


def test_n_eff_matches_the_kish_definition():
    """n_eff = (sum w_i)^2 / sum w_i^2 over EXAMPLES, computed the long way."""
    true_codes, _ = _codes(600, seed=13, skew=[0.6, 0.2, 0.15, 0.05])
    pi_data = np.bincount(true_codes, minlength=K) / len(true_codes)
    pi_target = np.array([0.25, 0.25, 0.25, 0.25])
    w = (pi_target / pi_data)[true_codes]
    kish = w.sum() ** 2 / (w**2).sum()
    block = prior_shift.weight_block(pi_target, pi_data, n_data=len(true_codes))
    assert block["n_eff"] == pytest.approx(kish, rel=1e-12)


# ---------------------------------------------------------------------------
# (f) the share gate
# ---------------------------------------------------------------------------

def test_share_gate_passes_a_large_well_separated_total():
    ok, reason = prior_shift.share_gate(
        {"point": 0.0923, "ci_lo": 0.0802, "ci_hi": 0.1042, "ci_valid": True}
    )
    assert ok is True and reason is None


def test_share_gate_suppresses_when_the_total_ci_spans_zero():
    ok, reason = prior_shift.share_gate(
        {"point": 0.0099, "ci_lo": -0.0027, "ci_hi": 0.0232, "ci_valid": True}
    )
    assert ok is False and "includes 0" in reason


def test_share_gate_suppresses_a_tiny_but_significant_total():
    ok, reason = prior_shift.share_gate(
        {"point": 0.005, "ci_lo": 0.001, "ci_hi": 0.009, "ci_valid": True}
    )
    assert ok is False and "macro-F1 points" in reason


def test_share_gate_suppresses_when_the_ci_is_invalid():
    ok, reason = prior_shift.share_gate(
        {"point": 0.09, "ci_lo": 0.08, "ci_hi": 0.10, "ci_valid": False}
    )
    assert ok is False and "ci_invalid" in reason


def test_ci_excludes_zero_helper():
    assert prior_shift.ci_excludes_zero({"ci_lo": 0.01, "ci_hi": 0.02}) is True
    assert prior_shift.ci_excludes_zero({"ci_lo": -0.02, "ci_hi": -0.01}) is True
    assert prior_shift.ci_excludes_zero({"ci_lo": -0.01, "ci_hi": 0.02}) is False
    assert prior_shift.ci_excludes_zero({"ci_lo": 0.0, "ci_hi": 0.02}) is False


# ---------------------------------------------------------------------------
# (g) one-at-a-time renormalisation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k0", range(K))
def test_one_at_a_time_mix_is_a_valid_distribution(k0):
    rng = np.random.default_rng(14)
    pi_ref = rng.dirichlet(np.ones(K))
    pi_year = rng.dirichlet(np.ones(K))
    mix = prior_shift.one_at_a_time_prior_mix(pi_ref, pi_year, k0)
    assert mix.sum() == pytest.approx(1.0, abs=1e-15)
    assert mix[k0] == pytest.approx(pi_year[k0])
    assert np.all(mix >= 0.0)
    # the other classes keep their RELATIVE proportions from the reference mix
    others = [k for k in range(K) if k != k0]
    ratios = mix[others] / pi_ref[others]
    np.testing.assert_allclose(ratios, ratios[0], rtol=1e-12)


def test_one_at_a_time_is_identity_when_the_share_did_not_move():
    pi_ref = np.array([0.5, 0.3, 0.15, 0.05])
    mix = prior_shift.one_at_a_time_prior_mix(pi_ref, pi_ref.copy(), 0)
    np.testing.assert_allclose(mix, pi_ref, atol=1e-15)


def test_one_at_a_time_handles_a_degenerate_reference():
    pi_ref = np.array([1.0, 0.0, 0.0, 0.0])
    pi_year = np.array([0.1, 0.3, 0.3, 0.3])
    mix = prior_shift.one_at_a_time_prior_mix(pi_ref, pi_year, 0)
    np.testing.assert_allclose(mix, pi_ref)  # no division by zero, zero effect


# ---------------------------------------------------------------------------
# (h) bootstrap determinism, pairing, and the ref_fixed variant
# ---------------------------------------------------------------------------

def _boot(seed=harness.BOOTSTRAP_SEED, variant="both", n=40, **kw):
    t_ref, p_ref = _codes(200, seed=15, skew=[0.5, 0.25, 0.15, 0.1])
    t_year, p_year = _codes(180, seed=16, skew=[0.1, 0.4, 0.3, 0.2])
    return prior_shift.bootstrap_components(
        t_ref, p_ref, t_year, p_year, K, oat_index=1,
        variant=variant, n_resamples=n, seed=seed, **kw
    )


def test_bootstrap_is_deterministic_and_json_stable():
    a, b = _boot(), _boot()
    for key in prior_shift.COMPONENT_KEYS:
        np.testing.assert_array_equal(a["replicates"][key], b["replicates"][key])
    def dump(out):
        return json.dumps(
            prior_shift._round_tree(
                {k: prior_shift._ci_block(0.0, out["replicates"][k], True)
                 for k in prior_shift.COMPONENT_KEYS}
            ),
            sort_keys=True,
        )

    assert dump(a) == dump(b)


def test_bootstrap_index_stream_depends_only_on_seed_and_sizes():
    """Why Haiku/Sonnet pair for free: equal (n_ref, n_year) -> identical index vectors."""
    def stream(n_ref, n_year, seed=harness.BOOTSTRAP_SEED, reps=5):
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(reps):
            out.append(rng.integers(0, n_ref, size=n_ref))
            out.append(rng.integers(0, n_year, size=n_year))
        return out

    for x, y in zip(stream(1500, 1500), stream(1500, 1500), strict=True):
        np.testing.assert_array_equal(x, y)
    # different sizes -> different stream (the pairing claim is size-conditional)
    assert not np.array_equal(stream(1500, 1500)[1], stream(1500, 1400)[1][:1500])


def test_ref_fixed_variant_holds_the_reference_cell_constant():
    both = _boot(variant="both")
    fixed = _boot(variant="ref_fixed")
    # cell A is a pure function of the reference sample, so under ref_fixed every
    # replicate's A is identical; A - C (= within::path_q) still varies through C.
    a_minus_b_fixed = fixed["replicates"]["prior::path_p"]
    a_minus_b_both = both["replicates"]["prior::path_p"]
    assert np.std(a_minus_b_fixed) < np.std(a_minus_b_both)
    assert prior_shift.bootstrap_components.__doc__  # documented draw order


def test_unknown_bootstrap_variant_is_rejected():
    with pytest.raises(ValueError, match="unknown bootstrap variant"):
        _boot(variant="year_fixed")


def test_empty_class_counters_are_populated_for_a_rare_class():
    """A class with 1 row of 200 vanishes from many replicates -> ci_valid must go false."""
    rng = np.random.default_rng(17)
    t_ref = np.zeros(200, dtype=np.int64)
    t_ref[:100] = 0
    t_ref[100:199] = 1
    t_ref[199] = 2                      # single row of a third class
    p_ref = t_ref.copy()
    p_ref[rng.random(200) < 0.2] = 0
    out = prior_shift.bootstrap_components(
        t_ref, p_ref, t_ref, p_ref, 3, oat_index=0, n_resamples=200
    )
    counts = out["empty_ref"] + out["empty_year"]
    assert counts[2] > prior_shift.EMPTY_CLASS_CI_FRACTION * 200
    assert counts[0] == 0 and counts[1] == 0


# ---------------------------------------------------------------------------
# Per-class attribution sums to the macro components
# ---------------------------------------------------------------------------

def test_per_class_contributions_sum_to_the_macro_components():
    t_ref, p_ref = _codes(400, seed=18, skew=[0.6, 0.2, 0.15, 0.05])
    t_year, p_year = _codes(400, seed=19, skew=[0.05, 0.45, 0.3, 0.2])
    pi_r, c_r, s_r = prior_shift.prior_and_conditional(t_ref, p_ref, K)
    pi_y, c_y, s_y = prior_shift.prior_and_conditional(t_year, p_year, K)
    labels = [f"c{i}" for i in range(K)]
    attrib = prior_shift.per_class_attribution(pi_r, c_r, pi_y, c_y, s_r, s_y, labels)
    comp, _ = prior_shift.component_vector(pi_r, c_r, s_r, pi_y, c_y, s_y, oat_index=0)
    assert sum(attrib[lbl]["prior_contrib"] for lbl in labels) == pytest.approx(
        comp["prior::path_p"], abs=1e-12
    )
    assert sum(attrib[lbl]["within_contrib"] for lbl in labels) == pytest.approx(
        comp["within::path_p"], abs=1e-12
    )
    # recall is reported straight off the diagonal (prior-invariant channel)
    assert attrib[labels[0]]["recall_ref"] == pytest.approx(c_r[0, 0])


# ---------------------------------------------------------------------------
# Support helpers: prior diagnostics, row restriction, JSON rounding, job selection
# ---------------------------------------------------------------------------

def test_chi2_divergence_conventions():
    p = np.array([0.5, 0.5, 0.0])
    assert prior_shift.chi2_divergence(p, p) == pytest.approx(0.0)
    assert prior_shift.chi2_divergence(np.array([0.5, 0.25, 0.25]), np.array([0.5, 0.5, 0.0])) \
        == np.inf
    assert prior_shift.chi2_divergence(
        np.array([0.6, 0.4, 0.0]), np.array([0.5, 0.5, 0.0])
    ) == pytest.approx(0.01 / 0.5 + 0.01 / 0.5)


def test_full_slice_priors_from_yaml(tmp_path):
    stats = tmp_path / "splits_stats.yaml"
    stats.write_text(
        "splits:\n"
        "  test_drift_2026h1:\n"
        "    class_year_counts:\n"
        "      card:\n"
        "        2026: 30\n"
        "      credit_reporting:\n"
        "        2026: 10\n"
        "      debt_collection:\n"
        "        2026: 60\n"
    )
    pi = prior_shift.full_slice_priors(
        "test_drift_2026h1", ["card", "credit_reporting", "debt_collection"], stats
    )
    np.testing.assert_allclose(pi, [0.3, 0.1, 0.6])
    assert pi.sum() == pytest.approx(1.0)


class _StubArtifact:
    def __init__(self, ids, y_true, y_pred):
        self.complaint_id = np.asarray(ids, dtype=np.int64)
        self.y_true = np.asarray(y_true, dtype=object)
        self.y_pred = np.asarray(y_pred, dtype=object)


def test_restrict_selects_the_right_rows_regardless_of_artifact_order():
    art = _StubArtifact([50, 10, 30, 20], ["a", "b", "c", "d"], ["a", "x", "c", "y"])
    ids, y_true, y_pred = prior_shift._restrict(art, np.array([10, 30], dtype=np.int64))
    np.testing.assert_array_equal(ids, [10, 30])
    assert list(y_true) == ["b", "c"]
    assert list(y_pred) == ["x", "c"]


def test_restrict_rejects_an_id_the_artifact_does_not_cover():
    art = _StubArtifact([1, 2, 3], ["a", "b", "c"], ["a", "b", "c"])
    with pytest.raises(ValueError, match="paired_subsample restriction failed"):
        prior_shift._restrict(art, np.array([2, 99], dtype=np.int64))


def test_round_tree_makes_non_finite_values_json_safe():
    out = prior_shift._round_tree(
        {"a": float("inf"), "b": float("nan"), "c": np.float64(1 / 3), "d": True, "e": None,
         "f": [np.int64(2), 0.123456789012345]}
    )
    assert out == {"a": None, "b": None, "c": round(1 / 3, prior_shift.JSON_ROUND),
                   "d": True, "e": None, "f": [2, round(0.123456789012345, 10)]}
    json.dumps(out)  # must not raise


def test_default_job_set_is_the_documented_twelve():
    jobs = prior_shift.default_jobs()
    assert len(jobs) == 12
    assert sum(1 for j in jobs if j[2] == prior_shift.SCOPE_NATIVE) == 9
    assert sum(1 for j in jobs if j[2] == prior_shift.SCOPE_PAIRED) == 3
    assert all(j[1] in prior_shift.DEFAULT_YEARS for j in jobs)
    assert prior_shift.REF_YEAR not in {j[1] for j in jobs}  # the reference is not a job


def test_job_selection_filters():
    jobs = prior_shift.select_jobs(["tier_a"], ["2026h1"], None)
    assert jobs == [("tier_a", "2026h1", prior_shift.SCOPE_NATIVE),
                    ("tier_a", "2026h1", prior_shift.SCOPE_PAIRED)]
    assert prior_shift.select_jobs(["tier_c_haiku"], None, [prior_shift.SCOPE_PAIRED]) == []


def test_output_names():
    assert prior_shift.output_name("tier_a", "2026h1", prior_shift.SCOPE_NATIVE) \
        == "tier_a__2026h1.json"
    assert prior_shift.output_name("tier_a", "2026h1", prior_shift.SCOPE_PAIRED) \
        == "tier_a__2026h1__paired_subsample.json"


def test_config_stem_resolution_and_unknown_tier():
    assert prior_shift.config_stem("tier_c_sonnet", "2025") \
        == "tier_c_sonnet_zeroshot_test_drift_2025"
    with pytest.raises(ValueError, match="unknown tier"):
        prior_shift.config_stem("tier_z", "2025")
