"""Prior-shift decomposition of yearly macro-F1 degradation (Phase 5, UPGRADE_PLAN §7.2).

The rolling yearly slices change *two* things at once between 2023 and 2026-H1: the class
mix (``credit_reporting`` collapses 9,445 -> 585 rows of 20,000) and the model's per-class
behaviour. This module measures how much of each system's macro-F1 drop is explained by
each, using an exact re-parameterisation rather than an approximation.

**The two coordinates.** Every macro-F1 in this lab is an exact function of exactly two
objects computed from the confusion matrix ``N``:

- the prior ``pi[k] = N[k, :].sum() / N.sum()``
- the row-normalised confusion ``C[k, j] = N[k, j] / N[k, :].sum()`` = measured
  ``P(y_pred = j | y_true = k)`` -- the *within-class behaviour*

With ``rho_k(pi, C) = sum_m pi[m] * C[m, k]`` (predicted mass of class k)::

    F1_k(pi, C) = 2 * pi[k] * C[k, k] / (pi[k] + rho_k)        # 0 when pi[k] + rho_k == 0
    macroF1(pi, C) = mean over ALL K classes of F1_k

That denominator is ``2TP + FP + FN = (TP + FN) + (TP + FP) = support_k + predicted_k``,
so this reproduces ``metrics.macro_f1_from_codes`` exactly, zero-convention included (a
regression test pins it, and `build_decomposition` hard-fails if cell A or cell D does not
equal the logged ``macro_f1`` of the run it claims to describe at 1e-12).

Re-weighting examples by their TRUE class with ``w_k = pi_target[k] / pi_data[k]`` produces
weighted counts ``Ntilde[k, j] = w_k * N[k, j]``, whose normalisation is exactly
``pi_target[k] * C[k, j]``; F1 is scale-invariant, so the weighted-count form and the
``(pi, C)`` form are algebraically identical. We implement the ``(pi, C)`` form: it never
divides by a zero class share, and the weights survive only as reported diagnostics
(``max``/``min``/Kish ``n_eff``). Weights are applied by true class ONLY -- weighting by
predicted class would destroy the invariance the decomposition rests on:

**Recall is prior-invariant; precision is not.** ``recall_k = C[k, k]`` depends only on row
k, so no true-class re-weighting can move it. ``precision_k = pi[k] * C[k, k] / rho_k``
does move. Therefore *all* of macro-F1's prior sensitivity flows through precision, and
macro-recall (``balanced_accuracy``, already logged with CIs) is the counterfactual-free
"model behaviour only" channel. It is emitted in every record as the anchor exhibit.

**The four cells** (reference R = 2023 of the same tier; sign convention: positive =
degradation)::

    A = macroF1(pi_R, C_R)     as logged, 2023
    B = macroF1(pi_Y, C_R)     2023 behaviour, year-Y mix   <- prior counterfactual, safe direction
    C = macroF1(pi_R, C_Y)     year-Y behaviour, 2023 mix   <- prior counterfactual, explosive direction
    D = macroF1(pi_Y, C_Y)     as logged, year Y
    total = A - D

**Path P is primary** (``prior = A - B``, ``within = B - D``): it is the forward,
operationally meaningful counterfactual ("what would the mix change alone have cost the
2023 system?"), and it never up-weights a collapsed class -- cell B *down*-weights
credit_reporting on 2023 data (Kish n_eff 10,902/20,000 for Tier A 2026-H1) where cell C
up-weights ~585 rows by 16x (n_eff 2,526; only 242/1,500 for Tier C). Measured CI widths
follow: 0.0070 for the Path-P prior term vs 0.0225 for Path Q's.

Path Q, the Shapley average, and the ANOVA (main effects + interaction) form are ALL
computed and stored as labelled sensitivities, and ``interaction = B + C - A - D`` plus
``prior_bracket`` (Path Q .. Path P) are mandatory fields, because path dependence here is
large and signed: for Tier A 2026-H1 the interaction is -0.038, 41% of the total. The
honest narration is "4.2 of 9.2 points under Path P; the two paths bracket the prior
contribution at [0.4, 4.2]; the effects are sub-additive because both hit the same class".
Never print a single path alone.

Also emitted, per §3 of the spec:

- ``accuracy_decomposition`` -- ``acc = sum_k pi_k * r_k`` is bilinear, so it decomposes
  EXACTLY (no counterfactual) into prior + within + interaction, per class. Reported as a
  decomposition *validator*, not a performance claim (accuracy is prior-dominated here).
- ``one_at_a_time_prior`` for credit_reporting -- move only that class's share to its
  year-Y value, renormalise the other eight proportionally, hold C_R fixed. Explicitly
  labelled non-additive: one-at-a-time effects do not sum to the prior term.

**Share suppression.** ``prior / total`` is a ratio of estimates whose denominator can
cross zero; measured bootstrap CIs are [-5.0, +8.3] for Tier A 2024 and [-22.6, +22.8] for
Haiku 2026-H1. It is emitted only when the total CI excludes 0 AND |total| >= 0.02;
otherwise ``share_prior`` is null with ``share_suppressed_reason``.

**Bootstrap** reuses the harness contract (``N_RESAMPLES``, ``BOOTSTRAP_SEED``, percentile
interval). Two slices are involved, so each replicate draws ``idx_ref`` and THEN
``idx_year`` from one ``default_rng(seed)`` stream, in that fixed order. Because the stream
depends only on (seed, n_ref, n_year), Haiku and Sonnet -- equal-sized artifacts on the
same rows -- get byte-identical index vectors, making their component deltas paired for
free. All four cells come from the same replicate pair, so components sum to the total
exactly on every replicate (checked, not assumed); the stored CIs are marginal percentile
intervals and do NOT sum. Three bootstrap variants:

- ``both`` (primary): resample both slices.
- ``ref_fixed``: hold the reference sample at its observed value, resample only year Y.
  The 2023 resample is common to the 2024/2025/2026 comparisons, so the year-over-year
  curve carries a shared offset; this variant isolates the shape of the trend.
- ``pi_full_slice``: resample both slices but pin ``pi`` to the frozen full-slice mix from
  ``splits_stats.yaml``. For Tier C (uniform-random 1,500-row subsample) ``C`` is unbiased
  regardless of mix while the full-slice ``pi`` is known exactly, so this is a
  lower-variance estimator of the same quantity -- it is a sensitivity, not the primary,
  only because it breaks the ``cell D == logged macro_f1`` identity gate.

Scopes: ``native`` (each tier on its own rows) and ``paired_subsample`` (Tier A restricted
to the exact complaint_ids of the Tier C subsample for the same year, so the "Tier A
collapses / the LLMs do not" claim is made on identical rows rather than 20,000 vs 1,500).
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

from triage_lab import harness, metrics, predictions

REPO_ROOT = harness.REPO_ROOT
DEFAULT_PREDS_DIR = harness.DEFAULT_PREDS_DIR
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "prior_shift"
DEFAULT_SPLITS_STATS_PATH = harness.DEFAULT_SPLITS_STATS_PATH

SCHEMA_VERSION = 1
JSON_ROUND = 10

# Frozen protocol constants. Changing any of these is a methodology change (CLAUDE.md
# workflow conventions), not a tweak, so none of them is a CLI flag.
REF_YEAR = "2023"
DEFAULT_YEARS: tuple[str, ...] = ("2024", "2025", "2026h1")
SCOPE_NATIVE = "native"
SCOPE_PAIRED = "paired_subsample"
PI_ARTIFACT = "artifact"
PI_FULL_SLICE = "full_slice"
OAT_CLASS = "credit_reporting"
SHARE_MIN_ABS_TOTAL = 0.02
EMPTY_CLASS_CI_FRACTION = 0.005
IDENTITY_TOL = 1e-12
ADDITIVITY_TOL = 1e-12

# Tier registry, data-driven so Tier B drops in with one line and no format change.
# Values are the config-file stem template whose run record names the artifact.
TIER_CONFIGS: dict[str, str] = {
    "tier_a": "tier_a_logreg_test_drift_{year}",
    "tier_b2": "tier_b2_distilbert_s0_test_drift_{year}",
    "tier_c_haiku": "tier_c_haiku_zeroshot_test_drift_{year}",
    "tier_c_sonnet": "tier_c_sonnet_zeroshot_test_drift_{year}",
}
TIER_ORDER: tuple[str, ...] = ("tier_a", "tier_b2", "tier_c_haiku", "tier_c_sonnet")
# Which tiers can be restricted to another tier's rows, and whose id set defines them.
# The listed id-source tiers must agree exactly (asserted) -- that agreement IS the
# pairing claim, so it is verified rather than trusted.
PAIRED_SUBSAMPLE_ID_SOURCES: dict[str, tuple[str, ...]] = {
    "tier_a": ("tier_c_haiku", "tier_c_sonnet"),
}

SPLIT_FMT = "test_drift_{year}"

# Ordered component vector carried through the bootstrap. Keys encode the path so the
# flat summary rows are unambiguous.
COMPONENT_KEYS: tuple[str, ...] = (
    "total",
    "prior::path_p",
    "within::path_p",
    "prior::path_q",
    "within::path_q",
    "prior::shapley",
    "within::shapley",
    "prior_main::anova",
    "within_main::anova",
    "interaction",
    f"one_at_a_time_prior::{OAT_CLASS}",
    "balanced_accuracy_delta",
)
PRIMARY_COMPONENTS = ("total", "prior::path_p", "within::path_p")


# ---------------------------------------------------------------------------
# Core numerics: the (pi, C) coordinates and the macro-F1 cell
# ---------------------------------------------------------------------------

def confusion_counts(true_codes, pred_codes, n_classes: int) -> np.ndarray:
    """(K, K) float counts; rows = true code, cols = predicted code (metrics.py orientation).

    bincount on the flattened (true * K + pred) index is the vectorised form used inside
    the bootstrap: it is the whole per-replicate cost, so it must not be a Python loop.
    """
    true_codes = np.asarray(true_codes, dtype=np.int64)
    pred_codes = np.asarray(pred_codes, dtype=np.int64)
    flat = np.bincount(true_codes * n_classes + pred_codes, minlength=n_classes * n_classes)
    return flat.reshape(n_classes, n_classes).astype(np.float64)


def prior_and_conditional(true_codes, pred_codes, n_classes: int):
    """Return (pi, C, support) -- the two coordinates plus raw per-class true support.

    A class with zero true support has an undefined row (0/0); the convention is
    ``C[k, :] = 0``, which makes ``F1_k = 0`` in any cell using that row (the numerator is
    ``pi_k * C[k, k]``) while leaving the class's FP inflow to other rows intact. This is
    the deterministic, conservative reading of "no observed behaviour for this class"; it
    is also why the weighted-count form (``w_k = inf``) is not used.
    """
    cm = confusion_counts(true_codes, pred_codes, n_classes)
    support = cm.sum(axis=1)
    total = support.sum()
    pi = support / total if total > 0 else np.zeros(n_classes, dtype=np.float64)
    safe = np.where(support > 0, support, 1.0)[:, None]
    cond = np.where(support[:, None] > 0, cm / safe, 0.0)
    return pi, cond, support


def per_class_f1_cell(pi: np.ndarray, cond: np.ndarray) -> np.ndarray:
    """F1_k(pi, C) = 2 pi_k C_kk / (pi_k + rho_k); 0 when the denominator is 0."""
    rho = pi @ cond
    denom = pi + rho
    num = 2.0 * pi * np.diag(cond)
    return np.where(denom > 0, num / np.where(denom > 0, denom, 1.0), 0.0)


def macro_f1_cell(pi: np.ndarray, cond: np.ndarray) -> float:
    """Unweighted mean of F1_k over ALL K classes (metrics.py macro convention).

    Averaging over all declared labels -- never only the present ones -- is what keeps
    cells A..D commensurable and the decomposition exactly additive.
    """
    return float(per_class_f1_cell(pi, cond).mean())


def balanced_accuracy_cell(cond: np.ndarray, support: np.ndarray) -> float:
    """Mean recall over classes with non-zero true support (metrics.balanced_accuracy)."""
    present = support > 0
    if not present.any():
        return 0.0
    return float(np.diag(cond)[present].mean())


def one_at_a_time_prior_mix(pi_ref: np.ndarray, pi_year: np.ndarray, k0: int) -> np.ndarray:
    """pi_ref with ONLY class k0 moved to its year-Y share, others renormalised pro rata.

    ``pi[k0] = pi_year[k0]``; ``pi[k] = pi_ref[k] * (1 - pi_year[k0]) / (1 - pi_ref[k0])``.
    Sums to 1 by construction (tested). Degenerate reference (``pi_ref[k0] == 1``) returns
    the reference mix unchanged, i.e. a zero effect, rather than dividing by zero.
    """
    rest = 1.0 - pi_ref[k0]
    if rest <= 0.0:
        return pi_ref.copy()
    mix = pi_ref * ((1.0 - pi_year[k0]) / rest)
    mix[k0] = pi_year[k0]
    return mix


# ---------------------------------------------------------------------------
# Cells -> components
# ---------------------------------------------------------------------------

def components_from_cells(cell_a: float, cell_b: float, cell_c: float, cell_d: float) -> dict:
    """All four decomposition forms from the four cells. Every form sums to `total`.

    Path P's within-term carries the whole interaction; Path Q's prior-term carries it;
    Shapley splits it evenly; the ANOVA form states it separately. Which one you print
    changes the headline by the size of the interaction, so all four are stored.
    """
    total = cell_a - cell_d
    p_prior, p_within = cell_a - cell_b, cell_b - cell_d
    q_prior, q_within = cell_c - cell_d, cell_a - cell_c
    interaction = cell_b + cell_c - cell_a - cell_d
    return {
        "total": total,
        "prior::path_p": p_prior,
        "within::path_p": p_within,
        "prior::path_q": q_prior,
        "within::path_q": q_within,
        "prior::shapley": 0.5 * (p_prior + q_prior),
        "within::shapley": 0.5 * (p_within + q_within),
        "prior_main::anova": p_prior,      # main effect measured at the reference cell
        "within_main::anova": q_within,    # = A - C
        "interaction": interaction,
    }


def _check_additivity(comp: dict, where: str) -> None:
    """Every path must reconstruct the total exactly. Checked, not assumed."""
    total = comp["total"]
    for path, (prior_key, within_key) in {
        "path_p": ("prior::path_p", "within::path_p"),
        "path_q": ("prior::path_q", "within::path_q"),
        "shapley": ("prior::shapley", "within::shapley"),
        "anova": ("prior_main::anova", "within_main::anova"),
    }.items():
        parts = comp[prior_key] + comp[within_key]
        if path == "anova":
            parts += comp["interaction"]
        if abs(total - parts) > ADDITIVITY_TOL:
            raise ValueError(
                f"{where}: {path} components do not sum to the total "
                f"({parts!r} vs {total!r}, delta {total - parts!r} > {ADDITIVITY_TOL})"
            )


def component_vector(
    pi_ref, cond_ref, support_ref, pi_year, cond_year, support_year, oat_index: int
) -> tuple[dict, dict]:
    """(components, cells) for one (ref, year) pair of coordinates.

    Also carries the two scalars that are not differences of the four cells: the
    one-at-a-time credit_reporting prior effect (non-additive by construction) and the
    balanced-accuracy delta (the prior-invariant anchor).
    """
    cell_a = macro_f1_cell(pi_ref, cond_ref)
    cell_b = macro_f1_cell(pi_year, cond_ref)
    cell_c = macro_f1_cell(pi_ref, cond_year)
    cell_d = macro_f1_cell(pi_year, cond_year)
    comp = components_from_cells(cell_a, cell_b, cell_c, cell_d)
    _check_additivity(comp, "component_vector")
    oat_mix = one_at_a_time_prior_mix(pi_ref, pi_year, oat_index)
    comp[f"one_at_a_time_prior::{OAT_CLASS}"] = cell_a - macro_f1_cell(oat_mix, cond_ref)
    comp["balanced_accuracy_delta"] = balanced_accuracy_cell(
        cond_ref, support_ref
    ) - balanced_accuracy_cell(cond_year, support_year)
    cells = {
        "A_ref_mix_ref_behavior": cell_a,
        "B_year_mix_ref_behavior": cell_b,
        "C_ref_mix_year_behavior": cell_c,
        "D_year_mix_year_behavior": cell_d,
    }
    return comp, cells


# ---------------------------------------------------------------------------
# Bootstrap over two independent slices
# ---------------------------------------------------------------------------

def bootstrap_components(
    true_ref,
    pred_ref,
    true_year,
    pred_year,
    n_classes: int,
    oat_index: int,
    *,
    variant: str = "both",
    pi_fixed_ref: np.ndarray | None = None,
    pi_fixed_year: np.ndarray | None = None,
    n_resamples: int = harness.N_RESAMPLES,
    seed: int = harness.BOOTSTRAP_SEED,
) -> dict:
    """Percentile-bootstrap replicates of every component.

    Draw order per replicate is ``idx_ref`` THEN ``idx_year`` from a single
    ``default_rng(seed)`` stream. The stream depends only on (seed, n_ref, n_year), so two
    systems evaluated on the same rows -- Haiku and Sonnet -- receive byte-identical index
    vectors and their component deltas are paired without any extra machinery.

    ``variant="ref_fixed"`` holds the reference sample at its observed value but still
    DRAWS ``idx_ref`` and discards it, so the year-slice index stream stays aligned with
    the primary variant and the two are directly comparable.

    ``pi_fixed_*`` pins the priors (used by the ``pi_full_slice`` sensitivity): ``C`` is
    still recomputed from the resampled rows, only the mix is treated as known.
    """
    if variant not in {"both", "ref_fixed"}:
        raise ValueError(f"unknown bootstrap variant {variant!r}")
    true_ref = np.asarray(true_ref, dtype=np.int64)
    pred_ref = np.asarray(pred_ref, dtype=np.int64)
    true_year = np.asarray(true_year, dtype=np.int64)
    pred_year = np.asarray(pred_year, dtype=np.int64)
    n_ref, n_year = len(true_ref), len(true_year)

    reps = {k: np.empty(n_resamples, dtype=np.float64) for k in COMPONENT_KEYS}
    empty_ref = np.zeros(n_classes, dtype=np.int64)
    empty_year = np.zeros(n_classes, dtype=np.int64)

    rng = np.random.default_rng(seed)
    for i in range(n_resamples):
        idx_ref = rng.integers(0, n_ref, size=n_ref)
        idx_year = rng.integers(0, n_year, size=n_year)
        if variant == "ref_fixed":
            pi_r, cond_r, sup_r = prior_and_conditional(true_ref, pred_ref, n_classes)
        else:
            pi_r, cond_r, sup_r = prior_and_conditional(
                true_ref[idx_ref], pred_ref[idx_ref], n_classes
            )
        pi_y, cond_y, sup_y = prior_and_conditional(
            true_year[idx_year], pred_year[idx_year], n_classes
        )
        empty_ref += (sup_r == 0).astype(np.int64)
        empty_year += (sup_y == 0).astype(np.int64)
        if pi_fixed_ref is not None:
            pi_r = pi_fixed_ref
        if pi_fixed_year is not None:
            pi_y = pi_fixed_year
        comp, _ = component_vector(pi_r, cond_r, sup_r, pi_y, cond_y, sup_y, oat_index)
        _check_additivity(comp, f"bootstrap replicate {i}")
        for key in COMPONENT_KEYS:
            reps[key][i] = comp[key]

    return {
        "replicates": reps,
        "empty_ref": empty_ref,
        "empty_year": empty_year,
        "n_resamples": n_resamples,
        "seed": seed,
        "variant": variant,
    }


def _percentile_ci(values: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(values, [harness.CI_LOWER_PCT, harness.CI_UPPER_PCT])
    return float(lo), float(hi)


def _ci_block(point: float, values: np.ndarray, ci_valid: bool) -> dict:
    lo, hi = _percentile_ci(values)
    return {"point": float(point), "ci_lo": lo, "ci_hi": hi, "ci_valid": ci_valid}


def ci_excludes_zero(block: dict) -> bool:
    return (block["ci_lo"] > 0.0 and block["ci_hi"] > 0.0) or (
        block["ci_lo"] < 0.0 and block["ci_hi"] < 0.0
    )


def share_gate(total_block: dict) -> tuple[bool, str | None]:
    """Emit prior/total only when the ratio's denominator is safely away from zero.

    Measured failure cases this gate exists for: Tier A 2024 share CI [-5.0, +8.3] and
    Haiku 2026-H1 [-22.6, +22.8]. A percentage of a near-zero total is not a finding.
    """
    if not total_block.get("ci_valid", True):
        return False, "ci_invalid: too many bootstrap replicates with an empty class"
    if not ci_excludes_zero(total_block):
        return False, (
            f"total degradation CI [{total_block['ci_lo']:.4f}, {total_block['ci_hi']:.4f}] "
            "includes 0; a share of an indistinguishable-from-zero total is not a finding"
        )
    if abs(total_block["point"]) < SHARE_MIN_ABS_TOTAL:
        return False, (
            f"|total| = {abs(total_block['point']):.4f} < {SHARE_MIN_ABS_TOTAL} macro-F1 "
            "points; the ratio is numerically unstable below this floor"
        )
    return True, None


# ---------------------------------------------------------------------------
# Diagnostics: weights, effective sample size, prior distances
# ---------------------------------------------------------------------------

def weight_block(pi_target: np.ndarray, pi_data: np.ndarray, n_data: int) -> dict:
    """Re-weighting diagnostics for evaluating `pi_data`'s rows at `pi_target`'s mix.

    ``w_k = pi_target[k] / pi_data[k]`` (true-class weights). No cap is applied anywhere:
    capping would silently change the estimand -- the reported number would no longer be
    macro-F1 at ``pi_target`` -- so the exposure is published instead. Kish effective
    sample size has a closed form here::

        n_eff = n_data / sum_k pi_target[k]^2 / pi_data[k] = n_data / (1 + chi2(target||data))

    A class with positive target share but zero observed share makes the weight infinite;
    ``n_eff`` is then 0 and ``max`` is null, which is the honest report rather than a
    silently finite number.
    """
    valid = pi_data > 0
    degenerate = bool(np.any((pi_target > 0) & ~valid))
    weights = np.where(valid, pi_target / np.where(valid, pi_data, 1.0), np.inf)
    if degenerate:
        n_eff = 0.0
        w_max: float | None = None
    else:
        denom = float(np.sum(np.where(valid, pi_target**2 / np.where(valid, pi_data, 1.0), 0.0)))
        n_eff = float(n_data / denom) if denom > 0 else 0.0
        w_max = float(np.max(weights[valid])) if valid.any() else None
    return {
        "w": weights,
        "max": w_max,
        "min": float(np.min(weights[valid])) if valid.any() else None,
        "n_eff": n_eff,
        "n_data": int(n_data),
        "degenerate_zero_share_class": degenerate,
    }


def chi2_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Pearson chi-square divergence chi2(p||q) = sum_k (p_k - q_k)^2 / q_k.

    Terms with ``q_k == 0`` and ``p_k == 0`` contribute 0; ``q_k == 0 < p_k`` is infinite.
    """
    diff = p - q
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(q > 0, diff**2 / np.where(q > 0, q, 1.0), np.where(p > 0, np.inf, 0.0))
    return float(np.sum(terms))


def full_slice_priors(split: str, class_labels, splits_stats_path=DEFAULT_SPLITS_STATS_PATH):
    """Frozen full-slice class mix from splits_stats.yaml (zero sampling error).

    ``class_year_counts`` is {class: {year: n}}; a drift slice spans one year but the sum
    is taken over whatever years the split declares, so this stays correct for any slice.
    """
    stats = yaml.safe_load(Path(splits_stats_path).read_text())
    counts = stats["splits"][split]["class_year_counts"]
    vec = np.array(
        [float(sum((counts.get(lbl) or {}).values())) for lbl in class_labels],
        dtype=np.float64,
    )
    total = vec.sum()
    if total <= 0:
        raise ValueError(f"split {split!r} has no class_year_counts in {splits_stats_path}")
    return vec / total


# ---------------------------------------------------------------------------
# Exact accuracy decomposition (complementary exhibit; no counterfactual)
# ---------------------------------------------------------------------------

def accuracy_decomposition(pi_ref, cond_ref, pi_year, cond_year, class_labels) -> dict:
    """`acc = sum_k pi_k r_k` is bilinear, so it splits exactly, per class::

        acc_R - acc_Y = sum_k dpi_k r_R,k  +  sum_k pi_R,k dr_k  -  sum_k dpi_k dr_k
                        (prior)               (within)              (interaction)

    with ``dpi = pi_R - pi_Y`` and ``dr = r_R - r_Y``, ``r_k = C[k, k]``. This is the
    path-free ANOVA form: the two one-sided paths differ only in where the third term is
    absorbed. Reported as a *validator* for the macro-F1 counterfactual, never as a
    performance claim -- accuracy under the 2023 mix rewards majority-class prediction.
    The same formula with a uniform ``pi`` has an identically-zero prior term, which is
    the algebraic statement that balanced accuracy is prior-free.
    """
    r_ref, r_year = np.diag(cond_ref), np.diag(cond_year)
    dpi, dr = pi_ref - pi_year, r_ref - r_year
    prior_k = dpi * r_ref
    within_k = pi_ref * dr
    inter_k = -dpi * dr
    return {
        "acc_ref": float(pi_ref @ r_ref),
        "acc_year": float(pi_year @ r_year),
        "total": float(pi_ref @ r_ref - pi_year @ r_year),
        "prior": float(prior_k.sum()),
        "within": float(within_k.sum()),
        "interaction": float(inter_k.sum()),
        "per_class": {
            lbl: {
                "prior": float(prior_k[k]),
                "within": float(within_k[k]),
                "interaction": float(inter_k[k]),
            }
            for k, lbl in enumerate(class_labels)
        },
    }


# ---------------------------------------------------------------------------
# Per-class attribution (exact, additive within each path)
# ---------------------------------------------------------------------------

def per_class_attribution(
    pi_ref, cond_ref, pi_year, cond_year, support_ref, support_year, class_labels
) -> dict:
    """Path-P per-class contributions; they sum EXACTLY to the macro components.

    ``prior_contrib_k = (F1_k(A) - F1_k(B)) / K``, ``within_contrib_k = (F1_k(B) - F1_k(D)) / K``.

    These are contributions of class k's F1 *to* each component -- NOT "caused by class k's
    own prior change". ``F1_k`` depends on every class's prior through the FP inflow
    ``rho_k``, so growing classes legitimately show negative prior contributions (they gain
    precision). The causal single-class question is answered by ``one_at_a_time_prior``,
    which is reported separately and labelled non-additive.
    """
    n_classes = len(class_labels)
    f1_a = per_class_f1_cell(pi_ref, cond_ref)
    f1_b = per_class_f1_cell(pi_year, cond_ref)
    f1_d = per_class_f1_cell(pi_year, cond_year)
    out = {}
    for k, lbl in enumerate(class_labels):
        out[lbl] = {
            "prior_contrib": float((f1_a[k] - f1_b[k]) / n_classes),
            "within_contrib": float((f1_b[k] - f1_d[k]) / n_classes),
            "f1_A": float(f1_a[k]),
            "f1_B": float(f1_b[k]),
            "f1_D": float(f1_d[k]),
            "recall_ref": float(cond_ref[k, k]),
            "recall_year": float(cond_year[k, k]),
            "pi_ref": float(pi_ref[k]),
            "pi_year": float(pi_year[k]),
            "support_ref": int(support_ref[k]),
            "support_year": int(support_year[k]),
        }
    return out


# ---------------------------------------------------------------------------
# Artifact resolution + scope restriction
# ---------------------------------------------------------------------------

def _current_git_sha() -> str:
    """SHA of the code producing this derivation (the artifacts carry their own, in
    `source.*.git_sha`). Single implementation lives in the harness; not duplicated."""
    return harness._git_sha()


def config_stem(tier: str, year: str) -> str:
    if tier not in TIER_CONFIGS:
        raise ValueError(f"unknown tier {tier!r}; known: {sorted(TIER_CONFIGS)}")
    return TIER_CONFIGS[tier].format(year=year)


def _load_verified(record, preds_dir):
    # Imported here rather than at module scope purely to keep the numeric core of this
    # module import-light (mirrors risk_coverage's lazy cost_model import). The verified
    # loader is used, not the raw reader, because these are headline numbers and the
    # gate's structural layer is exactly what catches a permuted id column -- which the
    # paired_subsample scope's id join would otherwise silently honour.
    from triage_lab import cost_model

    return cost_model.load_artifact_verified(record, preds_dir)


def _restrict(art, keep_ids: np.ndarray):
    """Row-subset an artifact's (id, y_true, y_pred) to `keep_ids`, id-sorted."""
    order = np.argsort(art.complaint_id, kind="stable")
    ids_sorted = art.complaint_id[order]
    pos = np.searchsorted(ids_sorted, keep_ids)
    if np.any(pos >= len(ids_sorted)) or not np.array_equal(ids_sorted[pos], keep_ids):
        missing = int(np.sum(ids_sorted[np.minimum(pos, len(ids_sorted) - 1)] != keep_ids))
        raise ValueError(
            f"paired_subsample restriction failed: {missing} of {len(keep_ids)} ids are "
            "absent from the artifact being restricted"
        )
    take = order[pos]
    return art.complaint_id[take], art.y_true[take], art.y_pred[take]


def paired_subsample_ids(tier: str, year: str, records, preds_dir) -> np.ndarray:
    """Sorted complaint_ids of the Tier C subsample for `year`, verified identical across
    the id-source tiers. Their agreement is the pairing claim, so it is asserted here."""
    sources = PAIRED_SUBSAMPLE_ID_SOURCES[tier]
    id_sets = {}
    for src in sources:
        stem = config_stem(src, year)
        art = _load_verified(records[stem], preds_dir)
        id_sets[src] = np.sort(art.complaint_id)
    first = id_sets[sources[0]]
    for src in sources[1:]:
        if not np.array_equal(first, id_sets[src]):
            raise ValueError(
                f"paired_subsample scope for {tier}/{year}: {sources[0]} and {src} do not "
                "cover identical rows, so the 'same rows' pairing claim is false"
            )
    return first


# ---------------------------------------------------------------------------
# Decomposition assembly
# ---------------------------------------------------------------------------

def _logged(record: dict, key: str) -> float | None:
    block = (record.get("metrics") or {}).get(key)
    return None if block is None else float(block["point"])


def _identity_check(name: str, computed: float, logged: float | None, applicable: bool) -> dict:
    if not applicable or logged is None:
        return {"name": name, "applicable": False, "computed": float(computed),
                "logged": logged, "ok": None}
    delta = abs(float(computed) - float(logged))
    return {"name": name, "applicable": True, "computed": float(computed),
            "logged": float(logged), "abs_delta": delta, "ok": bool(delta <= IDENTITY_TOL)}


def build_decomposition(
    tier: str,
    year: str,
    scope: str,
    *,
    records: dict,
    preds_dir=DEFAULT_PREDS_DIR,
    splits_stats_path=DEFAULT_SPLITS_STATS_PATH,
    ref_year: str = REF_YEAR,
    n_resamples: int = harness.N_RESAMPLES,
    seed: int = harness.BOOTSTRAP_SEED,
    generated_at: str | None = None,
    git_sha: str | None = None,
) -> dict:
    """Full JSON object for one (tier, year, scope) decomposition. See module docstring."""
    stem_ref, stem_year = config_stem(tier, ref_year), config_stem(tier, year)
    rec_ref, rec_year = records[stem_ref], records[stem_year]
    art_ref = _load_verified(rec_ref, preds_dir)
    art_year = _load_verified(rec_year, preds_dir)

    class_labels = list(art_ref.class_labels)
    if list(art_year.class_labels) != class_labels:
        raise ValueError(
            f"{tier}/{year}: reference and year artifacts declare different class label "
            "orders; the decomposition's coordinates would not be comparable"
        )
    n_classes = len(class_labels)
    if OAT_CLASS not in class_labels:
        raise ValueError(f"one-at-a-time class {OAT_CLASS!r} not among {class_labels}")
    oat_index = class_labels.index(OAT_CLASS)

    if scope == SCOPE_NATIVE:
        y_ref, p_ref = art_ref.y_true, art_ref.y_pred
        y_year, p_year = art_year.y_true, art_year.y_pred
        restricted_to = None
    elif scope == SCOPE_PAIRED:
        if tier not in PAIRED_SUBSAMPLE_ID_SOURCES:
            raise ValueError(f"tier {tier!r} has no paired_subsample id source")
        keep_ref = paired_subsample_ids(tier, ref_year, records, preds_dir)
        keep_year = paired_subsample_ids(tier, year, records, preds_dir)
        _, y_ref, p_ref = _restrict(art_ref, keep_ref)
        _, y_year, p_year = _restrict(art_year, keep_year)
        restricted_to = list(PAIRED_SUBSAMPLE_ID_SOURCES[tier])
    else:
        raise ValueError(f"unknown scope {scope!r}")

    true_ref = metrics.encode_labels(y_ref, class_labels)
    pred_ref = metrics.encode_labels(p_ref, class_labels)
    true_year = metrics.encode_labels(y_year, class_labels)
    pred_year = metrics.encode_labels(p_year, class_labels)

    pi_ref, cond_ref, sup_ref = prior_and_conditional(true_ref, pred_ref, n_classes)
    pi_year, cond_year, sup_year = prior_and_conditional(true_year, pred_year, n_classes)
    point, cells = component_vector(
        pi_ref, cond_ref, sup_ref, pi_year, cond_year, sup_year, oat_index
    )

    # --- identity gates -----------------------------------------------------
    # Cells A and D are the runs' own logged numbers under a different but algebraically
    # identical formula; if they disagree, either the artifact is not the run's or this
    # module's macro convention has drifted from metrics.py. Only meaningful at native
    # scope with artifact priors -- a row subset legitimately has a different macro-F1.
    gate_applicable = scope == SCOPE_NATIVE
    checks = [
        _identity_check("cell_A_vs_logged_macro_f1", cells["A_ref_mix_ref_behavior"],
                        _logged(rec_ref, "macro_f1"), gate_applicable),
        _identity_check("cell_D_vs_logged_macro_f1", cells["D_year_mix_year_behavior"],
                        _logged(rec_year, "macro_f1"), gate_applicable),
        _identity_check("balanced_accuracy_ref_vs_logged",
                        balanced_accuracy_cell(cond_ref, sup_ref),
                        _logged(rec_ref, "balanced_accuracy"), gate_applicable),
        _identity_check("balanced_accuracy_year_vs_logged",
                        balanced_accuracy_cell(cond_year, sup_year),
                        _logged(rec_year, "balanced_accuracy"), gate_applicable),
        _identity_check("accuracy_ref_vs_logged", float(pi_ref @ np.diag(cond_ref)),
                        _logged(rec_ref, "accuracy"), gate_applicable),
    ]
    failed = [c for c in checks if c["ok"] is False]
    if failed:
        detail = "; ".join(
            f"{c['name']}: computed {c['computed']!r} vs logged {c['logged']!r} "
            f"(|delta| = {c['abs_delta']:.3e} > {IDENTITY_TOL:.0e})"
            for c in failed
        )
        raise ValueError(
            f"{tier}/{year}/{scope} fails its identity gate -- {detail}. A decomposition "
            "whose endpoint cells are not the run's own logged numbers is not a "
            "decomposition of that run"
        )

    # --- bootstrap variants -------------------------------------------------
    boot = bootstrap_components(
        true_ref, pred_ref, true_year, pred_year, n_classes, oat_index,
        variant="both", n_resamples=n_resamples, seed=seed,
    )
    empty_counts = boot["empty_ref"] + boot["empty_year"]
    ci_valid = bool(np.all(empty_counts <= EMPTY_CLASS_CI_FRACTION * n_resamples))
    reps = boot["replicates"]
    blocks = {k: _ci_block(point[k], reps[k], ci_valid) for k in COMPONENT_KEYS}

    boot_ref_fixed = bootstrap_components(
        true_ref, pred_ref, true_year, pred_year, n_classes, oat_index,
        variant="ref_fixed", n_resamples=n_resamples, seed=seed,
    )
    ref_fixed_blocks = {
        k: _ci_block(point[k], boot_ref_fixed["replicates"][k], ci_valid)
        for k in PRIMARY_COMPONENTS
    }

    # pi_full_slice sensitivity: C resampled, mix pinned to the frozen full-slice counts.
    split_ref = SPLIT_FMT.format(year=ref_year)
    split_year = SPLIT_FMT.format(year=year)
    pi_full_ref = full_slice_priors(split_ref, class_labels, splits_stats_path)
    pi_full_year = full_slice_priors(split_year, class_labels, splits_stats_path)
    point_full, cells_full = component_vector(
        pi_full_ref, cond_ref, sup_ref, pi_full_year, cond_year, sup_year, oat_index
    )
    boot_full = bootstrap_components(
        true_ref, pred_ref, true_year, pred_year, n_classes, oat_index,
        variant="both", pi_fixed_ref=pi_full_ref, pi_fixed_year=pi_full_year,
        n_resamples=n_resamples, seed=seed,
    )
    full_blocks = {
        k: _ci_block(point_full[k], boot_full["replicates"][k], ci_valid)
        for k in PRIMARY_COMPONENTS
    }

    # --- share gate ---------------------------------------------------------
    gate_ok, gate_reason = share_gate(blocks["total"])
    if gate_ok:
        ratio = reps["prior::path_p"] / reps["total"]
        lo, hi = _percentile_ci(ratio)
        share_block = {
            "point": float(point["prior::path_p"] / point["total"]),
            "ci_lo": lo, "ci_hi": hi, "ci_valid": ci_valid,
            "estimator": "ratio_of_estimates",
            "gate": {"passed": True, "rule": "total-CI excludes 0 AND |total| >= "
                                             f"{SHARE_MIN_ABS_TOTAL}"},
        }
    else:
        share_block = None

    # --- provenance / diagnostics -------------------------------------------
    cfg_ref = predictions.load_config_checked(rec_ref)
    cfg_year = predictions.load_config_checked(rec_year)
    model_ref = cfg_ref.get("model", {})
    model_year = cfg_year.get("model", {})
    # `model.name` is per-RUN (it embeds the year), so it cannot stand as the system's id
    # in a two-year comparison. Prefer the provider slug (Tier C); fall back to the runner
    # (Tier A/B), and carry both per-run names explicitly so nothing is guessed downstream.
    model_id = model_ref.get("slug") or model_ref.get("runner", "")
    if model_id != (model_year.get("slug") or model_year.get("runner", "")):
        raise ValueError(
            f"{tier}: reference and year runs use different models "
            f"({model_id!r} vs {model_year.get('slug') or model_year.get('runner')!r}); "
            "a drift decomposition compares one system across years, not two systems"
        )
    scope_suffix = "" if scope == SCOPE_NATIVE else f" --scope {scope}"
    obj = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "git_sha": git_sha if git_sha is not None else _current_git_sha(),
        "repro_command": (
            f"uv run python -m triage_lab.prior_shift --tier {tier} --year {year}{scope_suffix}"
        ),
        "tier": tier,
        "model_id": model_id,
        "model_name": {"ref": model_ref.get("name", ""), "year": model_year.get("name", "")},
        "runner": model_ref.get("runner", ""),
        "year": year,
        "ref_year": ref_year,
        "scope": scope,
        "pi_source": PI_ARTIFACT,
        "sign_convention": "positive = degradation relative to ref_year",
        "class_labels": class_labels,
        "primary_path": "P_prior_first",
        "source": {
            "ref": _source_block(rec_ref, art_ref, stem_ref, len(true_ref)),
            "year": _source_block(rec_year, art_year, stem_year, len(true_year)),
            "logged_macro_f1": {
                "ref": _logged(rec_ref, "macro_f1"),
                "year": _logged(rec_year, "macro_f1"),
            },
            "restricted_to_rows_of": restricted_to,
            "identity_check": {
                "tol": IDENTITY_TOL,
                "applicable": gate_applicable,
                "checks": checks,
            },
        },
        "priors": {
            "ref": dict(zip(class_labels, pi_ref.tolist(), strict=True)),
            "year": dict(zip(class_labels, pi_year.tolist(), strict=True)),
            "full_slice_ref": dict(zip(class_labels, pi_full_ref.tolist(), strict=True)),
            "full_slice_year": dict(zip(class_labels, pi_full_year.tolist(), strict=True)),
            "pi_max_abs_dev_vs_full_ref": float(np.max(np.abs(pi_ref - pi_full_ref))),
            "pi_max_abs_dev_vs_full_year": float(np.max(np.abs(pi_year - pi_full_year))),
            "chi2_sub_vs_full_ref": chi2_divergence(pi_ref, pi_full_ref),
            "chi2_sub_vs_full_year": chi2_divergence(pi_year, pi_full_year),
        },
        "weights": {
            "to_year_mix": _weights_json(
                weight_block(pi_year, pi_ref, len(true_ref)), class_labels,
                "w_k = pi_year/pi_ref applied to REFERENCE rows (cell B; Path P's direction)",
            ),
            "to_ref_mix": _weights_json(
                weight_block(pi_ref, pi_year, len(true_year)), class_labels,
                "w_k = pi_ref/pi_year applied to YEAR rows (cell C; Path Q's direction)",
            ),
        },
        "cells": {
            name: {
                "macro_f1": value,
                "per_class_f1": dict(zip(
                    class_labels,
                    per_class_f1_cell(*_cell_parts(name, pi_ref, cond_ref, pi_year, cond_year)
                                      ).tolist(),
                    strict=True,
                )),
            }
            for name, value in cells.items()
        },
        "decomposition": {
            "total": blocks["total"],
            "primary": {
                "path": "P_prior_first",
                "prior": blocks["prior::path_p"],
                "within": blocks["within::path_p"],
                "note": "within-term carries the whole interaction; never quote without it",
            },
            "sensitivity": {
                "path_Q_behavior_first": {
                    "prior": blocks["prior::path_q"],
                    "within": blocks["within::path_q"],
                },
                "shapley": {
                    "prior": blocks["prior::shapley"],
                    "within": blocks["within::shapley"],
                },
                "anova": {
                    "prior_main": blocks["prior_main::anova"],
                    "within_main": blocks["within_main::anova"],
                    "interaction": blocks["interaction"],
                },
                "ref_fixed_bootstrap": {
                    "note": "reference sample held at its observed value; isolates the "
                            "shape of the year-over-year trend from the shared 2023 offset",
                    "total": ref_fixed_blocks["total"],
                    "prior": ref_fixed_blocks["prior::path_p"],
                    "within": ref_fixed_blocks["within::path_p"],
                },
                "pi_full_slice": {
                    "note": "priors pinned to the frozen full-slice mix (zero sampling "
                            "error); lower variance but cell D no longer equals the "
                            "logged macro_f1, which is why it is not primary",
                    "pi_source": PI_FULL_SLICE,
                    "cells": cells_full,
                    "total": full_blocks["total"],
                    "prior": full_blocks["prior::path_p"],
                    "within": full_blocks["within::path_p"],
                },
            },
            "interaction": blocks["interaction"],
            "prior_bracket": {
                "lo": min(point["prior::path_p"], point["prior::path_q"]),
                "hi": max(point["prior::path_p"], point["prior::path_q"]),
                "note": "Path Q .. Path P; the width IS the interaction magnitude",
            },
            "share_prior": share_block,
            "share_suppressed_reason": gate_reason,
        },
        "per_class": per_class_attribution(
            pi_ref, cond_ref, pi_year, cond_year, sup_ref, sup_year, class_labels
        ),
        "one_at_a_time_prior": {
            OAT_CLASS: {
                **blocks[f"one_at_a_time_prior::{OAT_CLASS}"],
                "additive": False,
                "note": "only this class's share moves to its year-Y value; the other "
                        "classes are renormalised pro rata and C_ref is held fixed. "
                        "One-at-a-time effects do NOT sum to the prior term",
            }
        },
        "accuracy_decomposition": accuracy_decomposition(
            pi_ref, cond_ref, pi_year, cond_year, class_labels
        ),
        "balanced_accuracy": {
            "ref": balanced_accuracy_cell(cond_ref, sup_ref),
            "year": balanced_accuracy_cell(cond_year, sup_year),
            **blocks["balanced_accuracy_delta"],
            "note": "macro-recall; population target independent of class mix, so this is "
                    "the counterfactual-free within-class drift channel",
        },
        "bootstrap": {
            "n_resamples": n_resamples,
            "seed": seed,
            "method": harness.BOOTSTRAP_METHOD,
            "ci_pct": [harness.CI_LOWER_PCT, harness.CI_UPPER_PCT],
            "resample": "both_slices_independent",
            "draw_order": "ref_then_year",
            "components_sum_to_total_per_replicate": True,
            "ci_note": "marginal percentile intervals; components sum exactly at the "
                       "point estimate, not at the interval endpoints",
            "ci_valid": ci_valid,
            "ci_valid_rule": f"no class empty in more than {EMPTY_CLASS_CI_FRACTION:.1%} "
                             "of replicates",
            "n_replicates_with_empty_class": dict(
                zip(class_labels, empty_counts.tolist(), strict=True)
            ),
        },
    }
    return obj


def _cell_parts(name, pi_ref, cond_ref, pi_year, cond_year):
    return {
        "A_ref_mix_ref_behavior": (pi_ref, cond_ref),
        "B_year_mix_ref_behavior": (pi_year, cond_ref),
        "C_ref_mix_year_behavior": (pi_ref, cond_year),
        "D_year_mix_year_behavior": (pi_year, cond_year),
    }[name]


def _weights_json(block: dict, class_labels, note: str) -> dict:
    out = {k: v for k, v in block.items() if k != "w"}
    out["w"] = dict(zip(class_labels, block["w"].tolist(), strict=True))
    out["note"] = note
    out["cap"] = None
    out["cap_note"] = "no cap by design: capping would change the estimand, so exposure " \
                      "is reported via n_eff/max instead"
    return out


def _source_block(record, art, config_name, n_used) -> dict:
    prov = art.provenance
    return {
        "run_id": record["run_id"],
        "config_name": config_name,
        "config_sha256": prov.get("config_sha256", ""),
        "git_sha": prov.get("git_sha", ""),
        "predictions_path": (record.get("extra") or {}).get("predictions_path", ""),
        "split": prov.get("split", ""),
        "split_sha256": prov.get("split_sha256", ""),
        "input_sha256": prov.get("input_sha256", ""),
        "n_artifact": len(art),
        "n_used": int(n_used),
    }


# ---------------------------------------------------------------------------
# Flat summary rows (drift-chart input)
# ---------------------------------------------------------------------------

def summary_rows(obj: dict) -> list[dict]:
    """One flat row per (tier, year, scope, pi_source, component)."""
    dec = obj["decomposition"]
    base = {
        "tier": obj["tier"],
        "year": obj["year"],
        "ref_year": obj["ref_year"],
        "scope": obj["scope"],
    }
    named = [
        (PI_ARTIFACT, "total", dec["total"]),
        (PI_ARTIFACT, "prior::path_p", dec["primary"]["prior"]),
        (PI_ARTIFACT, "within::path_p", dec["primary"]["within"]),
        (PI_ARTIFACT, "prior::path_q", dec["sensitivity"]["path_Q_behavior_first"]["prior"]),
        (PI_ARTIFACT, "within::path_q", dec["sensitivity"]["path_Q_behavior_first"]["within"]),
        (PI_ARTIFACT, "prior::shapley", dec["sensitivity"]["shapley"]["prior"]),
        (PI_ARTIFACT, "within::shapley", dec["sensitivity"]["shapley"]["within"]),
        (PI_ARTIFACT, "prior_main::anova", dec["sensitivity"]["anova"]["prior_main"]),
        (PI_ARTIFACT, "within_main::anova", dec["sensitivity"]["anova"]["within_main"]),
        (PI_ARTIFACT, "interaction", dec["interaction"]),
        (PI_ARTIFACT, f"one_at_a_time_prior::{OAT_CLASS}", obj["one_at_a_time_prior"][OAT_CLASS]),
        (PI_ARTIFACT, "balanced_accuracy_delta", obj["balanced_accuracy"]),
        (PI_ARTIFACT, "total::ref_fixed", dec["sensitivity"]["ref_fixed_bootstrap"]["total"]),
        (PI_ARTIFACT, "prior::path_p::ref_fixed",
         dec["sensitivity"]["ref_fixed_bootstrap"]["prior"]),
        (PI_ARTIFACT, "within::path_p::ref_fixed",
         dec["sensitivity"]["ref_fixed_bootstrap"]["within"]),
        (PI_FULL_SLICE, "total", dec["sensitivity"]["pi_full_slice"]["total"]),
        (PI_FULL_SLICE, "prior::path_p", dec["sensitivity"]["pi_full_slice"]["prior"]),
        (PI_FULL_SLICE, "within::path_p", dec["sensitivity"]["pi_full_slice"]["within"]),
    ]
    rows = []
    for pi_source, component, block in named:
        rows.append({
            **base,
            "pi_source": pi_source,
            "component": component,
            "point": block["point"],
            "ci_lo": block["ci_lo"],
            "ci_hi": block["ci_hi"],
            "ci_valid": block.get("ci_valid", True),
            "is_primary": bool(
                pi_source == PI_ARTIFACT
                and obj["scope"] == SCOPE_NATIVE
                and component in PRIMARY_COMPONENTS
            ),
        })
    share = dec["share_prior"]
    rows.append({
        **base,
        "pi_source": PI_ARTIFACT,
        "component": "share_prior",
        "point": None if share is None else share["point"],
        "ci_lo": None if share is None else share["ci_lo"],
        "ci_hi": None if share is None else share["ci_hi"],
        "ci_valid": None if share is None else share["ci_valid"],
        "is_primary": False,
        "suppressed_reason": dec["share_suppressed_reason"],
    })
    return rows


# ---------------------------------------------------------------------------
# Deterministic JSON output
# ---------------------------------------------------------------------------

def _round_tree(value):
    """Round floats to JSON_ROUND; NaN/inf -> None (valid JSON, honest about undefined)."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, dict):
        return {k: _round_tree(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_round_tree(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, JSON_ROUND)
    return value


def write_json(obj: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_round_tree(obj), sort_keys=True, indent=2) + "\n")
    return path


def output_name(tier: str, year: str, scope: str) -> str:
    suffix = "" if scope == SCOPE_NATIVE else f"__{scope}"
    return f"{tier}__{year}{suffix}.json"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def default_jobs() -> list[tuple[str, str, str]]:
    """12 native decompositions (4 tiers x 3 years) + Tier A paired_subsample x 3 years."""
    jobs = [(tier, year, SCOPE_NATIVE) for tier in TIER_ORDER for year in DEFAULT_YEARS]
    jobs += [(tier, year, SCOPE_PAIRED)
             for tier in PAIRED_SUBSAMPLE_ID_SOURCES for year in DEFAULT_YEARS]
    return jobs


def select_jobs(tiers, years, scopes) -> list[tuple[str, str, str]]:
    jobs = default_jobs()
    if tiers:
        jobs = [j for j in jobs if j[0] in set(tiers)]
    if years:
        jobs = [j for j in jobs if j[1] in set(years)]
    if scopes:
        jobs = [j for j in jobs if j[2] in set(scopes)]
    return jobs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m triage_lab.prior_shift")
    parser.add_argument("--all", action="store_true",
                        help="the full default set (12 native + 3 tier_a paired_subsample)")
    parser.add_argument("--tier", action="append", choices=sorted(TIER_CONFIGS),
                        help="restrict to this tier (repeatable)")
    parser.add_argument("--year", action="append", choices=list(DEFAULT_YEARS),
                        help="restrict to this year (repeatable)")
    parser.add_argument("--scope", action="append", choices=[SCOPE_NATIVE, SCOPE_PAIRED],
                        help="restrict to this scope (repeatable)")
    parser.add_argument("--preds-dir", type=Path, default=DEFAULT_PREDS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--results", type=Path, default=harness.DEFAULT_RESULTS_PATH)
    parser.add_argument("--splits-stats", type=Path, default=DEFAULT_SPLITS_STATS_PATH)
    args = parser.parse_args(argv)

    if not args.all and not (args.tier or args.year or args.scope):
        parser.error("give --all or at least one of --tier/--year/--scope")

    jobs = default_jobs() if args.all else select_jobs(args.tier, args.year, args.scope)
    if not jobs:
        print("no decompositions match the given selectors")
        return 0

    records = predictions.records_by_config(args.results)
    generated_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    git_sha = _current_git_sha()

    all_rows: list[dict] = []
    for tier, year, scope in jobs:
        obj = build_decomposition(
            tier, year, scope,
            records=records, preds_dir=args.preds_dir,
            splits_stats_path=args.splits_stats,
            generated_at=generated_at, git_sha=git_sha,
        )
        out_path = write_json(obj, args.out_dir / output_name(tier, year, scope))
        all_rows.extend(summary_rows(obj))
        dec = obj["decomposition"]
        share = dec["share_prior"]
        share_txt = "suppressed" if share is None else f"{share['point']:.3f}"
        print(
            f"[{tier:14s} {year:6s} {scope:16s}] "
            f"total={dec['total']['point']:+.6f} "
            f"[{dec['total']['ci_lo']:+.6f},{dec['total']['ci_hi']:+.6f}]  "
            f"prior={dec['primary']['prior']['point']:+.6f} "
            f"[{dec['primary']['prior']['ci_lo']:+.6f},{dec['primary']['prior']['ci_hi']:+.6f}]  "
            f"within={dec['primary']['within']['point']:+.6f}  "
            f"inter={dec['interaction']['point']:+.6f}  share={share_txt} -> {out_path}"
        )

    if args.all:
        summary_path = write_json(
            {
                "schema_version": SCHEMA_VERSION,
                "generated_at": generated_at,
                "git_sha": git_sha,
                "ref_year": REF_YEAR,
                "sign_convention": "positive = degradation relative to ref_year",
                "primary_path": "P_prior_first",
                "repro_command": "uv run python -m triage_lab.prior_shift --all",
                "rows": all_rows,
            },
            args.out_dir / "summary.json",
        )
        print(f"summary: {len(all_rows)} rows -> {summary_path}")
    else:
        print("summary.json not rewritten (partial selection; use --all)")
    return 0


if __name__ == "__main__":
    import sys

    from triage_lab.prior_shift import main as _main

    sys.exit(_main())
