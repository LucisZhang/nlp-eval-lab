"""Threshold optimization on CAL (Phase 4 task 3).

Where `cost_model` prices a *fixed* policy, this module searches over one: it sweeps the
Tier A confidence gate tau_A across every distinct `p_max` in a CAL prediction artifact,
prices each operating point under `configs/cost_model_v1.yaml`, and reports the argmin —
plus the fixed reference policies that argmin has to beat, and a sensitivity grid showing
how the answer moves when the two ESTIMATED dollar parameters move.

**Scope, per the owner's 2026-08-07 amendment narrowing UPGRADE_PLAN §4.2** (recorded in
`AMENDMENT_NOTE` and echoed into every output file):

- Confidence thresholds exist at **Tier A only** (and Tier B when it lands). There is no
  tau_C: Tier C is an unconditional TERMINAL stop once escalated. Its `p_max` is a
  degenerate one-hot (Phase 4 task 1), so it carries no rankable confidence to threshold.
- The **only** Tier C -> human signal is `parse_failed` on that row's receipt.
- The §4.2 3-sample self-consistency proxy is **considered-and-deferred**, not built: it
  needs temperature > 0, i.e. a new protocol version and new spend, for a signal whose S/N
  looks poor given the observed temperature-0 rerun flips.

Everything here is offline: CAL artifacts and committed receipts only, no API calls, no
TEST-* artifact is opened anywhere in this module, and `results/runs.jsonl` is read-only.

**Four policy families**, all expressed as one structure — "Tier A answers above the gate,
a per-row *escalation arm* handles the rest" — so a single swept implementation serves them
all and the families differ only in how their arm is built:

- ``a_to_human`` (full CAL, and the paired subset for like-for-like comparison): the arm is
  the human queue. Escalated rows pay `c_human`, no API, and are assumed resolved
  correctly.
- ``a_to_c_parsefail_human`` (paired 1,500-row subset only): the arm is Tier C. Every
  escalated row pays that row's MEASURED per-call cost, joined from the zero-shot CAL
  receipts. If the receipt says `parse_failed`, the row continues to the human queue and
  additionally pays `c_human` — the Tier C spend is still charged, because incurred spend
  is not refunded by a later routing decision (`cost_model`'s semantics). Otherwise Tier C's
  label answers and a wrong label costs `c_misroute`.

  A parse-failed row's `y_pred` in the Tier C artifact is the *fallback* label, which may be
  right or wrong by luck. This module ignores it: `parse_failed` overrides, the row routes
  to a human, and it can contribute neither a misroute charge nor a correct answer.

- ``a_to_b`` (full CAL and the paired subset; needs a cost config that prices Tier B): the
  arm is Tier B, TERMINAL — the frontier slot's own definition, so this family has one gate,
  not two. Escalated rows pay Tier B's declared amortized compute and a wrong Tier B label
  costs `c_misroute`. There is no human hand-off: a local classifier always emits a label,
  so Tier C's parse-failure signal has no analogue here.

- ``a_to_b_to_c`` (paired subset only, same reason as ``a_to_c``): the arm is Tier B gated
  at `tau_B`, falling through to Tier C terminally below it. This is the ONLY two-gate
  family, and its `(tau_A, tau_B)` pair is fit by an exhaustive JOINT argmin: the arm is
  rebuilt at every candidate `tau_B` and the ordinary `tau_A` sweep is run against each one
  (see `joint_sweep`). Escalated rows pay Tier B unconditionally — the forward pass happened
  before its confidence could be read — and additionally pay Tier C when they fall through.

The fixed reference points fall out of the same sweep as its endpoints, so they are priced
by identical code on identical data: ``a_only`` is the k = n endpoint (Tier A answers
everything) of every family, and the k = 0 endpoint is ``all_human`` / ``c_only`` /
``b_only`` / ``b_to_c`` respectively (see `NO_GATE_LABEL`).

**Statistics.** Point estimates are computed for the whole tau grid by prefix sums (O(n)
for all n thresholds, which is what makes an 86,972-point grid free). Bootstrap CIs — the
frozen harness constants, shared-index resampling, reusing `cost_model.bootstrap_cost` —
are computed ONLY at the operating points (tau*, a_only, c_only, all_human), because a CI
per grid point would be 86,972 bootstraps for a curve nobody reads a CI off.

**Two honesty notes ride in every output** (`SELECTION_OPTIMISM_NOTE`, `PMAX_SPACE_NOTE`):
tau* is chosen on the same CAL rows it is scored on, so its CAL cost is optimistically
biased and unbiased evaluation must happen on TEST (task 4); and the CAL rungs are
`calibration: none` while the TEST-IID final config is isotonic-calibrated, so a tau*
selected here is a threshold in a DIFFERENT probability space than the one it would be
applied in. Every operating point is therefore recorded twice — as tau* and as a target
coverage — and choosing between threshold-transfer and coverage-quantile-transfer is
explicitly a task-4 decision, not something this module papers over.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from triage_lab import cost_model, harness, predictions

REPO_ROOT = harness.REPO_ROOT
DEFAULT_PREDS_DIR = harness.DEFAULT_PREDS_DIR
DEFAULT_RESULTS_PATH = harness.DEFAULT_RESULTS_PATH
DEFAULT_THRESHOLDS_DIR = REPO_ROOT / "results" / "thresholds"

SCHEMA_VERSION = "thresholds-v1"

# Grid rows kept in the JSON. The sweep itself always runs at full resolution (every
# distinct p_max); this only downsamples what gets written, exactly as risk_coverage does,
# and the endpoints plus the tau* row are always kept whatever the sampling says.
DEFAULT_MAX_GRID_POINTS = 512

# The CAL rungs, and which one is PRIMARY. Primary = the rung whose feature/model config
# is identical to the TEST-IID final config (tier_a_logreg_test_iid) except for the two
# reporting switches (calibration none->isotonic, split cal->test_iid), so that a tau*
# selected on it describes the model that actually ships. NOT the best-scoring CAL rung:
# tier_a_logreg_word_cal beats it on CAL accuracy/macro_f1/aurc (see the task-3 report).
TIER_A_CAL_CONFIGS = (
    "tier_a_logreg_word_cal",
    "tier_a_logreg_wordchar_cal",
    "tier_a_cnb_wordchar_cal",
)
PRIMARY_TIER_A_CONFIG = "tier_a_logreg_wordchar_cal"

# ---------------------------------------------------------------------------
# Threshold DERIVATIONS
# ---------------------------------------------------------------------------
# v1 fits tau* on RAW CAL probabilities (calibration: none) and the router then applies it
# to the isotonic-calibrated TEST-IID artifact. That crosses a probability-space boundary
# and cost 5-16 points of realized coverage — measured, reported, and kept in the repo as
# the documented lesson rather than deleted.
#
# v2 fits tau* on tier_a_logreg_wordchar_isocal_cal: the SAME rung with calibration
# isotonic, i.e. in the deployment point's own probability space, so the constant transfers
# without a space change. v2 is primary for reported router numbers; v1 files are never
# rewritten (a test asserts byte-identity), and files carry `derivation` so the two
# generations can never be silently mixed. Files predating this field are v1 by definition,
# which is why the reader defaults to DERIVATION_V1 rather than requiring the key.
DERIVATION_V1 = "v1-raw"
DERIVATION_V2 = "v2-isocal"
V2_TIER_A_CAL_CONFIG = "tier_a_logreg_wordchar_isocal_cal"

DERIVATIONS: dict[str, dict] = {
    DERIVATION_V1: {
        "tier_a_configs": TIER_A_CAL_CONFIGS,
        "primary": PRIMARY_TIER_A_CONFIG,
        "summary_name": "summary__cost-{sha8}.json",
    },
    DERIVATION_V2: {
        "tier_a_configs": (V2_TIER_A_CAL_CONFIG,),
        "primary": V2_TIER_A_CAL_CONFIG,
        "summary_name": "summary__v2-isocal__cost-{sha8}.json",
    },
}

ISOCAL_IN_SAMPLE_NOTE = (
    "v2 derivation caveat, by design: tier_a_logreg_wordchar_isocal_cal fits its isotonic "
    "calibrator ON CAL and predicts CAL (same CalibratedClassifierCV/FrozenEstimator path "
    "as the TEST-IID final, base model never refit). CAL is the split thresholds have "
    "always been fit on, but these CAL probabilities are in-sample and are better "
    "calibrated here than they would be on held-out data. The thresholds derived from them "
    "are evaluated only on TEST."
)

# The paired Tier C CAL run whose 1,500 ids define the subset and supply the measured
# per-call costs and parse_failed flags. Zero-shot, matching the Tier C protocol taken to
# TEST (the few-shot ablation is not the reported arm).
TIER_C_CAL_CONFIG = "tier_c_haiku_ablation_zeroshot_cal"

FAMILY_A_TO_HUMAN = "a_to_human"
FAMILY_A_TO_C = "a_to_c_parsefail_human"
FAMILY_A_TO_B = "a_to_b"
FAMILY_A_TO_B_TO_C = "a_to_b_to_c"
DATASET_FULL_CAL = "full_cal"
DATASET_PAIRED = "paired_subset"

# The k = 0 endpoint of each family's tau_A sweep — "Tier A answers nothing" — is a real
# policy, and which one it is depends on what the escalation arm does. Naming it per family
# keeps `b_only` from being reported as `c_only` in the A->B cascade.
NO_GATE_LABEL = {
    FAMILY_A_TO_HUMAN: "all_human",
    FAMILY_A_TO_C: "c_only",
    FAMILY_A_TO_B: "b_only",
    FAMILY_A_TO_B_TO_C: "b_to_c",
}

# The Tier B CAL rung every Tier B threshold is fit against: DistilBERT (B2), the
# frozen-protocol Tier B winner on TEST-IID (macro-F1 0.7950 [0.7909, 0.7988] vs B1's
# 0.7878 / 0.7878 / 0.7863, paired CIs excluding zero, McNemar p <= 6e-8, and better ECE
# and AURC) — which is also the cheaper model, so using it as the cascade rung costs
# nothing on either axis. The three B1 seeds keep their own single-tier frontier points;
# they are not cascade rungs.
TIER_B_CAL_CONFIG = "tier_b2_distilbert_s0_cal"

# Sensitivity grid: log-spaced (x2 per step) around the v1 defaults, which are an EXACT
# cell (6.00, 2.50) so the headline number is a member of its own sensitivity grid rather
# than an interpolation of it. 6 x 6 = 36 cells.
SENSITIVITY_C_MISROUTE: tuple[float, ...] = (0.75, 1.5, 3.0, 6.0, 12.0, 24.0)
SENSITIVITY_C_HUMAN: tuple[float, ...] = (0.3125, 0.625, 1.25, 2.5, 5.0, 10.0)

JSON_ROUND = cost_model.JSON_ROUND
_round = cost_model._round
_round_ci = cost_model._round_ci

AMENDMENT_NOTE = (
    "Owner-approved amendment (2026-08-07) narrowing UPGRADE_PLAN §4.2: confidence "
    "thresholds exist at Tier A only (Tier B when it lands). Tier C is an unconditional "
    "TERMINAL stop once escalated — there is no tau_C and Tier C confidence is never used, "
    "because its p_max is a degenerate one-hot. The only Tier C -> human signal is "
    "parse_failed on that row's receipt. The §4.2 3-sample self-consistency proxy is "
    "considered-and-deferred (it would require temperature > 0 = a new protocol version "
    "and new spend, for questionable signal-to-noise), and no part of it is implemented."
)

SELECTION_OPTIMISM_NOTE = (
    "tau* is selected by argmin on the SAME CAL rows it is scored on, so the cost reported "
    "at tau* is optimistically biased (selection optimism) and its CI covers sampling "
    "noise at a FIXED tau, not the variability of the selection itself. Unbiased "
    "evaluation of the selected operating point happens once, on TEST, in task 4."
)

PMAX_SPACE_NOTE = (
    "tau* is a threshold in THIS artifact's p_max space. All three CAL rungs are "
    "calibration: none (raw model probabilities), while the TEST-IID final config "
    "tier_a_logreg_test_iid is calibration: isotonic (fit on CAL). A tau selected here is "
    "therefore NOT directly comparable to a tau in the calibrated TEST-IID p_max space. "
    "Each operating point is recorded both as tau* and as a target coverage (fraction "
    "answered by Tier A); which transfer rule to use — threshold matching vs "
    "coverage-quantile matching — and how to handle the calibration-space mismatch is a "
    "task-4 decision and is deliberately NOT resolved here."
)

SENSITIVITY_EVIDENCE_NOTE = (
    "Evidence class: estimated parameters, measured predictions. The grid axes "
    "(c_misroute, c_human) are ESTIMATED business defaults being varied on purpose; the "
    "per-row predictions, correctness and Tier C API costs underneath every cell are "
    "MEASURED. Point estimates only — no CIs on the grid."
)

TIE_BREAK_NOTE = (
    "Ties in the cost argmin are broken toward the LARGEST Tier A coverage (fewest "
    "escalations): at equal modeled cost, the policy that escalates less is cheaper in "
    "latency and operational load, neither of which this cost model prices."
)

JOINT_TIE_BREAK_NOTE = (
    "Two-gate families (a_to_b_to_c) break ties in the joint argmin lexicographically: "
    "lowest cost, then LARGEST Tier A coverage, then LARGEST Tier B coverage. Same "
    "reasoning as the single-gate rule, applied to each rung in cascade order — at equal "
    "modeled cost, answer as early and as cheaply in the cascade as possible."
)

TIER_B_ARM_NOTE = (
    "The Tier B arm has no human hand-off: a local classifier always emits a label, so "
    "the parse-failure signal that gives Tier C its only C->human route has no analogue "
    "here. Every escalated row is answered by Tier B and a wrong answer costs c_misroute. "
    "Tier B compute is charged per escalated row at the ESTIMATED amortized figure "
    "declared in the cost config (measured throughput x an estimated GPU rental rate); it "
    "is ~5 orders of magnitude below the misroute term, so no operating point moves with "
    "it."
)

TIER_B_IN_SAMPLE_NOTE = (
    "Tier B CAL caveat, by design and symmetric with ISOCAL_IN_SAMPLE_NOTE: the B2 CAL "
    "rung fits its temperature scaling ON CAL and predicts CAL (same trained checkpoint as "
    "the TEST-IID final, never refit), so tau_B is fit on in-sample-calibrated "
    "probabilities that are better calibrated here than they would be on held-out data. "
    "CAL is the split thresholds have always been fit on, and the thresholds derived from "
    "them are evaluated only on TEST."
)

JOINT_SWEEP_NOTE = (
    "a_to_b_to_c's (tau_A, tau_B) pair is fit JOINTLY and exhaustively, not sequentially: "
    "for every distinct Tier B p_max the escalation arm is rebuilt (B answers above "
    "tau_B, otherwise C answers terminally and the row pays BOTH B's compute and C's "
    "measured call — incurred spend is never refunded) and the full tau_A sweep is run "
    "against it, then the global argmin is taken over the product. Affordable because "
    "the family lives only on the 1,500-row paired subset. A sequential fit would choose "
    "tau_A against a different arm than the one it ends up facing."
)

PAIRED_COMPARISON_NOTE = (
    "The per-operating-point CIs under `operating_points` are MARGINAL: two operating "
    "points of the same policy share every example, so their marginal bands overlap even "
    "when one dominates. Any 'tau* beats X' claim must cite `paired_deltas` (bootstrap CI "
    "on the per-example cost difference, same shared-index contract) and requires that CI "
    "to exclude zero — CLAUDE.md's comparison rule."
)

HUMAN_CORRECT_NOTE = (
    "Rows routed to a human are recorded as correct (the cost model's P(error|human)=0 "
    "assumption). This affects the reported accuracy_system only; it cannot affect cost, "
    "because the misroute charge is gated on the row being machine-answered."
)


# ---------------------------------------------------------------------------
# Policy data: Tier A gate + a per-row escalation arm
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EscalationArm:
    """What happens to a row Tier A does NOT answer, per row.

    `api_cost_usd` is spend incurred by escalating that row, `to_human` marks rows that
    end in the human queue, and `correct` says whether the arm's answer is right (True for
    human-queued rows, by the model's human-correct assumption).
    """

    name: str
    api_cost_usd: np.ndarray
    to_human: np.ndarray
    correct: np.ndarray
    # Two-gate families only: which escalated rows the MIDDLE tier answered (the rest fall
    # through to the terminal tier). `None` for one-arm families, where there is no middle
    # tier whose coverage could be reported.
    b_answered: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.api_cost_usd)


@dataclass(frozen=True)
class PolicyData:
    """Everything a sweep needs: the Tier A gate signal and the escalation arm."""

    family: str
    dataset: str
    ids: np.ndarray
    p_max: np.ndarray
    correct_a: np.ndarray
    arm: EscalationArm
    inputs: dict  # provenance of every artifact/receipt log consumed

    def __len__(self) -> int:
        return len(self.ids)


def human_arm(n: int) -> EscalationArm:
    """Escalate straight to a human: c_human, no API spend, assumed correct."""
    return EscalationArm(
        name="human",
        api_cost_usd=np.zeros(n, dtype=np.float64),
        to_human=np.ones(n, dtype=bool),
        correct=np.ones(n, dtype=bool),
    )


def tier_c_arm(y_true, y_pred, api_cost_usd, parse_failed) -> EscalationArm:
    """Escalate to Tier C, terminal, with parse failure as the only hand-off to a human.

    A parse-failed row's `y_pred` is the fallback label; it is discarded here rather than
    scored, because under this policy that row was never answered by the model — it went
    to a human. Charging its fallback label as a correct or incorrect answer would credit
    (or blame) the router for a coin flip.
    """
    parse_failed = np.asarray(parse_failed, dtype=bool)
    c_correct = np.asarray(y_pred, dtype=object) == np.asarray(y_true, dtype=object)
    return EscalationArm(
        name="tier_c_terminal_parsefail_human",
        api_cost_usd=np.asarray(api_cost_usd, dtype=np.float64),
        to_human=parse_failed,
        correct=np.where(parse_failed, True, c_correct),
    )


def tier_b_arm(y_true, y_pred, per_example_usd: float) -> EscalationArm:
    """Escalate to Tier B, terminal, no human hand-off. See TIER_B_ARM_NOTE."""
    y_true = np.asarray(y_true, dtype=object)
    n = len(y_true)
    return EscalationArm(
        name="tier_b_terminal",
        api_cost_usd=np.full(n, float(per_example_usd), dtype=np.float64),
        to_human=np.zeros(n, dtype=bool),
        correct=np.asarray(y_pred, dtype=object) == y_true,
    )


def tier_b_then_c_arm(b_y_true, b_y_pred, b_p_max, tau_b: float, b_per_example_usd: float,
                      c_y_pred, c_api_cost_usd, parse_failed) -> EscalationArm:
    """Escalate to Tier B gated at `tau_b`, then to Tier C terminally below it.

    Every row reaching this arm has already run Tier B, so it pays Tier B's compute
    unconditionally; a row that then falls through to Tier C pays that call as well. That
    is `cost_model`'s incurred-spend semantics, not a modeling choice made here: the B
    forward pass happened before its confidence could be read.

    Tier C's terminal/parse-fail handling is identical to `tier_c_arm` — a parse-failed row
    goes to a human and its fallback label is discarded rather than scored.
    """
    b_y_true = np.asarray(b_y_true, dtype=object)
    answered_b = np.asarray(b_p_max, dtype=np.float64) >= tau_b
    parse_failed = np.asarray(parse_failed, dtype=bool)
    b_correct = np.asarray(b_y_pred, dtype=object) == b_y_true
    c_correct = np.asarray(c_y_pred, dtype=object) == b_y_true
    api = np.full(len(b_y_true), float(b_per_example_usd), dtype=np.float64)
    api = api + np.where(answered_b, 0.0, np.asarray(c_api_cost_usd, dtype=np.float64))
    return EscalationArm(
        name="tier_b_gate_then_tier_c_terminal_parsefail_human",
        api_cost_usd=api,
        to_human=(~answered_b) & parse_failed,
        correct=np.where(answered_b, b_correct,
                         np.where(parse_failed, True, c_correct)),
        b_answered=answered_b,
    )


def materialize(policy: PolicyData, tau: float):
    """Per-example (correct, api_cost_usd, to_human) for the policy at gate `tau`.

    The reference implementation: Tier A answers iff `p_max >= tau` (free, its own
    correctness), otherwise the escalation arm's row applies. `sweep` computes the same
    numbers for every tau at once via prefix sums; a test pins the two against each other
    at every grid point.
    """
    answered = policy.p_max >= tau
    return (
        np.where(answered, policy.correct_a, policy.arm.correct),
        np.where(answered, 0.0, policy.arm.api_cost_usd),
        np.where(answered, False, policy.arm.to_human),
    )


def cost_at(policy: PolicyData, tau: float, *, c_misroute: float,
            c_human: float) -> dict[str, float]:
    """Expected cost per 1k at one gate, via the `cost_model` scorer (reference path)."""
    correct, api, to_human = materialize(policy, tau)
    return cost_model.expected_cost_per_1k(
        correct, api, to_human, c_misroute=c_misroute, c_human=c_human)


def bootstrap_at(policy: PolicyData, tau: float, *, c_misroute: float, c_human: float,
                 n_resamples: int = harness.N_RESAMPLES,
                 seed: int = harness.BOOTSTRAP_SEED) -> dict[str, dict[str, float]]:
    """Bootstrap bands at ONE gate, through `cost_model`'s shared-index resampler.

    The gate is held FIXED across replicates: this is the CI of "what does this operating
    point cost", not of "what would the optimizer have picked". See SELECTION_OPTIMISM_NOTE.
    """
    correct, api, to_human = materialize(policy, tau)
    return cost_model.bootstrap_cost(
        correct, api, to_human, c_misroute=c_misroute, c_human=c_human,
        n_resamples=n_resamples, seed=seed,
    )


# ---------------------------------------------------------------------------
# The sweep (prefix sums over the confidence-sorted rows)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Grid:
    """Tie-aware tau grid, coverage-ascending, plus the prefix sums every cost needs.

    Row j means "Tier A answers the k_j most confident rows". k = 0 (answer nothing) is a
    real, reachable operating point — it is `all_human` / `c_only` — so it leads the grid,
    carried as tau = +inf ("no row clears the gate") and serialized as null.
    """

    tau: np.ndarray          # descending; tau[0] = +inf
    k: np.ndarray            # ascending count answered by Tier A
    n: int
    cum_a_wrong: np.ndarray  # prefix counts over confidence-sorted rows, indexed by k
    cum_a_correct: np.ndarray
    tail_arm_misroute: np.ndarray  # suffix counts (rows NOT answered by A)
    tail_arm_correct: np.ndarray
    tail_arm_human: np.ndarray
    tail_arm_api: np.ndarray


def build_grid(policy: PolicyData) -> Grid:
    """Sort by confidence, then prefix/suffix-sum every quantity the sweep needs."""
    p_max = np.asarray(policy.p_max, dtype=np.float64)
    n = len(p_max)
    if n == 0:
        raise ValueError("cannot sweep an empty policy")
    if not np.all(np.isfinite(p_max)):
        # NaN is the dangerous one: `nan >= tau` is False at EVERY threshold, so a NaN row
        # would silently escalate always, never appear as a threshold, and quietly shift
        # every coverage figure. Refuse rather than sweep a signal we cannot order.
        bad = np.flatnonzero(~np.isfinite(p_max))
        raise ValueError(
            f"{len(bad)} row(s) have non-finite p_max (ids "
            f"{policy.ids[bad][:cost_model.MAX_OFFENDERS_SHOWN].tolist()}"
            f"{' ...' if len(bad) > cost_model.MAX_OFFENDERS_SHOWN else ''}); a confidence "
            "gate cannot be swept over an unorderable signal"
        )
    order = np.argsort(-p_max, kind="stable")  # desc, ties by index: metrics.py convention

    a_wrong = (~np.asarray(policy.correct_a, dtype=bool))[order].astype(np.float64)
    arm = policy.arm
    arm_human = np.asarray(arm.to_human, dtype=bool)[order]
    arm_correct = np.asarray(arm.correct, dtype=bool)[order]
    arm_api = np.asarray(arm.api_cost_usd, dtype=np.float64)[order]
    # An escalated row costs a misroute only if a MACHINE answered it and got it wrong.
    arm_misroute = (~arm_human & ~arm_correct).astype(np.float64)
    arm_machine_correct = (~arm_human & arm_correct).astype(np.float64)

    def prefix(a):
        return np.concatenate(([0.0], np.cumsum(a)))

    cum_a_wrong = prefix(a_wrong)
    cum_a_correct = prefix(1.0 - a_wrong)
    cum_arm_misroute = prefix(arm_misroute)
    cum_arm_correct = prefix(arm_machine_correct)
    cum_arm_human = prefix(arm_human.astype(np.float64))
    cum_arm_api = prefix(arm_api)

    # Tie-aware k for each distinct tau: k = #{i : p_max_i >= tau}.
    values, counts = np.unique(p_max, return_counts=True)          # ascending
    taus_desc = values[::-1]
    k_desc = np.cumsum(counts[::-1])
    tau = np.concatenate(([np.inf], taus_desc))
    k = np.concatenate(([0], k_desc)).astype(np.int64)

    return Grid(
        tau=tau,
        k=k,
        n=n,
        cum_a_wrong=cum_a_wrong[k],
        cum_a_correct=cum_a_correct[k],
        tail_arm_misroute=cum_arm_misroute[-1] - cum_arm_misroute[k],
        tail_arm_correct=cum_arm_correct[-1] - cum_arm_correct[k],
        tail_arm_human=cum_arm_human[-1] - cum_arm_human[k],
        tail_arm_api=cum_arm_api[-1] - cum_arm_api[k],
    )


def sweep(grid: Grid, *, c_misroute: float, c_human: float) -> dict[str, np.ndarray]:
    """Point estimates for EVERY gate on the grid, in one vectorized pass.

    Identical arithmetic to `cost_at` (which goes row by row through `cost_model`), just
    reassociated: sums over per-row indicators become differences of prefix sums.
    """
    n = grid.n
    scale = cost_model.PER_N_COMPLAINTS / n

    misroute_count = grid.cum_a_wrong + grid.tail_arm_misroute
    misroute = float(c_misroute) * misroute_count * scale
    api = grid.tail_arm_api * scale
    human = float(c_human) * grid.tail_arm_human * scale
    total = misroute + api + human

    n_human = grid.tail_arm_human
    n_machine = n - n_human
    machine_correct = grid.cum_a_correct + grid.tail_arm_correct
    with np.errstate(invalid="ignore", divide="ignore"):
        accuracy_machine = np.where(n_machine > 0, machine_correct / n_machine, np.nan)
    return {
        "tau": grid.tau,
        "n_answered_a": grid.k.astype(np.float64),
        "coverage_a": grid.k / n,
        "escalation_rate": (n - grid.k) / n,
        "human_rate": n_human / n,
        "accuracy_machine": accuracy_machine,
        "accuracy_system": (machine_correct + n_human) / n,
        "cost_per_1k": total,
        "misroute_per_1k": misroute,
        "api_per_1k": api,
        "human_per_1k": human,
    }


def tau_star(policy: PolicyData, cfg: cost_model.CostConfig) -> float:
    """The cost-minimizing gate, at FULL float precision (never the rounded JSON value).

    Re-materializing a policy from a rounded tau could move rows across the gate, so any
    code that needs the threshold as a number — rather than as a report field — takes it
    from here.
    """
    rows = sweep(build_grid(policy), c_misroute=cfg.c_misroute_usd,
                 c_human=cfg.c_human_usd)
    return float(rows["tau"][argmin_index(rows["cost_per_1k"])])


def argmin_index(cost: np.ndarray) -> int:
    """Index of the cheapest gate; ties -> largest Tier A coverage (see TIE_BREAK_NOTE).

    The grid is coverage-ascending, so the LAST minimum is the highest-coverage one.
    """
    winners = np.flatnonzero(cost == cost.min())
    return int(winners[-1])


# ---------------------------------------------------------------------------
# The joint (tau_A, tau_B) sweep — two-gate families only
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JointFit:
    """The joint argmin plus everything needed to report the second dimension."""

    tau_b: float
    coverage_b_marginal: float   # fraction of ALL rows clearing the B gate
    policy: PolicyData           # the arm frozen at tau_b*
    rows: dict[str, np.ndarray]  # the tau_A sweep at tau_b*
    j_star: int                  # index of the joint argmin within `rows`
    grid: list[dict]             # one entry per tau_B candidate (best tau_A at that tau_B)
    j_grid_star: int             # index of the winning entry within `grid`


def tau_b_candidates(p_max_b) -> np.ndarray:
    """Descending distinct Tier B gates, led by +inf ("no row clears the Tier B gate").

    +inf is a real operating point — it is the pure `a_to_c` cascade with a Tier B forward
    pass paid for and thrown away — so it leads the grid rather than being excluded, for
    the same reason `build_grid` keeps k = 0.
    """
    values = np.unique(np.asarray(p_max_b, dtype=np.float64))
    if not np.all(np.isfinite(values)):
        raise ValueError(
            "Tier B p_max contains non-finite values; a confidence gate cannot be swept "
            "over an unorderable signal"
        )
    return np.concatenate(([np.inf], values[::-1]))


def joint_sweep(make_policy, tau_b_values, *, c_misroute: float, c_human: float) -> JointFit:
    """Exhaustive joint argmin over (tau_A, tau_B). See JOINT_SWEEP_NOTE.

    `make_policy(tau_b)` rebuilds the escalation arm at that Tier B gate; the inner tau_A
    sweep is the SAME prefix-sum machinery every one-gate family uses, so the two-gate
    family is priced by identical code on identical data. Ties are broken lexicographically
    (cost, then Tier A coverage, then Tier B coverage) per JOINT_TIE_BREAK_NOTE.
    """
    best = None
    grid: list[dict] = []
    for i, tau_b in enumerate(np.asarray(tau_b_values, dtype=np.float64)):
        policy = make_policy(float(tau_b))
        rows = sweep(build_grid(policy), c_misroute=c_misroute, c_human=c_human)
        j = argmin_index(rows["cost_per_1k"])
        coverage_b = float(np.count_nonzero(policy.arm.b_answered)) / len(policy)
        key = (float(rows["cost_per_1k"][j]), -float(rows["coverage_a"][j]), -coverage_b)
        grid.append({
            "tau_b": _tau_json(tau_b),
            "coverage_b_marginal": _round(coverage_b),
            "best_tau_a": _tau_json(rows["tau"][j]),
            "coverage_a": _round(float(rows["coverage_a"][j])),
            "human_rate": _round(float(rows["human_rate"][j])),
            "cost_per_1k": _round(float(rows["cost_per_1k"][j])),
        })
        if best is None or key < best[0]:
            best = (key, i, float(tau_b), coverage_b, policy, rows, j)
    _, i_star, tau_b_star, coverage_b_star, policy_star, rows_star, j_star = best
    return JointFit(
        tau_b=tau_b_star, coverage_b_marginal=coverage_b_star, policy=policy_star,
        rows=rows_star, j_star=j_star, grid=grid, j_grid_star=i_star,
    )


# ---------------------------------------------------------------------------
# Loading CAL inputs (offline; TEST-* is never opened here)
# ---------------------------------------------------------------------------

# Run selection by config stem lives in `predictions` (shared with the router simulator);
# kept under the module-local name this module and its tests already use.
_records_by_config = predictions.records_by_config


def load_artifact_checked(record: dict, preds_dir=DEFAULT_PREDS_DIR):
    """CAL-only wrapper around the repo's full artifact gate.

    Threshold fitting must never see a TEST-* slice, so the whitelist is enforced by the
    loader rather than by convention: there is no code path in this module that can open
    one. The gate itself (provenance + `predictions.verify_artifact`) lives in
    `cost_model.load_artifact_verified` and is shared with the router simulator.
    """
    return cost_model.load_artifact_verified(record, preds_dir, allowed_splits={"cal"})


def _artifact_block(record: dict, art, config_name: str) -> dict:
    return {
        "run_id": record["run_id"],
        "config_name": config_name,
        "config_sha256": art.provenance.get("config_sha256", ""),
        "split": art.provenance.get("split", ""),
        "split_sha256": art.provenance.get("split_sha256", ""),
        "n_examples": len(art),
    }


def restrict_to_ids(art, ids):
    """Positional index of `ids` inside `art`, requiring an EXACT id-set match.

    The paired policy joins three sources per row (Tier A probabilities, Tier C labels,
    Tier C receipts). If the Tier A restriction were merely a subset — or silently
    reordered — every downstream row would mix one complaint's confidence with another's
    label while all aggregate metrics stayed plausible. So: every requested id must exist
    exactly once, and the result is returned in the REQUESTED order, not the artifact's.
    """
    ids = np.asarray(ids, dtype=np.int64)
    art_ids = np.asarray(art.complaint_id, dtype=np.int64)
    position = {int(cid): i for i, cid in enumerate(art_ids)}
    if len(position) != len(art_ids):
        raise ValueError("artifact has duplicate complaint_id values; cannot restrict")
    missing = [int(cid) for cid in ids if int(cid) not in position]
    if missing:
        raise ValueError(
            f"{len(missing)} paired id(s) absent from the Tier A artifact: "
            f"{missing[:cost_model.MAX_OFFENDERS_SHOWN]}"
            f"{' ...' if len(missing) > cost_model.MAX_OFFENDERS_SHOWN else ''}; the Tier A "
            "subset must match the Tier C artifact's ids exactly"
        )
    return np.asarray([position[int(cid)] for cid in ids], dtype=np.int64)


def build_a_to_human(art_a, record_a, config_name: str, *,
                     dataset: str = DATASET_FULL_CAL, index=None) -> PolicyData:
    """`a_to_human` policy over a Tier A artifact (optionally restricted to `index`)."""
    idx = slice(None) if index is None else index
    ids = art_a.complaint_id[idx]
    correct_a = (art_a.y_true[idx] == art_a.y_pred[idx])
    return PolicyData(
        family=FAMILY_A_TO_HUMAN,
        dataset=dataset,
        ids=np.asarray(ids, dtype=np.int64),
        p_max=np.asarray(art_a.p_max[idx], dtype=np.float64),
        correct_a=np.asarray(correct_a, dtype=bool),
        arm=human_arm(len(ids)),
        inputs={"tier_a": _artifact_block(record_a, art_a, config_name)},
    )


def build_a_to_c(art_a, record_a, config_name_a, art_c, record_c, config_name_c,
                 *, api_cost_usd, parse_failed, cost_sum_check: dict,
                 receipts_sha256: str = "") -> PolicyData:
    """`a_to_c_parsefail_human` over the paired subset (Tier C artifact's ids, its order)."""
    index = restrict_to_ids(art_a, art_c.complaint_id)
    y_true_a = art_a.y_true[index]
    if not np.array_equal(y_true_a, art_c.y_true):
        raise ValueError(
            "Tier A and Tier C artifacts disagree on y_true for the paired ids; the two "
            "artifacts are not describing the same rows"
        )
    return PolicyData(
        family=FAMILY_A_TO_C,
        dataset=DATASET_PAIRED,
        ids=np.asarray(art_c.complaint_id, dtype=np.int64),
        p_max=np.asarray(art_a.p_max[index], dtype=np.float64),
        correct_a=np.asarray(y_true_a == art_a.y_pred[index], dtype=bool),
        arm=tier_c_arm(art_c.y_true, art_c.y_pred, api_cost_usd, parse_failed),
        inputs={
            "tier_a": _artifact_block(record_a, art_a, config_name_a),
            "tier_c": {
                **_artifact_block(record_c, art_c, config_name_c),
                "raw_log_path": (record_c.get("extra") or {}).get("raw_log_path", ""),
                "receipts_sha256": receipts_sha256,
                "prompt_bundle_sha256": art_c.provenance.get("prompt_bundle_sha256", ""),
                "model_slug": (record_c.get("extra") or {}).get("model_slug", ""),
                "logged_cost_usd": record_c.get("cost_usd"),
                "cost_sum_check": cost_sum_check,
                "n_parse_failed": int(np.count_nonzero(parse_failed)),
            },
        },
    )


def _aligned_tier_b(art_a, ids, art_b):
    """Tier B rows for `ids`, in that order, with the ground truth cross-checked.

    Same contract as the Tier A/Tier C join: an exact id match (no subsets, no reordering)
    and identical `y_true`, because a cascade mixes one complaint's gate decision with
    another's label if either fails while every aggregate metric stays plausible.
    """
    index_b = restrict_to_ids(art_b, ids)
    y_true_b = art_b.y_true[index_b]
    y_true_a = art_a.y_true[restrict_to_ids(art_a, ids)]
    if not np.array_equal(y_true_a, y_true_b):
        raise ValueError(
            "Tier A and Tier B CAL artifacts disagree on y_true for the joined ids; the "
            "two artifacts are not describing the same rows"
        )
    return index_b


def build_a_to_b(art_a, record_a, config_name_a, art_b, record_b, config_name_b, *,
                 b_per_example_usd: float, dataset: str = DATASET_FULL_CAL,
                 index=None) -> PolicyData:
    """`a_to_b`: Tier A answers above tau, otherwise Tier B answers TERMINALLY.

    One gate, by construction — the family is the A gate plus a fixed arm, exactly like
    `a_to_human` and `a_to_c_parsefail_human`. The two-gate cascade is `a_to_b_to_c`.
    """
    idx = slice(None) if index is None else index
    ids = np.asarray(art_a.complaint_id[idx], dtype=np.int64)
    index_b = _aligned_tier_b(art_a, ids, art_b)
    return PolicyData(
        family=FAMILY_A_TO_B,
        dataset=dataset,
        ids=ids,
        p_max=np.asarray(art_a.p_max[idx], dtype=np.float64),
        correct_a=np.asarray(art_a.y_true[idx] == art_a.y_pred[idx], dtype=bool),
        arm=tier_b_arm(art_b.y_true[index_b], art_b.y_pred[index_b], b_per_example_usd),
        inputs={
            "tier_a": _artifact_block(record_a, art_a, config_name_a),
            "tier_b": {
                **_artifact_block(record_b, art_b, config_name_b),
                "per_example_usd": float(b_per_example_usd),
                "evidence_class": "estimated (amortized compute; see the cost config note)",
            },
        },
    )


@dataclass(frozen=True)
class ABCFactory:
    """An `a_to_b_to_c` policy at ANY tau_B, with every id join done exactly once.

    The joint fit re-materializes the arm ~1,500 times per price cell and ~55,000 times
    across the sensitivity grid. Re-running the builder each time would redo the id joins
    (a dict over the whole CAL artifact) on every one of them, turning a 0.2 s fit into
    minutes; the template holds everything tau_B cannot change, and `at()` rebuilds only
    the arm.
    """

    template: PolicyData
    b_y_true: np.ndarray
    b_y_pred: np.ndarray
    b_p_max: np.ndarray
    c_y_pred: np.ndarray
    c_api_cost_usd: np.ndarray
    parse_failed: np.ndarray
    b_per_example_usd: float

    def at(self, tau_b: float) -> PolicyData:
        return replace(self.template, arm=tier_b_then_c_arm(
            self.b_y_true, self.b_y_pred, self.b_p_max, tau_b, self.b_per_example_usd,
            self.c_y_pred, self.c_api_cost_usd, self.parse_failed))

    def tau_b_candidates(self) -> np.ndarray:
        return tau_b_candidates(self.b_p_max)


def build_a_to_b_to_c(art_a, record_a, config_name_a, art_b, record_b, config_name_b,
                      art_c, record_c, config_name_c, *, tau_b: float,
                      b_per_example_usd: float, api_cost_usd, parse_failed,
                      cost_sum_check: dict, receipts_sha256: str = "") -> PolicyData:
    """`a_to_b_to_c` at a FIXED tau_B, over the paired subset (Tier C's ids, its order)."""
    return build_a_to_b_to_c_factory(
        art_a, record_a, config_name_a, art_b, record_b, config_name_b,
        art_c, record_c, config_name_c, b_per_example_usd=b_per_example_usd,
        api_cost_usd=api_cost_usd, parse_failed=parse_failed,
        cost_sum_check=cost_sum_check, receipts_sha256=receipts_sha256).at(tau_b)


def build_a_to_b_to_c_factory(art_a, record_a, config_name_a, art_b, record_b,
                              config_name_b, art_c, record_c, config_name_c, *,
                              b_per_example_usd: float, api_cost_usd, parse_failed,
                              cost_sum_check: dict,
                              receipts_sha256: str = "") -> ABCFactory:
    """Align the three tiers on the paired ids once; return a tau_B-parameterized family.

    `tau_b` is not an argument here because it is not an input to the family — it is what
    the joint fit chooses (see `joint_sweep` and JOINT_SWEEP_NOTE).
    """
    ids = np.asarray(art_c.complaint_id, dtype=np.int64)
    index_a = restrict_to_ids(art_a, ids)
    y_true_a = art_a.y_true[index_a]
    if not np.array_equal(y_true_a, art_c.y_true):
        raise ValueError(
            "Tier A and Tier C artifacts disagree on y_true for the paired ids; the two "
            "artifacts are not describing the same rows"
        )
    index_b = _aligned_tier_b(art_a, ids, art_b)
    template = PolicyData(
        family=FAMILY_A_TO_B_TO_C,
        dataset=DATASET_PAIRED,
        ids=ids,
        p_max=np.asarray(art_a.p_max[index_a], dtype=np.float64),
        correct_a=np.asarray(y_true_a == art_a.y_pred[index_a], dtype=bool),
        # Placeholder arm, replaced by `ABCFactory.at`; never swept as-is.
        arm=human_arm(len(ids)),
        inputs={
            "tier_a": _artifact_block(record_a, art_a, config_name_a),
            "tier_b": {
                **_artifact_block(record_b, art_b, config_name_b),
                "per_example_usd": float(b_per_example_usd),
                "evidence_class": "estimated (amortized compute; see the cost config note)",
            },
            "tier_c": {
                **_artifact_block(record_c, art_c, config_name_c),
                "raw_log_path": (record_c.get("extra") or {}).get("raw_log_path", ""),
                "receipts_sha256": receipts_sha256,
                "prompt_bundle_sha256": art_c.provenance.get("prompt_bundle_sha256", ""),
                "model_slug": (record_c.get("extra") or {}).get("model_slug", ""),
                "logged_cost_usd": record_c.get("cost_usd"),
                "cost_sum_check": cost_sum_check,
                "n_parse_failed": int(np.count_nonzero(parse_failed)),
            },
        },
    )
    return ABCFactory(
        template=template,
        b_y_true=np.asarray(art_b.y_true[index_b], dtype=object),
        b_y_pred=np.asarray(art_b.y_pred[index_b], dtype=object),
        b_p_max=np.asarray(art_b.p_max[index_b], dtype=np.float64),
        c_y_pred=np.asarray(art_c.y_pred, dtype=object),
        c_api_cost_usd=np.asarray(api_cost_usd, dtype=np.float64),
        parse_failed=np.asarray(parse_failed, dtype=bool),
        b_per_example_usd=float(b_per_example_usd),
    )


def load_tier_c_arm_inputs(art_c, record_c):
    """Measured per-row cost + parse_failed for the Tier C CAL run, through the cost gates.

    Returns (api_cost_usd, parse_failed, cost_sum_check, receipts_sha256).

    Reuses `cost_model`'s receipt machinery unchanged: per-receipt token/slug/price
    re-derivation, id-keyed join, duplicate detection, and the run-level cost-sum
    cross-check against the record's logged `cost_usd`. A failure of any of them is a hard
    failure here too — a router optimized against unverified prices is optimizing fiction.
    """
    raw_log_path = (record_c.get("extra") or {}).get("raw_log_path")
    if not raw_log_path:
        raise ValueError(
            f"tier_c run {record_c['run_id'][:8]} has no extra.raw_log_path; there are no "
            "measured per-call costs to escalate against"
        )
    api = cost_model.join_receipt_costs(art_c.complaint_id, raw_log_path, record=record_c)
    parse_failed = cost_model.join_parse_failed(
        art_c.complaint_id, raw_log_path, record=record_c)
    receipts_hash = cost_model.receipts_sha256(raw_log_path)
    check = cost_model.check_cost_sum(api, record_c)
    if not check["ok"]:
        raise ValueError(
            f"cost_sum_check FAILED for tier_c run {record_c['run_id'][:8]}: joined "
            f"${check['joined_cost_usd']:.6f} vs logged ${check['logged_cost_usd']:.6f} "
            f"(|delta| = {check['abs_delta']:.3e} > {check['tol']:.0e})"
        )
    return api, parse_failed, check, receipts_hash


# ---------------------------------------------------------------------------
# Result assembly (deterministic JSON)
# ---------------------------------------------------------------------------

def _tau_json(tau: float):
    """Serialize a threshold EXACTLY; +inf ('no row clears the gate') becomes null.

    Deliberately NOT routed through `_round`. Every other number here is a measurement
    being reported, but a threshold is a decision boundary that gets replayed against
    `p_max >= tau`: rounding it to 10 dp moves rows across an inclusive comparison, so a
    reader replaying the published tau would reproduce a different answered set and a
    different cost than the file claims (observed on 6 of 9 files before this fix — e.g.
    1,278 answers instead of 1,279, $962.00 instead of $960.33). `json.dumps` writes a
    float64 with `repr`, which round-trips exactly, so the published value IS the value
    that was optimized. `n_answered_at_tau_star` is published alongside so any replay can
    assert itself rather than trust this.
    """
    return None if not math.isfinite(float(tau)) else float(tau)


def _grid_row(rows: dict[str, np.ndarray], j: int) -> dict:
    out = {"tau": _tau_json(rows["tau"][j])}
    for key, values in rows.items():
        if key == "tau":
            continue
        value = values[j]
        out[key] = int(value) if key == "n_answered_a" else _round(float(value))
    return out


def _downsample(n_rows: int, max_points: int | None, keep: set[int]) -> list[int]:
    """<=max_points evenly index-spaced rows, always keeping `keep` (endpoints + tau*)."""
    if max_points is None or n_rows <= max_points:
        return list(range(n_rows))
    idx = set(np.linspace(0, n_rows - 1, max_points).round().astype(np.int64).tolist())
    return sorted(idx | keep)


def operating_point(policy: PolicyData, rows: dict[str, np.ndarray], j: int, *,
                    cfg: cost_model.CostConfig, label: str,
                    n_resamples: int = harness.N_RESAMPLES,
                    seed: int = harness.BOOTSTRAP_SEED) -> dict:
    """One CI'd operating point: the grid row plus bootstrap bands at that fixed gate."""
    bands = bootstrap_at(
        policy, rows["tau"][j],
        c_misroute=cfg.c_misroute_usd, c_human=cfg.c_human_usd,
        n_resamples=n_resamples, seed=seed,
    )
    return {
        "label": label,
        **_grid_row(rows, j),
        "expected_cost_per_1k": {k: _round_ci(v) for k, v in bands.items()},
    }


def paired_delta(policy: PolicyData, tau_a: float, tau_b: float, *,
                 cfg: cost_model.CostConfig, n_resamples: int = harness.N_RESAMPLES,
                 seed: int = harness.BOOTSTRAP_SEED) -> dict:
    """Bootstrap CI on cost(tau_a) - cost(tau_b), PAIRED on the same rows.

    Two operating points of the same policy are evaluated on identical examples, so their
    marginal CIs overlap heavily even when one dominates the other everywhere: the shared
    sampling noise cancels in the difference and only shows up in the marginals. CLAUDE.md
    requires comparison claims to rest on a paired CI excluding zero, so every "tau* beats
    X" statement must cite this block, never the two marginal bands.

    The difference is taken per example first, then resampled with the frozen shared-index
    contract — the same reason the cost components decompose exactly.
    """
    return _delta_from_costs(
        _per_example_cost_at(policy, tau_a, cfg),
        _per_example_cost_at(policy, tau_b, cfg),
        n_resamples=n_resamples, seed=seed,
    )


def _per_example_cost_at(policy: PolicyData, tau: float,
                         cfg: cost_model.CostConfig) -> np.ndarray:
    return cost_model.per_example_cost(
        *materialize(policy, tau), c_misroute=cfg.c_misroute_usd, c_human=cfg.c_human_usd)


def _delta_from_costs(per_a: np.ndarray, per_b: np.ndarray, *, n_resamples: int,
                      seed: int, favors=("tau_star", "reference")) -> dict:
    diff = per_a - per_b
    reps = cost_model.resample_means(
        {"delta": diff}, scale=cost_model.PER_N_COMPLAINTS,
        n_resamples=n_resamples, seed=seed)["delta"]
    lo, hi = np.percentile(reps, [harness.CI_LOWER_PCT, harness.CI_UPPER_PCT])
    point = float(diff.mean()) * cost_model.PER_N_COMPLAINTS
    return {
        "delta_cost_per_1k": _round_ci({"point": point, "ci_lo": float(lo),
                                        "ci_hi": float(hi)}),
        "excludes_zero": bool((lo > 0.0) or (hi < 0.0)),
        "favors": favors[0] if point < 0 else (favors[1] if point > 0 else "neither"),
    }


def paired_delta_across_policies(policy_a: PolicyData, tau_a: float, policy_b: PolicyData,
                                 tau_b: float, *, cfg: cost_model.CostConfig,
                                 n_resamples: int = harness.N_RESAMPLES,
                                 seed: int = harness.BOOTSTRAP_SEED) -> dict:
    """Paired CI on cost(policy_a @ tau_a) - cost(policy_b @ tau_b), same rows.

    This is the test behind the headline "escalating to Tier C beats escalating to a
    human" claim. It requires the two policies to be defined on the SAME ids in the SAME
    order — otherwise the difference would pair unrelated complaints — and refuses
    otherwise rather than comparing marginals and hoping.
    """
    if not np.array_equal(policy_a.ids, policy_b.ids):
        raise ValueError(
            f"cannot pair {policy_a.family}/{policy_a.dataset} with "
            f"{policy_b.family}/{policy_b.dataset}: their ids differ, so a per-example "
            "difference would compare different complaints"
        )
    return _delta_from_costs(
        _per_example_cost_at(policy_a, tau_a, cfg),
        _per_example_cost_at(policy_b, tau_b, cfg),
        n_resamples=n_resamples, seed=seed,
        favors=(policy_a.family, policy_b.family),
    )


def sensitivity_grid(policy: PolicyData, grid: Grid, *, cfg: cost_model.CostConfig,
                     c_misroute_values=SENSITIVITY_C_MISROUTE,
                     c_human_values=SENSITIVITY_C_HUMAN) -> list[dict]:
    """tau* and its mix for every (c_misroute, c_human) cell. Point estimates only."""
    cells = []
    for c_misroute in c_misroute_values:
        for c_human in c_human_values:
            rows = sweep(grid, c_misroute=c_misroute, c_human=c_human)
            j = argmin_index(rows["cost_per_1k"])
            cells.append({
                "c_misroute_usd": float(c_misroute),
                "c_human_usd": float(c_human),
                "is_cost_config_default": bool(
                    c_misroute == cfg.c_misroute_usd and c_human == cfg.c_human_usd),
                "tau_star": _tau_json(rows["tau"][j]),
                "coverage_a": _round(float(rows["coverage_a"][j])),
                "escalation_rate": _round(float(rows["escalation_rate"][j])),
                "human_rate": _round(float(rows["human_rate"][j])),
                "cost_per_1k": _round(float(rows["cost_per_1k"][j])),
                "cost_per_1k_a_only": _round(float(rows["cost_per_1k"][-1])),
                "cost_per_1k_no_gate": _round(float(rows["cost_per_1k"][0])),
            })
    return cells


def joint_sensitivity_grid(factory: ABCFactory, tau_b_values, *,
                           cfg: cost_model.CostConfig,
                           c_misroute_values=SENSITIVITY_C_MISROUTE,
                           c_human_values=SENSITIVITY_C_HUMAN) -> list[dict]:
    """The 36-cell grid for a two-gate family: the JOINT fit is re-run in every cell.

    Holding tau_B at the value chosen under the default prices and varying only tau_A would
    answer a question nobody asked ("what if the operator re-tuned one gate but not the
    other"); the point of the grid is where the whole operating point moves when the two
    estimated dollar parameters move.
    """
    cells = []
    for c_misroute in c_misroute_values:
        for c_human in c_human_values:
            fit = joint_sweep(factory.at, tau_b_values, c_misroute=c_misroute,
                              c_human=c_human)
            rows, j = fit.rows, fit.j_star
            cells.append({
                "c_misroute_usd": float(c_misroute),
                "c_human_usd": float(c_human),
                "is_cost_config_default": bool(
                    c_misroute == cfg.c_misroute_usd and c_human == cfg.c_human_usd),
                "tau_star": _tau_json(rows["tau"][j]),
                "tau_b_star": _tau_json(fit.tau_b),
                "coverage_a": _round(float(rows["coverage_a"][j])),
                "coverage_b_marginal": _round(fit.coverage_b_marginal),
                "escalation_rate": _round(float(rows["escalation_rate"][j])),
                "human_rate": _round(float(rows["human_rate"][j])),
                "cost_per_1k": _round(float(rows["cost_per_1k"][j])),
                "cost_per_1k_a_only": _round(float(rows["cost_per_1k"][-1])),
                "cost_per_1k_no_gate": _round(float(rows["cost_per_1k"][0])),
            })
    return cells


def build_policy_result(policy: PolicyData, cfg: cost_model.CostConfig, *,
                        is_primary: bool, max_points: int | None = DEFAULT_MAX_GRID_POINTS,
                        n_resamples: int = harness.N_RESAMPLES,
                        seed: int = harness.BOOTSTRAP_SEED,
                        derivation: str = DERIVATION_V1,
                        rows: dict[str, np.ndarray] | None = None,
                        j_star: int | None = None,
                        tier_b_gate: dict | None = None,
                        sensitivity_cells: list[dict] | None = None) -> dict:
    """Full deterministic result object for one (family, dataset, Tier A rung).

    `rows`/`j_star` are threaded only for the two-gate family, whose tau_A sweep and argmin
    were already computed against the winning tau_B by `joint_sweep`; recomputing them here
    would re-derive the same numbers from a re-materialized arm, i.e. give the joint fit a
    second chance to disagree with itself. Every other family leaves them None and this
    function does the sweep, exactly as before.
    """
    grid = build_grid(policy)
    if rows is None:
        rows = sweep(grid, c_misroute=cfg.c_misroute_usd, c_human=cfg.c_human_usd)
    if j_star is None:
        j_star = argmin_index(rows["cost_per_1k"])
    n_rows = len(rows["tau"])
    j_a_only = n_rows - 1     # every row answered by Tier A
    j_no_gate = 0             # no row answered by Tier A

    no_gate_label = NO_GATE_LABEL[policy.family]
    points = {
        "tau_star": operating_point(policy, rows, j_star, cfg=cfg, label="tau_star",
                                    n_resamples=n_resamples, seed=seed),
        "a_only": operating_point(policy, rows, j_a_only, cfg=cfg, label="a_only",
                                  n_resamples=n_resamples, seed=seed),
        no_gate_label: operating_point(policy, rows, j_no_gate, cfg=cfg,
                                       label=no_gate_label, n_resamples=n_resamples,
                                       seed=seed),
    }
    deltas = {
        f"tau_star_minus_{label}": paired_delta(
            policy, rows["tau"][j_star], rows["tau"][j],
            cfg=cfg, n_resamples=n_resamples, seed=seed)
        for label, j in (("a_only", j_a_only), (no_gate_label, j_no_gate))
    }
    keep = {0, n_rows - 1, j_star}
    kept = _downsample(n_rows, max_points, keep)

    # v1 files predate the derivation fields and are never rewritten, so they are emitted
    # exactly as before; only v2 carries the extra keys (a test asserts v1 byte-identity).
    version_block = {} if derivation == DERIVATION_V1 else {
        "derivation": derivation,
        "is_primary_v2": bool(is_primary),
        "derivation_note": ISOCAL_IN_SAMPLE_NOTE,
    }
    # Notes are attached only where they apply, so the families that predate Tier B keep
    # emitting exactly the note set they were published with.
    notes = {
        "amendment": AMENDMENT_NOTE,
        "selection_optimism": SELECTION_OPTIMISM_NOTE,
        "p_max_space": PMAX_SPACE_NOTE,
        "tie_break": TIE_BREAK_NOTE,
        "human_correct": HUMAN_CORRECT_NOTE,
        "human_assumption": cost_model.HUMAN_ASSUMPTION,
        "comparisons": PAIRED_COMPARISON_NOTE,
    }
    if policy.family in (FAMILY_A_TO_B, FAMILY_A_TO_B_TO_C):
        notes["tier_b_arm"] = TIER_B_ARM_NOTE
        notes["tier_b_cal_in_sample"] = TIER_B_IN_SAMPLE_NOTE
    if policy.family == FAMILY_A_TO_B_TO_C:
        notes["joint_sweep"] = JOINT_SWEEP_NOTE
        notes["joint_tie_break"] = JOINT_TIE_BREAK_NOTE
    return {
        "schema_version": SCHEMA_VERSION,
        **version_block,
        **({} if tier_b_gate is None else {"tier_b_gate": tier_b_gate}),
        "policy_family": policy.family,
        "dataset": policy.dataset,
        "is_primary": bool(is_primary),
        "n_examples": len(policy),
        "escalation_arm": policy.arm.name,
        "inputs": policy.inputs,
        "cost_config": cost_model.config_block(cfg),
        "tau_star": _tau_json(rows["tau"][j_star]),
        "n_answered_at_tau_star": int(rows["n_answered_a"][j_star]),
        "target_coverage_a": _round(float(rows["coverage_a"][j_star])),
        "operating_points": points,
        "paired_deltas": deltas,
        "grid": {
            "n_thresholds_full": n_rows,
            "n_thresholds_written": len(kept),
            "max_points": max_points,
            "rows": [_grid_row(rows, j) for j in kept],
        },
        "sensitivity": {
            "c_misroute_usd_values": [float(v) for v in SENSITIVITY_C_MISROUTE],
            "c_human_usd_values": [float(v) for v in SENSITIVITY_C_HUMAN],
            "evidence_class_note": SENSITIVITY_EVIDENCE_NOTE,
            "cells": (sensitivity_grid(policy, grid, cfg=cfg)
                      if sensitivity_cells is None else sensitivity_cells),
        },
        "bootstrap": {
            "n_resamples": int(n_resamples),
            "seed": int(seed),
            "method": (
                f"percentile [{harness.CI_LOWER_PCT}, {harness.CI_UPPER_PCT}] over "
                "resampled example indices (one integers(0, n, size=n) draw per replicate, "
                "shared across cost components); computed at operating points only"
            ),
        },
        "notes": notes,
    }


def build_joint_policy_result(factory: ABCFactory, cfg: cost_model.CostConfig, *,
                              is_primary: bool,
                              max_points: int | None = DEFAULT_MAX_GRID_POINTS,
                              n_resamples: int = harness.N_RESAMPLES,
                              seed: int = harness.BOOTSTRAP_SEED,
                              derivation: str = DERIVATION_V1) -> dict:
    """Fit (tau_A, tau_B) jointly, then emit the standard result object at tau_B*.

    Freezing tau_B* and reporting the ordinary tau_A sweep against it keeps the file
    readable by everything that already reads a threshold file (including the router's
    replay gate); the second dimension is not hidden by that choice — `tier_b_gate` carries
    tau_B*, its coverage, the realized four-way routing mix at the joint operating point,
    and the tau_B grid the argmin was taken over.
    """
    tau_b_values = factory.tau_b_candidates()
    fit = joint_sweep(factory.at, tau_b_values,
                      c_misroute=cfg.c_misroute_usd, c_human=cfg.c_human_usd)
    rows, j = fit.rows, fit.j_star
    policy = fit.policy

    # The realized four-way mix at (tau_A*, tau_B*): who actually answered each complaint.
    answered_a = policy.p_max >= rows["tau"][j]
    answered_b = (~answered_a) & policy.arm.b_answered
    to_c = (~answered_a) & (~policy.arm.b_answered)
    n = len(policy)
    kept = _downsample(len(fit.grid), max_points, {0, len(fit.grid) - 1, fit.j_grid_star})

    tier_b_gate = {
        "tau_b_star": _tau_json(fit.tau_b),
        "n_answered_b_at_tau_star": int(answered_b.sum()),
        "coverage_b_marginal": _round(fit.coverage_b_marginal),
        "fit": "joint_2d_argmin",
        "note": JOINT_SWEEP_NOTE,
        "tie_break": JOINT_TIE_BREAK_NOTE,
        "routing_mix_at_joint_operating_point": {
            "answered_tier_a": int(answered_a.sum()),
            "answered_tier_b": int(answered_b.sum()),
            "sent_to_tier_c": int(to_c.sum()),
            "to_human_parse_failed": int((to_c & policy.arm.to_human).sum()),
            "rate_tier_a": _round(float(answered_a.sum()) / n),
            "rate_tier_b": _round(float(answered_b.sum()) / n),
            "rate_tier_c": _round(float(to_c.sum()) / n),
        },
        "grid": {
            "n_tau_b_full": len(fit.grid),
            "n_tau_b_written": len(kept),
            "argmin_index": fit.j_grid_star,
            "rows": [fit.grid[i] for i in kept],
        },
    }
    return build_policy_result(
        policy, cfg, is_primary=is_primary, max_points=max_points,
        n_resamples=n_resamples, seed=seed, derivation=derivation,
        rows=rows, j_star=j, tier_b_gate=tier_b_gate,
        sensitivity_cells=joint_sensitivity_grid(factory, tau_b_values, cfg=cfg),
    )


def result_filename(obj: dict, cfg: cost_model.CostConfig) -> str:
    run8 = obj["inputs"]["tier_a"]["run_id"][:8]
    return (f"{obj['policy_family']}__{obj['dataset']}__{run8}"
            f"__cost-{cfg.sha256[:8]}.json")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def build_all(cfg: cost_model.CostConfig, *, preds_dir=DEFAULT_PREDS_DIR,
              results_path=DEFAULT_RESULTS_PATH,
              tier_a_configs=None,
              tier_c_config=TIER_C_CAL_CONFIG,
              tier_b_config=TIER_B_CAL_CONFIG,
              max_points: int | None = DEFAULT_MAX_GRID_POINTS,
              n_resamples: int = harness.N_RESAMPLES,
              seed: int = harness.BOOTSTRAP_SEED,
              derivation: str = DERIVATION_V1,
              primary_config: str | None = None) -> list[dict]:
    """Every (family, dataset, rung) result for one derivation. The caller writes.

    The Tier B families exist iff the cost config prices Tier B (`cost_model.prices_tier_b`)
    — not behind a flag, because an unpriced tier is a hard failure at scoring time, so
    under `cost_model_v1.yaml` an `a_to_b` policy is unscorable rather than merely
    unreported.
    """
    profile = DERIVATIONS[derivation]
    tier_a_configs = tier_a_configs or profile["tier_a_configs"]
    primary_config = primary_config or profile["primary"]
    records = _records_by_config(results_path)
    with_tier_b = cost_model.prices_tier_b(cfg)
    needed = (*tier_a_configs, tier_c_config, *((tier_b_config,) if with_tier_b else ()))
    missing = [name for name in needed if name not in records]
    if missing:
        raise ValueError(f"no run record for config(s) {missing} in {results_path}")

    record_c = records[tier_c_config]
    art_c = load_artifact_checked(record_c, preds_dir)
    api_cost, parse_failed, cost_sum_check, receipts_hash = load_tier_c_arm_inputs(
        art_c, record_c)

    record_b = art_b = None
    b_per_example_usd = 0.0
    if with_tier_b:
        record_b = records[tier_b_config]
        art_b = load_artifact_checked(record_b, preds_dir)
        b_per_example_usd = cost_model.amortized_per_example_usd(
            cfg, cost_model.tier_of_config_name(tier_b_config, cfg))

    results = []
    for config_name in tier_a_configs:
        record_a = records[config_name]
        art_a = load_artifact_checked(record_a, preds_dir)
        is_primary = config_name == primary_config
        index = restrict_to_ids(art_a, art_c.complaint_id)

        policies = [
            build_a_to_human(art_a, record_a, config_name, dataset=DATASET_FULL_CAL),
            build_a_to_human(art_a, record_a, config_name, dataset=DATASET_PAIRED,
                             index=index),
            build_a_to_c(art_a, record_a, config_name, art_c, record_c, tier_c_config,
                         api_cost_usd=api_cost, parse_failed=parse_failed,
                         cost_sum_check=cost_sum_check, receipts_sha256=receipts_hash),
        ]
        built = [
            build_policy_result(policy, cfg, is_primary=is_primary,
                                max_points=max_points, n_resamples=n_resamples, seed=seed,
                                derivation=derivation)
            for policy in policies
        ]
        # The headline cross-family claim ("escalate to Tier C rather than to a human")
        # compares two different policies on the same 1,500 rows, so it needs its own
        # paired CI — the marginal bands of two policies overlap for the same reason two
        # operating points of one policy do. Attached to the a_to_c result, which is the
        # side making the claim.
        human_paired, c_paired = policies[1], policies[2]
        built[2]["cross_family_paired_delta"] = {
            "vs": f"{human_paired.family}@tau_star ({human_paired.dataset})",
            "note": PAIRED_COMPARISON_NOTE,
            **paired_delta_across_policies(
                c_paired, tau_star(c_paired, cfg),
                human_paired, tau_star(human_paired, cfg),
                cfg=cfg, n_resamples=n_resamples, seed=seed,
            ),
        }
        results.extend(built)

        if not with_tier_b:
            continue

        # Tier B families. `a_to_b` is one-gate (Tier B is terminal, per the frontier
        # slot's own definition) and so goes through the ordinary sweep on both datasets;
        # `a_to_b_to_c` is the only two-gate family and is paired-subset-only, because
        # Tier C is observed on 1,500 CAL rows.
        b_policies = [
            build_a_to_b(art_a, record_a, config_name, art_b, record_b, tier_b_config,
                         b_per_example_usd=b_per_example_usd, dataset=DATASET_FULL_CAL),
            build_a_to_b(art_a, record_a, config_name, art_b, record_b, tier_b_config,
                         b_per_example_usd=b_per_example_usd, dataset=DATASET_PAIRED,
                         index=index),
        ]
        b_built = [
            build_policy_result(policy, cfg, is_primary=is_primary,
                                max_points=max_points, n_resamples=n_resamples, seed=seed,
                                derivation=derivation)
            for policy in b_policies
        ]
        # "Escalate to a fine-tune rather than to a human", on the rows where both are
        # defined — the CAL-side preview of the phase question, and the reason `a_to_b`
        # is also fit on the paired subset.
        b_paired = b_policies[1]
        b_built[1]["cross_family_paired_delta"] = {
            "vs": f"{human_paired.family}@tau_star ({human_paired.dataset})",
            "note": PAIRED_COMPARISON_NOTE,
            **paired_delta_across_policies(
                b_paired, tau_star(b_paired, cfg),
                human_paired, tau_star(human_paired, cfg),
                cfg=cfg, n_resamples=n_resamples, seed=seed,
            ),
        }
        results.extend(b_built)

        factory = build_a_to_b_to_c_factory(
            art_a, record_a, config_name, art_b, record_b, tier_b_config,
            art_c, record_c, tier_c_config, b_per_example_usd=b_per_example_usd,
            api_cost_usd=api_cost, parse_failed=parse_failed,
            cost_sum_check=cost_sum_check, receipts_sha256=receipts_hash)
        abc_built = build_joint_policy_result(
            factory, cfg, is_primary=is_primary, max_points=max_points,
            n_resamples=n_resamples, seed=seed, derivation=derivation)
        # "Does inserting the Tier B rung into the LLM cascade pay", paired on the same
        # 1,500 rows, each family at its own tau*.
        abc_policy = factory.at(float(abc_built["tier_b_gate"]["tau_b_star"])
                                if abc_built["tier_b_gate"]["tau_b_star"] is not None
                                else math.inf)
        abc_built["cross_family_paired_delta"] = {
            "vs": f"{c_paired.family}@tau_star ({c_paired.dataset})",
            "note": PAIRED_COMPARISON_NOTE,
            **paired_delta_across_policies(
                abc_policy, float(abc_built["tau_star"]) if abc_built["tau_star"] is not None
                else math.inf,
                c_paired, tau_star(c_paired, cfg),
                cfg=cfg, n_resamples=n_resamples, seed=seed,
            ),
        }
        results.append(abc_built)
    return results


def _cell_key(cell: dict) -> str:
    return f"{cell['c_misroute_usd']:g}|{cell['c_human_usd']:g}"


def build_summary(results: list[dict], cfg: cost_model.CostConfig, *,
                  derivation: str = DERIVATION_V1) -> dict:
    """Cross-family summary the EXPERIMENT_LOG entry cites.

    Family comparison happens ONLY on the paired subset: `a_to_human` on full CAL and
    `a_to_c_parsefail_human` on 1,500 rows are priced on different populations, and
    comparing them directly would be an artifact of which rows each saw.
    """
    primary = [r for r in results if r["is_primary"]]
    by_key = {(r["policy_family"], r["dataset"]): r for r in primary}
    a_to_human_paired = by_key[(FAMILY_A_TO_HUMAN, DATASET_PAIRED)]
    a_to_c_paired = by_key[(FAMILY_A_TO_C, DATASET_PAIRED)]

    # Tier B columns appear only when the Tier B families were built (i.e. when the cost
    # config prices Tier B), so a v1-cost summary keeps exactly the columns it shipped with.
    a_to_b_paired = by_key.get((FAMILY_A_TO_B, DATASET_PAIRED))
    a_to_b_to_c_paired = by_key.get((FAMILY_A_TO_B_TO_C, DATASET_PAIRED))

    cells_h = {_cell_key(c): c for c in a_to_human_paired["sensitivity"]["cells"]}
    cells_c = {_cell_key(c): c for c in a_to_c_paired["sensitivity"]["cells"]}
    cells_b = ({_cell_key(c): c for c in a_to_b_paired["sensitivity"]["cells"]}
               if a_to_b_paired else {})
    cells_abc = ({_cell_key(c): c for c in a_to_b_to_c_paired["sensitivity"]["cells"]}
                 if a_to_b_to_c_paired else {})
    comparison = []
    for key, cell_h in cells_h.items():
        cell_c = cells_c[key]
        costs = {
            FAMILY_A_TO_HUMAN: cell_h["cost_per_1k"],
            FAMILY_A_TO_C: cell_c["cost_per_1k"],
        }
        tier_b_columns: dict = {}
        if cells_b:
            cell_b = cells_b[key]
            costs[FAMILY_A_TO_B] = cell_b["cost_per_1k"]
            tier_b_columns.update({
                "cost_per_1k_a_to_b": cell_b["cost_per_1k"],
                "cost_per_1k_b_only": cell_b["cost_per_1k_no_gate"],
                "coverage_a_a_to_b": cell_b["coverage_a"],
            })
        if cells_abc:
            cell_abc = cells_abc[key]
            costs[FAMILY_A_TO_B_TO_C] = cell_abc["cost_per_1k"]
            tier_b_columns.update({
                "cost_per_1k_a_to_b_to_c": cell_abc["cost_per_1k"],
                "cost_per_1k_b_to_c": cell_abc["cost_per_1k_no_gate"],
                "coverage_a_a_to_b_to_c": cell_abc["coverage_a"],
                "tau_b_star_a_to_b_to_c": cell_abc["tau_b_star"],
                "coverage_b_marginal_a_to_b_to_c": cell_abc["coverage_b_marginal"],
            })
        winner = min(costs, key=lambda k: (costs[k], k))
        a_only = cell_h["cost_per_1k_a_only"]
        comparison.append({
            "c_misroute_usd": cell_h["c_misroute_usd"],
            "c_human_usd": cell_h["c_human_usd"],
            "is_cost_config_default": cell_h["is_cost_config_default"],
            "cost_per_1k_a_to_human": cell_h["cost_per_1k"],
            "cost_per_1k_a_to_c": cell_c["cost_per_1k"],
            **tier_b_columns,
            "cost_per_1k_a_only": a_only,
            "cost_per_1k_all_human": cell_h["cost_per_1k_no_gate"],
            "cost_per_1k_c_only": cell_c["cost_per_1k_no_gate"],
            "winner_family": winner,
            "coverage_a_a_to_human": cell_h["coverage_a"],
            "coverage_a_a_to_c": cell_c["coverage_a"],
            "beats_a_only": bool(min(costs.values()) < a_only),
        })

    profile = DERIVATIONS[derivation]
    version_block = {} if derivation == DERIVATION_V1 else {
        "derivation": derivation,
        "derivation_note": ISOCAL_IN_SAMPLE_NOTE,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "summary",
        **version_block,
        "primary_tier_a_config": profile["primary"],
        "tier_a_configs_swept": list(profile["tier_a_configs"]),
        "tier_c_config": TIER_C_CAL_CONFIG,
        **({} if a_to_b_paired is None else {"tier_b_config": TIER_B_CAL_CONFIG}),
        "cost_config": cost_model.config_block(cfg),
        "results": [
            {
                "policy_family": r["policy_family"],
                "dataset": r["dataset"],
                "is_primary": r["is_primary"],
                "tier_a_run_id": r["inputs"]["tier_a"]["run_id"],
                "tier_a_config": r["inputs"]["tier_a"]["config_name"],
                "n_examples": r["n_examples"],
                "tau_star": r["tau_star"],
                "target_coverage_a": r["target_coverage_a"],
                **({} if "tier_b_gate" not in r else {
                    "tau_b_star": r["tier_b_gate"]["tau_b_star"],
                    "coverage_b_marginal": r["tier_b_gate"]["coverage_b_marginal"],
                    "routing_mix_at_joint_operating_point":
                        r["tier_b_gate"]["routing_mix_at_joint_operating_point"],
                }),
                "operating_points": {
                    label: {
                        "tau": point["tau"],
                        "coverage_a": point["coverage_a"],
                        "escalation_rate": point["escalation_rate"],
                        "human_rate": point["human_rate"],
                        "accuracy_machine": point["accuracy_machine"],
                        "accuracy_system": point["accuracy_system"],
                        "expected_cost_per_1k": point["expected_cost_per_1k"],
                    }
                    for label, point in r["operating_points"].items()
                },
                "paired_deltas": r["paired_deltas"],
                "cross_family_paired_delta": r.get("cross_family_paired_delta"),
                "file": result_filename(r, cfg),
            }
            for r in results
        ],
        "sensitivity_comparison_paired_subset": {
            "note": (
                "Family comparison is made ONLY on the paired 1,500-row subset, where both "
                "families are defined on identical rows. " + SENSITIVITY_EVIDENCE_NOTE
            ),
            "cells": comparison,
        },
        "n_parse_failed_tier_c_cal": a_to_c_paired["inputs"]["tier_c"]["n_parse_failed"],
        "notes": {
            "amendment": AMENDMENT_NOTE,
            "selection_optimism": SELECTION_OPTIMISM_NOTE,
            "p_max_space": PMAX_SPACE_NOTE,
            "tie_break": TIE_BREAK_NOTE,
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m triage_lab.threshold_opt")
    parser.add_argument("--preds-dir", type=Path, default=DEFAULT_PREDS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_THRESHOLDS_DIR)
    parser.add_argument("--cost-config", type=Path, default=cost_model.DEFAULT_COST_CONFIG)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--max-points", type=int, default=DEFAULT_MAX_GRID_POINTS)
    parser.add_argument(
        "--tier-a-config", action="append", default=None, dest="tier_a_configs",
        help="CAL rung config name to sweep (repeatable); default: all three rungs",
    )
    parser.add_argument("--tier-c-config", default=TIER_C_CAL_CONFIG)
    parser.add_argument(
        "--tier-b-config", default=TIER_B_CAL_CONFIG,
        help="the Tier B CAL rung the a_to_b / a_to_b_to_c families escalate to; used "
             "only when the cost config prices Tier B",
    )
    parser.add_argument(
        "--derivation", choices=sorted(DERIVATIONS), default=DERIVATION_V1,
        help="threshold derivation: v1-raw (raw CAL p_max) or v2-isocal (calibrated CAL); "
             "v1 is the default so its committed files regenerate byte-identically",
    )
    args = parser.parse_args(argv)

    cfg = cost_model.load_cost_config(args.cost_config)
    print(f"cost config {cfg.path.name} ({cfg.version}) sha256={cfg.sha256[:12]} "
          f"c_misroute=${cfg.c_misroute_usd:.2f} c_human=${cfg.c_human_usd:.2f}")

    # Build everything before writing anything (batch atomicity, as in cost_model).
    results = build_all(
        cfg, preds_dir=args.preds_dir, results_path=args.results,
        tier_a_configs=tuple(args.tier_a_configs) if args.tier_a_configs else None,
        tier_c_config=args.tier_c_config, tier_b_config=args.tier_b_config,
        max_points=args.max_points, derivation=args.derivation,
    )
    summary = build_summary(results, cfg, derivation=args.derivation)

    written = []
    for obj in results:
        path = cost_model.write_result_json(obj, args.out_dir / result_filename(obj, cfg))
        written.append(path)
        star = obj["operating_points"]["tau_star"]
        total = star["expected_cost_per_1k"]["total"]
        tau = "none" if obj["tau_star"] is None else f"{obj['tau_star']:.6f}"
        print(
            f"[{'PRIMARY' if obj['is_primary'] else '       '}] "
            f"{obj['policy_family']:24s} {obj['dataset']:14s} "
            f"{obj['inputs']['tier_a']['config_name']:28s} n={obj['n_examples']:6d} "
            f"tau*={tau:>10s} cov_A={star['coverage_a']:.4f} "
            f"esc={star['escalation_rate']:.4f} hum={star['human_rate']:.4f} "
            f"cost/1k=${total['point']:8.2f} [{total['ci_lo']:8.2f}, {total['ci_hi']:8.2f}]"
        )
    summary_path = cost_model.write_result_json(
        summary,
        args.out_dir / DERIVATIONS[args.derivation]["summary_name"].format(
            sha8=cfg.sha256[:8]),
    )
    print(f"\nsummary -> {summary_path}  ({len(written)} policy file(s))")
    return 0


if __name__ == "__main__":
    import sys

    from triage_lab.threshold_opt import main as _main

    sys.exit(_main())
