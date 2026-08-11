"""Business cost model (Phase 4 task 2).

Accuracy alone cannot rank a triage policy: a 2-point macro-F1 gain bought with an LLM
call on every complaint may or may not be worth paying for, and the answer is a dollar
question. This module turns a run's per-example artifact (``data/preds/<run_id>.parquet``)
into the money number every Phase-4 frontier claim is denominated in — **expected cost per
1,000 complaints** — under the parameterization of UPGRADE_PLAN §4.2:

    cost = c_misroute * P(error) + c_api * E[tokens] + c_human * P(human)

Per example, with ``correct``, ``api_cost_usd`` and ``to_human`` aligned id-wise:

    cost_i = c_misroute * [not to_human_i and not correct_i]
           + api_cost_usd_i
           + c_human * [to_human_i]

``api_cost_usd`` is **incurred spend**, charged unconditionally: a cascade that pays for a
Tier C call and *then* defers to a human still paid for that call, and the money is gone
whatever the deferral decision was. A policy that abstains *before* any paid call passes
``0.0`` for that example. This keeps the API term a property of what was spent rather than
of what was decided, which is the only form that stays correct under composition.

Three things this module is deliberate about:

- **The parameters live in a hashed file, not in code.** ``configs/cost_model_v1.yaml``
  carries the two ESTIMATED dollar defaults (misroute $6.00, human review $2.50) and the
  per-tier API-cost policy; its sha256 is bound into every output JSON. A reader who
  disagrees with $6.00 can re-derive every number by editing a v2 file, and the hash makes
  the two generations impossible to confuse.
- **The API term is measured, never modeled.** Tier A is charged $0 by an explicit,
  labeled *estimate* (``amortized_zero``: CPU linear inference, no vendor charge; the
  figure must be written out in the cost config, never defaulted in code). Tier C is
  charged from the run's own committed per-call receipts joined on ``complaint_id``
  (``computed_cost_usd`` = real token counts x published per-MTok prices, CLAUDE.md rule
  6). The join is gated at three depths, all hard failures:

  1. *Per receipt* — prompt/completion/total token counts are positive integers with
     ``prompt + completion == total``; the receipt's ``slug`` is the model the run record
     says it called; and ``computed_cost_usd`` reproduces to 1e-12 from those token counts
     times the run's own ``extra.pricing_snapshot`` per-token rates. A receipt whose price
     does not follow from its tokens is not a measurement.
  2. *Per example* — every predicted ``complaint_id`` has exactly one receipt (duplicates
     make the per-example cost depend on file order).
  3. *Per run* — the joined per-example sum equals the run record's logged ``cost_usd`` to
     1e-6, reported as ``cost_sum_check``. This is what proves the vector scored here IS
     that run's spend, not a subset, a superset, or another run's receipts.

  The artifact's embedded provenance (run_id, config_sha256, split, split_sha256, and the
  Tier C prompt bundle hash) is likewise checked against the run record before scoring: a
  cost number stamped with a run's identity must be computed from that run's data.
- **Uncertainty comes from the same bootstrap contract as everything else.** Example
  indices are resampled with ``default_rng(BOOTSTRAP_SEED)`` drawing one
  ``integers(0, n, size=n)`` vector per replicate for ``N_RESAMPLES`` replicates,
  percentile interval — the harness constants ``risk_coverage`` binds, imported rather
  than restated. Total and all three components are computed from the *same* resample, so
  the component point estimates sum to the total exactly and their bands are mutually
  consistent.

Modeling assumption, stated because downstream claims inherit it (it is echoed into every
output JSON as ``human_assumption``):

**A human-queued complaint is resolved correctly, so it pays ``c_human`` and no misroute
charge** (P(error | human) = 0). It still pays whatever API spend it incurred on the way.
The assumption is optimistic toward the human queue: a real analyst error rate would raise
the all-human arm's true cost, so router-vs-human comparisons are conservative in the human
arm's favor.

The only policy implemented here is ``single_tier_all_answered`` — one tier answers every
example — which is the baseline set the router (Phase 4 task 3) has to beat.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from triage_lab import harness, predictions, risk_coverage
from triage_lab.snapshot import sha256_file

REPO_ROOT = harness.REPO_ROOT
DEFAULT_PREDS_DIR = harness.DEFAULT_PREDS_DIR
DEFAULT_RESULTS_PATH = harness.DEFAULT_RESULTS_PATH
DEFAULT_COST_DIR = REPO_ROOT / "results" / "cost_model"
DEFAULT_COST_CONFIG = REPO_ROOT / "configs" / "cost_model_v1.yaml"

# Cost components, in the order they appear in the §4.2 formula. `total` is their sum.
COMPONENT_KEYS: tuple[str, ...] = ("misroute", "api", "human")
TOTAL_KEY = "total"

POLICY_SINGLE_TIER = "single_tier_all_answered"

# API-cost modes. Both amortized modes charge the SAME declared per-example figure through
# the same code path; they differ only in what they claim. `amortized_zero` says "this tier
# costs nothing per complaint and we are charging exactly that"; `amortized_estimate` says
# "this tier costs something we did not receive an invoice for, and here is the figure we
# derived". Charging a nonzero amortized figure under the name `amortized_zero` would be a
# mislabeled measurement, which is the one thing this module exists to prevent.
MODE_AMORTIZED_ZERO = "amortized_zero"
MODE_AMORTIZED_ESTIMATE = "amortized_estimate"
MODE_MEASURED_RECEIPTS = "measured_receipts"
AMORTIZED_MODES = (MODE_AMORTIZED_ZERO, MODE_AMORTIZED_ESTIMATE)

# Tier B is priced per fine-tune SIZE, not as one tier: `tier_of_config_name` matches on
# `<tier>_` prefixes and the run configs are `tier_b1_modernbert_*` / `tier_b2_distilbert_*`,
# which a single `tier_b` key would match neither of. Two keys also let the ~2.5x slower
# ModernBERT carry its own figure instead of borrowing DistilBERT's.
TIER_B_TIERS: tuple[str, ...] = ("tier_b1", "tier_b2")

# Output schema id. Bumped when the meaning of a field changes, so two generations of
# results/cost_model/*.json can never be silently compared. `cost-v1` charges api_cost_usd
# unconditionally (incurred spend); there is no earlier released generation.
SCHEMA_VERSION = "cost-v1"

# Costs are reported per 1,000 complaints (the unit every UPGRADE_PLAN claim uses).
PER_N_COMPLAINTS = 1000

# Tolerance of the Tier C verification gate: joined per-example receipt costs vs the run
# record's logged cost_usd. 1e-6 USD is a ten-thousandth of a cent — far below any real
# per-call price and far above float64 summation noise (observed: ~1e-15 on 5k calls).
COST_SUM_TOL = 1e-6

# Tolerance for re-deriving a single receipt's computed_cost_usd from its token counts and
# the run's pricing snapshot. The receipt writer uses the same tokens x per-token-price
# arithmetic with no rounding, so agreement is exact in practice (observed: 0.0 across all
# 16,500 committed calls); 1e-12 USD leaves room only for float64 association order.
RECEIPT_COST_TOL = 1e-12

# Same rounding and NaN handling as the risk-coverage artifacts, aliased rather than
# restated so the two committed evidence families cannot drift apart.
JSON_ROUND = risk_coverage.JSON_ROUND
_round = risk_coverage._round
_round_ci = risk_coverage._round_ci

MAX_OFFENDERS_SHOWN = predictions.MAX_OFFENDERS_SHOWN

HUMAN_ASSUMPTION = (
    "Examples routed to the human queue pay c_human and are not charged the misroute "
    "cost: human resolution is assumed correct (P(error|human)=0). That is optimistic "
    "toward the human queue — a real analyst error rate would raise the all-human arm's "
    "true cost — so router-vs-all-human comparisons are conservative in the human arm's "
    "favor. API cost is INCURRED spend and is charged unconditionally: a policy that "
    "defers to a human after paying for a model call still pays for that call, and a "
    "policy that defers before any paid call passes api_cost_usd = 0.0 for that example."
)


# ---------------------------------------------------------------------------
# Cost configuration (hashed file -> in-memory params)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CostConfig:
    """Loaded cost parameters plus the identity of the file they came from.

    `sha256` is the hash of the raw file bytes (same hasher as `harness.config_sha256`),
    so comments and formatting are part of the cost model's identity: any edit yields a
    different hash and therefore visibly different cost artifacts.
    """

    path: Path
    sha256: str
    version: str
    c_misroute_usd: float
    c_human_usd: float
    api_cost: dict
    evidence_class: dict
    raw: dict

    @property
    def params(self) -> dict:
        return {"c_misroute_usd": self.c_misroute_usd, "c_human_usd": self.c_human_usd}

    def api_policy(self, tier: str) -> dict:
        """API-cost policy for `tier`. Missing tier is a hard failure, never a silent $0.

        Charging an unpriced tier zero would hand it a free cost advantage — precisely the
        error the cost model exists to prevent — so an unknown tier must be priced in the
        config file (a new version) before its runs can be scored.
        """
        api_cost = self.api_cost or {}
        if tier not in api_cost:
            raise ValueError(
                f"cost config {self.path} (v{self.version}) has no api_cost policy for "
                f"tier {tier!r}; priced tiers are {sorted(api_cost)}. Refusing to score an "
                "unpriced tier at $0 — add it in a new cost-config version instead."
            )
        policy = api_cost[tier]
        if not isinstance(policy, dict):
            raise TypeError(f"cost config {self.path} api_cost.{tier} is not a mapping")
        return policy


def load_cost_config(path=DEFAULT_COST_CONFIG) -> CostConfig:
    """Load + validate the versioned cost-parameter YAML and hash its raw bytes."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise TypeError(f"cost config {path} did not parse to a mapping")
    version = raw.get("version")
    if not version:
        raise ValueError(f"cost config {path} missing required key 'version'")
    if "params" not in raw:
        raise ValueError(f"cost config {path} missing required key 'params'")
    params = raw["params"]
    if not isinstance(params, dict):
        raise TypeError(f"cost config {path} key 'params' is not a mapping")
    for key in ("c_misroute_usd", "c_human_usd"):
        if key not in params:
            raise ValueError(f"cost config {path} missing required param {key!r}")
        value = float(params[key])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"cost config {path} param {key!r} is {params[key]!r}; a price must be a "
                "finite non-negative number (NaN/inf would silently poison every cost)"
            )
    api_cost = raw.get("api_cost")
    if not isinstance(api_cost, dict) or not api_cost:
        raise ValueError(f"cost config {path} missing required mapping 'api_cost'")
    return CostConfig(
        path=path,
        sha256=sha256_file(path),
        version=str(version),
        c_misroute_usd=float(params["c_misroute_usd"]),
        c_human_usd=float(params["c_human_usd"]),
        api_cost=api_cost,
        evidence_class=raw.get("evidence_class") or {},
        raw=raw,
    )


def prices_tier_b(cfg: CostConfig) -> bool:
    """Whether this cost generation prices Tier B, i.e. whether Tier B policies can exist.

    Used as the switch for every Tier B frontier point rather than a command-line flag:
    an unpriced tier is a hard failure at scoring time (`CostConfig.api_policy`), so under
    `cost_model_v1.yaml` a Tier B policy is not merely unreported, it is unscorable. Making
    the policy set a function of the prices keeps the two cost generations from ever
    disagreeing about which points exist.
    """
    api_cost = cfg.api_cost or {}
    return all(tier in api_cost for tier in TIER_B_TIERS)


def amortized_per_example_usd(cfg: CostConfig, tier: str) -> float:
    """The declared per-example charge for an amortized tier, validated.

    No `.get` default: an amortized charge is a modeling DECISION and must be written down
    in the hashed config, not implied by code. A missing field means nobody decided, and
    "nobody decided" must not silently become $0.
    """
    policy_cfg = cfg.api_policy(tier)
    mode = policy_cfg.get("mode")
    if mode not in AMORTIZED_MODES:
        raise ValueError(
            f"cost config {cfg.path} prices tier {tier!r} with mode {mode!r}, not one of "
            f"{list(AMORTIZED_MODES)}; there is no declared per-example figure to charge"
        )
    if "per_example_usd" not in policy_cfg:
        raise ValueError(
            f"cost config {cfg.path} api_cost.{tier} uses mode {mode!r} but declares no "
            "per_example_usd; the amortized charge must be stated explicitly in the "
            "config, never defaulted in code"
        )
    per_example = float(policy_cfg["per_example_usd"])
    if not math.isfinite(per_example) or per_example < 0.0:
        raise ValueError(
            f"cost config {cfg.path} api_cost.{tier}.per_example_usd is "
            f"{policy_cfg['per_example_usd']!r}; must be finite and non-negative"
        )
    if mode == MODE_AMORTIZED_ZERO and per_example != 0.0:
        raise ValueError(
            f"cost config {cfg.path} api_cost.{tier} declares mode 'amortized_zero' with "
            f"per_example_usd {per_example!r}; a nonzero amortized charge must use mode "
            f"{MODE_AMORTIZED_ESTIMATE!r} so the figure is not filed under a name that "
            "denies its existence"
        )
    return per_example


def config_block(cfg: CostConfig) -> dict:
    """The `cost_config` block bound into every output JSON (path is repo-relative)."""
    try:
        rel = str(cfg.path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        rel = str(cfg.path)
    return {
        "path": rel,
        "sha256": cfg.sha256,
        "version": cfg.version,
        "params": cfg.params,
        "evidence_class": dict(cfg.evidence_class),
    }


# ---------------------------------------------------------------------------
# Core scoring (aligned per-example arrays)
# ---------------------------------------------------------------------------

def cost_components(correct, api_cost_usd, to_human, *, c_misroute: float,
                    c_human: float) -> dict[str, np.ndarray]:
    """Per-example USD, split into the three §4.2 terms (arrays, id-aligned to the inputs).

    Human-assigned examples pay `c_human` instead of the misroute charge (human resolution
    assumed correct) but still pay `api_cost_usd`, which is INCURRED spend: money already
    handed to the vendor cannot be refunded by a routing decision taken afterwards. A
    caller that defers before any paid call passes 0.0 for that example. See
    HUMAN_ASSUMPTION and this module's docstring.
    """
    correct = np.asarray(correct, dtype=bool)
    api = np.asarray(api_cost_usd, dtype=np.float64)
    human = np.asarray(to_human, dtype=bool)
    n = len(correct)
    if len(api) != n or len(human) != n:
        raise ValueError(
            f"misaligned inputs: correct={n}, api_cost_usd={len(api)}, to_human={len(human)}"
        )
    if n == 0:
        raise ValueError("cannot score an empty policy: expected cost of no examples is undefined")
    if not np.all(np.isfinite(api)) or np.any(api < 0.0):
        raise ValueError("api_cost_usd must be finite and non-negative")

    answered = ~human
    return {
        "misroute": float(c_misroute) * (answered & ~correct).astype(np.float64),
        "api": api.astype(np.float64, copy=True),
        "human": float(c_human) * human.astype(np.float64),
    }


def per_example_cost(correct, api_cost_usd, to_human, *, c_misroute: float,
                     c_human: float) -> np.ndarray:
    """Total per-example USD (sum of the three components)."""
    comps = cost_components(correct, api_cost_usd, to_human,
                            c_misroute=c_misroute, c_human=c_human)
    return sum(comps[k] for k in COMPONENT_KEYS)


def expected_cost_per_1k(correct, api_cost_usd, to_human, *, c_misroute: float,
                         c_human: float) -> dict[str, float]:
    """Point estimates: mean per-example cost x 1,000, for the total and each component."""
    comps = cost_components(correct, api_cost_usd, to_human,
                            c_misroute=c_misroute, c_human=c_human)
    out = {k: float(comps[k].mean()) * PER_N_COMPLAINTS for k in COMPONENT_KEYS}
    out[TOTAL_KEY] = float(sum(out[k] for k in COMPONENT_KEYS))
    return out


def resample_means(
    arrays: dict[str, np.ndarray],
    *,
    scale: float = 1.0,
    n_resamples: int = harness.N_RESAMPLES,
    seed: int = harness.BOOTSTRAP_SEED,
) -> dict[str, np.ndarray]:
    """Bootstrap replicate means (x`scale`) for several id-aligned arrays, SHARED indices.

    Public because sharing the index vector across the total and its components is a
    correctness property, not an implementation detail: it is what makes the component
    bands decompose the total band instead of being three unrelated intervals. Exposed so
    a test can assert, replicate by replicate, that the components sum to the total, and
    reused for non-cost quantities (e.g. paired accuracy deltas) whose unit is not dollars
    per 1,000 — hence `scale` rather than a hardcoded PER_N_COMPLAINTS.
    """
    lengths = {len(a) for a in arrays.values()}
    if len(lengths) != 1:
        raise ValueError(f"arrays must be id-aligned; got lengths {sorted(lengths)}")
    n = lengths.pop()
    reps = {k: np.empty(n_resamples, dtype=np.float64) for k in arrays}
    rng = np.random.default_rng(seed)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)  # ONE draw per replicate, shared by every array
        for k, arr in arrays.items():
            reps[k][i] = arr[idx].mean() * scale
    return reps


def bootstrap_cost(
    correct,
    api_cost_usd,
    to_human,
    *,
    c_misroute: float,
    c_human: float,
    n_resamples: int = harness.N_RESAMPLES,
    seed: int = harness.BOOTSTRAP_SEED,
) -> dict[str, dict[str, float]]:
    """Percentile CIs for total + component cost/1k, from the harness bootstrap contract.

    One `default_rng(seed).integers(0, n, size=n)` index vector per replicate — the same
    frozen constants (N_RESAMPLES, BOOTSTRAP_SEED, 2.5/97.5) `risk_coverage` uses, imported
    from `harness` rather than restated. Total and components are read off the *same*
    resample, so the component points sum to the total exactly and the bands are mutually
    consistent (they are not independent intervals).
    """
    comps = cost_components(correct, api_cost_usd, to_human,
                            c_misroute=c_misroute, c_human=c_human)
    total = sum(comps[k] for k in COMPONENT_KEYS)
    arrays = {TOTAL_KEY: total, **comps}
    reps = resample_means(arrays, scale=PER_N_COMPLAINTS, n_resamples=n_resamples,
                          seed=seed)

    out: dict[str, dict[str, float]] = {}
    for k, arr in arrays.items():
        lo, hi = np.percentile(reps[k], [harness.CI_LOWER_PCT, harness.CI_UPPER_PCT])
        out[k] = {
            "point": float(arr.mean()) * PER_N_COMPLAINTS,
            "ci_lo": float(lo),
            "ci_hi": float(hi),
        }
    return out


# ---------------------------------------------------------------------------
# Tier C receipts: load + join + verification gate
# ---------------------------------------------------------------------------

def _repo_path(rel_or_abs) -> Path:
    p = Path(rel_or_abs)
    return p if p.is_absolute() else REPO_ROOT / p


def _offenders(ids: list) -> str:
    shown = sorted(set(ids))
    tail = " ..." if len(shown) > MAX_OFFENDERS_SHOWN else ""
    return f"{shown[:MAX_OFFENDERS_SHOWN]}{tail} ({len(shown)} id(s))"


def _positive_int(value) -> bool:
    """True iff `value` is a genuine positive integer token count (bools excluded)."""
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def pricing_rates(record: dict) -> tuple[float, float]:
    """(prompt, completion) USD-per-token from the run's own frozen pricing snapshot.

    The snapshot is what the run captured from OpenRouter /models at call time; using it
    (rather than today's prices) is what makes the re-derivation below a check on the
    receipts instead of a check on the current price list.
    """
    extra = record.get("extra") or {}
    if "pricing_snapshot" not in extra:
        raise ValueError(
            f"run {record.get('run_id', '?')[:8]} has no extra.pricing_snapshot; its "
            "per-call costs cannot be re-derived from token counts, so they cannot be "
            "verified as MEASURED (CLAUDE.md rule 6)"
        )
    pricing = extra["pricing_snapshot"]
    if not isinstance(pricing, dict):
        raise TypeError(
            f"run {record.get('run_id', '?')[:8]} extra.pricing_snapshot is not a mapping"
        )
    rates = []
    for key in ("prompt_usd_per_token", "completion_usd_per_token"):
        if key not in pricing:
            raise ValueError(f"pricing_snapshot missing {key!r} for run "
                             f"{record.get('run_id', '?')[:8]}")
        rate = float(pricing[key])
        if not math.isfinite(rate) or rate < 0.0:
            raise ValueError(f"pricing_snapshot {key!r} is {pricing[key]!r}; must be a "
                             "finite non-negative per-token price")
        rates.append(rate)
    slug = extra.get("model_slug")
    if slug and pricing.get("slug") and pricing["slug"] != slug:
        raise ValueError(
            f"pricing_snapshot is for {pricing['slug']!r} but the run called {slug!r}; "
            "the prices do not belong to this run's model"
        )
    return rates[0], rates[1]


def load_receipt_records(raw_log_path, *, model_slug: str, prompt_rate: float,
                         completion_rate: float,
                         tol: float = RECEIPT_COST_TOL) -> dict[int, dict]:
    """Map complaint_id -> VERIFIED receipt dict from a tier_c receipts jsonl.

    Every receipt must survive four checks before its dollar figure counts as measured:

    - **token counts** are positive integers with ``prompt + completion == total``. A zero
      or missing count means the usage block was never populated, and a total that does not
      decompose means the receipt's own arithmetic is inconsistent.
    - **slug** equals the model the run record says it called, so a receipts directory from
      a different model cannot be costed against this run.
    - **cost re-derivation**: ``prompt*prompt_rate + completion*completion_rate`` reproduces
      ``computed_cost_usd`` to `tol`. This is the check that makes "MEASURED" mean
      something — a price that does not follow from the tokens and the run's own frozen
      pricing snapshot is an assertion, not a measurement.
    - **cost value** is non-null, finite, non-negative.

    Plus the structural rule inherited from `predictions.load_receipts_by_id`: a duplicate
    complaint_id is a hard error, never last-write-wins, because with concurrent calls the
    line order is completion order and the per-example cost would depend on it.

    Every failure is collected and reported by category (offending ids named) rather than
    raising on the first bad line, so one pass tells you the whole story.

    Returns the whole verified line (not just the price) because downstream policies need
    other measured fields off the same receipt — notably `parse_failed`, the router's only
    Tier C -> human signal. Keeping one verified loader means those fields can never be
    read from a receipt that failed the cost gate.
    """
    path = _repo_path(raw_log_path)
    out: dict[int, dict] = {}
    bad: dict[str, list[int]] = {
        "duplicate complaint_id": [],
        "non-positive/non-integer token count": [],
        "prompt+completion != total_tokens": [],
        f"slug is not the run's model ({model_slug!r})": [],
        "null/negative/non-finite computed_cost_usd": [],
        "computed_cost_usd does not follow from tokens x pricing_snapshot": [],
    }
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = int(rec["complaint_id"])
            if cid in out:
                bad["duplicate complaint_id"].append(cid)
                continue
            prompt_tokens = rec.get("prompt_tokens")
            completion_tokens = rec.get("completion_tokens")
            total_tokens = rec.get("total_tokens")
            if not all(_positive_int(v) for v in
                       (prompt_tokens, completion_tokens, total_tokens)):
                bad["non-positive/non-integer token count"].append(cid)
                continue
            if prompt_tokens + completion_tokens != total_tokens:
                bad["prompt+completion != total_tokens"].append(cid)
                continue
            if rec.get("slug") != model_slug:
                bad[f"slug is not the run's model ({model_slug!r})"].append(cid)
                continue
            cost = rec.get("computed_cost_usd")
            if cost is None or not math.isfinite(float(cost)) or float(cost) < 0.0:
                bad["null/negative/non-finite computed_cost_usd"].append(cid)
                continue
            expected = prompt_tokens * prompt_rate + completion_tokens * completion_rate
            if abs(float(cost) - expected) > tol:
                bad["computed_cost_usd does not follow from tokens x pricing_snapshot"
                    ].append(cid)
                continue
            out[cid] = rec

    failures = [(name, ids) for name, ids in bad.items() if ids]
    if failures:
        detail = "; ".join(f"{name}: {_offenders(ids)}" for name, ids in failures)
        raise ValueError(
            f"receipt verification failed for {path} — {detail}. API cost must be MEASURED "
            "per call from real token usage x published prices (CLAUDE.md rule 6); a "
            "receipt that fails any of these checks cannot be scored"
        )
    return out


def receipts_sha256(raw_log_path) -> str:
    """sha256 of the receipts file actually consumed (same hasher as every other artifact).

    The run record names a receipts PATH, not its contents. Binding the file's hash into
    each derived artifact makes the receipts themselves part of the provenance chain, so a
    later edit, truncation or re-download of a `calls.jsonl` is detectable by comparing
    hashes rather than by re-deriving costs and noticing they moved.
    """
    return sha256_file(_repo_path(raw_log_path))


def join_receipts(ids, raw_log_path, *, record: dict) -> list[dict]:
    """Verified receipt per id, joined ON complaint_id. Missing id = hard fail.

    The join key is the complaint id, never position: receipt line order is completion
    order under concurrency, so a positional zip would silently mis-assign every cost.
    """
    prompt_rate, completion_rate = pricing_rates(record)
    model_slug = (record.get("extra") or {}).get("model_slug")
    if not model_slug:
        raise ValueError(
            f"run {record.get('run_id', '?')[:8]} has no extra.model_slug; the receipts "
            "cannot be confirmed to come from the model this run called"
        )
    by_id = load_receipt_records(
        raw_log_path, model_slug=model_slug,
        prompt_rate=prompt_rate, completion_rate=completion_rate,
    )
    missing = [int(cid) for cid in ids if int(cid) not in by_id]
    if missing:
        raise KeyError(
            f"{len(missing)} predicted complaint_id(s) have no receipt in "
            f"{_repo_path(raw_log_path)}: {missing[:MAX_OFFENDERS_SHOWN]}"
            f"{' ...' if len(missing) > MAX_OFFENDERS_SHOWN else ''}; every scored example "
            "must carry exactly one measured per-call cost"
        )
    return [by_id[int(cid)] for cid in ids]


def join_receipt_costs(ids, raw_log_path, *, record: dict) -> np.ndarray:
    """Per-example verified USD for `ids`, joined ON complaint_id."""
    return np.asarray(
        [r["computed_cost_usd"] for r in join_receipts(ids, raw_log_path, record=record)],
        dtype=np.float64,
    )


def join_parse_failed(ids, raw_log_path, *, record: dict) -> np.ndarray:
    """Per-example `parse_failed` flag, from the SAME verified receipts as the costs.

    Parse failure is the router's only Tier C -> human signal (UPGRADE_PLAN §4.2 as
    narrowed by the 2026-08-07 amendment), so it is a routing input and gets the same
    treatment as a price: it must be an explicit boolean on a receipt that already passed
    the cost gate. A missing or non-boolean flag is a hard failure — defaulting it to
    False would silently route a garbled response's fallback label to a customer.
    """
    receipts = join_receipts(ids, raw_log_path, record=record)
    bad = [int(r["complaint_id"]) for r in receipts
           if not isinstance(r.get("parse_failed"), bool)]
    if bad:
        raise ValueError(
            f"receipt(s) with missing/non-boolean parse_failed in "
            f"{_repo_path(raw_log_path)}: {_offenders(bad)}; parse failure is a routing "
            "decision input and cannot be defaulted"
        )
    return np.asarray([bool(r["parse_failed"]) for r in receipts], dtype=bool)


def check_cost_sum(joined: np.ndarray, record: dict, *, tol: float = COST_SUM_TOL) -> dict:
    """Cross-check the joined per-example costs against the run record's logged cost_usd.

    This is the verification gate for Tier C: it proves the per-example vector this module
    scores IS the spend that run reported, not a subset, a superset, or a different run's
    receipts. Returns the reported block; the caller fails the run when `ok` is False.
    """
    logged = record.get("cost_usd")
    if logged is None:
        raise ValueError(
            f"tier_c run {record.get('run_id', '?')[:8]} logged no cost_usd; the joined "
            "receipt costs cannot be cross-checked, so this run is not scorable"
        )
    joined_sum = float(joined.sum())
    delta = abs(joined_sum - float(logged))
    return {
        "logged_cost_usd": float(logged),
        "joined_cost_usd": joined_sum,
        "abs_delta": delta,
        "tol": float(tol),
        "ok": bool(delta <= tol),
    }


# ---------------------------------------------------------------------------
# Single-tier policy builder
# ---------------------------------------------------------------------------

@dataclass
class Policy:
    """One scorable policy: aligned decision arrays plus how their API cost was obtained."""

    run_id: str
    config_name: str
    tier: str
    policy: str
    correct: np.ndarray
    api_cost_usd: np.ndarray
    to_human: np.ndarray
    api_policy: dict
    cost_sum_check: dict | None = None
    receipts_sha256: str = ""
    raw_log_path: str = ""

    def __len__(self) -> int:
        return len(self.correct)


def tier_of_config_name(name: str, cfg: CostConfig) -> str:
    """Tier for a run config name (`tier_a_logreg_test_iid` -> `tier_a`), from the config.

    The priced tiers in the cost config are the vocabulary; an unmatched name is a hard
    failure for the same reason an unpriced tier is (see `CostConfig.api_policy`).
    """
    for tier in sorted(cfg.api_cost or {}, key=len, reverse=True):
        if name.startswith(f"{tier}_"):
            return tier
    raise ValueError(
        f"config name {name!r} does not name a tier priced in {cfg.path} "
        f"(priced tiers: {sorted(cfg.api_cost or {})})"
    )


def build_single_tier_policy(record: dict, art: predictions.PredictionsArtifact,
                             cfg: CostConfig) -> Policy:
    """All-answered single-tier policy for one run: correct, api_cost_usd, to_human.

    `to_human` is all False by construction — this policy set is the "one tier answers
    everything" baseline the router must beat. API cost comes from the tier's policy in
    the cost config: the amortized modes charge the declared per-example figure (an
    explicit estimate), `measured_receipts` joins the run's committed receipts and runs the
    cost-sum verification gate.
    """
    config_name = Path(record.get("config_path", "")).stem
    tier = tier_of_config_name(config_name, cfg)
    policy_cfg = cfg.api_policy(tier)
    mode = policy_cfg.get("mode")

    correct = (art.y_true == art.y_pred)
    n = len(art)
    cost_sum_check = None
    receipts_hash = ""

    if mode in AMORTIZED_MODES:
        api_cost = np.full(n, amortized_per_example_usd(cfg, tier), dtype=np.float64)
    elif mode == MODE_MEASURED_RECEIPTS:
        raw_log_path = (record.get("extra") or {}).get("raw_log_path")
        if not raw_log_path:
            raise ValueError(
                f"run {record.get('run_id', '?')[:8]} ({config_name}) is priced "
                "measured_receipts but its record carries no extra.raw_log_path; "
                "there is nothing measured to join"
            )
        api_cost = join_receipt_costs(art.complaint_id, raw_log_path, record=record)
        receipts_hash = receipts_sha256(raw_log_path)
        cost_sum_check = check_cost_sum(api_cost, record)
        if not cost_sum_check["ok"]:
            raise ValueError(
                f"cost_sum_check FAILED for run {record.get('run_id', '?')[:8]} "
                f"({config_name}): joined receipt costs total "
                f"${cost_sum_check['joined_cost_usd']:.6f} but the run record logged "
                f"${cost_sum_check['logged_cost_usd']:.6f} "
                f"(|delta| = {cost_sum_check['abs_delta']:.3e} > {cost_sum_check['tol']:.0e}); "
                "the per-example costs are not this run's measured spend"
            )
    else:
        raise ValueError(
            f"cost config {cfg.path} gives tier {tier!r} unknown api-cost mode {mode!r}; "
            f"expected one of {[*AMORTIZED_MODES, MODE_MEASURED_RECEIPTS]}"
        )

    return Policy(
        run_id=record["run_id"],
        config_name=config_name,
        tier=tier,
        policy=POLICY_SINGLE_TIER,
        correct=correct,
        api_cost_usd=api_cost,
        to_human=np.zeros(n, dtype=bool),
        api_policy=dict(policy_cfg),
        cost_sum_check=cost_sum_check,
        receipts_sha256=receipts_hash,
        raw_log_path=str((record.get("extra") or {}).get("raw_log_path", "")),
    )


# ---------------------------------------------------------------------------
# Result assembly (deterministic JSON)
# ---------------------------------------------------------------------------

def build_result(policy: Policy, art: predictions.PredictionsArtifact, cfg: CostConfig, *,
                 n_resamples: int = harness.N_RESAMPLES,
                 seed: int = harness.BOOTSTRAP_SEED) -> dict:
    """Assemble the deterministic cost JSON object for one scored policy."""
    bands = bootstrap_cost(
        policy.correct,
        policy.api_cost_usd,
        policy.to_human,
        c_misroute=cfg.c_misroute_usd,
        c_human=cfg.c_human_usd,
        n_resamples=n_resamples,
        seed=seed,
    )
    prov = art.provenance
    # Incurred spend: summed over ALL examples, not just answered ones (a deferred example
    # that already paid for a call still spent the money).
    api_block = {
        "mode": policy.api_policy.get("mode", ""),
        "evidence_class": policy.api_policy.get("evidence_class", ""),
        "charged": "unconditionally (incurred spend, including examples sent to a human)",
        "note": " ".join(str(policy.api_policy.get("note", "")).split()),
        "total_usd": _round(float(policy.api_cost_usd.sum())),
        "mean_per_example_usd": _round(float(policy.api_cost_usd.mean())),
        # Receipts identity: the run record names a PATH, this names the BYTES.
        "raw_log_path": policy.raw_log_path,
        "receipts_sha256": policy.receipts_sha256,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": policy.run_id,
        "config_name": policy.config_name,
        "config_sha256": prov.get("config_sha256", ""),
        "tier": policy.tier,
        "split": prov.get("split", ""),
        "split_sha256": prov.get("split_sha256", ""),
        "n_examples": len(policy),
        "cost_config": config_block(cfg),
        "policy": policy.policy,
        "accuracy": _round(float(np.mean(policy.correct))),
        "n_to_human": int(policy.to_human.sum()),
        "expected_cost_per_1k": {k: _round_ci(v) for k, v in bands.items()},
        "api_cost": api_block,
        "cost_sum_check": (
            {k: (_round(v) if isinstance(v, float) else v)
             for k, v in policy.cost_sum_check.items()}
            if policy.cost_sum_check is not None else None
        ),
        "human_assumption": HUMAN_ASSUMPTION,
        "bootstrap": {
            "n_resamples": int(n_resamples),
            "seed": int(seed),
            "method": (
                f"percentile [{harness.CI_LOWER_PCT}, {harness.CI_UPPER_PCT}] over "
                "resampled example indices (one integers(0, n, size=n) draw per replicate)"
            ),
        },
    }


def write_result_json(obj: dict, path) -> Path:
    """Write one result JSON atomically (tmp file in the same dir + os.replace).

    Never a partially written artifact: a reader either sees the previous generation or
    the new one, including if the process dies mid-batch.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(obj, sort_keys=True, indent=2) + "\n")
    os.replace(tmp, path)
    return path


# ---------------------------------------------------------------------------
# Provenance gate (artifact identity vs the run record)
# ---------------------------------------------------------------------------

def check_provenance(art: predictions.PredictionsArtifact, record: dict) -> None:
    """Hard-fail unless the artifact's embedded provenance IS this run's.

    The output JSON stamps the run's id, config hash and split hash onto numbers computed
    from whatever parquet happened to sit at `data/preds/<run_id>.parquet`. If that file
    was produced from a different config or a different split, the stamp launders a
    provenance break into a clean-looking cost artifact — the same failure
    `predictions.load_config_checked` refuses at backfill time, refused again here because
    this module can be pointed at any `--preds-dir`.

    The Tier C prompt hash is compared whenever either side carries one, so a
    present-vs-absent asymmetry fails too ("not verified" is not "verified"). Every other
    field must be present and equal on BOTH sides — an empty string is treated as a
    mismatch, because "this artifact does not say which code produced it" is not a pass.
    """
    prov = art.provenance
    dataset = record.get("dataset") or {}
    pairs = [
        ("run_id", prov.get("run_id", ""), record.get("run_id", "")),
        ("config_sha256", prov.get("config_sha256", ""), record.get("config_sha256", "")),
        ("split", prov.get("split", ""), dataset.get("split", "")),
        ("split_sha256", prov.get("split_sha256", ""), dataset.get("split_sha256", "")),
        # The CODE that produced the artifact and the DATA SNAPSHOT the splits were cut
        # from. Without these two, an artifact regenerated by different code, or from a
        # re-downloaded CFPB snapshot, matches on every other field while describing a
        # different experiment.
        ("git_sha", prov.get("git_sha", ""), record.get("git_sha", "")),
        ("input_sha256", prov.get("input_sha256", ""), dataset.get("input_sha256", "")),
    ]
    prompt_art = prov.get("prompt_bundle_sha256", "") or ""
    prompt_rec = (record.get("extra") or {}).get("prompt_bundle_sha256", "") or ""
    if prompt_art or prompt_rec:
        pairs.append(("prompt_bundle_sha256", prompt_art, prompt_rec))

    mismatches = [(field, a, b) for field, a, b in pairs if not a or not b or a != b]
    if mismatches:
        detail = "; ".join(
            f"{field}: artifact {a!r} vs record {b!r}" for field, a, b in mismatches
        )
        raise ValueError(
            f"provenance mismatch for run {record.get('run_id', '?')[:8]} — {detail}. The "
            "artifact was not produced by the inputs this record names, so a cost number "
            "computed from it must not be stamped with this run's identity"
        )


def load_artifact_verified(record: dict, preds_dir=DEFAULT_PREDS_DIR, *,
                           allowed_splits: set[str] | None = None):
    """Read a run's artifact and refuse it unless it passes the repo's FULL gate.

    Three layers, cheapest-to-strongest, all hard failures:

    1. `check_provenance` — the artifact's embedded ids/hashes are this run's.
    2. `allowed_splits` — an optional whitelist, so a caller that must not touch a slice
       (e.g. CAL-only threshold fitting) is stopped structurally rather than by review.
    3. `predictions.verify_artifact` — the repo's own structural + aggregate gate: ids
       unique and non-null, every id a member of the frozen split, `y_true` agreeing with
       that split, `p_max` exactly `probs.max`, `y_pred` the argmax (or the one-hot column
       for tier_c), and the artifact's recomputed accuracy/macro_f1/aurc/acc_at_cov::*
       matching the logged record to 1e-9.

    Layer 3 is the one worth arguing for: anything that reads `p_max` as a *ranking* signal
    or joins several artifacts per row is broken by exactly the faults that leave aggregate
    metrics untouched — a permuted id column, a p_max that is not the row's max
    probability. Re-running the gate costs under a second per artifact and removes any need
    to trust that `make preds` was run after the last change.
    """
    art_path = Path(preds_dir) / f"{record['run_id']}.parquet"
    art = predictions.read_artifact(art_path)
    check_provenance(art, record)
    split = (record.get("dataset") or {}).get("split", "")
    if allowed_splits is not None and split not in allowed_splits:
        raise ValueError(
            f"run {record['run_id'][:8]} is on split {split!r}, which is outside the "
            f"allowed set {sorted(allowed_splits)} for this task"
        )
    config = predictions.load_config_checked(record)
    rows = predictions.verify_artifact(art, record, config, art_path=art_path)
    failed = [r for r in rows if not r["ok"]]
    if failed:
        detail = "; ".join(
            f"{r['check']}: {r.get('detail', f'{r.get("abs_delta")!r} off')}"
            for r in failed
        )
        raise ValueError(
            f"artifact {record['run_id'][:8]} ({art_path}) fails the predictions "
            f"verification gate — {detail}. A number computed from an unverified artifact "
            "is not a number for the run it claims to describe"
        )
    return art


def score_run(record: dict, cfg: CostConfig, *, preds_dir=DEFAULT_PREDS_DIR,
              n_resamples: int = harness.N_RESAMPLES,
              seed: int = harness.BOOTSTRAP_SEED) -> dict:
    """Read one run's artifact, verify its provenance, score it, return the result dict."""
    art_path = Path(preds_dir) / f"{record['run_id']}.parquet"
    art = predictions.read_artifact(art_path)
    check_provenance(art, record)
    policy = build_single_tier_policy(record, art, cfg)
    return build_result(policy, art, cfg, n_resamples=n_resamples, seed=seed)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_run_ids(selectors, *, select_all: bool, preds_dir: Path) -> list[str]:
    """Artifact stems (= run_ids) matching the given prefixes, or all of them."""
    all_ids = [p.stem for p in sorted(preds_dir.glob("*.parquet"))]
    if select_all:
        return all_ids
    chosen: list[str] = []
    for sel in selectors:
        matches = [rid for rid in all_ids if rid.startswith(sel)]
        if not matches:
            raise ValueError(f"no artifact in {preds_dir} matches prefix {sel!r}")
        chosen.extend(matches)
    return chosen


def _run_ids_for_config_prefixes(prefixes, *, results_path: Path,
                                 preds_dir: Path) -> list[str]:
    """Run ids whose config stem starts with one of `prefixes` (e.g. `tier_b`).

    Selecting by CONFIG rather than by run id is what makes "score the Tier B runs under
    the new cost config" a stable command: it keeps working when a new Tier B run lands,
    without a hash being pasted into the Makefile.
    """
    records = predictions.load_records(results_path)
    chosen: list[str] = []
    for prefix in prefixes:
        matches = [r["run_id"] for r in records
                   if Path(r.get("config_path", "")).stem.startswith(prefix)]
        if not matches:
            raise ValueError(f"no run record in {results_path} whose config starts with "
                             f"{prefix!r}")
        missing = [rid for rid in matches
                   if not (Path(preds_dir) / f"{rid}.parquet").exists()]
        if missing:
            raise ValueError(
                f"config prefix {prefix!r} selects run(s) {[m[:8] for m in missing]} with "
                f"no artifact under {preds_dir}; run `make preds` first"
            )
        chosen.extend(matches)
    return chosen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m triage_lab.cost_model")
    parser.add_argument("run_id", nargs="*", help="run_id prefix(es) whose artifact to score")
    parser.add_argument("--all", action="store_true", help="every artifact under --preds-dir")
    parser.add_argument(
        "--config-prefix", action="append", default=[], dest="config_prefixes",
        help="score every run whose config stem starts with this (repeatable), e.g. "
             "`--config-prefix tier_b` to price the Tier B runs under a new cost config",
    )
    parser.add_argument("--preds-dir", type=Path, default=DEFAULT_PREDS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_COST_DIR)
    parser.add_argument("--cost-config", type=Path, default=DEFAULT_COST_CONFIG)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH)
    args = parser.parse_args(argv)

    if not args.all and not args.run_id and not args.config_prefixes:
        parser.error("give run_id prefix(es), --config-prefix, or --all")

    cfg = load_cost_config(args.cost_config)
    run_ids = _resolve_run_ids(args.run_id, select_all=args.all, preds_dir=args.preds_dir)
    if args.config_prefixes:
        run_ids += _run_ids_for_config_prefixes(
            args.config_prefixes, results_path=args.results, preds_dir=args.preds_dir)
    if not run_ids:
        # Empty selection is a failure, not a no-op: `make cost-model` finding nothing to
        # score means the preds artifacts are missing, and exiting 0 would let CI go green
        # on an evidence directory that was never regenerated.
        print(f"ERROR: no prediction artifacts found under {args.preds_dir}; "
              "run `make preds` first")
        return 1

    records = {r["run_id"]: r for r in predictions.load_records(args.results)}
    print(f"cost config {cfg.path.name} ({cfg.version}) sha256={cfg.sha256[:12]} "
          f"c_misroute=${cfg.c_misroute_usd:.2f} c_human=${cfg.c_human_usd:.2f}")

    # Score EVERY selected run before writing ANY output: a batch that half-fails must not
    # leave results/cost_model/ holding a mix of generations. Failures raise out of here
    # (nonzero exit) with nothing written.
    scored: list[tuple[str, dict]] = []
    for run_id in run_ids:
        record = records.get(run_id)
        if record is None:
            raise ValueError(
                f"artifact {run_id[:8]} has no record in {args.results}; a cost number "
                "needs the run's logged cost_usd and receipts path to be verifiable"
            )
        scored.append((run_id, score_run(record, cfg, preds_dir=args.preds_dir)))

    for run_id, obj in scored:
        out_path = write_result_json(obj, args.out_dir / f"{run_id}.json")
        tot = obj["expected_cost_per_1k"][TOTAL_KEY]
        api = obj["expected_cost_per_1k"]["api"]
        check = obj["cost_sum_check"]
        gate = "" if check is None else (" ✓sum" if check["ok"] else " ✗sum")
        print(
            f"[{run_id[:8]}] {obj['config_name']:46s} {obj['split']:15s} "
            f"n={obj['n_examples']:5d} acc={obj['accuracy']:.4f} "
            f"cost/1k=${tot['point']:8.2f} [{tot['ci_lo']:8.2f}, {tot['ci_hi']:8.2f}] "
            f"api/1k=${api['point']:7.2f}{gate} -> {out_path}"
        )
    return 0


if __name__ == "__main__":
    import sys

    from triage_lab.cost_model import main as _main

    sys.exit(_main())
