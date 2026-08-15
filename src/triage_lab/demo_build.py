"""Static demo data builder (Phase 6) — every file `demo/DATA_CONTRACT.md` specifies.

Repro:

    make demo-data        # uv run python -m triage_lab.demo_build --all

`demo/DATA_CONTRACT.md` is the normative spec; this module is its producer side and the
static site (`demo/index.html` + `demo/assets/`) is its consumer side. Ten files land in
``demo/data/`` and are committed, so the demo is fully static: no server, no API call, no
model run at page load.

**The case study page is data, not copy.** ``case_study.json`` (added 2026-08-13) carries
the narrative panel's prose AND a declared `numbers` array per section: every numeric token
that appears in a paragraph is COPIED at build time from a run record or a committed
derived artifact, with its source path, run ids, repro command, unit and evidence class.
The prose lives here as templates with the displays interpolated, so a number can never be
typed into the page by hand — and `tests/test_demo_build.py` enforces exactly that
(paragraph token ⊆ declared displays, display ↔ value, value ↔ source artifact).

**Nothing here is a new measurement.** Every number is either COPIED from an append-only
results record / committed artifact (metrics, CIs, costs, thresholds, drift rollup) or
DERIVED from a frozen per-example artifact by arithmetic that carries no free parameters
(calibration bins, the CAL tau sweep, the per-sample router path). Each object says which,
via ``evidence_class`` and a ``run_id`` / ``source`` provenance field, because a demo is
exactly where an unattributed number does the most damage.

**The provenance section is the one exception, and it says so.** The case study's
coursework-seed section quotes self-reported class results out of the READ-ONLY archive
`docs/seed-evidence/` (this module only ever reads it). Those figures carry their own
evidence class — `provenance` — and their display strings must be exact substrings of the
archived file, or the build fails. They are lineage, not evidence, and nothing else on the
page depends on them.

**Run ids are resolved, never transcribed.** ``results/runs.jsonl`` is the only registry;
runs are looked up by (config stem, split) and the id falls out. A hardcoded id in this
file would be a second source of truth that no gate compares against the log.

**Determinism.** The build writes no wall-clock timestamp — the only timestamps in the
output are the ones copied out of results records. JSON is ``sort_keys=True, indent=2,
ensure_ascii=False`` with a trailing newline, so two builds of the same repo state produce
byte-identical files and `git diff` shows only real changes.

**The curated sample set is frozen like a split** (CLAUDE.md rule 2). It is selected once
from the paired Haiku ∩ Sonnet TEST-IID receipt ids under seed 20260806 and committed as
``demo/data/curated_ids.json``; every later build REGENERATES the selection and hard-fails
on any difference. It is never reselected to make an exhibit look better.

**Tier B is real data (backfilled 2026-08-12).** The scaffold's pending slots are replaced
in place: frontier points (B1 x3 seeds, B2, the a_to_b and a_to_b_to_c routers), policy
blocks for both Tier B cascades, per-sample tier_b1/tier_b2 cards, temperature-scaled
calibration exhibits, and the tier_b2 drift series with the a_to_b escalation arms. The
one slot that remains pending is the Tier B1 yearly drift series — descoped by owner
(2026-08-12), retained as a labeled slot because "not measured" and "measured to be
absent" are different claims. The build now requires a cost config that prices Tier B
(``configs/cost_model_v2.yaml``) and hard-fails otherwise.

**The headline router is ``a_to_b`` (owner decision 2026-08-12)** — the only certified
two-axis win (cost AND macro-F1 paired CIs excluding zero vs b2_only), dominating 3 model
baselines on the full slice. The Haiku-terminal cascade ``a_to_c_parsefail_human`` stays
in the payload as the LLM-cascade contrast exhibit; the drift-chapter Sonnet-terminal
variant is unchanged.

**Reused, never reimplemented.** The router path shown per sample comes from
``router_sim.build_paired_policies`` under the frozen ``v2-isocal`` operating point, the
thresholds from ``router_sim.load_cal_thresholds`` (which validates and replays each tau*
against the CAL run that produced it), the CAL sweep from ``threshold_opt.build_grid`` /
``sweep``, and every receipt from ``cost_model``'s verifying loader. The demo cannot
disagree with Phase 4 about the op semantics because it does not own them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

import duckdb
import numpy as np
import yaml

from triage_lab import (
    cost_model,
    frontier,
    harness,
    metrics,
    predictions,
    router_sim,
    threshold_opt,
)

REPO_ROOT = harness.REPO_ROOT
DEFAULT_OUT_DIR = REPO_ROOT / "demo" / "data"
DEFAULT_RESULTS_PATH = harness.DEFAULT_RESULTS_PATH
DEFAULT_PREDS_DIR = harness.DEFAULT_PREDS_DIR
DEFAULT_SPLITS_DIR = predictions.DEFAULT_SPLITS_DIR
DEFAULT_DRIFT_SUMMARY = REPO_ROOT / "results" / "drift" / "summary.json"
DEFAULT_FRONTIER_DIR = REPO_ROOT / "results" / "frontier"
DEFAULT_ROUTER_DIR = router_sim.DEFAULT_ROUTER_DIR
DEFAULT_COST_DIR = cost_model.DEFAULT_COST_DIR

# Committed derived artifacts the case study copies from (no new measurement anywhere).
DEFAULT_PRIOR_SHIFT_SUMMARY = REPO_ROOT / "results" / "prior_shift" / "summary.json"
DEFAULT_PAIRED_WITHIN = (REPO_ROOT / "results" / "prior_shift"
                         / "paired_within_tier_a_vs_tier_b2_2026h1.json")
DEFAULT_TIER_B_COMPARE = REPO_ROOT / "results" / "tier_b_compare" / "summary.json"
# The three Tier C paired comparisons the case study cites. Read by path rather than
# through a rollup: three files with three repro commands are already the index, and a
# summary.json would be a second place the same numbers live.
DEFAULT_TIER_C_COMPARE_DIR = REPO_ROOT / "results" / "tier_c_compare"
TIER_C_COMPARE_SONNET_IID = "sonnet_minus_haiku__test_iid"
TIER_C_COMPARE_SONNET_POSTCUTOFF = "sonnet_minus_haiku__test_postcutoff"
TIER_C_COMPARE_FEWSHOT = "haiku_fewshot_minus_zeroshot__cal"
DEFAULT_PERTURBATION_SUMMARY = REPO_ROOT / "results" / "perturbation" / "summary.json"
DEFAULT_OOV_SUMMARY = REPO_ROOT / "results" / "oov" / "summary.json"
DEFAULT_ONNX_PARITY = REPO_ROOT / "results" / "onnx_parity" / "tier_b2_s0_parity.json"
DEFAULT_LIVE_AGREEMENT = REPO_ROOT / "demo" / "live" / "agreement_report.json"
DEFAULT_SNAPSHOT_MANIFEST = REPO_ROOT / "SNAPSHOT_MANIFEST.yaml"

SCHEMA_VERSION = "demo-v1"

# --- repo URL base (ONE constant; the GitHub-push session fills it) ----------------------
# Empty until this repo has a public URL. Every repo-relative path the demo renders
# (seed-evidence files today; commit and Actions links later) goes through `repo_block()`
# -> `demo/assets/app.js`'s repoRef(), which renders a non-link <code> chip labeled
# "link resolves after GitHub push" while this is "" and a real <a href> once it is set.
# Filling it is a one-line change plus `make demo-data`; nothing else moves.
REPO_URL_BASE: str = "https://github.com/LucisZhang/triage-router"
REPO_DEFAULT_BRANCH: str = "main"


def repo_block(url_base: str = REPO_URL_BASE,
               default_branch: str = REPO_DEFAULT_BRANCH) -> dict:
    """The `repo` object `case_study.json` carries: pure function of the two constants.

    A trailing slash on the base would produce `…//blob/…` hrefs downstream, so it is
    normalized away here rather than in three places in the renderer.
    """
    return {"url_base": url_base.rstrip("/"), "default_branch": default_branch}

# The primary operating point for every reported router number (STATUS.md Phase 4 task 5).
OP_VERSION = router_sim.OP_V2

# The demo builds under the cost generation that prices Tier B (Phase 4 backfill,
# 2026-08-11): v2 = v1 verbatim + tier_b1/tier_b2 amortized-estimate pricing, so every
# pre-Tier-B number reproduces exactly while the Tier B policies become scorable. This is
# deliberately NOT cost_model.DEFAULT_COST_CONFIG (still v1 for the v1-generation
# artifacts' sake).
DEMO_COST_CONFIG = REPO_ROOT / "configs" / "cost_model_v2.yaml"

# Owner decision 2026-08-12, executed in this payload: a_to_b is the headline router;
# the Haiku-terminal cascade is retained as the LLM-cascade contrast exhibit.
HEADLINE_ROUTER = threshold_opt.FAMILY_A_TO_B
HEADLINE_NOTE = (
    "Owner decision 2026-08-12: headline_router = a_to_b (full TEST-IID) — the only "
    "certified two-axis win (cost AND macro-F1 paired CIs excl. 0 vs b2_only), dominating "
    "3 model baselines. a_to_c_parsefail_human is retained as the LLM-cascade contrast "
    "exhibit; the drift-chapter Sonnet-terminal variant is unchanged."
)

# --- frozen curated-set selection (CLAUDE.md rule 2) -------------------------------------
CURATED_VERSION = "v1"
CURATED_SEED = 20260806
CURATED_N = 200
CURATED_METHOD = (
    "class-stratified proportional (largest remainder, min 1/class), ids sorted before draw"
)
CURATED_POOL = "TEST-IID complaint_ids scored by BOTH Haiku and Sonnet finals (paired 1,500)"

# --- run selection, by config stem + split (never by hardcoded run id) -------------------
TIER_A_LOGREG_TEST = "tier_a_logreg_test_iid"
TIER_A_CNB_TEST = "tier_a_cnb_test_iid"
TIER_A_RAW_CAL = "tier_a_logreg_wordchar_cal"
TIER_A_ISOCAL_CAL = threshold_opt.V2_TIER_A_CAL_CONFIG
HAIKU_TEST = "tier_c_haiku_zeroshot_test_iid"
SONNET_TEST = "tier_c_sonnet_zeroshot_test_iid"
TEST_IID = "test_iid"
CAL = "cal"

# The four frozen Tier B TEST-IID finals (Phase 2 eval backfill, 2026-08-10), as
# (config stem, frontier/calibration key, label). B1's three seeds are three separate
# points everywhere — seed variance is the exhibit, and averaging them would be a number
# no run record carries.
TIER_B_TESTS: tuple[tuple[str, str, str], ...] = (
    ("tier_b1_modernbert_sa", "tier_b1_sa", "Tier B1 — ModernBERT-base (seed a)"),
    ("tier_b1_modernbert_sb", "tier_b1_sb", "Tier B1 — ModernBERT-base (seed b)"),
    ("tier_b1_modernbert_sc", "tier_b1_sc", "Tier B1 — ModernBERT-base (seed c)"),
    ("tier_b2_distilbert_s0", "tier_b2", "Tier B2 — DistilBERT (deployment point)"),
)
# The per-sample tier_b1 card shows one seed; sa is the first of the frozen seed list
# (20260805, the same seed B2 trained under), not a metric-based pick — sb ties sa on
# macro-F1 and all three runs ship as their own frontier + calibration exhibits.
TIER_B1_SAMPLE_CONFIG = "tier_b1_modernbert_sa"
TIER_B2_SAMPLE_CONFIG = router_sim.TIER_B_CASCADE_CONFIG  # B2, the cascade rung

CALIBRATION_N_BINS = metrics.DEFAULT_N_BINS  # 15, the repo's ECE binning
# The bins must be the ones the logged ECE was computed over, not merely 15 bins that look
# like them, so the recomputation is pinned against the record at this tolerance.
ECE_REPLAY_TOL = 1e-9

# <= this many rows in the published CAL tau sweep (the full grid is one row per distinct
# p_max, ~87k on CAL — a slider does not need them and a demo payload cannot carry them).
TAU_SWEEP_MAX_POINTS = 256

ROUND = cost_model.JSON_ROUND

# --- the one remaining pending slot (real object, per the contract) ----------------------
# Everything else the scaffold marked "pending Tier B" is real data as of 2026-08-12. The
# B1 yearly drift series was descoped by owner the same day (~8h MPS for a model B2
# dominates on every TEST-IID metric); the slot stays labeled rather than deleted.
PENDING_DRIFT_SERIES = (
    ("tier_b1", ("Tier B1 yearly drift series — descoped by owner 2026-08-12 "
                 "(slot retained; not measured ≠ measured absent)")),
)
PENDING_TIER_B_SLOTS = tuple(slot for slot, _ in PENDING_DRIFT_SERIES)

EVIDENCE_LEGEND = {
    "measured": (
        "read off a real observation: a logged run metric, a per-call receipt, or a "
        "deterministic derivation from a frozen per-example artifact"
    ),
    "estimated": (
        "a modeling assumption the reader is invited to move (c_misroute, c_human, the "
        "Tier A amortized-zero compute charge) — never a measurement"
    ),
    "projected": (
        "an extrapolation to conditions not evaluated (e.g. cost at a volume no run "
        "reached); carries the assumption that produced it"
    ),
    "derived": (
        "arithmetic over measured inputs with no free parameters (calibration bins, the "
        "CAL threshold sweep, per-sample router paths under a frozen tau)"
    ),
    # Used ONLY by the case study's provenance section. It exists so a coursework score can
    # never wear the same badge as a run this lab actually made.
    "provenance": (
        "a self-reported coursework result, copied verbatim from the read-only seed "
        "archive under docs/seed-evidence/ — NOT verified or reproduced by this lab; "
        "provenance, not evidence"
    ),
}

DRIFT_ANNOTATIONS = (
    {"x": "2023-04", "label": "CFPB taxonomy consolidation"},
    {"x": "2026-H1", "label": "credit_reporting prior-shift cliff"},
)

TIER_C_CALIBRATION_NOTE = (
    "Tier C structured output emits a degenerate one-hot p_max — no calibration signal to "
    "plot; parse-failure is its only self-signal (UPGRADE_PLAN §4.2 amendment)."
)

FROZEN_TAU_NOTE = (
    "a_to_c, a_to_b and a_to_b_to_c taus are frozen from Phase 4 CAL optimization; the "
    "demo does not re-solve them (only the a_to_human arm carries a published CAL sweep; "
    "Haiku scored only the paired subset on CAL)."
)

TAU_SWEEP_NOTE = (
    "sweep computed from the frozen CAL per-example artifact; enables live threshold "
    "re-solve for the A->human arm only"
)

SLIDER_NOTE = (
    "Expected cost is linear in (c_misroute, c_human, api price scale) given each policy's "
    "MEASURED rates, so the frontend recomputes it client-side from `rates`, "
    "`p_error_machine` and `api_cost_per_1k_usd`. Only `a_to_human` may have its tau "
    "re-solved (via tau_sweep_a_to_human); every other policy's tau is frozen."
)

SUPPORT_NOTE = (
    "Supports differ by point and are NOT comparable as populations: Tier A, all four "
    "Tier B points, and the a_to_human / a_to_b routers are scored on the full "
    "104,443-row TEST-IID slice; Haiku, the a_to_c cascade and the a_to_b_to_c cascade "
    "on Haiku's 5,000-row uniform subsample; Sonnet on the paired 1,500."
)

ROUTER_PATH_NOTE = (
    "decision recomputed with the frozen Phase 4 op (reuse router_sim logic)"
)

NARRATIVE_SOURCE = "frozen split parquet (CFPB, US-gov public domain)"

RECEIPTS_REPRO = {
    "results_log": "results/runs.jsonl",
    "note": "append-only; corrections reference superseded run ids",
}

MODEL_SLUG_LABELS = {
    "anthropic/claude-haiku-4.5": "Claude Haiku 4.5",
    "anthropic/claude-sonnet-5": "Claude Sonnet 5",
}


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def write_json(obj, path) -> Path:
    """Deterministic, atomic JSON write: sorted keys, 2-space indent, trailing newline.

    `ensure_ascii=False` because the narratives are real consumer text: escaping them to
    \\uXXXX triples the payload and makes the committed file unreadable in review, while
    UTF-8 round-trips exactly.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return path


def _rel(path) -> str:
    """Repo-relative POSIX path, for the `source` provenance field on every exhibit.

    Falls back to the absolute path for anything outside the repo (a `--out-dir` pointed at
    a tmp dir in tests); every provenance field written into the payload names a committed
    input and is therefore always inside.
    """
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _round(value):
    return None if value is None else round(float(value), ROUND)


def git_head_sha() -> str:
    """HEAD at build time. Same helper the harness stamps into every run record."""
    return harness._git_sha()


# ---------------------------------------------------------------------------
# Run resolution (by config stem + split; ids are an OUTPUT of this, never an input)
# ---------------------------------------------------------------------------

def _config_name(record: dict) -> str:
    return Path(record.get("config_path", "")).stem


def _slice_of(record: dict) -> str:
    return (record.get("dataset") or {}).get("split", "")


def _superseded_ids(records: list[dict]) -> set[str]:
    """Run ids that a LATER record declares it supersedes (CLAUDE.md rule 3 corrections).

    A correction is a new append-only record that references the run it replaces, so the
    reference is the only signal that an older record is no longer current. Both spellings
    are accepted because the log predates a fixed field name; neither is present today, so
    this path is inert until the first correction lands.
    """
    out: set[str] = set()
    for record in records:
        extra = record.get("extra") or {}
        for key in ("supersedes", "supersedes_run_id", "superseded_run_id"):
            value = extra.get(key) or record.get(key)
            if isinstance(value, str):
                out.add(value)
            elif isinstance(value, (list, tuple)):
                out.update(str(v) for v in value)
    return out


def resolve_records(records: list[dict]) -> dict[tuple[str, str], dict]:
    """(config stem, split) -> the CURRENT record for that run.

    Duplicates are resolved by an explicit rule rather than by file order: any record a
    later one supersedes is dropped, and of what remains the newest `timestamp_utc` wins.
    If that still leaves a tie the build fails — two same-second records for one
    (config, split) differ in something neither the key nor the log names, and picking one
    would make every demo number depend on line order.
    """
    superseded = _superseded_ids(records)
    grouped: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        if record["run_id"] in superseded:
            continue
        grouped.setdefault((_config_name(record), _slice_of(record)), []).append(record)

    out: dict[tuple[str, str], dict] = {}
    for key, group in grouped.items():
        if len(group) == 1:
            out[key] = group[0]
            continue
        newest = max(r["timestamp_utc"] for r in group)
        winners = [r for r in group if r["timestamp_utc"] == newest]
        if len(winners) != 1:
            ids = ", ".join(sorted(r["run_id"][:8] for r in winners))
            raise ValueError(
                f"cannot resolve {key}: {len(winners)} records share the newest timestamp "
                f"{newest} ({ids}) and none supersedes the other"
            )
        out[key] = winners[0]
    return out


def record_for(resolved: dict, config_name: str, slice_name: str) -> dict:
    key = (config_name, slice_name)
    if key not in resolved:
        available = ", ".join(sorted(n for n, s in resolved if s == slice_name))
        raise ValueError(
            f"no run record for config {config_name!r} on slice {slice_name!r} in the "
            f"results log; {slice_name} runs present: {available}"
        )
    return resolved[key]


def tier_of(record: dict) -> str:
    """`A`, `B` or `C` for a record, from its config stem.

    Tier B configs are named per fine-tune SIZE (`tier_b1_modernbert_*`,
    `tier_b2_distilbert_*`), so `tier_b_` alone matches neither — the same prefix trap
    `cost_model.tier_of_config_name` has to handle. Both spellings are accepted here.
    """
    extra = record.get("extra") or {}
    if extra.get("tier"):
        return str(extra["tier"]).replace("tier_", "").upper()
    name = _config_name(record)
    for prefix, tier in (("tier_a_", "A"), ("tier_b_", "B"), ("tier_b1_", "B"),
                         ("tier_b2_", "B"), ("tier_c_", "C")):
        if name.startswith(prefix):
            return tier
    raise ValueError(f"cannot infer tier for config {name!r}")


def model_label(record: dict) -> str:
    """Human label for a run's MODEL, derived from the config stem + logged model slug."""
    name = _config_name(record)
    extra = record.get("extra") or {}
    slug = extra.get("model_slug")
    if slug:
        base = f"Tier C — {MODEL_SLUG_LABELS.get(slug, slug)}"
        base += " (few-shot)" if "fewshot" in name else " (zero-shot)"
    elif tier_of(record) == "B":
        # Named from the record's own logged base model, not from the config stem: the
        # stem says which frontier point it is, the record says what was actually fine-tuned.
        base = f"Tier B — {extra.get('base_model') or name} (fine-tuned)"
    elif "_cnb_" in name:
        base = "Tier A — TF-IDF ComplementNB (word+char)"
    elif "_logreg_word_" in name or name.endswith("_logreg_word_cal"):
        base = "Tier A — TF-IDF LogReg (word-only)"
    else:
        base = "Tier A — TF-IDF LogReg (word+char)"

    qualifiers = []
    if "isocal" in name:
        qualifiers.append("isotonic CAL rung")
    if "smoke" in name:
        qualifiers.append("smoke")
    if "ablation" in name:
        qualifiers.append("ablation")
    perturbation = extra.get("perturbation")
    if perturbation:
        qualifiers.append(f"{perturbation['family']} @ {perturbation['rate']:g}")
    return base + (f" [{', '.join(qualifiers)}]" if qualifiers else "")


# ---------------------------------------------------------------------------
# Metric objects (copied, never recomputed)
# ---------------------------------------------------------------------------

def metric_from_record(record: dict, key: str) -> dict:
    """`{point, ci_lo, ci_hi}` copied VERBATIM out of a run record's metrics block."""
    block = (record.get("metrics") or {}).get(key)
    if block is None:
        raise ValueError(f"run {record['run_id'][:8]} has no logged metric {key!r}")
    return {"point": block["point"], "ci_lo": block["ci_lo"], "ci_hi": block["ci_hi"]}


def ci_block(band: dict) -> dict:
    """`{point, ci_lo, ci_hi}` copied verbatim from an artifact's CI band."""
    return {"point": band["point"], "ci_lo": band["ci_lo"], "ci_hi": band["ci_hi"]}


def pending_slot(slot: str, label: str | None = None) -> dict:
    out = {"pending": True, "slot": slot}
    if label is not None:
        out["label"] = label
    return out


# ---------------------------------------------------------------------------
# 1. meta.json
# ---------------------------------------------------------------------------

def snapshot_sha256(records: list[dict]) -> str:
    """The single dataset snapshot every run was cut from; disagreement is a hard failure."""
    values = {(r.get("dataset") or {}).get("input_sha256", "") for r in records}
    values.discard("")
    if len(values) != 1:
        raise ValueError(
            f"expected exactly one dataset input_sha256 across the results log, found "
            f"{sorted(v[:12] for v in values)}; the demo cannot name one snapshot"
        )
    return values.pop()


def build_meta(records: list[dict], cfg: cost_model.CostConfig) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "git_sha": git_head_sha(),
        "snapshot_sha256": snapshot_sha256(records),
        "op_version": OP_VERSION,
        "cost_model": {"path": _rel(cfg.path), "sha256": cfg.sha256},
        "evidence_classes": EVIDENCE_LEGEND,
        "headline_router": {
            "policy": HEADLINE_ROUTER,
            "evaluation_set": router_sim.EVAL_FULL,
            "note": HEADLINE_NOTE,
        },
        "pending_tier_b": list(PENDING_TIER_B_SLOTS),
    }


# ---------------------------------------------------------------------------
# 2. runs_index.json
# ---------------------------------------------------------------------------

def build_runs_index(records: list[dict]) -> dict:
    """One entry per non-header record in `results/runs.jsonl`, metrics verbatim.

    `extra` is always present (empty when the record has none) so a consumer never has to
    branch on key existence — the same rule the contract states for pending slots.
    """
    out: dict[str, dict] = {}
    for record in records:
        out[record["run_id"]] = {
            "config_name": _config_name(record),
            "config_path": record.get("config_path", ""),
            "config_sha256": record.get("config_sha256", ""),
            "slice": _slice_of(record),
            "timestamp_utc": record.get("timestamp_utc", ""),
            "git_sha": record.get("git_sha", ""),
            "dataset": record.get("dataset") or {},
            "metrics": record.get("metrics") or {},
            "cost_usd": record.get("cost_usd"),
            "wall_clock_seconds": record.get("wall_clock_seconds"),
            "tier": tier_of(record),
            "model_label": model_label(record),
            "extra": record.get("extra") or {},
        }
    return out


# ---------------------------------------------------------------------------
# 3. frontier.json
# ---------------------------------------------------------------------------

def primary_frontier_path(cfg: cost_model.CostConfig | None = None,
                          frontier_dir=DEFAULT_FRONTIER_DIR) -> Path:
    """The opv2 frontier file for THIS cost config.

    Resolved by name (like `router_sim_path`) rather than by "the only one there": one
    frontier file exists per cost generation, and once `cost_model_v2.yaml` adds Tier B
    pricing there are legitimately several. Picking by the config the demo is being built
    under is the only selection that cannot silently mix generations. Passing `cfg=None`
    keeps the old behaviour for callers that have no config in hand: exactly one file, or
    a hard failure.
    """
    frontier_dir = Path(frontier_dir)
    if cfg is not None:
        path = frontier_dir / frontier.result_filename(cfg, OP_VERSION)
        if not path.exists():
            raise ValueError(
                f"missing frontier artifact {path} for cost config {cfg.path.name}; "
                "run `make frontier` under that cost config first"
            )
        return path
    candidates = sorted(frontier_dir.glob("frontier__opv2__*.json"))
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one primary (opv2) frontier file under {frontier_dir}, "
            f"found {[p.name for p in candidates]}; `make frontier` writes one per cost "
            "config — pass the cost config to select one"
        )
    return candidates[0]


def router_sim_path(name: str, cfg: cost_model.CostConfig,
                    router_dir=DEFAULT_ROUTER_DIR) -> Path:
    path = Path(router_dir) / router_sim.result_filename(name, cfg, OP_VERSION)
    if not path.exists():
        raise ValueError(f"missing router_sim artifact {path}; run `make router-sim` first")
    return path


def cost_artifact_path(run_id: str, cost_dir=DEFAULT_COST_DIR) -> Path:
    path = Path(cost_dir) / f"{run_id}.json"
    if not path.exists():
        raise ValueError(f"missing cost-model artifact {path}; run `make cost-model` first")
    return path


def _single_tier_point(key: str, label: str, record: dict, *, cost_dir) -> dict:
    """A single-tier frontier point: macro-F1 from the run record, cost from cost_model."""
    run_id = record["run_id"]
    path = cost_artifact_path(run_id, cost_dir)
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact["run_id"] != run_id:
        raise ValueError(f"cost artifact {path.name} is not run {run_id[:8]}'s")
    return {
        "key": key,
        "label": label,
        "kind": "single",
        "run_id": run_id,
        "n": artifact["n_examples"],
        "evaluation_set": artifact["split"],
        "cost_model_source": _rel(path),
        "cost_basis": (
            "expected_cost_per_1k.total from the run's cost_model artifact "
            "(c_misroute·P(error) + measured api spend; no human arm)"
        ),
        "cost_per_1k_usd": ci_block(artifact["expected_cost_per_1k"]["total"]),
        "api_cost_per_1k_usd": ci_block(artifact["expected_cost_per_1k"]["api"]),
        "macro_f1": metric_from_record(record, "macro_f1"),
        "macro_f1_basis": "macro_f1 (run record, 95% bootstrap CI)",
        "evidence_class": "measured",
    }


def _router_point(key: str, label: str, policy: dict, *, source: Path,
                  run_refs: list[str], headline: bool = False) -> dict:
    """A router frontier point, copied from the frozen v2 router_sim evaluation.

    `macro_f1` carries no CI: `router_sim` logs `macro_f1_system` as a point estimate (the
    bootstrap it does run is on PAIRED deltas, which is what a comparison claim needs).
    The contract's `{"point": ...}`-only form is used rather than inventing an interval.
    """
    return {
        "key": key,
        "label": label,
        "kind": "router",
        "headline": headline,
        "run_id": run_refs[0],
        "run_refs": run_refs,
        "n": policy["n_examples"],
        "evaluation_set": policy["evaluation_set"],
        "cost_model_source": _rel(source),
        "cost_basis": (
            "expected_cost_per_1k.total from the frozen v2-isocal router_sim evaluation "
            "(misroute + measured api + human arm)"
        ),
        "cost_per_1k_usd": ci_block(policy["expected_cost_per_1k"]["total"]),
        "api_cost_per_1k_usd": ci_block(policy["expected_cost_per_1k"]["api"]),
        "macro_f1": {"point": policy["macro_f1_system"]},
        "macro_f1_basis": "macro_f1_system (router_sim; human-credited, no CI logged)",
        "evidence_class": "measured",
    }


def build_frontier(resolved: dict, cfg: cost_model.CostConfig, *,
                   frontier_dir=DEFAULT_FRONTIER_DIR, router_dir=DEFAULT_ROUTER_DIR,
                   cost_dir=DEFAULT_COST_DIR) -> dict:
    frontier_path = primary_frontier_path(cfg, frontier_dir)
    claims = json.loads(frontier_path.read_text(encoding="utf-8"))
    if claims.get("operating_point_version") != OP_VERSION:
        raise ValueError(
            f"{frontier_path.name} is operating point "
            f"{claims.get('operating_point_version')!r}, not the primary {OP_VERSION!r}"
        )

    full_path = router_sim_path(router_sim.EVAL_FULL, cfg, router_dir)
    paired_path = router_sim_path(router_sim.EVAL_PAIRED, cfg, router_dir)
    full = json.loads(full_path.read_text(encoding="utf-8"))
    paired = json.loads(paired_path.read_text(encoding="utf-8"))

    logreg = record_for(resolved, TIER_A_LOGREG_TEST, TEST_IID)
    cnb = record_for(resolved, TIER_A_CNB_TEST, TEST_IID)
    haiku = record_for(resolved, HAIKU_TEST, TEST_IID)
    sonnet = record_for(resolved, SONNET_TEST, TEST_IID)
    tier_b = {config: record_for(resolved, config, TEST_IID)
              for config, _, _ in TIER_B_TESTS}
    b2 = tier_b[TIER_B2_SAMPLE_CONFIG]

    points = [
        _single_tier_point("tier_a_logreg", "Tier A — TF-IDF LogReg", logreg,
                           cost_dir=cost_dir),
        _single_tier_point("tier_a_cnb", "Tier A — TF-IDF ComplementNB", cnb,
                           cost_dir=cost_dir),
        *[_single_tier_point(key, label, tier_b[config], cost_dir=cost_dir)
          for config, key, label in TIER_B_TESTS],
        _single_tier_point("tier_c_haiku", "Tier C — Claude Haiku 4.5 (TEST-IID)", haiku,
                           cost_dir=cost_dir),
        _single_tier_point("tier_c_sonnet",
                           "Tier C — Claude Sonnet 5 (TEST-IID subsample)", sonnet,
                           cost_dir=cost_dir),
        _router_point("a_to_human", "Router A→human", full["policies"]["a_to_human"],
                      source=full_path, run_refs=[logreg["run_id"]]),
        _router_point("a_to_b", "Router A→B2 (headline)",
                      full["policies"][threshold_opt.FAMILY_A_TO_B],
                      source=full_path, headline=True,
                      run_refs=[logreg["run_id"], b2["run_id"]]),
        _router_point("a_to_c_haiku", "Router A→Haiku (LLM-cascade contrast, "
                      "parse-fail→human)",
                      paired["policies"][threshold_opt.FAMILY_A_TO_C],
                      source=paired_path,
                      run_refs=[logreg["run_id"], haiku["run_id"]]),
        _router_point("a_to_b_to_c", "Router A→B2→Haiku (parse-fail→human)",
                      paired["policies"][threshold_opt.FAMILY_A_TO_B_TO_C],
                      source=paired_path,
                      run_refs=[logreg["run_id"], b2["run_id"], haiku["run_id"]]),
    ]
    return {
        "claims": {**claims, "source": _rel(frontier_path)},
        "points": points,
        "headline_note": HEADLINE_NOTE,
        "pending_points": [],
        "support_note": SUPPORT_NOTE,
    }


# ---------------------------------------------------------------------------
# 4. policies.json
# ---------------------------------------------------------------------------

def _policy_block(key: str, label: str, policy: dict, *, source: Path,
                  run_refs: list[str], headline: bool = False) -> dict:
    routing = policy["routing"]
    gate = policy["gate"]
    accuracy_machine = policy["accuracy_machine"]
    # The two-gate cascade carries its frozen second threshold too; single-gate policies
    # have no tau_b key and the contract keeps the absence (never a null placeholder).
    tau_b_block = {} if "tau_b" not in gate else {
        "tau_b": {
            "value": gate["tau_b"],
            "cal_tau_b_star": gate["tau_source"].get("cal_tau_b_star"),
            "cal_coverage_b_marginal": gate["tau_source"].get("cal_coverage_b_marginal"),
        },
    }
    return {
        "key": key,
        "label": label,
        "headline": headline,
        "router_sim_policy": policy["policy"],
        "evaluation_set": policy["evaluation_set"],
        "n": policy["n_examples"],
        "tau": {
            "value": gate["tau"],
            "transfer_mode": gate["transfer_mode"],
            "cal_tau_star": gate["tau_source"]["cal_tau_star"],
            "cal_target_coverage_a": gate["tau_source"]["cal_target_coverage_a"],
            "source": f"results/thresholds/{gate['tau_source']['file']}",
            "sha256": gate["tau_source"]["sha256"],
        },
        **tau_b_block,
        "run_refs": run_refs,
        "rates": {
            "answered_a": routing["coverage_a"],
            "escalated": routing["escalation_rate"],
            "human": routing["human_rate"],
        },
        "p_error_machine": (None if accuracy_machine is None
                            else _round(1.0 - accuracy_machine)),
        "api_cost_per_1k_usd": ci_block(policy["expected_cost_per_1k"]["api"]),
        "macro_f1_system": {"point": policy["macro_f1_system"]},
        "macro_f1_answered": {"point": policy["macro_f1_answered"]},
        "accuracy_system": {"point": policy["accuracy_system"]},
        "accuracy_machine": {"point": accuracy_machine},
        "expected_cost_per_1k": policy["expected_cost_per_1k"],
        "evidence_class": "measured",
        "source": _rel(source),
    }


def tau_sweep(art_cal, record_cal, *, max_points: int = TAU_SWEEP_MAX_POINTS) -> list[dict]:
    """The a_to_human cost curve on CAL, as (tau, coverage, quality, human rate) rows.

    Built with `threshold_opt`'s own grid + sweep, so the demo's slider re-solve walks
    exactly the operating points Phase 4 optimized over. The grid is one row per distinct
    `p_max` (~87k on CAL); it is index-downsampled to `max_points` evenly spaced rows,
    endpoints kept, because the shape of the curve is what a slider needs and the payload
    is committed.

    The `tau = +inf` row (Tier A answers nothing, i.e. all-human) is dropped: it has no
    answered set, so `acc_answered` is undefined there and a null would be a trap for a
    frontend that interpolates.
    """
    policy = threshold_opt.build_a_to_human(
        art_cal, record_cal, _config_name(record_cal), dataset=threshold_opt.DATASET_FULL_CAL)
    rows = threshold_opt.sweep(threshold_opt.build_grid(policy),
                               c_misroute=0.0, c_human=0.0)
    finite = np.flatnonzero(np.isfinite(rows["tau"]))
    n_rows = len(finite)
    keep = {0, n_rows - 1}
    out = []
    for j in threshold_opt._downsample(n_rows, max_points, keep):
        i = int(finite[j])
        accuracy = float(rows["accuracy_machine"][i])
        out.append({
            "tau": float(rows["tau"][i]),
            "coverage": _round(rows["coverage_a"][i]),
            "acc_answered": _round(accuracy),
            "misroute_rate_answered": _round(1.0 - accuracy),
            "human_rate": _round(rows["human_rate"][i]),
        })
    return out


def build_policies(resolved: dict, cfg: cost_model.CostConfig, *, art_cal, record_cal,
                   router_dir=DEFAULT_ROUTER_DIR) -> dict:
    full_path = router_sim_path(router_sim.EVAL_FULL, cfg, router_dir)
    paired_path = router_sim_path(router_sim.EVAL_PAIRED, cfg, router_dir)
    full = json.loads(full_path.read_text(encoding="utf-8"))
    paired = json.loads(paired_path.read_text(encoding="utf-8"))
    for obj, path in ((full, full_path), (paired, paired_path)):
        if obj.get("operating_point_version") != OP_VERSION:
            raise ValueError(f"{path.name} is not the primary {OP_VERSION} operating point")

    logreg = record_for(resolved, TIER_A_LOGREG_TEST, TEST_IID)
    haiku = record_for(resolved, HAIKU_TEST, TEST_IID)
    b2 = record_for(resolved, TIER_B2_SAMPLE_CONFIG, TEST_IID)
    return {
        "op_version": OP_VERSION,
        "cost_defaults": {
            "c_misroute": cfg.c_misroute_usd,
            "c_human": cfg.c_human_usd,
            "source": _rel(cfg.path),
            "sha256": cfg.sha256,
            "evidence_class": "estimated",
        },
        "headline_note": HEADLINE_NOTE,
        "policies": [
            _policy_block("a_to_human", "Tier A gate → human queue",
                          full["policies"]["a_to_human"], source=full_path,
                          run_refs=[logreg["run_id"]]),
            _policy_block("a_to_b", "Tier A gate → DistilBERT terminal (headline router)",
                          full["policies"][threshold_opt.FAMILY_A_TO_B],
                          source=full_path, headline=True,
                          run_refs=[logreg["run_id"], b2["run_id"]]),
            _policy_block("a_to_c_haiku",
                          "Tier A gate → Haiku terminal (LLM-cascade contrast, "
                          "parse-fail→human)",
                          paired["policies"][threshold_opt.FAMILY_A_TO_C],
                          source=paired_path,
                          run_refs=[logreg["run_id"], haiku["run_id"]]),
            _policy_block("a_to_b_to_c",
                          "Tier A gate → DistilBERT gate → Haiku terminal "
                          "(parse-fail→human)",
                          paired["policies"][threshold_opt.FAMILY_A_TO_B_TO_C],
                          source=paired_path,
                          run_refs=[logreg["run_id"], b2["run_id"], haiku["run_id"]]),
        ],
        "tau_sweep_a_to_human": {
            "slice": _slice_of(record_cal),
            "calibration": "isotonic",
            "run_id": record_cal["run_id"],
            "config_name": _config_name(record_cal),
            "n": len(art_cal),
            "grid": tau_sweep(art_cal, record_cal),
            "note": TAU_SWEEP_NOTE,
            "evidence_class": "derived",
        },
        "frozen_tau_note": FROZEN_TAU_NOTE,
        "slider_note": SLIDER_NOTE,
        "human_credit_note": router_sim.HUMAN_CREDIT_NOTE,
    }


# ---------------------------------------------------------------------------
# 5. drift.json
# ---------------------------------------------------------------------------

def build_drift(summary_path=DEFAULT_DRIFT_SUMMARY) -> dict:
    """Verbatim copy of the drift rollup (which now carries the tier_b2 yearly series and
    the two a_to_b escalation arms) plus annotations and the one remaining pending slot."""
    path = Path(summary_path)
    return {
        "summary": json.loads(path.read_text(encoding="utf-8")),
        "annotations": [dict(a) for a in DRIFT_ANNOTATIONS],
        "pending_series": [pending_slot(slot, label)
                           for slot, label in PENDING_DRIFT_SERIES],
        "source": _rel(path),
    }


# ---------------------------------------------------------------------------
# 6. calibration.json
# ---------------------------------------------------------------------------

def reliability_bins(p_max, correct, n_bins: int = CALIBRATION_N_BINS) -> list[dict]:
    """Equal-width reliability bins over [0, 1], the repo's ECE binning (metrics.py §38).

    Bin membership is `min(floor(p * n_bins), n_bins - 1)` — lower-closed intervals with
    `p = 1.0` folded into the top bin — so these ARE the bins behind the logged ECE, not a
    lookalike. Empty bins are emitted with `n = 0` and null statistics rather than dropped:
    a missing bin and an empty bin are different facts about a model's confidence.
    """
    p_max = np.asarray(p_max, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    idx = np.minimum((p_max * n_bins).astype(np.int64), n_bins - 1)
    out = []
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        out.append({
            "lo": _round(b / n_bins),
            "hi": _round((b + 1) / n_bins),
            "n": count,
            "conf_mean": _round(float(p_max[mask].mean())) if count else None,
            "acc": _round(float(correct[mask].mean())) if count else None,
        })
    return out


def _ece_from_bins(bins: list[dict], n_total: int) -> float:
    return float(sum((b["n"] / n_total) * abs(b["acc"] - b["conf_mean"])
                     for b in bins if b["n"]))


def calibration_exhibit(key: str, label: str, record: dict, art, *,
                        calibration: str) -> dict:
    correct = np.asarray(art.y_true) == np.asarray(art.y_pred)
    bins = reliability_bins(art.p_max, correct)
    logged = metric_from_record(record, "ece")
    replay = _ece_from_bins(bins, len(art))
    if abs(replay - logged["point"]) > ECE_REPLAY_TOL:
        raise ValueError(
            f"reliability bins for {key} do not reproduce the logged ECE: bins give "
            f"{replay!r}, run {record['run_id'][:8]} logged {logged['point']!r} "
            f"(|delta| > {ECE_REPLAY_TOL:g}); the plotted bins are not the ones behind "
            "the reported number"
        )
    return {
        "key": key,
        "label": label,
        "run_id": record["run_id"],
        "config_name": _config_name(record),
        "slice": _slice_of(record),
        "calibration": calibration,
        "n": len(art),
        "n_bins": CALIBRATION_N_BINS,
        "bins": bins,
        "ece": logged,
        "brier": metric_from_record(record, "brier"),
        "evidence_class": "measured (bins derived from frozen per-example artifact)",
        "source": f"data/preds/{record['run_id']}.parquet",
    }


def build_calibration(resolved: dict, artifacts: dict) -> dict:
    """Tier A raw/isotonic/deployment exhibits plus the four temperature-scaled Tier B ones.

    There is deliberately no "Tier A logreg RAW on TEST-IID" exhibit, because no such run
    exists: every reported TEST-IID final is `calibration: isotonic` (fit on CAL). The raw
    vs isotonic contrast is therefore shown where it was actually measured — on CAL, on the
    two rungs that differ by exactly that one switch — and each exhibit states its own
    slice rather than borrowing the other's. All four Tier B TEST-IID finals ship
    (temperature fit on CAL): B1's three seeds are three exhibits for the same reason the
    frontier carries three points — seed variance is the exhibit.
    """
    raw_cal = record_for(resolved, TIER_A_RAW_CAL, CAL)
    isocal = record_for(resolved, TIER_A_ISOCAL_CAL, CAL)
    test_iid = record_for(resolved, TIER_A_LOGREG_TEST, TEST_IID)
    haiku = record_for(resolved, HAIKU_TEST, TEST_IID)
    sonnet = record_for(resolved, SONNET_TEST, TEST_IID)
    tier_b_exhibits = []
    for config, key, label in TIER_B_TESTS:
        record = record_for(resolved, config, TEST_IID)
        tier_b_exhibits.append(calibration_exhibit(
            key, f"{label} — temperature-scaled (TEST-IID)",
            record, artifacts[record["run_id"]], calibration="temperature"))
    return {
        "exhibits": [
            calibration_exhibit(
                "tier_a_logreg_raw", "Tier A LogReg — raw probabilities (CAL)",
                raw_cal, artifacts[raw_cal["run_id"]], calibration="raw"),
            calibration_exhibit(
                "tier_a_logreg_isotonic", "Tier A LogReg — isotonic (CAL, in-sample)",
                isocal, artifacts[isocal["run_id"]], calibration="isotonic"),
            calibration_exhibit(
                "tier_a_logreg_test_iid",
                "Tier A LogReg — isotonic, deployment point (TEST-IID)",
                test_iid, artifacts[test_iid["run_id"]], calibration="isotonic"),
            *tier_b_exhibits,
        ],
        "no_raw_test_iid_note": (
            "No raw-probability TEST-IID exhibit exists: every reported TEST-IID final is "
            "calibration: isotonic (fit on CAL, disjoint from TEST-IID). The raw-vs-"
            "isotonic contrast is shown on CAL, where the two rungs differ by exactly that "
            "one config switch."
        ),
        "isotonic_in_sample_note": threshold_opt.ISOCAL_IN_SAMPLE_NOTE,
        "tier_c_note": {
            "text": TIER_C_CALIBRATION_NOTE,
            "run_ids": [haiku["run_id"], sonnet["run_id"]],
        },
        "pending": [],
    }


# ---------------------------------------------------------------------------
# 7 + 8. curated_ids.json + samples.json
# ---------------------------------------------------------------------------

def receipt_ids(record: dict) -> list[int]:
    """Sorted complaint ids present in a Tier C run's committed per-call receipts."""
    raw_log_path = (record.get("extra") or {}).get("raw_log_path")
    if not raw_log_path:
        raise ValueError(f"run {record['run_id'][:8]} logs no extra.raw_log_path")
    return sorted(int(cid) for cid in predictions.load_receipts_by_id(raw_log_path))


def paired_pool(haiku_record: dict, sonnet_record: dict) -> list[int]:
    """Ids scored by BOTH TEST-IID finals — the only rows with a full tier-by-tier row."""
    pool = sorted(set(receipt_ids(haiku_record)) & set(receipt_ids(sonnet_record)))
    if not pool:
        raise ValueError("Haiku and Sonnet TEST-IID receipts share no complaint_id")
    return pool


def stratified_allocation(class_counts: dict[str, int], n: int) -> dict[str, int]:
    """Proportional allocation with largest-remainder rounding and >= 1 per class.

    Largest remainder (Hamilton) rather than independent rounding, so the parts sum to `n`
    exactly without a fudge row. The min-1 floor is applied FIRST and the remainder is
    distributed over what is left, so a rare class cannot be rounded out of the demo — the
    tail classes are precisely the ones a triage reviewer needs to see. Ties in the
    remainder break on class name ascending, so the allocation does not depend on dict
    order.
    """
    classes = sorted(class_counts)
    total = sum(class_counts[c] for c in classes)
    if n > total:
        raise ValueError(f"cannot draw {n} rows from a pool of {total}")
    if n < len(classes):
        raise ValueError(f"cannot give each of {len(classes)} classes >=1 row out of {n}")

    quota = {c: n * class_counts[c] / total for c in classes}
    alloc = {c: min(max(1, int(quota[c])), class_counts[c]) for c in classes}
    remaining = n - sum(alloc.values())
    while remaining > 0:
        eligible = [c for c in classes if alloc[c] < class_counts[c]]
        if not eligible:
            raise ValueError("pool exhausted before the allocation was filled")
        eligible.sort(key=lambda c: (-(quota[c] - int(quota[c])), c))
        for c in eligible[:remaining]:
            alloc[c] += 1
        remaining = n - sum(alloc.values())
    while remaining < 0:  # only reachable if min-1 floors overshoot n
        donors = sorted((c for c in classes if alloc[c] > 1),
                        key=lambda c: (quota[c] - int(quota[c]), c))
        if not donors:
            raise ValueError("cannot shrink the allocation without dropping a class")
        for c in donors[:-remaining]:
            alloc[c] -= 1
        remaining = n - sum(alloc.values())
    return alloc


def select_curated_ids(pool: list[int], y_true_by_id: dict[int, str], *,
                       n: int = CURATED_N, seed: int = CURATED_SEED) -> list[int]:
    """The frozen n=200 curated draw. Deterministic given (pool, labels, n, seed).

    `RandomState` (not `default_rng`): its `choice` stream is guaranteed stable across
    NumPy versions, which is what "frozen" has to mean for a committed selection. Ids are
    sorted before the draw and classes are visited in sorted order, so nothing depends on
    the order the pool was assembled in.
    """
    pool = sorted(int(cid) for cid in pool)
    by_class: dict[str, list[int]] = {}
    for cid in pool:
        by_class.setdefault(y_true_by_id[cid], []).append(cid)
    alloc = stratified_allocation({c: len(ids) for c, ids in by_class.items()}, n)

    rng = np.random.RandomState(seed)
    chosen: list[int] = []
    for label in sorted(by_class):
        candidates = np.asarray(sorted(by_class[label]), dtype=np.int64)
        picked = rng.choice(candidates, size=alloc[label], replace=False)
        chosen.extend(int(v) for v in picked)
    return sorted(chosen)


def build_curated_ids(pool: list[int], y_true_by_id: dict[int, str]) -> dict:
    """The committed selection record. `pool_sha256` makes the freeze checkable without data/.

    The draw itself needs `y_true`, which lives only in the gitignored split parquet, so a
    receipts-only check cannot replay it. Hashing the canonical pool id list closes that
    gap: CI can prove the committed selection was drawn from exactly the pool the committed
    receipts still describe, and the full replay runs wherever `data/` exists.
    """
    ids = select_curated_ids(pool, y_true_by_id)
    canonical = json.dumps(pool, separators=(",", ":")).encode("utf-8")
    return {
        "version": CURATED_VERSION,
        "seed": CURATED_SEED,
        "method": CURATED_METHOD,
        "pool": CURATED_POOL,
        "pool_n": len(pool),
        "pool_sha256": hashlib.sha256(canonical).hexdigest(),
        "n": len(ids),
        "complaint_ids": ids,
    }


def freeze_check(obj: dict, path: Path) -> dict:
    """Regenerate-and-compare: a committed selection may never change (CLAUDE.md rule 2)."""
    path = Path(path)
    if not path.exists():
        return obj
    committed = path.read_text(encoding="utf-8")
    regenerated = json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    if committed != regenerated:
        raise ValueError(
            f"FROZEN SELECTION CHANGED: regenerating {path} does not reproduce the "
            "committed file. A curated set is frozen the moment it is committed (CLAUDE.md "
            "rule 2) — investigate what moved (receipts, split, selection logic) rather "
            "than deleting the file"
        )
    return obj


def load_split_rows(complaint_ids, split: str, splits_dir=DEFAULT_SPLITS_DIR) -> dict:
    """complaint_id -> (narrative, class) from the frozen split parquet, ids required."""
    path = Path(splits_dir) / f"{split}.parquet"
    if not path.exists():
        raise ValueError(f"missing frozen split {path}; run `make data` first")
    wanted = [int(c) for c in complaint_ids]
    con = duckdb.connect()
    try:
        con.execute("SET threads=1")
        con.register("_wanted", {"complaint_id": np.asarray(wanted, dtype=np.int64)})
        rows = con.execute(
            "SELECT s.complaint_id, s.narrative, s.class "
            f"FROM read_parquet('{path.as_posix()}') s "
            "JOIN _wanted w ON s.complaint_id = w.complaint_id"
        ).fetchall()
    finally:
        con.close()
    out = {int(cid): (narrative, label) for cid, narrative, label in rows}
    missing = [cid for cid in wanted if cid not in out]
    if missing:
        raise ValueError(
            f"{len(missing)} curated id(s) absent from {path.name}: "
            f"{missing[:cost_model.MAX_OFFENDERS_SHOWN]}"
        )
    return out


def tier_c_sample_fields(ids, record: dict, class_labels: list[str]) -> dict[int, dict]:
    """Per-id Tier C demo fields, straight off the run's VERIFIED committed receipts.

    `cost_model.join_receipts` is the same loader the cost model uses: it re-derives every
    price from the receipt's own tokens and the run's frozen pricing snapshot, checks the
    slug, and refuses duplicates. The demo therefore cannot display a dollar figure that
    would not survive the cost gate.
    """
    raw_log_path = (record.get("extra") or {}).get("raw_log_path")
    fallback = (record.get("extra") or {}).get("fallback_label")
    if not fallback:
        raise ValueError(f"run {record['run_id'][:8]} logs no extra.fallback_label")
    receipts = cost_model.join_receipts(ids, raw_log_path, record=record)
    labels = predictions.reconstruct_tier_c_labels(
        ids, {int(r["complaint_id"]): r.get("content") for r in receipts},
        class_labels, fallback)
    out = {}
    for cid, receipt, label in zip(ids, receipts, labels, strict=True):
        out[int(cid)] = {
            "label": label,
            "cost_usd": receipt["computed_cost_usd"],
            "latency_ms": receipt["latency_ms"],
            "provider": receipt["provider"],
            "prompt_tokens": int(receipt["prompt_tokens"]),
            "completion_tokens": int(receipt["completion_tokens"]),
            "parse_failed": bool(receipt["parse_failed"]),
            "run_id": record["run_id"],
        }
    return out


def router_paths(inputs: router_sim.TestInputs, cal_thresholds: dict) -> tuple[dict, float]:
    """complaint_id -> the HEADLINE router's path for that row, plus the tau it used.

    The headline policy (a_to_b, owner decision 2026-08-12) is built by
    `router_sim.build_full_policies`, so the op semantics are the Phase 4 ones by
    construction. The A-gate outcome is then recomputed with the router's own expression
    (`p_max >= tau`) and cross-checked against the policy's full label vector — for a
    B2-terminal cascade every row is answered by machine, so the strongest replay check is
    that `np.where(answered, A's label, B2's label)` reproduces `policy.y_pred` exactly.
    If it does not, this module has drifted from the op it claims to replay and the build
    fails.
    """
    policies = {p.name: p
                for p in router_sim.build_full_policies(inputs, cal_thresholds)}
    policy = policies[HEADLINE_ROUTER]
    rung = inputs.tier_b_cascade
    if rung is None:
        raise ValueError("the headline a_to_b policy needs a Tier B cascade rung; the "
                         "cost config does not price Tier B")
    tau = float(policy.gate["tau"])
    answered = np.asarray(inputs.art_a.p_max, dtype=np.float64) >= tau
    expected = np.where(answered,
                        np.asarray(inputs.art_a.y_pred, dtype=object),
                        np.asarray(rung.art.y_pred[rung.index_full], dtype=object))
    if policy.to_human.any() or not np.array_equal(policy.y_pred, expected):
        raise ValueError(
            "recomputed a_to_b outcome disagrees with router_sim's policy vectors; the "
            "demo's router path is not the frozen op"
        )
    out = {}
    for cid, is_answered in zip(policy.ids, answered, strict=True):
        out[int(cid)] = (["A", "answered"] if is_answered
                         else ["A", "escalated", "B2", "answered"])
    return out, tau


def _tier_b_sample_fields(ids, rung: router_sim.TierBRung, record: dict) -> dict[int, dict]:
    """Per-id Tier B demo fields from the rung's already-verified frozen artifact."""
    index = threshold_opt.restrict_to_ids(rung.art, ids)
    out = {}
    for cid, pos in zip(ids, index, strict=True):
        label = str(rung.art.y_pred[pos])
        out[int(cid)] = {
            "label": label,
            "p_max": float(rung.art.p_max[pos]),
            "correct": label == str(rung.art.y_true[pos]),
            "run_id": record["run_id"],
        }
    return out


def _tier_b_rung(inputs: router_sim.TestInputs, config_name: str) -> router_sim.TierBRung:
    for rung in inputs.tier_b:
        if rung.config_name == config_name:
            return rung
    raise ValueError(f"no Tier B rung loaded for config {config_name!r}; the cost config "
                     "must price Tier B")


def build_samples(curated: dict, *, resolved: dict, inputs: router_sim.TestInputs,
                  cal_thresholds: dict, splits_dir=DEFAULT_SPLITS_DIR) -> dict:
    ids = list(curated["complaint_ids"])
    logreg = record_for(resolved, TIER_A_LOGREG_TEST, TEST_IID)
    haiku = record_for(resolved, HAIKU_TEST, TEST_IID)
    sonnet = record_for(resolved, SONNET_TEST, TEST_IID)
    b1_record = record_for(resolved, TIER_B1_SAMPLE_CONFIG, TEST_IID)
    b2_record = record_for(resolved, TIER_B2_SAMPLE_CONFIG, TEST_IID)

    class_labels = list(inputs.art_a.class_labels)
    split_rows = load_split_rows(ids, TEST_IID, splits_dir)
    index_a = threshold_opt.restrict_to_ids(inputs.art_a, ids)
    haiku_fields = tier_c_sample_fields(ids, haiku, class_labels)
    sonnet_fields = tier_c_sample_fields(ids, sonnet, class_labels)
    b1_fields = _tier_b_sample_fields(ids, _tier_b_rung(inputs, TIER_B1_SAMPLE_CONFIG),
                                      b1_record)
    b2_fields = _tier_b_sample_fields(ids, _tier_b_rung(inputs, TIER_B2_SAMPLE_CONFIG),
                                      b2_record)
    paths, tau = router_paths(inputs, cal_thresholds)

    samples = []
    for cid, pos in zip(ids, index_a, strict=True):
        narrative, y_true = split_rows[cid]
        if str(inputs.art_a.y_true[pos]) != y_true:
            raise ValueError(
                f"complaint {cid}: the frozen split says y_true={y_true!r} but the Tier A "
                f"artifact says {inputs.art_a.y_true[pos]!r}"
            )
        a_label = str(inputs.art_a.y_pred[pos])
        haiku_row = haiku_fields[cid]
        sonnet_row = sonnet_fields[cid]
        samples.append({
            "complaint_id": cid,
            "narrative": narrative,
            "y_true": y_true,
            "tiers": {
                "tier_a_logreg": {
                    "label": a_label,
                    "p_max": float(inputs.art_a.p_max[pos]),
                    "correct": a_label == y_true,
                    "run_id": logreg["run_id"],
                },
                "tier_b1": b1_fields[cid],
                "tier_b2": b2_fields[cid],
                "haiku": {**haiku_row, "correct": haiku_row["label"] == y_true},
                "sonnet": {**sonnet_row, "correct": sonnet_row["label"] == y_true},
            },
            "router": {
                "op_version": OP_VERSION,
                "policy": HEADLINE_ROUTER,
                "tau": tau,
                "path": paths[cid],
                "note": ROUTER_PATH_NOTE,
            },
        })
    return {
        "selection": {**curated, "narrative_source": NARRATIVE_SOURCE},
        "samples": samples,
        "class_labels": class_labels,
        "router_note": HEADLINE_NOTE,
        "tier_b1_note": (
            "The tier_b1 card shows seed sa (20260805, the first of the frozen seed "
            "list); seeds sb and sc are separate logged runs with their own frontier and "
            "calibration exhibits."
        ),
        "parse_failure_note": (
            "A row with parse_failed=true carries the run's frozen FALLBACK label "
            f"({(haiku.get('extra') or {}).get('fallback_label')!r}); `correct` scores "
            "that fallback. The headline a_to_b router never consults Tier C; the "
            "parse-fail→human arm lives in the LLM-cascade contrast policies (panel 3)."
        ),
        "evidence_class": "measured",
    }


# ---------------------------------------------------------------------------
# 9. receipts.json
# ---------------------------------------------------------------------------

def aggregate_receipts(record: dict) -> dict:
    """Per-run receipt rollup, recomputed from the committed calls.jsonl.

    The run-level total is cross-checked against the record's logged `cost_usd` at the
    cost model's own tolerance: the receipts drawer exists to let a reader audit a headline
    cost, so a drawer that disagrees with the log is worse than no drawer.
    """
    extra = record.get("extra") or {}
    raw_log_path = extra["raw_log_path"]
    prompt_rate, completion_rate = cost_model.pricing_rates(record)
    by_id = cost_model.load_receipt_records(
        raw_log_path, model_slug=extra["model_slug"],
        prompt_rate=prompt_rate, completion_rate=completion_rate)
    receipts = [by_id[cid] for cid in sorted(by_id)]

    total = float(np.asarray([r["computed_cost_usd"] for r in receipts],
                             dtype=np.float64).sum())
    logged = record.get("cost_usd")
    if logged is not None and abs(total - float(logged)) > cost_model.COST_SUM_TOL:
        raise ValueError(
            f"receipt total ${total:.6f} for run {record['run_id'][:8]} disagrees with the "
            f"logged cost_usd ${float(logged):.6f} (tol {cost_model.COST_SUM_TOL:g})"
        )
    providers = Counter(str(r.get("provider")) for r in receipts)
    return {
        "run_id": record["run_id"],
        "config_name": _config_name(record),
        "model": extra["model_slug"],
        "raw_log_path": raw_log_path,
        "receipts_sha256": cost_model.receipts_sha256(raw_log_path),
        "n_calls": len(receipts),
        "total_cost_usd": total,
        "logged_cost_usd": logged,
        "provider_mix": dict(sorted(providers.items())),
        "token_totals": {
            "prompt": int(sum(int(r["prompt_tokens"]) for r in receipts)),
            "completion": int(sum(int(r["completion_tokens"]) for r in receipts)),
        },
        "parse_failures": int(sum(1 for r in receipts if r["parse_failed"])),
        "evidence_class": "measured",
    }


def build_receipts(records: list[dict]) -> dict:
    """Every Tier C run that logged receipts, smoke runs included (they are in the log)."""
    runs = {}
    for record in records:
        extra = record.get("extra") or {}
        if not extra.get("raw_log_path"):
            continue
        runs[record["run_id"]] = aggregate_receipts(record)
    if not runs:
        raise ValueError("no Tier C run in the results log carries extra.raw_log_path")
    return {"runs": runs, "repro": dict(RECEIPTS_REPRO)}


# ---------------------------------------------------------------------------
# 10. case_study.json
# ---------------------------------------------------------------------------
#
# The case study is the one page where prose and numbers sit in the same sentence, which
# is exactly where an unattributed figure is cheapest to write and most expensive to be
# wrong about. So the page is built like every other exhibit: the paragraphs are templates
# in this module, each numeric DISPLAY is formatted from a value COPIED out of a run record
# or a committed derived artifact, and the section declares that value with its unit,
# source path, run ids, repro command and evidence class. A handful of entries are DERIVED
# (a ratio or a relative change over two copied values); they carry `basis: "derived"` and
# a formula in `note`, and the tests recompute them from the sources independently.
#
# Numbers the spec asked for that no committed artifact carries are NOT on the page. The
# paired Sonnet-vs-Haiku deltas and the few-shot ablation delta come from
# `triage_lab.tier_c_compare`, which prints and writes nothing; they live in
# EXPERIMENT_LOG.md with their repro commands, and each affected section says so in
# `gaps` rather than quoting a number this page cannot check.

CASE_STUDY_SCHEMA = "case-study-v1"

CASE_STUDY_TITLE = ("Case study — what was measured, how it was verified, "
                    "and what it does not prove")

CASE_STUDY_SOURCE_NOTE = (
    "Every numeric token in every paragraph below is declared in that section's `numbers` "
    "array and copied at build time from `results/runs.jsonl` or a committed derived "
    "artifact. Entries marked `derived` are arithmetic over copied values with no free "
    "parameters and carry their formula. The build fails if a paragraph contains a number "
    "no entry declares."
)

# The one number on the page with no results/ artifact behind it: a pytest run is not a
# logged experiment. It is recorded here as an explicit constant, stamped with the command
# that produced it, rather than written into the prose where nothing could check it.
SUITE_RESULT = {
    "passed": 675,
    "skipped": 1,
    "failed": 0,
    "command": "uv run pytest -q",
    "note": ("measured in this working tree at the git SHA meta.json records; a test run "
             "is not a logged experiment, so this is a declared constant in "
             "src/triage_lab/demo_build.py, not a copy from results/"),
}

# No slots left: `provenance_seeds` shipped 2026-08-13 and `reproduce_headline` shipped
# 2026-08-13 as verification item 11 (`make reproduce-headline`, driven by
# `src/triage_lab/reproduce_headline.py`). The tuple stays — a page that can never declare
# something pending would quietly turn "not measured" into "measured absent" the next time
# a slot is needed.
CASE_STUDY_PENDING: tuple[dict, ...] = ()


# --- display formatting (display strings are FORMATTED from the copied value) ------------

def _fmt_f(value, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def _fmt_sf(value, digits: int = 4) -> str:
    return f"{float(value):+.{digits}f}"


def _fmt_ci(metric: dict, digits: int = 4, signed: bool = False) -> str:
    fmt = _fmt_sf if signed else _fmt_f
    return (f"{fmt(metric['point'], digits)} [{fmt(metric['ci_lo'], digits)}, "
            f"{fmt(metric['ci_hi'], digits)}]")


def _fmt_usd(value, digits: int = 2, signed: bool = False) -> str:
    v = float(value)
    if v < 0:
        return f"-${abs(v):,.{digits}f}"
    return f"+${v:,.{digits}f}" if signed else f"${v:,.{digits}f}"


def _fmt_usd_ci(metric: dict, digits: int = 2, signed: bool = False) -> str:
    return (f"{_fmt_usd(metric['point'], digits, signed)} "
            f"[{_fmt_usd(metric['ci_lo'], digits, signed)}, "
            f"{_fmt_usd(metric['ci_hi'], digits, signed)}]")


def _fmt_pct(value, digits: int = 1) -> str:
    return f"{float(value) * 100:.{digits}f}%"


def _fmt_count(value) -> str:
    return f"{int(value):,d}"


def _fmt_pvalue(value) -> str:
    """Two decimals where a p-value is readable that way, one significant figure below.

    `1.0e+00` for a p of exactly 1 is technically correct and reads like a bug; `1.00` says
    "the discordant counts are as balanced as they get", which is the actual finding.
    """
    p = float(value)
    return f"{p:.2f}" if p >= 0.001 else f"{p:.1e}"


def tier_c_compare_artifact(key: str, compare_dir=DEFAULT_TIER_C_COMPARE_DIR) -> dict:
    """One paired Tier C comparison, with its own recorded repro command.

    Hard-fails on a missing file rather than degrading to a pending slot: these three
    comparisons are load-bearing sentences on the page, and a page that silently drops
    "statistically tied on IID" is telling a different story than the one that was
    measured.
    """
    path = Path(compare_dir) / f"{key}.json"
    if not path.exists():
        raise ValueError(
            f"missing Tier C paired-comparison artifact {path}; regenerate it with "
            "`uv run python -m triage_lab.tier_c_compare <A> <B> --out <path>` (see the "
            "artifact's own repro_command field)")
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact["key"] != key:
        raise ValueError(f"{path.name} declares key {artifact['key']!r}, not {key!r}")
    return {**artifact, "path": _rel(path)}


def _tier_c_delta_numbers(artifact: dict, prefix: str, metric: str = "macro_f1") -> list:
    """The (delta, McNemar p, n) trio a paired Tier C sentence needs, all copied."""
    band = ci_block_from_delta(artifact["deltas"][metric])
    source = artifact["path"]
    repro = artifact["repro_command"]
    run_ids = [arm["run_id"] for arm in (artifact["arm_a"], artifact["arm_b"])
               if arm["run_id"]]
    return [
        _cs_number(f"{prefix}_delta", _fmt_ci(band, signed=True), band, unit="raw",
                   source=source, run_ids=run_ids, repro=repro,
                   note=f"paired {metric} delta, {artifact['comparison']}, "
                        f"slice {artifact['split']}, pairing {artifact['pairing']}"),
        _cs_number(f"{prefix}_p", _fmt_pvalue(artifact["mcnemar"]["p_value"]),
                   artifact["mcnemar"]["p_value"], unit="pvalue", source=source,
                   run_ids=run_ids, repro=repro),
        _cs_number(f"{prefix}_n", _fmt_count(artifact["n_examples"]),
                   artifact["n_examples"], unit="count", source=source, run_ids=run_ids,
                   repro=repro),
    ]


def _cs_number(label: str, display: str, value, *, unit: str, source: str,
               run_ids=(), repro: str | None = None, basis: str = "copied",
               evidence_class: str = "measured", note: str | None = None) -> dict:
    """One declared number: what it says, what it is, and where it came from.

    `unit` is what makes the display checkable without re-implementing the formatter in
    the test: it says how to read the digits in `display` back into `value`
    (raw / usd / pct / count / pvalue).
    """
    entry = {
        "label": label,
        "display": display,
        "value": value,
        "unit": unit,
        "basis": basis,
        "evidence_class": evidence_class,
        "run_ids": list(run_ids),
        "source": source,
    }
    if repro is not None:
        entry["repro"] = repro
    if note is not None:
        entry["note"] = note
    return entry


def _cs_section(section_id: str, kind: str, title: str, *, paragraphs=(), numbers=(),
                repro=(), items=(), gaps=(), pending=None, lineage=None,
                caveats=None) -> dict:
    """A page section. `run_ids` is the ordered union of its numbers' ids (the chips)."""
    run_ids: list[str] = []
    for entry in numbers:
        for run_id in entry["run_ids"]:
            if run_id not in run_ids:
                run_ids.append(run_id)
    section = {
        "id": section_id,
        "kind": kind,
        "title": title,
        "paragraphs": list(paragraphs),
        "numbers": list(numbers),
        "repro": list(repro),
        "run_ids": run_ids,
        "items": list(items),
        "gaps": list(gaps),
    }
    if pending is not None:
        section["pending"] = pending
    # kind-specific keys, emitted only where they mean something (provenance today), so a
    # consumer never sees an empty `lineage` on a section that could not have one.
    if lineage is not None:
        section["lineage"] = list(lineage)
    if caveats is not None:
        section["caveats"] = list(caveats)
    return section


# --- artifact accessors (one lookup rule each; ambiguity is a hard failure) --------------

def _one(rows: list[dict], where: dict, what: str) -> dict:
    matches = [r for r in rows if all(r.get(k) == v for k, v in where.items())]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {what} row for {where}, found "
                         f"{len(matches)}")
    return matches[0]


def _drift_logged(summary: dict, tier: str, slice_name: str) -> dict:
    return _one(summary["series"]["logged"], {"tier": tier, "slice": slice_name},
                "drift logged")


def _drift_escalation(summary: dict, policy: str, slice_name: str,
                      dataset: str = "full_cal") -> dict:
    return _one(summary["series"]["escalation"],
                {"policy": policy, "slice": slice_name, "dataset": dataset},
                "drift escalation")


def _prior_shift(rows: list[dict], tier: str, year: str, component: str,
                 scope: str = "native") -> dict:
    """The `pi_source: artifact` row — the slice's OWN class mix, not the full-slice one.

    Every component ships twice, once per class-prior source; the artifact row is the
    primary and the full_slice row is the sensitivity companion. Selecting on the key
    rather than on `is_primary` is deliberate: `share_prior` is a real published row that
    is not flagged primary, and a filter that quietly dropped it would be a silent
    difference between what the analysis publishes and what the page can cite.
    """
    row = _one(rows, {"tier": tier, "year": year, "component": component, "scope": scope,
                      "pi_source": "artifact"}, "prior-shift")
    return ci_block(row)


def _perturbation(summary: dict, arm: str, family: str, rate: float = 0.1) -> dict:
    return _one(summary["rows"], {"arm": arm, "family": family, "rate": rate},
                "perturbation")


def _oov(summary: dict, slice_name: str, metric: str) -> dict:
    return _one(summary["rows"], {"slice": slice_name, "metric": metric}, "oov")


def _frontier_claim(doc: dict, router: str, baseline: str, evaluation_set: str) -> dict:
    return _one(doc["claims"], {"router": router, "baseline": baseline,
                                "evaluation_set": evaluation_set}, "frontier claim")


def _tier_b_comparison(doc: dict, a: str, b: str) -> dict:
    return _one(doc["comparisons"], {"a": a, "b": b}, "tier_b_compare")


def _verified_receipts(record: dict) -> dict[int, dict]:
    """The run's committed per-call receipts, through the cost model's verifying loader."""
    extra = record["extra"]
    prompt_rate, completion_rate = cost_model.pricing_rates(record)
    return cost_model.load_receipt_records(
        extra["raw_log_path"], model_slug=extra["model_slug"],
        prompt_rate=prompt_rate, completion_rate=completion_rate)


def prompt_token_inflation(clean_record: dict, perturbed_record: dict) -> float:
    """(perturbed - clean) / clean prompt tokens on the rows the perturbed run scored.

    DERIVED, not a new measurement: both sides are committed receipts. The clean side is
    restricted to the perturbed run's own complaint ids, which is a pairing rather than a
    join — the perturbed 1,500 are a byte-identical subset of the clean 5,000 under the
    same cap_seed (results/perturbation/summary.json, methods_notes.tier_c_join).
    """
    clean = _verified_receipts(clean_record)
    perturbed = _verified_receipts(perturbed_record)
    ids = sorted(perturbed)
    missing = [cid for cid in ids if cid not in clean]
    if missing:
        raise ValueError(
            f"{len(missing)} perturbed complaint id(s) absent from the clean run's "
            f"receipts: {missing[:cost_model.MAX_OFFENDERS_SHOWN]}")
    clean_tokens = sum(int(clean[cid]["prompt_tokens"]) for cid in ids)
    perturbed_tokens = sum(int(perturbed[cid]["prompt_tokens"]) for cid in ids)
    return perturbed_tokens / clean_tokens - 1.0


# --- the sections ------------------------------------------------------------------------

def _cs_intro(records: list[dict], drift: dict) -> dict:
    boot = drift["bootstrap"]
    ci_lo_pct, ci_hi_pct = boot["ci_pct"]
    numbers = [
        _cs_number("n_run_records", _fmt_count(len(records)), len(records), unit="count",
                   source=_rel(DEFAULT_RESULTS_PATH), basis="derived",
                   note="len(results/runs.jsonl) at build time",
                   repro="wc -l results/runs.jsonl"),
        _cs_number("ci_level", f"{ci_hi_pct - ci_lo_pct:.0f}%",
                   float(ci_hi_pct - ci_lo_pct), unit="pctpoint",
                   source=_rel(DEFAULT_DRIFT_SUMMARY), basis="derived",
                   note="bootstrap.ci_pct[1] - bootstrap.ci_pct[0]"),
        _cs_number("bootstrap_resamples", _fmt_count(boot["n_resamples"]),
                   boot["n_resamples"], unit="count",
                   source=_rel(DEFAULT_DRIFT_SUMMARY)),
        _cs_number("bootstrap_seed", str(int(boot["seed"])), int(boot["seed"]),
                   unit="count", source=_rel(DEFAULT_DRIFT_SUMMARY)),
    ]
    d = {e["label"]: e["display"] for e in numbers}
    return _cs_section(
        "intro", "narrative", CASE_STUDY_TITLE,
        paragraphs=[
            ("A three-tier consumer-complaint triage lab on the CFPB Consumer Complaint "
             "Database: TF-IDF linear models (Tier A), fine-tuned transformers (Tier B) "
             "and Claude LLMs via OpenRouter (Tier C), combined into a confidence-cascade "
             "router optimized against an explicit business cost model and stress-tested "
             "against measured 2015 to 2026 distribution drift."),
            (f"{d['n_run_records']} append-only run records back this page. Every headline "
             f"number carries a {d['ci_level']} bootstrap confidence interval "
             f"({d['bootstrap_resamples']} resamples, fixed seed {d['bootstrap_seed']}) and "
             "a reproduction command, and traces to results/runs.jsonl or to a committed "
             "derived artifact."),
        ],
        numbers=numbers,
        repro=["uv run python -m triage_lab.demo_build --all"],
    )


def _cs_tiers(resolved: dict, compare: dict, cost_dir, tier_c_compare_dir) -> dict:
    logreg = record_for(resolved, TIER_A_LOGREG_TEST, TEST_IID)
    b1_sa = record_for(resolved, "tier_b1_modernbert_sa", TEST_IID)
    b1_sb = record_for(resolved, "tier_b1_modernbert_sb", TEST_IID)
    b1_sc = record_for(resolved, "tier_b1_modernbert_sc", TEST_IID)
    b2 = record_for(resolved, TIER_B2_SAMPLE_CONFIG, TEST_IID)
    haiku = record_for(resolved, HAIKU_TEST, TEST_IID)
    sonnet = record_for(resolved, SONNET_TEST, TEST_IID)

    b1_vs_a = _tier_b_comparison(compare, "b1_sa", "baseline")
    b1_vs_b2 = _tier_b_comparison(compare, "b1_sa", "b2")

    haiku_cost = json.loads(cost_artifact_path(haiku["run_id"], cost_dir)
                            .read_text(encoding="utf-8"))
    sonnet_cost = json.loads(cost_artifact_path(sonnet["run_id"], cost_dir)
                             .read_text(encoding="utf-8"))
    logreg_cost = json.loads(cost_artifact_path(logreg["run_id"], cost_dir)
                             .read_text(encoding="utf-8"))

    sonnet_vs_haiku = tier_c_compare_artifact(TIER_C_COMPARE_SONNET_IID,
                                              tier_c_compare_dir)
    runs_src = _rel(DEFAULT_RESULTS_PATH)
    compare_src = _rel(DEFAULT_TIER_B_COMPARE)
    compare_repro = "uv run python scripts/compare_tier_b.py"

    def _f1(label, record, repro):
        metric = metric_from_record(record, "macro_f1")
        return _cs_number(label, _fmt_ci(metric), metric, unit="raw", source=runs_src,
                          run_ids=[record["run_id"]], repro=repro)

    harness_repro = ("uv run python -m triage_lab.harness "
                     "configs/tier_a_logreg_test_iid.yaml")
    numbers = [
        _f1("tier_a_macro_f1", logreg, harness_repro),
        _cs_number("n_test_iid", _fmt_count(logreg_cost["n_examples"]),
                   logreg_cost["n_examples"], unit="count",
                   source=f"results/cost_model/{logreg['run_id']}.json",
                   run_ids=[logreg["run_id"]]),
        _cs_number("b1_sa_macro_f1", _fmt_f(compare["provenance"]["b1_sa"]
                                            ["macro_f1_point"]),
                   compare["provenance"]["b1_sa"]["macro_f1_point"], unit="raw",
                   source=compare_src, run_ids=[b1_sa["run_id"]], repro=compare_repro),
        _cs_number("b1_sb_macro_f1", _fmt_f(compare["provenance"]["b1_sb"]
                                            ["macro_f1_point"]),
                   compare["provenance"]["b1_sb"]["macro_f1_point"], unit="raw",
                   source=compare_src, run_ids=[b1_sb["run_id"]], repro=compare_repro),
        _cs_number("b1_sc_macro_f1", _fmt_f(compare["provenance"]["b1_sc"]
                                            ["macro_f1_point"]),
                   compare["provenance"]["b1_sc"]["macro_f1_point"], unit="raw",
                   source=compare_src, run_ids=[b1_sc["run_id"]], repro=compare_repro),
        _cs_number("b1_minus_a_macro_f1",
                   _fmt_ci(ci_block_from_delta(b1_vs_a["macro_f1"]), signed=True),
                   ci_block_from_delta(b1_vs_a["macro_f1"]),
                   unit="raw", source=compare_src,
                   run_ids=[b1_sa["run_id"], logreg["run_id"]], repro=compare_repro),
        _f1("b2_macro_f1", b2, compare_repro),
        _cs_number("b1_minus_b2_macro_f1",
                   _fmt_ci(ci_block_from_delta(b1_vs_b2["macro_f1"]), signed=True),
                   ci_block_from_delta(b1_vs_b2["macro_f1"]),
                   unit="raw", source=compare_src,
                   run_ids=[b1_sa["run_id"], b2["run_id"]], repro=compare_repro),
        _cs_number("b1_minus_b2_mcnemar_p",
                   _fmt_pvalue(b1_vs_b2["mcnemar"]["p_value"]),
                   b1_vs_b2["mcnemar"]["p_value"], unit="pvalue", source=compare_src,
                   run_ids=[b1_sa["run_id"], b2["run_id"]], repro=compare_repro),
        _f1("haiku_macro_f1", haiku,
            "uv run --extra tierc python -m triage_lab.harness "
            "configs/tier_c_haiku_zeroshot_test_iid.yaml"),
        _cs_number("haiku_cost_per_1k",
                   _fmt_usd(haiku_cost["expected_cost_per_1k"]["api"]["point"], 3),
                   haiku_cost["expected_cost_per_1k"]["api"]["point"], unit="usd",
                   source=f"results/cost_model/{haiku['run_id']}.json",
                   run_ids=[haiku["run_id"]], repro="make cost-model"),
        _f1("sonnet_macro_f1", sonnet,
            "uv run --extra tierc python -m triage_lab.harness "
            "configs/tier_c_sonnet_zeroshot_test_iid.yaml"),
        _cs_number("sonnet_cost_per_1k",
                   _fmt_usd(sonnet_cost["expected_cost_per_1k"]["api"]["point"], 3),
                   sonnet_cost["expected_cost_per_1k"]["api"]["point"], unit="usd",
                   source=f"results/cost_model/{sonnet['run_id']}.json",
                   run_ids=[sonnet["run_id"]], repro="make cost-model"),
        _cs_number("n_sonnet_paired", _fmt_count(sonnet_cost["n_examples"]),
                   sonnet_cost["n_examples"], unit="count",
                   source=f"results/cost_model/{sonnet['run_id']}.json",
                   run_ids=[sonnet["run_id"]]),
        *_tier_c_delta_numbers(sonnet_vs_haiku, "sonnet_minus_haiku_iid"),
    ]
    d = {e["label"]: e["display"] for e in numbers}
    return _cs_section(
        "tiers", "narrative", "Three tiers on frozen IID data",
        paragraphs=[
            (f"Tier A is a word+char TF-IDF logistic regression with isotonic calibration: "
             f"macro-F1 {d['tier_a_macro_f1']} on the frozen TEST-IID slice, "
             f"n={d['n_test_iid']}."),
            (f"Fine-tuning buys a certified but modest gain. ModernBERT-base scores "
             f"{d['b1_sa_macro_f1']} / {d['b1_sb_macro_f1']} / {d['b1_sc_macro_f1']} "
             f"macro-F1 across the three frozen seeds, and the paired B1 minus A delta is "
             f"{d['b1_minus_a_macro_f1']} — an interval that excludes zero."),
            (f"The pre-registered surprise was CONFIRMED. DistilBERT — the smaller, cheaper "
             f"deployment point — reaches {d['b2_macro_f1']} and tops every ModernBERT "
             f"seed: the paired B1 minus B2 delta is {d['b1_minus_b2_macro_f1']} with "
             f"McNemar p={d['b1_minus_b2_mcnemar_p']}."),
            (f"Zero-shot LLMs land between the two learned tiers on quality and far above "
             f"both on price. Haiku 4.5 scores {d['haiku_macro_f1']} on TEST-IID at a "
             f"measured {d['haiku_cost_per_1k']}/1k calls; Sonnet 5 scores "
             f"{d['sonnet_macro_f1']} on its paired {d['n_sonnet_paired']}-row subsample at "
             f"{d['sonnet_cost_per_1k']}/1k. Supports differ by point and are not "
             "comparable as populations."),
            (f"On the rows both LLMs scored, the bigger model buys nothing in "
             f"distribution. Paired on the same {d['sonnet_minus_haiku_iid_n']} TEST-IID "
             f"complaints, Sonnet minus Haiku is "
             f"{d['sonnet_minus_haiku_iid_delta']} macro-F1 with McNemar "
             f"p={d['sonnet_minus_haiku_iid_p']} — statistically tied, at nearly three "
             f"times the measured price per call. That tie does not survive drift; the "
             f"next section is where it breaks."),
        ],
        numbers=numbers,
        repro=[harness_repro, compare_repro, "make cost-model",
               sonnet_vs_haiku["repro_command"]],
    )


def _cs_drift(resolved: dict, drift: dict, prior_shift: dict, paired_within: dict,
              tier_c_compare_dir) -> dict:
    summary = drift
    sonnet_pc = tier_c_compare_artifact(TIER_C_COMPARE_SONNET_POSTCUTOFF,
                                        tier_c_compare_dir)
    rows = prior_shift["rows"]
    drift_src = _rel(DEFAULT_DRIFT_SUMMARY)
    ps_src = _rel(DEFAULT_PRIOR_SHIFT_SUMMARY)
    pw_src = _rel(DEFAULT_PAIRED_WITHIN)
    drift_repro = "make drift-charts"
    ps_repro = "make prior-shift"
    pw_repro = "make prior-shift-paired"

    a_2023 = _drift_logged(summary, "tier_a", "test_drift_2023")
    a_2024 = _drift_logged(summary, "tier_a", "test_drift_2024")
    a_2025 = _drift_logged(summary, "tier_a", "test_drift_2025")
    a_2026 = _drift_logged(summary, "tier_a", "test_drift_2026h1")
    b2_2023 = _drift_logged(summary, "tier_b2", "test_drift_2023")
    b2_2026 = _drift_logged(summary, "tier_b2", "test_drift_2026h1")
    haiku_2026 = _drift_logged(summary, "tier_c_haiku", "test_drift_2026h1")
    sonnet_2026 = _drift_logged(summary, "tier_c_sonnet", "test_drift_2026h1")

    def _yearly(label, row, digits=4):
        return _cs_number(label, _fmt_f(row["macro_f1"]["point"], digits),
                          row["macro_f1"]["point"], unit="raw", source=drift_src,
                          run_ids=[row["run_id"]], repro=drift_repro)

    def _yearly_ci(label, row):
        return _cs_number(label, _fmt_ci(ci_block(row["macro_f1"])),
                          ci_block(row["macro_f1"]), unit="raw", source=drift_src,
                          run_ids=[row["run_id"]], repro=drift_repro)

    def _component(label, tier, component, run_ids):
        metric = _prior_shift(rows, tier, "2026h1", component)
        return _cs_number(label, _fmt_ci(metric, signed=True), metric, unit="raw",
                          source=ps_src, run_ids=run_ids, repro=ps_repro)

    a_runs = [a_2023["run_id"], a_2026["run_id"]]
    b2_runs = [b2_2023["run_id"], b2_2026["run_id"]]
    haiku_runs = [_drift_logged(summary, "tier_c_haiku", "test_drift_2023")["run_id"],
                  haiku_2026["run_id"]]

    cr_a_ref = record_for(resolved, "tier_a_logreg_test_drift_2023",
                          "test_drift_2023")
    cr_a_year = record_for(resolved, "tier_a_logreg_test_drift_2026h1",
                           "test_drift_2026h1")
    cr_b2_ref = record_for(resolved, "tier_b2_distilbert_s0_test_drift_2023",
                           "test_drift_2023")
    cr_b2_year = record_for(resolved, "tier_b2_distilbert_s0_test_drift_2026h1",
                            "test_drift_2026h1")
    runs_src = _rel(DEFAULT_RESULTS_PATH)

    def _credit(label, record):
        metric = metric_from_record(record, "f1::credit_reporting")
        return _cs_number(label, _fmt_f(metric["point"]), metric["point"], unit="raw",
                          source=runs_src, run_ids=[record["run_id"]])

    share_a = _prior_shift(rows, "tier_a", "2026h1", "share_prior")
    share_b2 = _prior_shift(rows, "tier_b2", "2026h1", "share_prior")
    pw_delta = paired_within["delta"]

    numbers = [
        _yearly("a_2023", a_2023), _yearly("a_2024", a_2024), _yearly("a_2025", a_2025),
        _yearly_ci("a_2026", a_2026),
        _cs_number("n_drift_slice",
                   _fmt_count(summary["slice_sizes"]["test_drift_2026h1"]),
                   summary["slice_sizes"]["test_drift_2026h1"], unit="count",
                   source=drift_src, repro=drift_repro),
        _component("a_total", "tier_a", "total", a_runs),
        _component("a_prior", "tier_a", "prior::path_p", a_runs),
        _component("a_within", "tier_a", "within::path_p", a_runs),
        _credit("a_credit_2023", cr_a_ref), _credit("a_credit_2026", cr_a_year),
        _yearly("b2_2023", b2_2023), _yearly_ci("b2_2026_ci", b2_2026),
        _component("b2_total", "tier_b2", "total", b2_runs),
        _component("b2_prior", "tier_b2", "prior::path_p", b2_runs),
        _component("b2_within", "tier_b2", "within::path_p", b2_runs),
        _cs_number("b2_share_prior", _fmt_ci(share_b2), share_b2, unit="raw",
                   source=ps_src, run_ids=b2_runs, repro=ps_repro),
        _cs_number("a_share_prior", _fmt_ci(share_a), share_a, unit="raw",
                   source=ps_src, run_ids=a_runs, repro=ps_repro),
        _credit("b2_credit_2023", cr_b2_ref), _credit("b2_credit_2026", cr_b2_year),
        _cs_number("paired_within_delta",
                   _fmt_ci(ci_block_from_delta(pw_delta), signed=True),
                   ci_block_from_delta(pw_delta),
                   unit="raw", source=pw_src, run_ids=[*a_runs, *b2_runs],
                   repro=pw_repro,
                   note="paired component delta: within::path_p(tier_a) − "
                        "within::path_p(tier_b2), identical bootstrap draws"),
        _cs_number("paired_within_path_q",
                   _fmt_ci(ci_block_from_delta(
                       paired_within["delta_sensitivity"]["within::path_q"]), signed=True),
                   ci_block_from_delta(
                       paired_within["delta_sensitivity"]["within::path_q"]),
                   unit="raw", source=pw_src, run_ids=[*a_runs, *b2_runs], repro=pw_repro,
                   note="decomposition-path sensitivity; cited as robustness, not in prose"),
        _cs_number("paired_within_shapley",
                   _fmt_ci(ci_block_from_delta(
                       paired_within["delta_sensitivity"]["within::shapley"]), signed=True),
                   ci_block_from_delta(
                       paired_within["delta_sensitivity"]["within::shapley"]),
                   unit="raw", source=pw_src, run_ids=[*a_runs, *b2_runs], repro=pw_repro,
                   note="decomposition-path sensitivity; cited as robustness, not in prose"),
        _component("haiku_prior", "tier_c_haiku", "prior::path_p", haiku_runs),
        _component("haiku_within", "tier_c_haiku", "within::path_p", haiku_runs),
        _yearly("haiku_2026", haiku_2026), _yearly("sonnet_2026", sonnet_2026),
        _yearly("a_2026_point", a_2026), _yearly("b2_2026_point", b2_2026),
        *_tier_c_delta_numbers(sonnet_pc, "sonnet_minus_haiku_pc"),
    ]
    d = {e["label"]: e["display"] for e in numbers}
    return _cs_section(
        "drift", "narrative", "Drift: one cliff, three different autopsies",
        paragraphs=[
            (f"Tier A walks down and then falls off a cliff. Its yearly macro-F1 goes "
             f"{d['a_2023']} to {d['a_2024']} to {d['a_2025']} to {d['a_2026']} from 2023 "
             f"to 2026-H1, on frozen slices of n={d['n_drift_slice']} each."),
            (f"It dies of both wounds. Against the 2023 reference the loss decomposes as "
             f"total {d['a_total']} = prior shift {d['a_prior']} plus within-class "
             f"degradation {d['a_within']}. The credit_reporting F1 goes "
             f"{d['a_credit_2023']} to {d['a_credit_2026']}."),
            (f"Tier B2 has the same shape at a smaller magnitude: {d['b2_2023']} to "
             f"{d['b2_2026_ci']} yearly; total {d['b2_total']} = prior {d['b2_prior']} + "
             f"within {d['b2_within']}; the prior share is {d['b2_share_prior']} against "
             f"Tier A's {d['a_share_prior']}, intervals that overlap. Its credit_reporting "
             f"F1 goes {d['b2_credit_2023']} to {d['b2_credit_2026']}. The pre-registered "
             "“B2 behaves like an LLM under drift” hypothesis is REFUTED: "
             "fine-tuning bought loss magnitude, not loss shape."),
            (f"One comparison is certified rather than directional. B2's within-class "
             f"damage is smaller than Tier A's: the paired within_A minus within_B2 delta "
             f"is {d['paired_within_delta']} on the same n={d['n_drift_slice']} rows with "
             f"identical bootstrap draws, and the interval excludes zero (robust across "
             f"decomposition paths). The certification covers evaluation-sample uncertainty "
             f"only — Tier B2 is a single fine-tune from one seed, so this is a statement "
             f"about these two fitted systems, not about the model families."),
            (f"The LLMs pay only the prior penalty. Haiku's prior component is "
             f"{d['haiku_prior']} while its within-class component is {d['haiku_within']}, "
             f"an interval containing zero. At the cliff the four points are Tier A "
             f"{d['a_2026_point']}, Tier B2 {d['b2_2026_point']}, Haiku {d['haiku_2026']} "
             f"and Sonnet {d['sonnet_2026']}."),
            (f"That last gap is where the two LLMs stop being interchangeable, and the "
             f"paired evidence for it comes from a different frozen slice — "
             f"TEST-POSTCUTOFF, whose boundary sits strictly after both models' training "
             f"cutoffs. On the same {d['sonnet_minus_haiku_pc_n']} rows there, Sonnet "
             f"minus Haiku is {d['sonnet_minus_haiku_pc_delta']} macro-F1 with McNemar "
             f"p={d['sonnet_minus_haiku_pc_p']}, after the two were statistically tied on "
             f"TEST-IID. The capability gap only shows up off the training distribution: "
             f"in distribution you are paying for it and not collecting it. (The 2026-H1 "
             f"points above are un-paired slice metrics, not that comparison — different "
             f"slice, different claim.)"),
        ],
        numbers=numbers,
        repro=[drift_repro, ps_repro, pw_repro, sonnet_pc["repro_command"]],
    )


def ci_block_from_delta(band: dict) -> dict:
    """`{point, ci_lo, ci_hi}` from a delta band, whose centre may be `point` or `delta`.

    The repo's analysis artifacts disagree on that one key name — `frontier`/`prior_shift`
    write `point`, `perturbation`/`tier_b_compare` write `delta` — so the accessor accepts
    both rather than each caller re-deciding. Carrying both would be a second source of
    truth; a band with neither is a hard failure, never a silent None.
    """
    if "point" in band:
        centre = band["point"]
    elif "delta" in band:
        centre = band["delta"]
    else:
        raise ValueError(f"delta band has neither 'point' nor 'delta': {sorted(band)}")
    return {"point": centre, "ci_lo": band["ci_lo"], "ci_hi": band["ci_hi"]}


def _cs_thresholds(drift: dict) -> dict:
    src = _rel(DEFAULT_DRIFT_SUMMARY)
    repro = "make drift-charts"
    slices = ("test_iid", "test_drift_2023", "test_drift_2024", "test_drift_2025")
    human = {s: _drift_escalation(drift, "a_to_human", s) for s in slices}
    human_2026 = _drift_escalation(drift, "a_to_human", "test_drift_2026h1")
    b_arm = {s: _drift_escalation(drift, "a_to_b", s) for s in slices}
    b_2026 = _drift_escalation(drift, "a_to_b", "test_drift_2026h1")
    cal_human = drift["thresholds"]["a_to_human__full_slice"]["cal_escalation_rate"]
    a_logged_2026 = _drift_logged(drift, "tier_a", "test_drift_2026h1")
    b2_logged_2026 = _drift_logged(drift, "tier_b2", "test_drift_2026h1")

    human_rise = (human_2026["escalation_rate"]["point"]
                  / human["test_drift_2025"]["escalation_rate"]["point"] - 1.0)
    b_rise = (b_2026["escalation_rate"]["point"]
              / b_arm["test_drift_2025"]["escalation_rate"]["point"] - 1.0)

    def _esc(label, row):
        return _cs_number(label, _fmt_f(row["escalation_rate"]["point"]),
                          row["escalation_rate"]["point"], unit="raw", source=src,
                          run_ids=[row["gate_run_id"]], repro=repro)

    numbers = [
        _esc("human_2022", human["test_iid"]),
        _esc("human_2023", human["test_drift_2023"]),
        _esc("human_2024", human["test_drift_2024"]),
        _esc("human_2025", human["test_drift_2025"]),
        _cs_number("human_cal_op", _fmt_f(cal_human), cal_human, unit="raw", source=src,
                   repro="make thresholds",
                   note="CAL cost-argmin operating point the frozen tau was fitted at"),
        _cs_number("human_2026", _fmt_ci(ci_block(human_2026["escalation_rate"])),
                   ci_block(human_2026["escalation_rate"]), unit="raw", source=src,
                   run_ids=[human_2026["gate_run_id"]], repro=repro),
        _cs_number("human_rise", f"{human_rise * 100:.0f}%", human_rise, unit="pct",
                   basis="derived", source=src, repro=repro,
                   note="escalation_rate(2026-H1) / escalation_rate(2025) - 1, a_to_human "
                        "full-slice arm"),
        _esc("b_2022", b_arm["test_iid"]), _esc("b_2023", b_arm["test_drift_2023"]),
        _esc("b_2024", b_arm["test_drift_2024"]), _esc("b_2025", b_arm["test_drift_2025"]),
        _cs_number("b_2026", _fmt_ci(ci_block(b_2026["escalation_rate"])),
                   ci_block(b_2026["escalation_rate"]), unit="raw", source=src,
                   run_ids=[b_2026["gate_run_id"], b_2026["terminal_run_id"]],
                   repro=repro),
        _cs_number("b_rise", f"{b_rise * 100:.0f}%", b_rise, unit="pct",
                   basis="derived", source=src, repro=repro,
                   note="escalation_rate(2026-H1) / escalation_rate(2025) - 1, a_to_b "
                        "full-slice arm"),
        _cs_number("answered_acc_2026",
                   _fmt_ci(ci_block(human_2026["accuracy_machine"])),
                   ci_block(human_2026["accuracy_machine"]), unit="raw", source=src,
                   run_ids=[human_2026["gate_run_id"]], repro=repro),
        _cs_number("full_slice_acc_2026", _fmt_ci(ci_block(a_logged_2026["accuracy"])),
                   ci_block(a_logged_2026["accuracy"]), unit="raw", source=src,
                   run_ids=[a_logged_2026["run_id"]], repro=repro),
        _cs_number("cascade_acc_2026", _fmt_ci(ci_block(b_2026["accuracy_system"])),
                   ci_block(b_2026["accuracy_system"]), unit="raw", source=src,
                   run_ids=[b_2026["gate_run_id"], b_2026["terminal_run_id"]],
                   repro=repro),
        _cs_number("b2_only_acc_2026", _fmt_ci(ci_block(b2_logged_2026["accuracy"])),
                   ci_block(b2_logged_2026["accuracy"]), unit="raw", source=src,
                   run_ids=[b2_logged_2026["run_id"]], repro=repro),
    ]
    d = {e["label"]: e["display"] for e in numbers}
    return _cs_section(
        "thresholds", "narrative", "Frozen thresholds go stale late and abruptly",
        paragraphs=[
            (f"The escalate-to-human rate under the frozen CAL threshold is flat for four "
             f"years — {d['human_2022']}, {d['human_2023']}, {d['human_2024']}, "
             f"{d['human_2025']} from 2022-H2 through 2025, against a CAL operating point "
             f"of {d['human_cal_op']} — and then jumps to {d['human_2026']} at 2026-H1, a "
             f"relative rise of {d['human_rise']}."),
            (f"The escalate-to-DistilBERT arm under the same frozen gate moves "
             f"{d['b_2022']}, {d['b_2023']}, {d['b_2024']}, {d['b_2025']} over the same "
             f"slices and then {d['b_2026']}, a rise of {d['b_rise']}."),
            (f"The gate is doing real work exactly where it is needed: at 2026-H1 the "
             f"answered set scores accuracy {d['answered_acc_2026']} against "
             f"{d['full_slice_acc_2026']} on the full slice."),
            (f"But the frozen-threshold cascade slightly trails the un-gated DistilBERT at "
             f"the same cliff — system accuracy {d['cascade_acc_2026']} against "
             f"{d['b2_only_acc_2026']}. That is threshold staleness: a gate fitted on CAL "
             "self-adjusts late and abruptly rather than degrading gracefully, which is a "
             "monitoring requirement, not a tuning bug."),
        ],
        numbers=numbers,
        repro=[repro, "make thresholds"],
    )


def _cs_router(cfg: cost_model.CostConfig, *, frontier_dir, router_dir) -> dict:
    frontier_path = primary_frontier_path(cfg, frontier_dir)
    claims = json.loads(frontier_path.read_text(encoding="utf-8"))
    full_path = router_sim_path(router_sim.EVAL_FULL, cfg, router_dir)
    paired_path = router_sim_path(router_sim.EVAL_PAIRED, cfg, router_dir)
    full = json.loads(full_path.read_text(encoding="utf-8"))
    paired = json.loads(paired_path.read_text(encoding="utf-8"))

    fsrc = _rel(frontier_path)
    full_src = _rel(full_path)
    paired_src = _rel(paired_path)
    frepro = "make tier-b-frontier"

    vs_b2 = _frontier_claim(claims, "a_to_b", "b2_only", router_sim.EVAL_FULL)
    vs_a = _frontier_claim(claims, "a_to_human", "a_only", router_sim.EVAL_FULL)
    btc_vs_b2 = _frontier_claim(claims, "a_to_b_to_c", "b2_only", router_sim.EVAL_PAIRED)

    def _delta_cost(label, claim, source, note=None):
        band = ci_block_from_delta(claim["delta_cost_per_1k"])
        return _cs_number(label, _fmt_usd_ci(band, signed=True), band, unit="usd",
                          source=source, repro=frepro, note=note)

    def _delta_f1(label, claim, source):
        band = ci_block_from_delta(claim["delta_macro_f1_system"])
        return _cs_number(label, _fmt_ci(band, signed=True), band, unit="raw",
                          source=source, repro=frepro)

    def _policy_cost(label, doc, name, source):
        band = ci_block(doc["policies"][name]["expected_cost_per_1k"]["total"])
        return _cs_number(label, _fmt_usd(band["point"]), band["point"], unit="usd",
                          source=source, repro="make router-sim")

    def _policy_cost_ci(label, doc, name, source):
        band = ci_block(doc["policies"][name]["expected_cost_per_1k"]["total"])
        return _cs_number(label, _fmt_usd_ci(band), band, unit="usd", source=source,
                          repro="make router-sim")

    numbers = [
        _delta_cost("a_to_b_vs_b2_cost", vs_b2, fsrc),
        _delta_f1("a_to_b_vs_b2_f1", vs_b2, fsrc),
        _policy_cost_ci("a_to_b_cost", full, "a_to_b", full_src),
        _cs_number("a_to_b_system_f1",
                   _fmt_f(full["policies"]["a_to_b"]["macro_f1_system"]),
                   full["policies"]["a_to_b"]["macro_f1_system"], unit="raw",
                   source=full_src, repro="make router-sim"),
        _policy_cost("a_only_cost", full, "a_only", full_src),
        _policy_cost("b1_only_cost", full, "b1_only_sa", full_src),
        _policy_cost("b2_only_cost", full, "b2_only", full_src),
        _delta_cost("a_to_human_vs_a_cost", vs_a, fsrc),
        _delta_f1("a_to_human_vs_a_f1", vs_a, fsrc),
        _policy_cost_ci("a_to_b_to_c_cost", paired, "a_to_b_to_c", paired_src),
        _cs_number("n_paired_subset", _fmt_count(paired["n_examples"]),
                   paired["n_examples"], unit="count", source=paired_src,
                   repro="make router-sim"),
        _delta_cost("a_to_b_to_c_vs_b2_cost", btc_vs_b2, fsrc),
    ]
    d = {e["label"]: e["display"] for e in numbers}
    return _cs_section(
        "router", "narrative", "What pays is routing to cheap capacity, not the LLM",
        paragraphs=[
            (f"The headline router is a confidence gate in front of Tier A with DistilBERT "
             f"as the terminal, evaluated on the full TEST-IID slice under cost-model v2 "
             f"(Tier B priced by an amortized ESTIMATE, not a measurement). It is the only "
             f"certified two-axis win in the study: against a DistilBERT-only baseline it "
             f"is both cheaper, {d['a_to_b_vs_b2_cost']} per 1k complaints, and better, "
             f"macro-F1 {d['a_to_b_vs_b2_f1']} — both paired intervals exclude zero."),
            (f"In absolute terms it runs at {d['a_to_b_cost']} per 1k at system macro-F1 "
             f"{d['a_to_b_system_f1']}, dominating Tier A alone ({d['a_only_cost']}), "
             f"ModernBERT alone ({d['b1_only_cost']}) and DistilBERT alone "
             f"({d['b2_only_cost']})."),
            (f"The same gate in front of Tier A alone buys {d['a_to_human_vs_a_cost']} per "
             f"1k and {d['a_to_human_vs_a_f1']} system macro-F1 against un-gated Tier A. "
             f"The gate is certified in front of DistilBERT too, but it is worth far less "
             f"per 1k there. What pays is not the gate; it is routing to cheap capacity."),
            (f"The cheapest policy measured is the two-gate cascade that ends at Haiku: "
             f"{d['a_to_b_to_c_cost']} per 1k on the paired {d['n_paired_subset']}-row "
             f"subset. But its edge over DistilBERT alone is NOT established: "
             f"{d['a_to_b_to_c_vs_b2_cost']}, an interval containing zero."),
        ],
        numbers=numbers,
        repro=[frepro, "make router-sim"],
    )


def _cs_robustness(resolved: dict, perturbation: dict) -> dict:
    src = _rel(DEFAULT_PERTURBATION_SUMMARY)
    repro = "make perturb && make perturb-report"
    wordchar_typo = _perturbation(perturbation, "logreg_wordchar", "typo")
    word_typo = _perturbation(perturbation, "logreg_word_only", "typo")
    wordchar_case = _perturbation(perturbation, "logreg_wordchar", "case")

    shield = 1.0 - (wordchar_typo["metrics"]["macro_f1"]["delta"]
                    / word_typo["metrics"]["macro_f1"]["delta"])

    # The perturbed Tier C runs are named by the summary row), never by string surgery on
    # a config stem: the row is the thing that knows which run is which arm.
    tier_c_rows = {family: _perturbation(perturbation, "tier_c_haiku", family)
                   for family in ("typo", "ocr", "case")}
    clean = record_for(resolved, HAIKU_TEST, TEST_IID)
    perturbed_records = {
        family: record_for(resolved, row["perturbed_config"], TEST_IID)
        for family, row in tier_c_rows.items()
    }
    inflation = {family: prompt_token_inflation(clean, record)
                 for family, record in perturbed_records.items()}
    haiku_typo = tier_c_rows["typo"]
    haiku_ocr = tier_c_rows["ocr"]
    haiku_case = tier_c_rows["case"]

    def _delta(label, row):
        band = ci_block_from_delta(row["metrics"]["macro_f1"])
        return _cs_number(label, _fmt_ci(band, signed=True), band, unit="raw", source=src,
                          run_ids=[row["perturbed_run_id"], row["clean_run_id"]],
                          repro=repro)

    def _inflation(label, family):
        record = perturbed_records[family]
        return _cs_number(label, _fmt_pct(inflation[family]), inflation[family],
                          unit="pct", basis="derived", evidence_class="derived",
                          source="results/tier_c_raw/**/calls.jsonl",
                          run_ids=[record["run_id"], clean["run_id"]],
                          repro="make perturb-tier-c",
                          note="sum(prompt_tokens) perturbed / clean - 1, over the "
                               "perturbed run's own complaint ids (a pairing: the "
                               "perturbed rows are a byte-identical subset of the clean "
                               "run under the same cap_seed)")

    numbers = [
        _cs_number("perturb_rate", _fmt_pct(wordchar_typo["rate"], 0),
                   wordchar_typo["rate"], unit="pct", source=src, repro=repro),
        _delta("wordchar_typo", wordchar_typo),
        _delta("word_only_typo", word_typo),
        _cs_number("char_shield_share", f"{shield * 100:.0f}%", shield, unit="pct",
                   basis="derived", evidence_class="derived", source=src, repro=repro,
                   note="1 - delta(word+char, typo) / delta(word-only, typo); the "
                        "artifact's own methods_notes.char_shield warns that a joint CI "
                        "for this difference-of-differences was NOT computed, so this is "
                        "directional only"),
        _cs_number("wordchar_case", _fmt_ci(
            ci_block_from_delta(wordchar_case["metrics"]["macro_f1"]), signed=True),
            ci_block_from_delta(wordchar_case["metrics"]["macro_f1"]), unit="raw",
            source=src, run_ids=[wordchar_case["perturbed_run_id"]], repro=repro,
            note="predicted structural zero: both TF-IDF blocks lowercase their input"),
        _delta("haiku_typo", haiku_typo),
        _delta("haiku_ocr", haiku_ocr),
        _delta("haiku_case", haiku_case),
        _cs_number("n_tier_c_perturb", _fmt_count(haiku_typo["n_rows"]),
                   haiku_typo["n_rows"], unit="count", source=src, repro=repro),
        _inflation("inflation_typo", "typo"),
        _inflation("inflation_ocr", "ocr"),
        _inflation("inflation_case", "case"),
    ]
    d = {e["label"]: e["display"] for e in numbers}
    return _cs_section(
        "robustness", "narrative", "Robustness has a price even when quality holds",
        paragraphs=[
            (f"Under {d['perturb_rate']} character corruption on TEST-IID, the word+char "
             f"Tier A model loses {d['wordchar_typo']} macro-F1 to typos while the "
             f"word-only sensitivity arm loses {d['word_only_typo']}. The char-n-gram "
             f"block therefore absorbs roughly {d['char_shield_share']} of the typo damage "
             f"— a directional read only: the two arms are different fitted models with "
             f"independently bootstrapped intervals and no joint CI was computed. The case "
             f"family is an exact structural zero for Tier A, {d['wordchar_case']}, "
             f"because both TF-IDF blocks lowercase their input; it is reported as an "
             f"end-to-end control on the plumbing, not as a robustness finding."),
            (f"Haiku loses least. On the same {d['n_tier_c_perturb']} paired rows its typo "
             f"delta is {d['haiku_typo']}, while ocr {d['haiku_ocr']} and case "
             f"{d['haiku_case']} are both null — their intervals contain zero. The "
             f"cross-tier ranking under character noise is LLM at least as robust as "
             f"word+char TF-IDF, and both ahead of word-only TF-IDF."),
            (f"Robustness carries a serving-cost tax that quality metrics do not show. On "
             f"those same rows, corrupted text inflates Haiku's prompt tokens by "
             f"{d['inflation_typo']} (typo), {d['inflation_ocr']} (ocr) and "
             f"{d['inflation_case']} (case). The case family has the largest inflation and "
             "no measurable quality effect at all — noisy inputs make escalation more "
             "expensive exactly when Tier A is least reliable."),
        ],
        numbers=numbers,
        repro=[repro, "make perturb-tier-c"],
    )


def _cs_negatives(resolved: dict, oov_summary: dict, cfg: cost_model.CostConfig, *,
                  frontier_dir, router_dir, tier_c_compare_dir) -> dict:
    ablation = tier_c_compare_artifact(TIER_C_COMPARE_FEWSHOT, tier_c_compare_dir)
    oov_src = _rel(DEFAULT_OOV_SUMMARY)
    oov_repro = "make oov"
    train_tok = _oov(oov_summary, "train", "model_vocab_oov_token_rate")
    year_tok = _oov(oov_summary, "test_drift_2026h1", "model_vocab_oov_token_rate")
    dist_2025 = _oov(oov_summary, "test_drift_2025", "tfidf_centroid_cosine_distance")
    dist_2026 = _oov(oov_summary, "test_drift_2026h1", "tfidf_centroid_cosine_distance")
    year_type = _oov(oov_summary, "test_drift_2026h1", "model_vocab_oov_type_rate")

    fewshot = record_for(resolved, "tier_c_haiku_ablation_fewshot_cal", CAL)
    zeroshot = record_for(resolved, "tier_c_haiku_ablation_zeroshot_cal", CAL)
    runs_src = _rel(DEFAULT_RESULTS_PATH)

    frontier_path = primary_frontier_path(cfg, frontier_dir)
    claims = json.loads(frontier_path.read_text(encoding="utf-8"))
    summary_path = Path(router_dir) / router_sim.result_filename(
        "summary", cfg, OP_VERSION)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cascade_vs_a = _frontier_claim(claims, "a_to_c_parsefail_human", "a_only",
                                   router_sim.EVAL_PAIRED)
    cross_family = summary["owner_decision_1_cross_family"]

    numbers = [
        _cs_number("oov_train", _fmt_pct(train_tok["point"], 3), train_tok["point"],
                   unit="pct", source=oov_src, repro=oov_repro),
        _cs_number("oov_2026", _fmt_pct(year_tok["point"], 3), year_tok["point"],
                   unit="pct", source=oov_src, repro=oov_repro),
        _cs_number("centroid_2025", _fmt_ci(ci_block(dist_2025)), ci_block(dist_2025),
                   unit="raw", source=oov_src, repro=oov_repro),
        _cs_number("centroid_2026", _fmt_ci(ci_block(dist_2026)), ci_block(dist_2026),
                   unit="raw", source=oov_src, repro=oov_repro),
        _cs_number("oov_type_2026", _fmt_pct(year_type["point"], 1), year_type["point"],
                   unit="pct", source=oov_src, repro=oov_repro),
        _cs_number("k_fewshot", str(int(fewshot["extra"]["num_exemplars"])),
                   int(fewshot["extra"]["num_exemplars"]), unit="count",
                   source=runs_src, run_ids=[fewshot["run_id"]]),
        _cs_number("k_zeroshot", str(int(zeroshot["extra"]["num_exemplars"])),
                   int(zeroshot["extra"]["num_exemplars"]), unit="count",
                   source=runs_src, run_ids=[zeroshot["run_id"]]),
        _cs_number("fewshot_f1", _fmt_ci(metric_from_record(fewshot, "macro_f1")),
                   metric_from_record(fewshot, "macro_f1"), unit="raw", source=runs_src,
                   run_ids=[fewshot["run_id"]]),
        _cs_number("zeroshot_f1", _fmt_ci(metric_from_record(zeroshot, "macro_f1")),
                   metric_from_record(zeroshot, "macro_f1"), unit="raw", source=runs_src,
                   run_ids=[zeroshot["run_id"]]),
        _cs_number("fewshot_spend", _fmt_usd(fewshot["cost_usd"]), fewshot["cost_usd"],
                   unit="usd", source=runs_src, run_ids=[fewshot["run_id"]]),
        _cs_number("zeroshot_spend", _fmt_usd(zeroshot["cost_usd"]), zeroshot["cost_usd"],
                   unit="usd", source=runs_src, run_ids=[zeroshot["run_id"]]),
        _cs_number("n_ablation", _fmt_count(fewshot["extra"]["n_examples"]),
                   fewshot["extra"]["n_examples"], unit="count", source=runs_src,
                   run_ids=[fewshot["run_id"]]),
        *_tier_c_delta_numbers(ablation, "fewshot_minus_zeroshot"),
        _cs_number("cascade_vs_a_cost",
                   _fmt_usd_ci(ci_block_from_delta(cascade_vs_a["delta_cost_per_1k"]),
                               signed=True),
                   ci_block_from_delta(cascade_vs_a["delta_cost_per_1k"]), unit="usd",
                   source=_rel(frontier_path), repro="make tier-b-frontier"),
        _cs_number("cascade_vs_a_f1",
                   _fmt_ci(ci_block_from_delta(cascade_vs_a["delta_macro_f1_system"]),
                           signed=True),
                   ci_block_from_delta(cascade_vs_a["delta_macro_f1_system"]), unit="raw",
                   source=_rel(frontier_path), repro="make tier-b-frontier"),
        _cs_number("cross_family_cost",
                   _fmt_usd_ci(ci_block_from_delta(cross_family["delta_cost_per_1k"]),
                               signed=True),
                   ci_block_from_delta(cross_family["delta_cost_per_1k"]), unit="usd",
                   source=_rel(summary_path), repro="make router-sim"),
        _cs_number("cross_family_n", _fmt_count(cross_family["n_examples"]),
                   cross_family["n_examples"], unit="count", source=_rel(summary_path),
                   repro="make router-sim"),
    ]
    d = {e["label"]: e["display"] for e in numbers}
    return _cs_section(
        "negatives", "narrative", "Honest negatives",
        paragraphs=[
            (f"The out-of-vocabulary explanation was REFUTED. Model-vocabulary token OOV "
             f"moves only from {d['oov_train']} on TRAIN to {d['oov_2026']} at 2026-H1, "
             f"and the TF-IDF centroid distance PEAKS in 2025 at {d['centroid_2025']} and "
             f"FALLS at 2026-H1 to {d['centroid_2026']} — disjoint intervals. Lexical "
             f"drift is ruled out; the cliff is prior shift. Types and tokens tell opposite "
             f"stories: {d['oov_type_2026']} of TYPES at 2026-H1 are outside the model "
             f"vocabulary, against {d['oov_2026']} of TOKENS."),
            (f"The few-shot hypothesis was REFUTED. On the same {d['n_ablation']} CAL rows, "
             f"Haiku with k={d['k_fewshot']} exemplars scores {d['fewshot_f1']} against "
             f"{d['zeroshot_f1']} at k={d['k_zeroshot']}, for double the measured spend "
             f"({d['fewshot_spend']} against {d['zeroshot_spend']}). Paired on those rows "
             f"the delta is {d['fewshot_minus_zeroshot_delta']} macro-F1, McNemar "
             f"p={d['fewshot_minus_zeroshot_p']} — an interval containing zero. The "
             f"exemplars bought no measurable quality for twice the money, so every Tier "
             f"C final in the study is zero-shot."),
            (f"The LLM cascade does not pay. Under the primary operating point the "
             f"Haiku-terminal cascade is cheaper than Tier A alone, "
             f"{d['cascade_vs_a_cost']} per 1k, but it buys no established quality: "
             f"macro-F1 {d['cascade_vs_a_f1']}, an interval containing zero, so the "
             f"two-axis claim is not certified. Head to head against the gate-to-human "
             f"router on the same {d['cross_family_n']} rows it resolves AGAINST the "
             f"cascade: {d['cross_family_cost']} per 1k, more expensive with the interval "
             f"excluding zero."),
        ],
        numbers=numbers,
        repro=[oov_repro, "make tier-b-frontier", "make router-sim",
               ablation["repro_command"]],
        gaps=[
            ("An earlier cost generation recorded the cascade-vs-Tier-A cost comparison as "
             "directional (interval containing zero). The figures above are the primary "
             "operating point and cost generation the rest of this page uses; mixing "
             "generations inside one paragraph is exactly what the artifact naming "
             "prevents."),
        ],
    )


def _cs_verification(records: list[dict], resolved: dict, manifest: dict, drift: dict,
                     parity: dict, live: dict, receipts: dict) -> dict:
    boot = drift["bootstrap"]
    ci_lo_pct, ci_hi_pct = boot["ci_pct"]
    b2 = record_for(resolved, TIER_B2_SAMPLE_CONFIG, TEST_IID)
    haiku = record_for(resolved, HAIKU_TEST, TEST_IID)

    provider_counts: Counter = Counter()
    for entry in receipts["runs"].values():
        provider_counts.update(entry["provider_mix"])
    total_calls = sum(provider_counts.values())
    bedrock_share = provider_counts["Amazon Bedrock"] / total_calls

    runs_src = _rel(DEFAULT_RESULTS_PATH)
    numbers = [
        _cs_number("snapshot_bytes", _fmt_count(manifest["size_bytes"]),
                   manifest["size_bytes"], unit="count",
                   source=_rel(DEFAULT_SNAPSHOT_MANIFEST), repro="make data"),
        _cs_number("n_run_records", _fmt_count(len(records)), len(records), unit="count",
                   source=runs_src, basis="derived",
                   note="len(results/runs.jsonl) at build time"),
        _cs_number("ci_level", f"{ci_hi_pct - ci_lo_pct:.0f}%",
                   float(ci_hi_pct - ci_lo_pct), unit="pctpoint",
                   source=_rel(DEFAULT_DRIFT_SUMMARY), basis="derived",
                   note="bootstrap.ci_pct[1] - bootstrap.ci_pct[0]"),
        _cs_number("bootstrap_resamples", _fmt_count(boot["n_resamples"]),
                   boot["n_resamples"], unit="count", source=_rel(DEFAULT_DRIFT_SUMMARY)),
        _cs_number("bootstrap_seed", str(int(boot["seed"])), int(boot["seed"]),
                   unit="count", source=_rel(DEFAULT_DRIFT_SUMMARY)),
        _cs_number("bedrock_share", _fmt_pct(bedrock_share, 2), bedrock_share,
                   unit="pct", basis="derived", evidence_class="derived",
                   source="results/tier_c_raw/**/calls.jsonl",
                   note="Amazon Bedrock calls / all committed Tier C receipt calls"),
        _cs_number("n_tier_c_calls", _fmt_count(total_calls), total_calls, unit="count",
                   basis="derived", evidence_class="derived",
                   source="results/tier_c_raw/**/calls.jsonl"),
        _cs_number("max_concurrency", str(int(haiku["extra"]["max_concurrency"])),
                   int(haiku["extra"]["max_concurrency"]), unit="count", source=runs_src,
                   run_ids=[haiku["run_id"]]),
        _cs_number("int8_agreement", _fmt_f(parity["agreement"]["int8_vs_pytorch_fp32"]),
                   parity["agreement"]["int8_vs_pytorch_fp32"], unit="raw",
                   source=_rel(DEFAULT_ONNX_PARITY), run_ids=[b2["run_id"]],
                   repro="uv run python scripts/export_onnx_distilbert.py"),
        _cs_number("fp32_agreement", _fmt_f(parity["agreement"]["onnx_fp32_vs_pytorch"]),
                   parity["agreement"]["onnx_fp32_vs_pytorch"], unit="raw",
                   source=_rel(DEFAULT_ONNX_PARITY), run_ids=[b2["run_id"]],
                   repro="uv run python scripts/export_onnx_distilbert.py"),
        _cs_number("parity_n", _fmt_count(parity["n_samples"]), parity["n_samples"],
                   unit="count", source=_rel(DEFAULT_ONNX_PARITY)),
        _cs_number("live_tier_a_agreement",
                   _fmt_pct(live["tier_a"]["label_agreement_vs_official"], 1),
                   live["tier_a"]["label_agreement_vs_official"], unit="pct",
                   source=_rel(DEFAULT_LIVE_AGREEMENT)),
        _cs_number("live_n", _fmt_count(live["tier_a"]["n"]), live["tier_a"]["n"],
                   unit="count", source=_rel(DEFAULT_LIVE_AGREEMENT)),
        _cs_number("live_tier_b2_agreement",
                   _fmt_pct(live["tier_b2"]["vs_official_fp32"]
                            ["label_agreement_vs_official"], 1),
                   live["tier_b2"]["vs_official_fp32"]["label_agreement_vs_official"],
                   unit="pct", source=_rel(DEFAULT_LIVE_AGREEMENT)),
        _cs_number("suite_passed", _fmt_count(SUITE_RESULT["passed"]),
                   SUITE_RESULT["passed"], unit="count", basis="declared",
                   source="src/triage_lab/demo_build.py", repro=SUITE_RESULT["command"],
                   note=SUITE_RESULT["note"]),
        _cs_number("suite_skipped", _fmt_count(SUITE_RESULT["skipped"]),
                   SUITE_RESULT["skipped"], unit="count", basis="declared",
                   source="src/triage_lab/demo_build.py", repro=SUITE_RESULT["command"],
                   note=SUITE_RESULT["note"]),
        _cs_number("suite_failed", _fmt_count(SUITE_RESULT["failed"]),
                   SUITE_RESULT["failed"], unit="count", basis="declared",
                   source="src/triage_lab/demo_build.py", repro=SUITE_RESULT["command"],
                   note=SUITE_RESULT["note"]),
    ]
    d = {e["label"]: e["display"] for e in numbers}
    items = [
        {"n": 1, "title": "Frozen dataset snapshot",
         "text": (f"CFPB complaints.csv.zip, downloaded {manifest['download_date']}, "
                  f"SHA-256 {manifest['sha256']}, {d['snapshot_bytes']} bytes. The splits "
                  f"reproduce byte-identically from that snapshot under `make data`."),
         "source": _rel(DEFAULT_SNAPSHOT_MANIFEST), "run_ids": [],
         "evidence_class": "measured"},
        {"n": 2, "title": "Locked environment",
         "text": ("A committed uv lockfile; `uv sync --frozen` is the only supported "
                  "install path. Tier B was trained on an NVIDIA RTX A6000 in bf16 "
                  "(bundle chain-of-custody verified) and evaluated locally on MPS; the "
                  "hardware is recorded in each run record."),
         "source": runs_src, "run_ids": [b2["run_id"]], "evidence_class": "measured"},
        {"n": 3, "title": "One config per run, one append-only log",
         "text": (f"One YAML config per run, one JSONL record appended to "
                  f"results/runs.jsonl — {d['n_run_records']} records, never edited. Every "
                  f"record carries its git SHA, dataset snapshot hash, config or prompt "
                  f"hash, wall-clock and cost."),
         "source": runs_src, "run_ids": [], "evidence_class": "measured"},
        {"n": 4, "title": "Uncertainty on every headline number",
         "text": (f"{d['ci_level']} bootstrap confidence intervals "
                  f"({d['bootstrap_resamples']} resamples, fixed seed "
                  f"{d['bootstrap_seed']}) on every headline number; every comparison "
                  f"claim uses a paired bootstrap plus McNemar on the shared rows."),
         "source": _rel(DEFAULT_DRIFT_SUMMARY), "run_ids": [],
         "evidence_class": "measured"},
        {"n": 5, "title": "Tier C cost receipts",
         "text": (f"Raw per-call logs are retained and committed. Costs are actual token "
                  f"usage multiplied by published per-MTok prices — never estimated from "
                  f"characters or averages — and the upstream provider is recorded per "
                  f"call: {d['bedrock_share']} of {d['n_tier_c_calls']} calls were served "
                  f"by Amazon Bedrock."),
         "source": "results/tier_c_raw/**/calls.jsonl", "run_ids": [],
         "evidence_class": "measured"},
        {"n": 6, "title": "Latency methodology, stated as a limit",
         "text": (f"Latency is client-side wall-clock through OpenRouter with no provider "
                  f"pinning at max_concurrency {d['max_concurrency']}. The figures "
                  f"characterize the OpenRouter-to-Bedrock route, NOT the Anthropic "
                  f"first-party API."),
         "source": runs_src, "run_ids": [haiku["run_id"]], "evidence_class": "measured"},
        {"n": 7, "title": "ONNX parity for the in-browser model",
         "text": (f"Argmax agreement between the int8 export (per-channel QInt8) and "
                  f"PyTorch fp32 is {d['int8_agreement']} on a held-out CAL sample of "
                  f"{d['parity_n']} rows; the fp32 ONNX export is exact at "
                  f"{d['fp32_agreement']}. A stock "
                  f"per-tensor quantization was measured first and REJECTED: the export "
                  f"config was fixed, the acceptance threshold was not relaxed. Official "
                  f"numbers remain the harness fp32 runs."),
         "source": _rel(DEFAULT_ONNX_PARITY), "run_ids": [b2["run_id"]],
         "evidence_class": "measured"},
        {"n": 8, "title": "Live in-browser agreement, frozen and disclosed in the UI",
         "text": (f"Tier A agrees with its official run on {d['live_tier_a_agreement']} "
                  f"of {d['live_n']} curated rows (a bit-identical export); the B2 int8 "
                  f"browser model agrees on {d['live_tier_b2_agreement']} against the "
                  f"official fp32 run. Both rates are frozen in the payload and shown in "
                  f"the playground panel."),
         "source": _rel(DEFAULT_LIVE_AGREEMENT), "run_ids": [], "evidence_class":
             "measured"},
        {"n": 9, "title": "Tests, including the one that guards this page",
         "text": (f"A smoke eval runs in CI; the full suite is {d['suite_passed']} "
                  f"passed / {d['suite_skipped']} skipped / {d['suite_failed']} failed. "
                  f"The demo's traceability is test-enforced: every run id under "
                  f"demo/data must exist in the results log, every copied number must "
                  f"equal its source, and every numeric token in this page's prose must "
                  f"be a declared, sourced value."),
         "source": "tests/test_demo_build.py", "run_ids": [],
         "evidence_class": "measured"},
        {"n": 10, "title": "Dev/test hygiene",
         "text": ("TEST-* slices are frozen and were touched only for final reported "
                  "runs; all iteration happened on CAL. EXPERIMENT_LOG.md is published "
                  "including the failed hypotheses."),
         "source": "EXPERIMENT_LOG.md", "run_ids": [], "evidence_class": "measured"},
        {"n": 11, "title": "One-command headline reproduction",
         "text": ("`make reproduce-headline` re-derives the headline claim chain from the "
                  "frozen snapshot: it re-materializes the splits and checks their hashes "
                  "against the frozen run records, re-derives every per-example artifact "
                  "the chain reads (each verified against its own logged metrics), re-runs "
                  "the threshold, router and frontier derivations under the Tier B cost "
                  "generation, rebuilds this payload, and FAILS unless every regenerated "
                  "file is byte-identical to the committed one. Its scope is the headline "
                  "claim chain — the drift, perturbation and OOV exhibits keep their own "
                  "reproduction commands in EXPERIMENT_LOG.md."),
         "source": "Makefile", "run_ids": [], "evidence_class": "measured"},
    ]
    return _cs_section(
        "verification", "verification", "How this was verified",
        numbers=numbers, items=items,
        repro=["uv run pytest -q", "make data", "make reproduce-headline"],
    )


CASE_STUDY_LIMITS = (
    ("Not a production serving system. There are no SLA, throughput or uptime claims; "
     "latency was bench-measured (and through OpenRouter), never fleet-measured."),
    ("Single-domain and English-only: US consumer-finance complaints. Generalization to "
     "other triage domains is not demonstrated."),
    ("Tier C costs are point-in-time prices for specific model IDs, recorded per call. "
     "Forward cost claims are projected, and the frontier is a method, not a leaderboard."),
    ("The business cost-model parameters — misroute cost, human-review cost and the Tier B "
     "amortized compute price — are ESTIMATED defaults, not measurements. Their sensitivity "
     "is exposed in the policy-builder panel rather than asserted away."),
    ("LLM contamination on pre-cutoff data is possible. It is mitigated by TEST-POSTCUTOFF "
     "(a boundary strictly after the latest Tier C training cutoff), not eliminated."),
    ("CFPB narratives are opt-in and CFPB-scrubbed, so there is selection bias against the "
     "full complaint population; the scrubbing artifacts are in-distribution quirks the "
     "models learn from."),
    ("No online learning and no human-in-the-loop feedback were measured, and the "
     "escalation arm assumes humans are always right — a modeling choice, stated as one."),
    ("The coursework seed scores that motivated this lab are self-reported class results: "
     "provenance, not evidence. The group NER project was collaborative, so role "
     "attribution there is biographical context, not a measured claim."),
)


def _cs_limits() -> dict:
    return _cs_section(
        "limits", "limits", "What this does not prove",
        items=[{"text": text} for text in CASE_STUDY_LIMITS],
    )


# --- provenance: the coursework seeds (docs/seed-evidence/, READ-ONLY) -------------------
#
# The one section on the page whose numbers are NOT this lab's evidence. They are
# self-reported class results, copied character-for-character out of a read-only archive of
# the original write-ups, and they carry their own evidence class ("provenance") so a reader
# can never mistake one for a run this lab made. Three rules keep that honest:
#
#   1. A display is an EXACT SUBSTRING of the file it cites. `_seed_number` re-reads the
#      file at build time and refuses to emit anything it cannot find, so the digits on the
#      page are the digits in the archive — including "5,787" with its comma and the
#      Naive Bayes F1 at its full absurd precision (rounding it would be a new number).
#   2. Every file in the archive is described. The build lists the directory and hard-fails
#      if the set of described paths differs, so a file cannot be added or removed without
#      the page saying what it is.
#   3. What the plan claims but the archive does not contain goes in `gaps`, not in the
#      prose (UPGRADE_PLAN Appendix A's Kaggle 0.83615 and the 0.8136 starter baseline both
#      cite documents that are NOT in this repo).
#
# Nothing here is edited, executed or regenerated: this module only reads.

SEED_EVIDENCE_DIR = REPO_ROOT / "docs" / "seed-evidence"

SEED_NB_REPORT = "docs/seed-evidence/task2-naive-bayes-report.md"
SEED_NER_SUMMARY = "docs/seed-evidence/task3-ner-memm-session-summary.md"
SEED_SCRAPING_REPORT = "docs/seed-evidence/task1-scraping-report.md"
SEED_NB_SWEEP = "docs/seed-evidence/classification_result_{n}.txt"
UPGRADE_PLAN = "UPGRADE_PLAN.md"

# role labels (rendered): deliberately digit-free, because every numeric token on this page
# has to be a declared, sourced number and a role label is neither.
ROLE_SEED = "seed exhibit lineage"
ROLE_METHOD = "methodology seed"
ROLE_ARCHIVE = "archived context — not a seed"


def _seed_text(rel_path: str) -> str:
    return (REPO_ROOT / rel_path).read_text(encoding="utf-8")


# A whole numeric token, on the same rule the case study's prose gate uses: not preceded by
# a letter or digit, and swallowing its own thousands separators and decimals. Substring
# matching would let a short display like "15" pass on the "15" inside "1155".
_SEED_TOKEN_RE = re.compile(r"(?<![0-9A-Za-z])\d[\d,]*(?:\.\d+)?(?:[eE][+-]?\d+)?")


def _seed_number(label: str, display: str, *, source: str, unit: str = "raw",
                 note: str | None = None) -> dict:
    """One coursework figure, quoted exactly as its source file writes it.

    Hard-fails the build when `display` is not a whole numeric token of the file — same
    string, same thousands separators, same precision. A provenance number that has drifted
    from the archive is worse than no provenance section at all.
    """
    text = _seed_text(source)
    if display not in {m.group(0) for m in _SEED_TOKEN_RE.finditer(text)}:
        raise ValueError(
            f"provenance number {label!r}: {display!r} is not a number {source} writes "
            f"(as that exact token); quote the archive, do not restate it")
    numeric = float(display.replace(",", ""))
    value = int(numeric) if unit == "count" else numeric
    return _cs_number(label, display, value, unit=unit, source=source, run_ids=[],
                      basis="copied", evidence_class="provenance",
                      repro=f"grep -n '{display}' {source}", note=note)


def _seed_items() -> list[dict]:
    """One card per file in the read-only archive, in the order the story needs them."""
    items = [
        {"path": SEED_NB_REPORT, "role": ROLE_SEED,
         "text": ("The from-scratch multinomial Naive Bayes report (Reuters, five news "
                  "classes): tokenizing, Porter stemming, frequency-based feature "
                  "selection, Laplace smoothing, log-space scoring and a macro-F1. This is "
                  "the ancestor of this lab's classical tier — which is sklearn, not this "
                  "code."),
         "evidence_class": "provenance"},
    ]
    for n_features, kind in (("3000", "the smallest sweep setting"),
                             ("5000", "a sweep setting"),
                             ("10000", "the assignment's required setting"),
                             ("15000", "the largest sweep setting")):
        items.append(
            {"path": SEED_NB_SWEEP.format(n=n_features), "role": ROLE_SEED,
             "text": (f"Per-document Naive Bayes predictions at {n_features} selected "
                      f"features — {kind}. The report's sweep row for this setting was "
                      f"scored from this file; it is kept as the output artifact behind a "
                      f"quoted number, not as a number itself."),
             "evidence_class": "provenance"})
    items.append(
        {"path": SEED_NER_SUMMARY, "role": ROLE_METHOD,
         "text": ("The dated iteration log of the CoNLL-2003 MEMM NER project (group work): "
                  "four feature phases with dev scores, four Kaggle submissions each with a "
                  "stated hypothesis and a stated verdict, harmful features identified and "
                  "removed with reasons, an out-of-memory failure and its fix. Its lesson "
                  "list is what this lab's method section was built from."),
         "evidence_class": "provenance"})
    items.append(
        {"path": SEED_SCRAPING_REPORT, "role": ROLE_ARCHIVE,
         "text": ("A crawl-and-preprocess pipeline over 200 scraped book descriptions, from "
                  "the earliest coursework project. The upgrade plan's "
                  "reuse-versus-discard audit discarded that project outright: nothing in "
                  "it seeds this lab, and it is archived here only so the citation set is "
                  "the whole archive rather than the flattering part of it."),
         "evidence_class": "provenance"})
    return items


SEED_LINEAGE = (
    {"lesson": ("Convergence is a hyperparameter, not a detail: raising the MEMM's "
                "MAX_ITER from 15 to 25 was the single biggest gain in the NER project, "
                "bigger than any feature it tried."),
     "practice": ("Every training and optimization constant lives in a per-run YAML config "
                  "whose hash is stamped into the run record, so no result depends on a "
                  "value nobody wrote down.")},
    {"lesson": ("Removing harmful features beat adding good ones — no_gaz and pl|s3 were "
                "dropped with mechanistic explanations, and the score moved."),
     "practice": ("Negative results are first-class: EXPERIMENT_LOG.md publishes the failed "
                  "hypotheses, and this page's 'what this does not prove' section is as "
                  "long as its results.")},
    {"lesson": ("Each Kaggle submission changed exactly one thing against the previous one, "
                "so every score delta had a candidate cause."),
     "practice": ("Single-variable experiment discipline: one config delta per run, one "
                  "appended record in results/runs.jsonl, one logged verdict.")},
    {"lesson": ("The dev-to-test drop of about 0.04 F1 was diagnosed as a temporal "
                "out-of-vocabulary gap between news periods, and reported as structural "
                "rather than explained away as noise."),
     "practice": ("The entire drift protocol: yearly evaluations, the prior-shift "
                  "decomposition, and OOV tracking against the frozen training vocabulary.")},
    {"lesson": ("The coursework could not afford bootstrap confidence intervals and said so "
                "in its own presentation notes, which left every improvement claim resting "
                "on a single point estimate."),
     "practice": ("This lab puts a bootstrap interval on every headline number and a paired "
                  "interval on every comparison claim; a delta whose interval contains zero "
                  "is reported as a tie.")},
)

SEED_CAVEATS = (
    ("The NER project was group work. Role attribution — that the owner led the modeling — "
     "is biographical context, not a measured claim, and nothing in this archive verifies "
     "it."),
    ("The coursework datasets (Reuters, CoNLL-2003) are research-licensed and were "
     "deliberately excluded from this repo. Only the write-ups are archived; no coursework "
     "dataset, notebook or submission file is redistributed here."),
)

SEED_GAPS = (
    ("The upgrade plan's appendix cites a best Kaggle public score of 0.83615, from a later "
     "submission whose only record is a notebook docstring that is NOT in this archive. The "
     "page quotes what the committed session summary states — the best of the four "
     "submissions it documents — and treats the higher figure as uncited."),
    ("The same appendix frames the NER progression from a starter-code baseline of 0.8136, "
     "which comes from a document that is not in this archive either. The page therefore "
     "starts the progression at the owner's own first phase, which the committed summary "
     "does state."),
    ("The 'coursework could not afford bootstrap CIs' line traces to the group's "
     "presentation speaker notes, which are not in this archive. It is stated as narrative "
     "lineage, not quoted as a source."),
    ("The plan's general caveat says both coursework reports acknowledge AI assistance. In "
     "the committed archive, the Naive Bayes report carries an explicit acknowledgement "
     "section, while the NER summary instead documents a feature catalogue synthesized from "
     "AI research sources. The page says what the files say."),
)


def _cs_provenance() -> dict:
    items = _seed_items()
    described = {item["path"] for item in items}
    on_disk = {f"docs/seed-evidence/{p.name}" for p in sorted(SEED_EVIDENCE_DIR.iterdir())
               if p.is_file() and not p.name.startswith(".")}
    if described != on_disk:
        raise ValueError(
            "docs/seed-evidence/ and the provenance section disagree: "
            f"undescribed on disk {sorted(on_disk - described)}, described but missing "
            f"{sorted(described - on_disk)} — the archive is the citation set, so every "
            "file in it must be named on the page")

    numbers = [
        _seed_number("nb_macro_f1", "0.9647679093041557", source=SEED_NB_REPORT,
                     note="macro-F1 on the provided Reuters test set at the required "
                          "feature count, as the report states it"),
        _seed_number("nb_features_baseline", "10000", source=SEED_NB_REPORT, unit="count"),
        _seed_number("nb_features_min", "3000", source=SEED_NB_REPORT, unit="count"),
        _seed_number("nb_features_mid", "5000", source=SEED_NB_REPORT, unit="count"),
        _seed_number("nb_features_max", "15000", source=SEED_NB_REPORT, unit="count"),
        _seed_number("nb_sweep_low", "0.9636702838478547", source=SEED_NB_REPORT,
                     note="lowest macro-F1 in the report's feature-count sweep"),
        _seed_number("nb_sweep_high", "0.9658523189692165", source=SEED_NB_REPORT,
                     note="highest macro-F1 in the report's feature-count sweep"),
        _seed_number("nb_train_docs", "5,787", source=SEED_NB_REPORT, unit="count"),
        _seed_number("nb_test_docs", "2,298", source=SEED_NB_REPORT, unit="count"),
        _seed_number("ner_dev_first", "0.8322", source=SEED_NER_SUMMARY,
                     note="dev macro-F1 of the owner's own first feature phase"),
        _seed_number("ner_dev_best", "0.8733", source=SEED_NER_SUMMARY,
                     note="dev macro-F1 after feature pruning and more iterations"),
        _seed_number("ner_kaggle_low", "0.82335", source=SEED_NER_SUMMARY,
                     note="weakest of the four public scores the summary documents"),
        _seed_number("ner_kaggle_high", "0.83522", source=SEED_NER_SUMMARY,
                     note="best of the four public scores the summary documents"),
        _seed_number("ner_dev_test_gap", "0.04", source=SEED_NER_SUMMARY,
                     note="the dev-to-test macro-F1 gap the summary calls structural"),
        _seed_number("ner_max_iter_from", "15", source=SEED_NER_SUMMARY, unit="count"),
        _seed_number("ner_max_iter_to", "25", source=SEED_NER_SUMMARY, unit="count"),
        _seed_number("scrape_books", "200", source=SEED_SCRAPING_REPORT, unit="count",
                     note="book descriptions crawled, cleaned, tokenized and stemmed"),
        _seed_number("appendix_kaggle_uncited", "0.83615", source=UPGRADE_PLAN,
                     note="quoted only to record the discrepancy in `gaps`; its source "
                          "document is not in docs/seed-evidence/"),
        _seed_number("appendix_starter_baseline", "0.8136", source=UPGRADE_PLAN,
                     note="quoted only to record the discrepancy in `gaps`; its source "
                          "document is not in docs/seed-evidence/"),
    ]
    d = {e["label"]: e["display"] for e in numbers}
    return _cs_section(
        "provenance", "provenance",
        "Provenance — the coursework seeds, and what they are not",
        paragraphs=[
            ("This lab began as the rewrite of three graduate NLP coursework projects, and "
             "the figures in this section are here for lineage only: coursework seed "
             "scores are self-reported class results — provenance, not evidence. No claim "
             "anywhere else on this page rests on them. Everything this lab asserts rests "
             "instead on artifacts a stranger can re-derive: a frozen CFPB snapshot, an "
             "append-only run log, a bootstrap interval on every headline number, and a "
             "reproduction command for each one."),
            ("The numbers below were copied character-for-character out of "
             "docs/seed-evidence/, a read-only citation archive of the original write-ups. "
             "They were not recomputed and have not been verified here; the build simply "
             "refuses to print a figure it cannot find in the archived file. The Naive "
             "Bayes report also carries an explicit AI-assistance acknowledgement, and the "
             "NER project's feature catalogue was synthesized from AI research sources — "
             "which is one more reason the portfolio narrative rests on this lab's "
             "evidence rather than on these."),
            (f"Naive Bayes on the five-class Reuters set, implemented from scratch: "
             f"macro-F1 {d['nb_macro_f1']} at the required "
             f"{d['nb_features_baseline']}-feature setting, over {d['nb_train_docs']} "
             f"training and {d['nb_test_docs']} test documents. Sweeping the feature count "
             f"across {d['nb_features_min']} / {d['nb_features_mid']} / "
             f"{d['nb_features_baseline']} / {d['nb_features_max']} moved the score only "
             f"between {d['nb_sweep_low']} and {d['nb_sweep_high']} — a spread with no "
             f"interval around it, which is exactly the habit this lab was built to break."),
            (f"The MEMM NER project logged four documented feature phases, from a dev "
             f"macro-F1 of {d['ner_dev_first']} to {d['ner_dev_best']}, and four Kaggle "
             f"submissions between {d['ner_kaggle_low']} and {d['ner_kaggle_high']}, each "
             f"with a stated hypothesis and a stated verdict — including one recorded, in "
             f"the summary's own words, as HYPOTHESIS WRONG. That summary also calls its "
             f"dev-to-test gap of about {d['ner_dev_test_gap']} F1 a structural property "
             f"of the corpus: the test split comes from a later period of news with a "
             f"higher out-of-vocabulary rate. A drift protocol grew out of that sentence."),
        ],
        numbers=numbers,
        items=items,
        lineage=SEED_LINEAGE,
        caveats=list(SEED_CAVEATS),
        gaps=list(SEED_GAPS),
        repro=["shasum -a 256 docs/seed-evidence/*",
               "uv run python -m triage_lab.demo_build --all"],
    )


def build_case_study(records: list[dict], resolved: dict, cfg: cost_model.CostConfig, *,
                     receipts: dict, frontier_dir=DEFAULT_FRONTIER_DIR,
                     router_dir=DEFAULT_ROUTER_DIR, cost_dir=DEFAULT_COST_DIR,
                     drift_summary=DEFAULT_DRIFT_SUMMARY,
                     prior_shift_summary=DEFAULT_PRIOR_SHIFT_SUMMARY,
                     paired_within=DEFAULT_PAIRED_WITHIN,
                     tier_b_compare=DEFAULT_TIER_B_COMPARE,
                     perturbation_summary=DEFAULT_PERTURBATION_SUMMARY,
                     oov_summary=DEFAULT_OOV_SUMMARY,
                     onnx_parity=DEFAULT_ONNX_PARITY,
                     live_agreement=DEFAULT_LIVE_AGREEMENT,
                     snapshot_manifest=DEFAULT_SNAPSHOT_MANIFEST,
                     tier_c_compare_dir=DEFAULT_TIER_C_COMPARE_DIR) -> dict:
    """The narrative page, assembled from copied numbers (see the header comment)."""
    def _json(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    drift = _json(drift_summary)
    sections = [
        _cs_intro(records, drift),
        _cs_tiers(resolved, _json(tier_b_compare), cost_dir, tier_c_compare_dir),
        _cs_drift(resolved, drift, _json(prior_shift_summary), _json(paired_within),
                  tier_c_compare_dir),
        _cs_thresholds(drift),
        _cs_router(cfg, frontier_dir=frontier_dir, router_dir=router_dir),
        _cs_robustness(resolved, _json(perturbation_summary)),
        _cs_negatives(resolved, _json(oov_summary), cfg, frontier_dir=frontier_dir,
                      router_dir=router_dir, tier_c_compare_dir=tier_c_compare_dir),
        _cs_verification(records, resolved,
                         yaml.safe_load(Path(snapshot_manifest).read_text("utf-8")),
                         drift, _json(onnx_parity), _json(live_agreement), receipts),
        _cs_limits(),
        _cs_provenance(),
    ]
    return {
        "schema_version": CASE_STUDY_SCHEMA,
        "title": CASE_STUDY_TITLE,
        "source_note": CASE_STUDY_SOURCE_NOTE,
        "repo": repo_block(),
        "sections": sections,
        "pending": [pending_slot(slot["slot"], slot["label"])
                    for slot in CASE_STUDY_PENDING],
        "evidence_classes": EVIDENCE_LEGEND,
    }


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_all(out_dir=DEFAULT_OUT_DIR, *, results_path=DEFAULT_RESULTS_PATH,
              preds_dir=DEFAULT_PREDS_DIR, splits_dir=DEFAULT_SPLITS_DIR,
              cost_config_path=DEMO_COST_CONFIG,
              thresholds_dir=router_sim.DEFAULT_THRESHOLDS_DIR,
              frontier_dir=DEFAULT_FRONTIER_DIR, router_dir=DEFAULT_ROUTER_DIR,
              cost_dir=DEFAULT_COST_DIR,
              drift_summary=DEFAULT_DRIFT_SUMMARY) -> list[Path]:
    """Write all ten contract files. Everything is computed before anything is written."""
    out_dir = Path(out_dir)
    cfg = cost_model.load_cost_config(cost_config_path)
    if not cost_model.prices_tier_b(cfg):
        raise ValueError(
            f"cost config {cfg.path.name} does not price Tier B; the demo ships Tier B "
            "exhibits and must build under the v2 cost generation "
            "(configs/cost_model_v2.yaml)"
        )
    records = predictions.load_records(results_path)
    resolved = resolve_records(records)

    # Frozen TEST-IID inputs (incl. the four Tier B rungs) + the CAL tau* constants,
    # through the router's own gates.
    inputs = router_sim.load_test_inputs(preds_dir, results_path, cost_config=cfg)
    cal_thresholds = router_sim.load_cal_thresholds(
        thresholds_dir, cost_sha256=cfg.sha256, results_path=results_path,
        derivation=OP_VERSION, cost_config=cfg, preds_dir=preds_dir)

    raw_cal_record = record_for(resolved, TIER_A_RAW_CAL, CAL)
    isocal_record = record_for(resolved, TIER_A_ISOCAL_CAL, CAL)
    test_record = record_for(resolved, TIER_A_LOGREG_TEST, TEST_IID)
    art_raw_cal = cost_model.load_artifact_verified(raw_cal_record, preds_dir,
                                                    allowed_splits={CAL})
    art_isocal = cost_model.load_artifact_verified(isocal_record, preds_dir,
                                                   allowed_splits={CAL})
    artifacts = {
        raw_cal_record["run_id"]: art_raw_cal,
        isocal_record["run_id"]: art_isocal,
        test_record["run_id"]: inputs.art_a,
    }
    # The Tier B calibration exhibits reuse the rungs' artifacts — already loaded through
    # `load_artifact_verified` by `router_sim.load_test_inputs`, never loaded twice.
    for config, _, _ in TIER_B_TESTS:
        record = record_for(resolved, config, TEST_IID)
        artifacts[record["run_id"]] = _tier_b_rung(inputs, config).art

    haiku = record_for(resolved, HAIKU_TEST, TEST_IID)
    sonnet = record_for(resolved, SONNET_TEST, TEST_IID)
    pool = paired_pool(haiku, sonnet)
    y_true_by_id = {cid: label for cid, (_, label)
                    in load_split_rows(pool, TEST_IID, splits_dir).items()}
    curated = freeze_check(build_curated_ids(pool, y_true_by_id),
                           out_dir / "curated_ids.json")

    receipts = build_receipts(records)
    payload = {
        "meta.json": build_meta(records, cfg),
        "runs_index.json": build_runs_index(records),
        "frontier.json": build_frontier(resolved, cfg, frontier_dir=frontier_dir,
                                        router_dir=router_dir, cost_dir=cost_dir),
        "policies.json": build_policies(resolved, cfg, art_cal=art_isocal,
                                        record_cal=isocal_record, router_dir=router_dir),
        "drift.json": build_drift(drift_summary),
        "calibration.json": build_calibration(resolved, artifacts),
        "curated_ids.json": curated,
        "samples.json": build_samples(curated, resolved=resolved, inputs=inputs,
                                      cal_thresholds=cal_thresholds, splits_dir=splits_dir),
        "receipts.json": receipts,
        "case_study.json": build_case_study(
            records, resolved, cfg, receipts=receipts, frontier_dir=frontier_dir,
            router_dir=router_dir, cost_dir=cost_dir, drift_summary=drift_summary),
    }
    return [write_json(obj, out_dir / name) for name, obj in sorted(payload.items())]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m triage_lab.demo_build")
    parser.add_argument("--all", action="store_true",
                        help="build every demo/data file (the only supported mode)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--preds-dir", type=Path, default=DEFAULT_PREDS_DIR)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--cost-config", type=Path, default=DEMO_COST_CONFIG)
    parser.add_argument("--thresholds-dir", type=Path,
                        default=router_sim.DEFAULT_THRESHOLDS_DIR)
    args = parser.parse_args(argv)
    if not args.all:
        parser.error("give --all (the contract is one payload; there is no partial mode)")

    paths = build_all(args.out_dir, results_path=args.results, preds_dir=args.preds_dir,
                      splits_dir=args.splits_dir, cost_config_path=args.cost_config,
                      thresholds_dir=args.thresholds_dir)
    for path in paths:
        print(f"{_rel(path):32s} {path.stat().st_size:>9,d} bytes")
    print(f"op_version={OP_VERSION} headline_router={HEADLINE_ROUTER} "
          f"pending_tier_b={list(PENDING_TIER_B_SLOTS)}")
    return 0


if __name__ == "__main__":
    import sys

    from triage_lab.demo_build import main as _main

    sys.exit(_main())
