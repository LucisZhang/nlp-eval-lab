"""Static demo data builder (Phase 6) — every file `demo/DATA_CONTRACT.md` specifies.

Repro:

    make demo-data        # uv run python -m triage_lab.demo_build --all

`demo/DATA_CONTRACT.md` is the normative spec; this module is its producer side and the
static site (`demo/index.html` + `demo/assets/`) is its consumer side. Nine files land in
``demo/data/`` and are committed, so the demo is fully static: no server, no API call, no
model run at page load.

**Nothing here is a new measurement.** Every number is either COPIED from an append-only
results record / committed artifact (metrics, CIs, costs, thresholds, drift rollup) or
DERIVED from a frozen per-example artifact by arithmetic that carries no free parameters
(calibration bins, the CAL tau sweep, the per-sample router path). Each object says which,
via ``evidence_class`` and a ``run_id`` / ``source`` provenance field, because a demo is
exactly where an unattributed number does the most damage.

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

**Tier B is pending, not omitted.** Every Tier B slot is a real object
``{"pending": true, "slot": ..., "label": ...}`` so the backfill replaces it in place and a
missing panel is visible as a missing panel rather than as silence.

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
from collections import Counter
from pathlib import Path

import duckdb
import numpy as np

from triage_lab import (
    cost_model,
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

SCHEMA_VERSION = "demo-v1"

# The primary operating point for every reported router number (STATUS.md Phase 4 task 5).
OP_VERSION = router_sim.OP_V2

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

CALIBRATION_N_BINS = metrics.DEFAULT_N_BINS  # 15, the repo's ECE binning
# The bins must be the ones the logged ECE was computed over, not merely 15 bins that look
# like them, so the recomputation is pinned against the record at this tolerance.
ECE_REPLAY_TOL = 1e-9

# <= this many rows in the published CAL tau sweep (the full grid is one row per distinct
# p_max, ~87k on CAL — a slider does not need them and a demo payload cannot carry them).
TAU_SWEEP_MAX_POINTS = 256

ROUND = cost_model.JSON_ROUND

# --- pending Tier B slots (real objects everywhere, per the contract) --------------------
PENDING_FRONTIER_POINTS = (
    ("tier_b1_modernbert", "Tier B1 — ModernBERT-base (3 seeds)"),
    ("tier_b2_distilbert", "Tier B2 — DistilBERT int8 ONNX"),
    ("router_a_b_c", "Router A→B→C"),
)
PENDING_SAMPLE_TIERS = (
    ("tier_b1", "Tier B1 — ModernBERT-base (3 seeds)"),
    ("tier_b2", "Tier B2 — DistilBERT int8 ONNX"),
)
PENDING_CALIBRATION = (
    ("tier_b1_temp_scaling", "Tier B1 — temperature scaling"),
    ("tier_b2_temp_scaling", "Tier B2 — temperature scaling"),
)
PENDING_TIER_B_SLOTS = tuple(
    slot for slot, _ in (*PENDING_FRONTIER_POINTS, *PENDING_SAMPLE_TIERS, *PENDING_CALIBRATION)
)

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
    "a_to_c tau is frozen from Phase 4 CAL optimization; the demo does not re-solve it "
    "(Haiku scored only the paired subset on CAL)."
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
    "Supports differ by point and are NOT comparable as populations: Tier A and the "
    "a_to_human router are scored on the full 104,443-row TEST-IID slice, Haiku and the "
    "a_to_c cascade on Haiku's 5,000-row uniform subsample, Sonnet on the paired 1,500."
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
    """`A` or `C` for a record. Tier B records do not exist yet (pending)."""
    extra = record.get("extra") or {}
    if extra.get("tier"):
        return str(extra["tier"]).replace("tier_", "").upper()
    name = _config_name(record)
    for prefix, tier in (("tier_a_", "A"), ("tier_b_", "B"), ("tier_c_", "C")):
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

def primary_frontier_path(frontier_dir=DEFAULT_FRONTIER_DIR) -> Path:
    """The one committed opv2 frontier file. Zero or many is a hard failure."""
    candidates = sorted(Path(frontier_dir).glob("frontier__opv2__*.json"))
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one primary (opv2) frontier file under {frontier_dir}, "
            f"found {[p.name for p in candidates]}; `make frontier` writes one per cost "
            "config and the demo cannot pick between them"
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
                  run_refs: list[str]) -> dict:
    """A router frontier point, copied from the frozen v2 router_sim evaluation.

    `macro_f1` carries no CI: `router_sim` logs `macro_f1_system` as a point estimate (the
    bootstrap it does run is on PAIRED deltas, which is what a comparison claim needs).
    The contract's `{"point": ...}`-only form is used rather than inventing an interval.
    """
    return {
        "key": key,
        "label": label,
        "kind": "router",
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
    frontier_path = primary_frontier_path(frontier_dir)
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

    points = [
        _single_tier_point("tier_a_logreg", "Tier A — TF-IDF LogReg", logreg,
                           cost_dir=cost_dir),
        _single_tier_point("tier_a_cnb", "Tier A — TF-IDF ComplementNB", cnb,
                           cost_dir=cost_dir),
        _single_tier_point("tier_c_haiku", "Tier C — Claude Haiku 4.5 (TEST-IID)", haiku,
                           cost_dir=cost_dir),
        _single_tier_point("tier_c_sonnet",
                           "Tier C — Claude Sonnet 5 (TEST-IID subsample)", sonnet,
                           cost_dir=cost_dir),
        _router_point("a_to_human", "Router A→human", full["policies"]["a_to_human"],
                      source=full_path, run_refs=[logreg["run_id"]]),
        _router_point("a_to_c_haiku", "Router A→Haiku (terminal, parse-fail→human)",
                      paired["policies"][threshold_opt.FAMILY_A_TO_C],
                      source=paired_path,
                      run_refs=[logreg["run_id"], haiku["run_id"]]),
    ]
    return {
        "claims": {**claims, "source": _rel(frontier_path)},
        "points": points,
        "pending_points": [pending_slot(slot, label)
                           for slot, label in PENDING_FRONTIER_POINTS],
        "support_note": SUPPORT_NOTE,
    }


# ---------------------------------------------------------------------------
# 4. policies.json
# ---------------------------------------------------------------------------

def _policy_block(key: str, label: str, policy: dict, *, source: Path,
                  run_refs: list[str]) -> dict:
    routing = policy["routing"]
    gate = policy["gate"]
    accuracy_machine = policy["accuracy_machine"]
    return {
        "key": key,
        "label": label,
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
    return {
        "op_version": OP_VERSION,
        "cost_defaults": {
            "c_misroute": cfg.c_misroute_usd,
            "c_human": cfg.c_human_usd,
            "source": _rel(cfg.path),
            "sha256": cfg.sha256,
            "evidence_class": "estimated",
        },
        "policies": [
            _policy_block("a_to_human", "Tier A gate → human queue",
                          full["policies"]["a_to_human"], source=full_path,
                          run_refs=[logreg["run_id"]]),
            _policy_block("a_to_c_haiku", "Tier A gate → Haiku terminal (parse-fail→human)",
                          paired["policies"][threshold_opt.FAMILY_A_TO_C],
                          source=paired_path,
                          run_refs=[logreg["run_id"], haiku["run_id"]]),
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
    path = Path(summary_path)
    return {
        "summary": json.loads(path.read_text(encoding="utf-8")),
        "annotations": [dict(a) for a in DRIFT_ANNOTATIONS],
        "pending_series": [pending_slot("tier_b1"), pending_slot("tier_b2")],
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
    """Three exhibits: the raw CAL rung, the same rung isotonic, the shipped TEST-IID point.

    There is deliberately no "Tier A logreg RAW on TEST-IID" exhibit, because no such run
    exists: every reported TEST-IID final is `calibration: isotonic` (fit on CAL). The raw
    vs isotonic contrast is therefore shown where it was actually measured — on CAL, on the
    two rungs that differ by exactly that one switch — and each exhibit states its own
    slice rather than borrowing the other's.
    """
    raw_cal = record_for(resolved, TIER_A_RAW_CAL, CAL)
    isocal = record_for(resolved, TIER_A_ISOCAL_CAL, CAL)
    test_iid = record_for(resolved, TIER_A_LOGREG_TEST, TEST_IID)
    haiku = record_for(resolved, HAIKU_TEST, TEST_IID)
    sonnet = record_for(resolved, SONNET_TEST, TEST_IID)
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
        "pending": [pending_slot(slot, label) for slot, label in PENDING_CALIBRATION],
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
    """complaint_id -> the frozen v2 cascade's path for that row, plus the tau it used.

    The policy is built by `router_sim.build_paired_policies`, so the op semantics are the
    Phase 4 ones by construction. The A-gate outcome is then recomputed with the router's
    own expression (`p_max >= tau`) and cross-checked against the policy's `to_human`
    vector, which is defined as `(~answered) & parse_failed` — if the two disagree, this
    module has drifted from the op it claims to replay and the build fails.
    """
    policies = {p.name: p
                for p in router_sim.build_paired_policies(inputs, cal_thresholds)}
    policy = policies[threshold_opt.FAMILY_A_TO_C]
    tau = float(policy.gate["tau"])
    answered = np.asarray(inputs.art_a.p_max[inputs.index_paired], dtype=np.float64) >= tau
    if not np.array_equal(policy.to_human, (~answered) & inputs.parse_failed):
        raise ValueError(
            "recomputed A-gate outcome disagrees with router_sim's to_human vector; the "
            "demo's router path is not the frozen op"
        )
    out = {}
    for cid, is_answered, to_human in zip(policy.ids, answered, policy.to_human,
                                          strict=True):
        if is_answered:
            path = ["A", "answered"]
        else:
            path = ["A", "escalated", "C", "human" if to_human else "answered"]
        out[int(cid)] = path
    return out, tau


def build_samples(curated: dict, *, resolved: dict, inputs: router_sim.TestInputs,
                  cal_thresholds: dict, splits_dir=DEFAULT_SPLITS_DIR) -> dict:
    ids = list(curated["complaint_ids"])
    logreg = record_for(resolved, TIER_A_LOGREG_TEST, TEST_IID)
    haiku = record_for(resolved, HAIKU_TEST, TEST_IID)
    sonnet = record_for(resolved, SONNET_TEST, TEST_IID)

    class_labels = list(inputs.art_a.class_labels)
    split_rows = load_split_rows(ids, TEST_IID, splits_dir)
    index_a = threshold_opt.restrict_to_ids(inputs.art_a, ids)
    haiku_fields = tier_c_sample_fields(ids, haiku, class_labels)
    sonnet_fields = tier_c_sample_fields(ids, sonnet, class_labels)
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
                "tier_b1": pending_slot("tier_b1", PENDING_SAMPLE_TIERS[0][1]),
                "tier_b2": pending_slot("tier_b2", PENDING_SAMPLE_TIERS[1][1]),
                "haiku": {**haiku_row, "correct": haiku_row["label"] == y_true},
                "sonnet": {**sonnet_row, "correct": sonnet_row["label"] == y_true},
            },
            "router": {
                "op_version": OP_VERSION,
                "policy": "a_to_c_haiku",
                "tau": tau,
                "path": paths[cid],
                "note": ROUTER_PATH_NOTE,
            },
        })
    return {
        "selection": {**curated, "narrative_source": NARRATIVE_SOURCE},
        "samples": samples,
        "class_labels": class_labels,
        "parse_failure_note": (
            "A row with parse_failed=true carries the run's frozen FALLBACK label "
            f"({(haiku.get('extra') or {}).get('fallback_label')!r}); `correct` scores that "
            "fallback, while the router discards it and sends the row to a human."
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
# Build
# ---------------------------------------------------------------------------

def build_all(out_dir=DEFAULT_OUT_DIR, *, results_path=DEFAULT_RESULTS_PATH,
              preds_dir=DEFAULT_PREDS_DIR, splits_dir=DEFAULT_SPLITS_DIR,
              cost_config_path=cost_model.DEFAULT_COST_CONFIG,
              thresholds_dir=router_sim.DEFAULT_THRESHOLDS_DIR,
              frontier_dir=DEFAULT_FRONTIER_DIR, router_dir=DEFAULT_ROUTER_DIR,
              cost_dir=DEFAULT_COST_DIR,
              drift_summary=DEFAULT_DRIFT_SUMMARY) -> list[Path]:
    """Write all nine contract files. Everything is computed before anything is written."""
    out_dir = Path(out_dir)
    cfg = cost_model.load_cost_config(cost_config_path)
    records = predictions.load_records(results_path)
    resolved = resolve_records(records)

    # Frozen TEST-IID inputs + the CAL tau* constants, through the router's own gates.
    inputs = router_sim.load_test_inputs(preds_dir, results_path)
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

    haiku = record_for(resolved, HAIKU_TEST, TEST_IID)
    sonnet = record_for(resolved, SONNET_TEST, TEST_IID)
    pool = paired_pool(haiku, sonnet)
    y_true_by_id = {cid: label for cid, (_, label)
                    in load_split_rows(pool, TEST_IID, splits_dir).items()}
    curated = freeze_check(build_curated_ids(pool, y_true_by_id),
                           out_dir / "curated_ids.json")

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
        "receipts.json": build_receipts(records),
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
    parser.add_argument("--cost-config", type=Path, default=cost_model.DEFAULT_COST_CONFIG)
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
    print(f"op_version={OP_VERSION} pending_tier_b={list(PENDING_TIER_B_SLOTS)}")
    return 0


if __name__ == "__main__":
    import sys

    from triage_lab.demo_build import main as _main

    sys.exit(_main())
