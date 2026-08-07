"""Eval harness: one YAML config -> one appended JSONL record in `results/`.

This module owns the *measurement contract* of the lab (UPGRADE_PLAN.md §4.3,
§6.1, §6.4). It is deliberately model-agnostic: Tier A/B/C runners register into
`RUNNERS` and return predictions + probabilities; the harness turns those into a
point-metric dict, bootstrap 95% CIs, and an append-only, fully-provenanced JSONL
record. No tier-specific logic lives here.

Determinism guarantees (the reason this file is careful):

- **Bootstrap** uses `numpy.random.default_rng(BOOTSTRAP_SEED)` with N_RESAMPLES
  replicates. For a given replicate we draw the resample index vector *once* and
  reuse it across *every* metric, so all metrics of a run are computed on mutually
  consistent replicates and the CIs are byte-reproducible run-to-run.
- **Paired comparisons** draw one index vector per replicate from a fresh
  `default_rng(BOOTSTRAP_SEED)` and apply it to *both* systems, so the delta CI is
  a proper paired bootstrap. Identical systems therefore give delta == 0 exactly on
  every replicate -> CI [0, 0].
- **McNemar** is the exact two-sided binomial test on the discordant pairs (no
  normal/continuity approximation), computed with `math.comb`.
- **Records are append-only.** `append_record` opens in "a" mode and writes one
  compact, sorted-key JSON line. It never rewrites. A correction is a *new* record
  carrying `supersedes_run_id`; the harness has no update/delete path by design
  (CLAUDE.md hard rule 3).

Hashing/provenance reuses `snapshot.sha256_file` so config hashing is identical to
the rest of the Phase-0 pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

from triage_lab import metrics
from triage_lab.snapshot import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_PATH = REPO_ROOT / "results" / "runs.jsonl"
DEFAULT_SPLITS_STATS_PATH = REPO_ROOT / "data" / "splits" / "splits_stats.yaml"
# Per-example prediction artifacts land here (data/ is gitignored; artifacts are
# regenerable via `python -m triage_lab.predictions`, never committed).
DEFAULT_PREDS_DIR = REPO_ROOT / "data" / "preds"

# Frozen bootstrap constants (repo convention: same seed as the Phase-0 split RNG
# salt; see splits.py SEED). Changing either is a methodology change, not a tweak.
N_RESAMPLES = 1000
BOOTSTRAP_SEED = 20260805
CI_LOWER_PCT = 2.5
CI_UPPER_PCT = 97.5

# IEEE-754 float64 range limits, used by `mcnemar` to decide when its exact-integer tail
# has to be scaled in log space: the largest finite value is just under 2^1024, and the
# smallest positive subnormal is 2^-1074 (so 0.5**n silently becomes 0.0 for n > 1074).
MAX_FLOAT_EXP = 1024
MIN_SUBNORMAL_EXP = 1074
BOOTSTRAP_METHOD = "percentile"


# ---------------------------------------------------------------------------
# Model-runner registry
# ---------------------------------------------------------------------------

RUNNERS: dict[str, Callable[[dict], RunnerResult]] = {}

# Runner modules imported on demand so their registration side effects fire
# without the harness hard-depending on tier-specific libraries at import time.
# tier_b pulls in torch/transformers (the optional `tierb` dep group); a light CI
# env without that group must still resolve tier_a, so import failures are skipped.
# tier_c keeps its `openai` import lazy (inside the runner), so its module import
# always succeeds and registers even in a light env — it only needs the `tierc`
# group + OPENROUTER_API_KEY when actually invoked.
_OPTIONAL_RUNNER_MODULES = (
    "triage_lab.tier_a",
    "triage_lab.tier_b",
    "triage_lab.tier_c",
)


def _load_optional_runners() -> None:
    """Import known runner modules so their `@register_runner` decorators run.

    Missing optional dependencies (e.g. torch for tier_b) are tolerated: the module is
    skipped so the other tiers stay usable. A truly-unknown runner still fails loud in
    `run()` when the registry lookup misses.
    """
    import importlib

    for module in _OPTIONAL_RUNNER_MODULES:
        try:
            importlib.import_module(module)
        except ImportError:
            continue


@dataclass(frozen=True)
class RunnerResult:
    """What a registered runner returns: predictions, probs, and dataset provenance.

    `dataset` must carry {split, split_sha256, input_sha256}; use `dataset_info` to
    populate it from splits_stats.yaml for real splits, or supply synthetic values
    in tests. `cost_usd` is None for local (Tier A/B) runners; Tier C fills it from
    measured token usage.
    """

    y_true: np.ndarray
    y_pred: np.ndarray
    probs: np.ndarray
    class_labels: list
    dataset: dict
    cost_usd: float | None = None
    extra: dict = field(default_factory=dict)
    # Optional per-example identifiers (complaint_id), id-aligned to y_true/y_pred/probs.
    # When a runner supplies these, `run()` persists a per-example predictions artifact
    # (data/preds/<run_id>.parquet) and records its path under extra.predictions_path.
    # Absent (None) -> no artifact, record schema unchanged (backward compatible).
    ids: np.ndarray | None = None


def register_runner(name: str) -> Callable[[Callable[[dict], RunnerResult]], Callable]:
    """Decorator registering a runner under `name` for `config["model"]["runner"]`."""

    def deco(fn: Callable[[dict], RunnerResult]) -> Callable[[dict], RunnerResult]:
        if name in RUNNERS:
            raise ValueError(f"runner {name!r} already registered")
        RUNNERS[name] = fn
        return fn

    return deco


# ---------------------------------------------------------------------------
# Config loading + hashing
# ---------------------------------------------------------------------------

def load_config(path) -> dict:
    """Load + minimally validate a run config YAML.

    Required keys: `model.runner` (registry key) and `data.split`. The raw file
    bytes are what get hashed downstream (see `config_sha256`), so comments and
    formatting are part of the config identity — edit = new hash = new run.
    """
    path = Path(path)
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise TypeError(f"config {path} did not parse to a mapping")
    model = config.get("model")
    if not isinstance(model, dict) or "runner" not in model:
        raise ValueError(f"config {path} missing required key model.runner")
    data = config.get("data")
    if not isinstance(data, dict) or "split" not in data:
        raise ValueError(f"config {path} missing required key data.split")
    return config


def config_sha256(path) -> str:
    """sha256 of the raw config bytes (reuses the Phase-0 file hasher)."""
    return sha256_file(Path(path))


def dataset_info(split: str, splits_stats_path=DEFAULT_SPLITS_STATS_PATH) -> dict:
    """Build the record `dataset` block from the frozen splits_stats.yaml."""
    stats = yaml.safe_load(Path(splits_stats_path).read_text())
    split_stats = stats["splits"][split]
    return {
        "split": split,
        "split_sha256": split_stats["sha256"],
        "input_sha256": stats["input_sha256"],
    }


# ---------------------------------------------------------------------------
# Point metrics
# ---------------------------------------------------------------------------

def _scalar_metrics_from_codes(
    true_idx: np.ndarray,
    pred_idx: np.ndarray,
    probs: np.ndarray,
    class_labels: list,
    coverages: tuple[float, ...] = metrics.DEFAULT_COVERAGES,
) -> dict[str, float]:
    """Every scalar the harness CIs, computed from pre-encoded integer codes."""
    k = len(class_labels)
    out: dict[str, float] = {
        "macro_f1": metrics.macro_f1_from_codes(true_idx, pred_idx, k),
        "balanced_accuracy": metrics.balanced_accuracy_from_codes(true_idx, pred_idx, k),
        "accuracy": metrics.accuracy_from_codes(true_idx, pred_idx),
        "ece": metrics.expected_calibration_error_from_codes(true_idx, probs),
        "brier": metrics.brier_score_from_codes(true_idx, probs),
        "aurc": metrics.aurc_from_codes(true_idx, probs),
    }
    per_class = metrics.per_class_f1_from_codes(true_idx, pred_idx, k)
    for label, value in zip(class_labels, per_class, strict=True):
        out[f"f1::{label}"] = float(value)
    for cov_key, value in metrics.accuracy_at_coverage_from_codes(
        true_idx, probs, coverages
    ).items():
        out[f"acc_at_cov::{cov_key}"] = value
    return out


def evaluate(y_true, y_pred, probs, class_labels) -> dict[str, float]:
    """Point-metric dict for one system (no CIs). Keys are stable metric names."""
    true_idx = metrics.encode_labels(y_true, class_labels)
    pred_idx = metrics.encode_labels(y_pred, class_labels)
    probs = np.asarray(probs, dtype=np.float64)
    return _scalar_metrics_from_codes(true_idx, pred_idx, probs, class_labels)


# ---------------------------------------------------------------------------
# Bootstrap CIs
# ---------------------------------------------------------------------------

def bootstrap_ci(
    y_true,
    y_pred,
    probs,
    class_labels,
    n_resamples: int = N_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, dict[str, float]]:
    """Percentile 95% bootstrap CI for every scalar metric.

    One resample index vector per replicate, reused across all metrics so the
    replicate is consistent. Returns {metric: {point, ci_lo, ci_hi}}.
    """
    true_idx = metrics.encode_labels(y_true, class_labels)
    pred_idx = metrics.encode_labels(y_pred, class_labels)
    probs = np.asarray(probs, dtype=np.float64)
    n = len(true_idx)

    point = _scalar_metrics_from_codes(true_idx, pred_idx, probs, class_labels)
    keys = list(point)
    replicates = {k: np.empty(n_resamples, dtype=np.float64) for k in keys}

    rng = np.random.default_rng(seed)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        rep = _scalar_metrics_from_codes(
            true_idx[idx], pred_idx[idx], probs[idx], class_labels
        )
        for k in keys:
            replicates[k][i] = rep[k]

    out: dict[str, dict[str, float]] = {}
    for k in keys:
        lo, hi = np.percentile(replicates[k], [CI_LOWER_PCT, CI_UPPER_PCT])
        out[k] = {"point": float(point[k]), "ci_lo": float(lo), "ci_hi": float(hi)}
    return out


# ---------------------------------------------------------------------------
# Paired comparisons
# ---------------------------------------------------------------------------

# Single-system scalar metric resolvers for paired deltas. Each maps
# (true_idx, pred_idx, probs, n_classes) -> float.
_PAIRED_METRICS: dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray, int], float]] = {
    "macro_f1": lambda t, p, pr, k: metrics.macro_f1_from_codes(t, p, k),
    "balanced_accuracy": lambda t, p, pr, k: metrics.balanced_accuracy_from_codes(t, p, k),
    "accuracy": lambda t, p, pr, k: metrics.accuracy_from_codes(t, p),
    "ece": lambda t, p, pr, k: metrics.expected_calibration_error_from_codes(t, pr),
    "brier": lambda t, p, pr, k: metrics.brier_score_from_codes(t, pr),
    "aurc": lambda t, p, pr, k: metrics.aurc_from_codes(t, pr),
}


def paired_bootstrap_delta(
    y_true,
    pred_a,
    pred_b,
    probs_a,
    probs_b,
    metric: str,
    class_labels,
    n_resamples: int = N_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    """Paired bootstrap of (metric(A) - metric(B)); same indices hit both systems.

    A difference is only claimed where the returned CI excludes zero (§6.1).
    """
    if metric not in _PAIRED_METRICS:
        raise ValueError(f"unknown paired metric {metric!r}; choose from {sorted(_PAIRED_METRICS)}")
    fn = _PAIRED_METRICS[metric]
    k = len(class_labels)
    true_idx = metrics.encode_labels(y_true, class_labels)
    a_idx = metrics.encode_labels(pred_a, class_labels)
    b_idx = metrics.encode_labels(pred_b, class_labels)
    pa = np.asarray(probs_a, dtype=np.float64)
    pb = np.asarray(probs_b, dtype=np.float64)
    n = len(true_idx)

    point = fn(true_idx, a_idx, pa, k) - fn(true_idx, b_idx, pb, k)
    deltas = np.empty(n_resamples, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        deltas[i] = fn(true_idx[idx], a_idx[idx], pa[idx], k) - fn(
            true_idx[idx], b_idx[idx], pb[idx], k
        )
    lo, hi = np.percentile(deltas, [CI_LOWER_PCT, CI_UPPER_PCT])
    return {"delta": float(point), "ci_lo": float(lo), "ci_hi": float(hi)}


def mcnemar(y_true, pred_a, pred_b, class_labels=None) -> dict[str, float]:
    """Exact two-sided binomial McNemar test on discordant pairs.

    b = #(A right, B wrong), c = #(A wrong, B right), n = b + c.
    p = min(1, 2 * Σ_{i=0}^{min(b,c)} C(n, i) * 0.5^n). n == 0 -> p = 1.0.

    The tail is accumulated exactly in Python ints via the recurrence
    ``C(n, i+1) = C(n, i) * (n - i) // (i + 1)`` — same values as summing ``math.comb``,
    but O(k) big-int operations instead of recomputing each binomial from scratch, which
    matters once n reaches the tens of thousands (a full TEST-IID model-vs-model
    comparison).

    The final scaling is done in log space whenever the float expression
    ``2.0 * tail * 0.5**n`` cannot represent the answer, decided by EXPONENT BOUNDS rather
    than by catching an exception — because the two failure modes are not symmetric:

    - **overflow** (``2 * tail`` ≥ 2^1024) raises, and is loud;
    - **underflow** (``n > 1074``, so ``0.5**n`` is 0.0) does NOT raise. It silently
      returns p = 0.0 for any strongly imbalanced comparison on a slice of more than ~1k
      discordant pairs — a fabricated "infinitely significant" result. Probe: n = 1075
      with min(b, c) = 0 returned exactly 0.0 instead of ≈5e-324.

    Results are bit-identical to the float path wherever that path was well-defined; only
    the two out-of-range regimes changed, and one of them was previously silent.
    """
    yt = np.asarray(y_true)
    pa = np.asarray(pred_a)
    pb = np.asarray(pred_b)
    a_correct = pa == yt
    b_correct = pb == yt
    b = int(np.sum(a_correct & ~b_correct))
    c = int(np.sum(~a_correct & b_correct))
    n = b + c
    if n == 0:
        p_value = 1.0
    else:
        k = min(b, c)
        term = 1  # C(n, 0)
        tail = 1
        for i in range(k):
            term = term * (n - i) // (i + 1)
            tail += term
        # float64 can represent 2*tail only below 2^1024, and 0.5**n only for n <= 1074.
        if (2 * tail).bit_length() <= MAX_FLOAT_EXP and n <= MIN_SUBNORMAL_EXP:
            p_value = min(1.0, 2.0 * tail * (0.5**n))
        else:
            log_p = math.log(2.0) + math.log(tail) - n * math.log(2.0)
            p_value = min(1.0, math.exp(log_p)) if log_p < 0.0 else 1.0
    return {"b": b, "c": c, "n_discordant": n, "p_value": float(p_value)}


# ---------------------------------------------------------------------------
# Append-only JSONL records
# ---------------------------------------------------------------------------

def _git_sha() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO_ROOT,
        )
        return proc.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def build_record(
    config_path,
    config_hash: str,
    metrics_ci: dict[str, dict[str, float]],
    dataset: dict,
    wall_clock_seconds: float,
    cost_usd: float | None,
    *,
    git_sha: str | None = None,
    timestamp_utc: str | None = None,
    n_resamples: int = N_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    supersedes_run_id: str | None = None,
    extra: dict | None = None,
) -> dict:
    """Assemble a fully-provenanced result record. `run_id` = sha256(config+git+ts).

    `extra` is the runner's own provenance block (e.g. Tier B checkpoint sha, fitted
    temperature, hardware, truncation rate). It is persisted verbatim under the record's
    ``extra`` key when non-empty; local runners that supply nothing (Tier A) leave the
    key absent so their record schema is unchanged.
    """
    if timestamp_utc is None:
        timestamp_utc = datetime.now(UTC).isoformat()
    if git_sha is None:
        git_sha = _git_sha()
    run_id = hashlib.sha256(
        f"{config_hash}:{git_sha}:{timestamp_utc}".encode()
    ).hexdigest()
    record = {
        "run_id": run_id,
        "timestamp_utc": timestamp_utc,
        "git_sha": git_sha,
        "config_path": str(config_path),
        "config_sha256": config_hash,
        "dataset": dataset,
        "metrics": metrics_ci,
        "bootstrap": {
            "n_resamples": n_resamples,
            "seed": seed,
            "method": BOOTSTRAP_METHOD,
        },
        "wall_clock_seconds": wall_clock_seconds,
        "cost_usd": cost_usd,
    }
    if supersedes_run_id is not None:
        record["supersedes_run_id"] = supersedes_run_id
    if extra:
        record["extra"] = extra
    return record


def append_record(results_path, record: dict) -> None:
    """Append one compact, sorted-key JSON line. Never rewrites (CLAUDE.md rule 3)."""
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True, separators=(",", ":"))
    with open(results_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# End-to-end run
# ---------------------------------------------------------------------------

def _persist_predictions(record: dict, result: RunnerResult, cfg_hash: str, preds_dir) -> None:
    """Write the per-example predictions artifact and stamp extra.predictions_path.

    Only called when the runner supplied `ids`. Imported lazily to avoid a module-level
    cycle (predictions imports harness). The artifact is regenerable and gitignored; its
    path (relative to the repo when possible) is recorded so the run is self-describing.

    The bound provenance mirrors the record's own: code (git_sha), config bytes, data
    (split + split_sha256 + snapshot input_sha256) and, for Tier C, the frozen prompt's
    bundle hash out of the runner's `extra`. Anything a tier does not carry is written as
    "" by `ArtifactProvenance`.
    """
    from triage_lab import predictions

    art_path = Path(preds_dir) / f"{record['run_id']}.parquet"
    provenance = predictions.ArtifactProvenance(
        run_id=record["run_id"],
        config_sha256=cfg_hash,
        split=result.dataset["split"],
        split_sha256=result.dataset.get("split_sha256", ""),
        class_labels=list(result.class_labels),
        git_sha=record.get("git_sha", ""),
        input_sha256=result.dataset.get("input_sha256", ""),
        prompt_bundle_sha256=(result.extra or {}).get("prompt_bundle_sha256", ""),
    )
    predictions.write_artifact(
        art_path,
        ids=result.ids,
        y_true=result.y_true,
        y_pred=result.y_pred,
        probs=result.probs,
        class_labels=result.class_labels,
        provenance=provenance,
    )
    try:
        rel = str(art_path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        rel = str(art_path)
    record.setdefault("extra", {})["predictions_path"] = rel


def _resolve_preds_dir(preds_dir, results_path) -> Path:
    """Where the predictions artifact goes when a runner supplies ids.

    Explicit `preds_dir` wins. Otherwise real runs (default results log) land in
    `data/preds`; a redirected results log (tests, sandboxes) puts artifacts beside it
    so those runs never pollute the repo's `data/preds`.
    """
    if preds_dir is not None:
        return Path(preds_dir)
    if Path(results_path) == DEFAULT_RESULTS_PATH:
        return DEFAULT_PREDS_DIR
    return Path(results_path).parent / "preds"


def run(
    config_path,
    results_path=DEFAULT_RESULTS_PATH,
    *,
    append: bool = True,
    preds_dir=None,
) -> dict:
    """Resolve the runner from config, time it, evaluate + CI, (optionally) append.

    `append=False` (CLI `--no-append`) returns the built record without writing it —
    used for smoke/pipeline proofs that must never touch the append-only results log; a
    dry-run also writes no predictions artifact.
    """
    config_path = Path(config_path)
    config = load_config(config_path)
    cfg_hash = config_sha256(config_path)
    runner_name = config["model"]["runner"]
    if runner_name not in RUNNERS:
        _load_optional_runners()
    if runner_name not in RUNNERS:
        raise ValueError(f"unknown runner {runner_name!r}; registered: {sorted(RUNNERS)}")
    runner = RUNNERS[runner_name]

    t0 = time.perf_counter()
    result = runner(config)
    wall = time.perf_counter() - t0

    metrics_ci = bootstrap_ci(
        result.y_true, result.y_pred, result.probs, result.class_labels
    )
    record = build_record(
        config_path,
        cfg_hash,
        metrics_ci,
        result.dataset,
        wall,
        result.cost_usd,
        extra=result.extra,
    )
    # Auto-persist the per-example artifact for real (appended) runs whose runner
    # supplied ids. Dry-runs (--no-append) and id-less runners leave the log/schema
    # untouched (backward compatible).
    if append and getattr(result, "ids", None) is not None:
        _persist_predictions(record, result, cfg_hash, _resolve_preds_dir(preds_dir, results_path))
    if append:
        append_record(results_path, record)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m triage_lab.harness")
    parser.add_argument("config", type=Path, help="path to the run config YAML")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument(
        "--no-append",
        action="store_true",
        help="do not write to the results log; print the record JSON to stdout instead "
        "(for smoke/pipeline proofs)",
    )
    args = parser.parse_args(argv)

    record = run(args.config, args.results, append=not args.no_append)
    if args.no_append:
        print(json.dumps(record, sort_keys=True, indent=2))
        return 0
    print(f"run_id={record['run_id'][:16]} split={record['dataset']['split']}")
    for name in ("macro_f1", "balanced_accuracy", "ece", "brier", "aurc"):
        m = record["metrics"][name]
        print(f"  {name:18s} {m['point']:.4f}  [{m['ci_lo']:.4f}, {m['ci_hi']:.4f}]")
    return 0


if __name__ == "__main__":
    # Delegate to the canonically-imported module. `python -m triage_lab.harness`
    # runs this file as `__main__`, but runner registration goes through
    # `from triage_lab.harness import register_runner` (see tier_a.py), which loads
    # a SECOND module object under the real name `triage_lab.harness` with its own
    # RUNNERS dict. Calling this file's `main()` would read the (empty) __main__
    # registry. Re-importing and calling the canonical `main` guarantees the run()
    # path reads the same RUNNERS that decorators populate.
    from triage_lab.harness import main as _canonical_main

    sys.exit(_canonical_main())
