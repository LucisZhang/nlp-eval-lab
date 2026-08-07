"""Risk-coverage tables (Phase 4 task 1, part B).

A selective classifier answers only when confident; its operating curve is the tradeoff
between *coverage* (fraction answered) and *risk* (error rate on the answered set). This
module builds two consistent views of that curve from a run's per-example artifact
(``data/preds/<run_id>.parquet``) and emits a small, deterministic JSON evidence file per
run (``results/risk_coverage/<run_id>.json``) — the committed input the router phase and
the static demo consume.

Two views, one confidence signal (``p_max``):

- **Threshold domain** (the deployable operating-point table): for each threshold τ over
  the distinct ``p_max`` breakpoints, answer iff ``p_max >= τ`` and report
  ``(tau, n_covered, coverage, selective_accuracy, selective_risk)``. This is the table an
  operator reads to pick a confidence gate. Tests verify it against a brute-force mask at
  *every* distinct threshold (full resolution); the JSON downsamples to ≤512 evenly-spaced
  real breakpoints so the artifact stays small.
- **Rank domain** (the summary metric): rank by confidence descending, ties broken by
  ascending original index — the *identical* convention as ``metrics.risk_coverage_curve``
  — so ``aurc`` here EQUALS ``metrics.aurc_from_codes`` on the same inputs, and
  ``acc_at_cov::c`` EQUALS ``metrics.accuracy_at_coverage_from_codes``. Both are asserted
  in the tests.

CI bands (AURC + selective-accuracy at coverage {0.50, 0.80, 0.90, 0.95}) reuse the
harness bootstrap contract exactly: ``default_rng(BOOTSTRAP_SEED)`` drawing one
``integers(0, n, size=n)`` index vector per replicate for ``N_RESAMPLES`` replicates,
percentile interval. Because the RNG call sequence and the metric conventions match
``harness.bootstrap_ci``, the bands are byte-identical to the logged record's
``aurc`` / ``acc_at_cov::*`` CIs (a test pins this).

Degenerate one-hot inputs (Tier C, ``p_max`` ≡ 1.0) collapse the threshold table to a
single row at τ = 1.0, coverage 1.0 — the honest statement that an LLM label decision
carries no rankable confidence.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from triage_lab import harness, predictions

REPO_ROOT = harness.REPO_ROOT
DEFAULT_PREDS_DIR = harness.DEFAULT_PREDS_DIR
DEFAULT_RC_DIR = REPO_ROOT / "results" / "risk_coverage"

DEFAULT_COVERAGES: tuple[float, ...] = (0.50, 0.80, 0.90, 0.95)
DEFAULT_MAX_POINTS = 512
JSON_ROUND = 10


# ---------------------------------------------------------------------------
# Threshold-domain table
# ---------------------------------------------------------------------------

def _downsample_thresholds(taus_desc: np.ndarray, max_points: int) -> np.ndarray:
    """Pick ≤max_points evenly index-spaced REAL breakpoints (endpoints included)."""
    n = len(taus_desc)
    if max_points is None or n <= max_points:
        return taus_desc
    idx = np.unique(np.linspace(0, n - 1, max_points).round().astype(np.int64))
    return taus_desc[idx]


def threshold_table(p_max, correct, max_points: int | None = None) -> list[dict]:
    """Threshold-domain operating-point rows, τ descending (coverage ascending).

    Grid = distinct p_max values (full resolution); `max_points` downsamples to that many
    evenly-spaced real breakpoints. Coverage/accuracy at τ are over {i : p_max_i >= τ}.
    Coverage 0 is intentionally absent: the grid holds only realizable operating points
    (the standard empirical RC curve), and risk on an empty answered set is undefined.
    """
    p_max = np.asarray(p_max, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    n = len(p_max)
    if n == 0:
        return []
    taus = np.unique(p_max)[::-1]  # distinct, descending
    taus = _downsample_thresholds(taus, max_points) if max_points is not None else taus

    rows = []
    for tau in taus:
        mask = p_max >= tau
        n_cov = int(mask.sum())
        acc = float(correct[mask].mean()) if n_cov else float("nan")
        rows.append({
            "tau": float(tau),
            "n_covered": n_cov,
            "coverage": n_cov / n,
            "selective_accuracy": acc,
            "selective_risk": 1.0 - acc,
        })
    return rows


# ---------------------------------------------------------------------------
# Rank-domain curve / AURC / acc@coverage (metrics.py convention)
# ---------------------------------------------------------------------------

def _confidence_order(p_max: np.ndarray) -> np.ndarray:
    """Descending confidence, ties -> ascending index (stable). Matches metrics.py."""
    return np.argsort(-p_max, kind="stable")


def selective_curve(p_max, correct):
    """Rank-domain (coverage k/N, risk over top-k). Same convention as metrics.py."""
    p_max = np.asarray(p_max, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    n = len(p_max)
    if n == 0:
        return np.array([]), np.array([])
    order = _confidence_order(p_max)
    errors = 1.0 - correct[order]
    k = np.arange(1, n + 1, dtype=np.float64)
    return k / n, np.cumsum(errors) / k


def aurc(p_max, correct) -> float:
    """AURC = mean risk over k=1..N. EQUALS metrics.aurc_from_codes on the same inputs."""
    _, risks = selective_curve(p_max, correct)
    return float(risks.mean()) if risks.size else 0.0


def accuracy_at_coverage(
    p_max, correct, coverages: tuple[float, ...] = DEFAULT_COVERAGES
) -> dict[str, float]:
    """Selective accuracy over the top ceil(c*N) most confident. Matches metrics.py.

    Requires 0 < c <= 1. c = 0 is rejected rather than clamped: the clamp would answer a
    "cover nothing" request with the single most-confident example's accuracy — a number
    that reads as a real operating point and is not one (see `threshold_table`: coverage 0
    is not represented on the empirical RC curve at all).
    """
    p_max = np.asarray(p_max, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    n = len(p_max)
    order = _confidence_order(p_max)
    out: dict[str, float] = {}
    for c in coverages:
        if not (0.0 < float(c) <= 1.0):
            raise ValueError(f"coverage must satisfy 0 < c <= 1, got {c!r}")
        key = f"{c:.2f}"
        if n == 0:
            out[key] = 0.0
            continue
        n_accept = min(max(int(np.ceil(c * n)), 1), n)
        out[key] = float(correct[order[:n_accept]].mean())
    return out


# ---------------------------------------------------------------------------
# Bootstrap CI bands (harness contract, reused RNG sequence)
# ---------------------------------------------------------------------------

def bootstrap_summary(
    p_max,
    correct,
    coverages: tuple[float, ...] = DEFAULT_COVERAGES,
    n_resamples: int = harness.N_RESAMPLES,
    seed: int = harness.BOOTSTRAP_SEED,
) -> dict[str, dict[str, float]]:
    """AURC + acc_at_cov::c CIs via the harness bootstrap contract.

    One `default_rng(seed).integers(0, n, size=n)` per replicate, percentile interval —
    the exact RNG call sequence and metric conventions of `harness.bootstrap_ci`, so these
    bands are byte-identical to the logged record's aurc / acc_at_cov::* CIs.
    """
    p_max = np.asarray(p_max, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    n = len(p_max)

    keys = ["aurc"] + [f"acc_at_cov::{c:.2f}" for c in coverages]
    point = {"aurc": aurc(p_max, correct)}
    for c in coverages:
        point[f"acc_at_cov::{c:.2f}"] = accuracy_at_coverage(p_max, correct, (c,))[f"{c:.2f}"]

    reps = {k: np.empty(n_resamples, dtype=np.float64) for k in keys}
    rng = np.random.default_rng(seed)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        cp = p_max[idx]
        cc = correct[idx]
        reps["aurc"][i] = aurc(cp, cc)
        acc = accuracy_at_coverage(cp, cc, coverages)
        for c in coverages:
            reps[f"acc_at_cov::{c:.2f}"][i] = acc[f"{c:.2f}"]

    out: dict[str, dict[str, float]] = {}
    for k in keys:
        lo, hi = np.percentile(reps[k], [harness.CI_LOWER_PCT, harness.CI_UPPER_PCT])
        out[k] = {"point": float(point[k]), "ci_lo": float(lo), "ci_hi": float(hi)}
    return out


# ---------------------------------------------------------------------------
# JSON table assembly (deterministic)
# ---------------------------------------------------------------------------

def _round(value):
    if isinstance(value, float):
        if math.isnan(value):  # NaN -> None for valid JSON
            return None
        return round(value, JSON_ROUND)
    return value


def _round_ci(ci: dict) -> dict:
    return {k: _round(v) for k, v in ci.items()}


def build_table(art: predictions.PredictionsArtifact, *, max_points=DEFAULT_MAX_POINTS,
                coverages=DEFAULT_COVERAGES) -> dict:
    """Assemble the deterministic risk-coverage JSON object for one artifact."""
    correct = (art.y_pred == art.y_true).astype(np.float64)
    summary = bootstrap_summary(art.p_max, correct, coverages)
    table = threshold_table(art.p_max, correct, max_points=max_points)
    prov = art.provenance
    return {
        "run_id": prov.get("run_id", ""),
        "config_sha256": prov.get("config_sha256", ""),
        "split": prov.get("split", ""),
        "split_sha256": prov.get("split_sha256", ""),
        "n_examples": len(art),
        "class_labels": list(art.class_labels),
        "coverages": [float(c) for c in coverages],
        "summary": {k: _round_ci(v) for k, v in summary.items()},
        "threshold_table": [{k: _round(v) for k, v in row.items()} for row in table],
    }


def write_table_json(obj: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, sort_keys=True, indent=2) + "\n")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_artifacts(selectors, *, select_all: bool, preds_dir: Path) -> list[Path]:
    all_paths = sorted(preds_dir.glob("*.parquet"))
    if select_all:
        return all_paths
    chosen = []
    for sel in selectors:
        matches = [p for p in all_paths if p.stem.startswith(sel)]
        if not matches:
            raise ValueError(f"no artifact in {preds_dir} matches prefix {sel!r}")
        chosen.extend(matches)
    return chosen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m triage_lab.risk_coverage")
    parser.add_argument("run_id", nargs="*", help="run_id prefix(es) whose artifact to read")
    parser.add_argument("--all", action="store_true", help="every artifact under --preds-dir")
    parser.add_argument("--preds-dir", type=Path, default=DEFAULT_PREDS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RC_DIR)
    parser.add_argument("--max-points", type=int, default=DEFAULT_MAX_POINTS)
    args = parser.parse_args(argv)

    if not args.all and not args.run_id:
        parser.error("give run_id prefix(es) or --all")

    paths = _resolve_artifacts(args.run_id, select_all=args.all, preds_dir=args.preds_dir)
    if not paths:
        print(f"no artifacts found under {args.preds_dir}")
        return 0
    for path in paths:
        art = predictions.read_artifact(path)
        obj = build_table(art, max_points=args.max_points)
        out_path = args.out_dir / f"{obj['run_id']}.json"
        write_table_json(obj, out_path)
        s = obj["summary"]["aurc"]
        print(
            f"[{obj['run_id'][:8]}] {obj['split']:16s} n={obj['n_examples']:5d} "
            f"aurc={s['point']:.6f} [{s['ci_lo']:.6f}, {s['ci_hi']:.6f}] "
            f"rows={len(obj['threshold_table'])} -> {out_path}"
        )
    return 0


if __name__ == "__main__":
    import sys

    from triage_lab.risk_coverage import main as _main

    sys.exit(_main())
