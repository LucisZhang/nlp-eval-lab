"""Paired-comparison step of Tier B harness finals (docs/TIER_B_RUNBOOK.md §6).

Loads the four Tier B eval records (B1 seeds a/b/c + B2) plus the Tier A baseline
record from `results/runs.jsonl`, aligns their per-example prediction artifacts on
`complaint_id`, and computes paired bootstrap deltas (macro_f1, accuracy) + McNemar
for every B1-seed-vs-A, B1-seed-vs-B2 and B2-vs-A comparison, plus the B1 seed-variance
block. Uses `triage_lab.harness.paired_bootstrap_delta` / `.mcnemar` — no metric math is
reimplemented here.

Writes a single JSON summary to `--out` and prints a compact table to stdout. Does not
touch `results/runs.jsonl` or any other committed file (CLAUDE.md hard rule 3).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from triage_lab import harness, predictions

REPO_ROOT = harness.REPO_ROOT
DEFAULT_RUNS_JSONL = harness.DEFAULT_RESULTS_PATH
DEFAULT_PREDS_DIR = harness.DEFAULT_PREDS_DIR

DEFAULT_B1_CONFIGS = [
    "configs/tier_b1_modernbert_sa.yaml",
    "configs/tier_b1_modernbert_sb.yaml",
    "configs/tier_b1_modernbert_sc.yaml",
]
DEFAULT_B2_CONFIG = "configs/tier_b2_distilbert_s0.yaml"
DEFAULT_BASELINE_CONFIG = "configs/tier_a_logreg_test_iid.yaml"
DEFAULT_OUT = "results/tier_b_compare/summary.json"

PAIRED_METRICS = ("macro_f1", "accuracy")
EXTRA_PROVENANCE_KEYS = ("T", "temperature", "inference_hardware", "train_hardware")


def _repo_path(rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    return p if p.is_absolute() else REPO_ROOT / p


def select_latest_run(records: list[dict], config_path: str) -> tuple[dict, list[dict]]:
    """Latest-by-timestamp record whose config_path matches; superseded ones returned too."""
    matches = [r for r in records if r.get("config_path") == config_path]
    if not matches:
        raise ValueError(
            f"no run record in runs.jsonl has config_path={config_path!r}"
        )
    matches.sort(key=lambda r: r["timestamp_utc"])
    latest = matches[-1]
    superseded = matches[:-1]
    return latest, superseded


def artifact_path_for(record: dict, preds_dir: Path) -> Path:
    extra = record.get("extra") or {}
    if extra.get("predictions_path"):
        return _repo_path(extra["predictions_path"])
    return preds_dir / f"{record['run_id']}.parquet"


def load_system(record: dict, preds_dir: Path) -> dict:
    art_path = artifact_path_for(record, preds_dir)
    if not art_path.exists():
        raise FileNotFoundError(
            f"predictions artifact not found for run {record['run_id'][:8]} "
            f"(config {record.get('config_path')}) at {art_path}"
        )
    art = predictions.read_artifact(art_path)
    order = np.argsort(art.complaint_id, kind="stable")
    return {
        "record": record,
        "art_path": str(art_path),
        "ids": art.complaint_id[order],
        "y_true": art.y_true[order],
        "y_pred": art.y_pred[order],
        "probs": art.probs[order] if art.probs.shape[1] else None,
        "class_labels": list(art.class_labels),
    }


def assert_identical_ids(systems: dict[str, dict]) -> None:
    names = list(systems)
    ref_name = names[0]
    ref_ids = set(int(i) for i in systems[ref_name]["ids"])
    for name in names[1:]:
        ids = set(int(i) for i in systems[name]["ids"])
        if ids != ref_ids:
            only_ref = sorted(ref_ids - ids)[:5]
            only_other = sorted(ids - ref_ids)[:5]
            raise ValueError(
                f"id set mismatch between {ref_name!r} and {name!r}: "
                f"{len(ref_ids)} vs {len(ids)} ids; "
                f"in {ref_name} only (sample): {only_ref}; "
                f"in {name} only (sample): {only_other}"
            )


def _provenance_block(sysinfo: dict) -> dict:
    record = sysinfo["record"]
    extra = record.get("extra") or {}
    prov = {
        "run_id": record["run_id"],
        "config_path": record.get("config_path"),
        "config_sha256": record.get("config_sha256"),
        "timestamp_utc": record.get("timestamp_utc"),
        "macro_f1_point": record["metrics"]["macro_f1"]["point"],
    }
    for key in EXTRA_PROVENANCE_KEYS:
        if key in extra:
            prov[key] = extra[key]
    return prov


def compare_pair(name_a: str, sys_a: dict, name_b: str, sys_b: dict) -> dict:
    """Paired bootstrap deltas + McNemar for A vs B, on their (identical, sorted) ids."""
    class_labels = sys_a["class_labels"]
    out: dict = {"a": name_a, "b": name_b, "n": len(sys_a["ids"])}
    for metric in PAIRED_METRICS:
        out[metric] = harness.paired_bootstrap_delta(
            sys_a["y_true"],
            sys_a["y_pred"],
            sys_b["y_pred"],
            sys_a["probs"],
            sys_b["probs"],
            metric,
            class_labels,
        )
    out["mcnemar"] = harness.mcnemar(sys_a["y_true"], sys_a["y_pred"], sys_b["y_pred"])
    return out


def seed_variance_block(b1_systems: list[dict]) -> dict:
    points = [s["record"]["metrics"]["macro_f1"]["point"] for s in b1_systems]
    arr = np.asarray(points, dtype=np.float64)
    return {
        "metric": "macro_f1",
        "source": "run record metrics.macro_f1.point (TEST-IID)",
        "n_seeds": len(points),
        "points": points,
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if len(points) > 1 else 0.0,
        "sd_ddof": 1,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "range": float(arr.max() - arr.min()),
    }


def build_summary(
    runs_jsonl: Path,
    b1_configs: list[str],
    b2_config: str,
    baseline_config: str,
    preds_dir: Path,
) -> dict:
    records = predictions.load_records(runs_jsonl)

    selections: dict[str, tuple[dict, list[dict]]] = {}
    selections["baseline"] = select_latest_run(records, baseline_config)
    selections["b2"] = select_latest_run(records, b2_config)
    seed_names = ["b1_sa", "b1_sb", "b1_sc"]
    for name, cfg in zip(seed_names, b1_configs, strict=True):
        selections[name] = select_latest_run(records, cfg)

    systems = {name: load_system(rec, preds_dir) for name, (rec, _sup) in selections.items()}
    assert_identical_ids(systems)

    comparisons = []
    for name in seed_names:
        comparisons.append(compare_pair(name, systems[name], "baseline", systems["baseline"]))
        comparisons.append(compare_pair(name, systems[name], "b2", systems["b2"]))
    comparisons.append(compare_pair("b2", systems["b2"], "baseline", systems["baseline"]))

    provenance = {name: _provenance_block(systems[name]) for name in systems}
    superseded = {
        name: [
            {"run_id": r["run_id"], "timestamp_utc": r["timestamp_utc"]}
            for r in sup
        ]
        for name, (_rec, sup) in selections.items()
        if sup
    }

    return {
        "runs_jsonl": str(runs_jsonl),
        "bootstrap": {"n_resamples": harness.N_RESAMPLES, "seed": harness.BOOTSTRAP_SEED},
        "comparisons": comparisons,
        "seed_variance": seed_variance_block([systems[n] for n in seed_names]),
        "provenance": provenance,
        "superseded": superseded,
    }


def _fmt_delta(d: dict) -> str:
    return f"{d['delta']:+.4f} [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]"


def print_summary(summary: dict) -> None:
    print(f"bootstrap: n_resamples={summary['bootstrap']['n_resamples']} "
          f"seed={summary['bootstrap']['seed']}")
    print()
    header = f"{'A':10s} {'B':10s} {'n':>7s}  {'macro_f1 delta [CI]':32s}  {'accuracy delta [CI]':32s}  mcnemar(b,c,p)"
    print(header)
    print("-" * len(header))
    for cmp in summary["comparisons"]:
        mc = cmp["mcnemar"]
        print(
            f"{cmp['a']:10s} {cmp['b']:10s} {cmp['n']:7d}  "
            f"{_fmt_delta(cmp['macro_f1']):32s}  {_fmt_delta(cmp['accuracy']):32s}  "
            f"({mc['b']}, {mc['c']}, p={mc['p_value']:.3g})"
        )
    print()
    sv = summary["seed_variance"]
    print(
        f"B1 seed variance ({sv['metric']}, ddof={sv['sd_ddof']}): "
        f"mean={sv['mean']:.4f} sd={sv['sd']:.4f} "
        f"min={sv['min']:.4f} max={sv['max']:.4f} range={sv['range']:.4f} "
        f"points={sv['points']}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-jsonl", type=Path, default=DEFAULT_RUNS_JSONL)
    parser.add_argument("--b1-configs", nargs=3, default=DEFAULT_B1_CONFIGS)
    parser.add_argument("--b2-config", default=DEFAULT_B2_CONFIG)
    parser.add_argument("--baseline-config", default=DEFAULT_BASELINE_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--preds-dir", type=Path, default=DEFAULT_PREDS_DIR)
    args = parser.parse_args(argv)

    summary = build_summary(
        args.runs_jsonl,
        list(args.b1_configs),
        args.b2_config,
        args.baseline_config,
        args.preds_dir,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print_summary(summary)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
