"""Paired zero-shot-vs-few-shot comparison over two Tier C raw receipt logs.

Read-only bookkeeping helper for the approved CAL ablation (UPGRADE_PLAN.md §6.2). Given
the two arms' ``calls.jsonl`` receipt files (written by ``triage_lab.tier_c``), it:

1. reconstructs each arm's predictions ``{complaint_id -> label}`` by parsing the stored
   ``content`` JSON — a receipt flagged ``parse_failed`` (or whose content will not parse to
   a valid enum label) resolves to the frozen fallback label, exactly as the runner did;
2. **verifies the two id sets are identical** and fails loud otherwise — the arms must be
   the same paired row set (same ``eval_rows_cap`` + ``cap_seed``) or the paired bootstrap /
   McNemar are meaningless;
3. loads the gold labels for those ids from the split parquet (via the runner's loader);
4. runs ``harness.paired_bootstrap_delta`` for accuracy and macro_f1 (A = first CLI arg =
   few-shot by convention; the output is labelled with the actual receipt paths) plus
   ``harness.mcnemar`` on the paired predictions;
5. prints a compact JSON report to stdout.

It never calls an API, never runs a model, and never touches ``results/runs.jsonl``.
``probs`` for the delta call are the same degenerate one-hot the runner emits; for
accuracy/macro_f1 the deltas depend on the predictions only, so the one-hot is a formality
that keeps the call signature consistent.

**Two additive options (2026-08-13), both opt-in; the default invocation is unchanged.**

``--out PATH`` also writes a committed derived artifact under ``results/tier_c_compare/``.
Without it the tool stays exactly what it was — stdout only, nothing on disk — so every
earlier invocation reproduces byte-for-byte. The artifact exists because a number that
only ever appeared on stdout cannot be cited: the case-study page refuses to display a
figure no committed file can be checked against, so three of its comparisons were simply
missing until this tool could write one.

``--pair-on shared`` intersects the two id sets instead of requiring them to be identical.
The default is still ``exact`` (fail loud), because for an ablation with one
``eval_rows_cap``/``cap_seed`` an unequal id set means something is wrong. ``shared`` is
for the legitimately-asymmetric case — Sonnet scored 1,500 of the rows Haiku scored 5,000 —
which previously required hand-filtering the committed receipts into a temp directory. The
intersection is taken on ids alone and the surviving rows are the same rows that filtering
produced, so it reproduces those reports exactly; the counts dropped from each arm are
recorded in the artifact rather than left implicit.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from triage_lab import harness, predictions
from triage_lab.tier_c import DEFAULT_SPLITS_DIR, load_split_frame
from triage_lab.tier_c_prompt import load_prompt_bundle

SCHEMA_VERSION = 1
ANALYSIS = "tier_c_paired_compare"
DEFAULT_OUT_DIR = harness.REPO_ROOT / "results" / "tier_c_compare"

PAIR_EXACT = "exact"
PAIR_SHARED = "shared"

COST_NOTE = "derivation only: no model was run and results/runs.jsonl is untouched"


def read_receipt_predictions(path, labels: list[str], fallback_label: str) -> dict[int, str]:
    """Reconstruct ``{complaint_id -> predicted label}`` from one arm's receipt jsonl.

    Mirrors the runner's decision: parse the stored ``content`` to a valid enum label; a
    ``parse_failed`` receipt (or any content that will not parse to a member of ``labels``)
    resolves to ``fallback_label``. Fails loud on a duplicated ``complaint_id``.
    """
    from triage_lab.tier_c import parse_label

    preds: dict[int, str] = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        receipt = json.loads(line)
        complaint_id = int(receipt["complaint_id"])
        if complaint_id in preds:
            raise ValueError(f"duplicate complaint_id {complaint_id} in receipts {path}")
        label = parse_label(receipt.get("content"), labels)
        if label is None or receipt.get("parse_failed"):
            label = fallback_label
        preds[complaint_id] = label
    return preds


def load_true_labels(split: str, ids, splits_dir: Path, label_column: str = "class",
                     order_column: str = "complaint_id",
                     text_column: str = "narrative") -> dict[int, str]:
    """Gold ``{complaint_id -> class}`` for ``ids`` from the split parquet (runner's loader)."""
    eval_path = Path(splits_dir) / f"{split}.parquet"
    split_ids, _texts, split_labels = load_split_frame(
        eval_path, text_column, label_column, order_column
    )
    lookup = {int(cid): lab for cid, lab in zip(split_ids, split_labels, strict=True)}
    wanted = {int(i) for i in ids}
    missing = wanted - set(lookup)
    if missing:
        raise ValueError(
            f"{len(missing)} receipt ids absent from split {split!r} parquet "
            f"(e.g. {sorted(missing)[:5]})"
        )
    return {cid: lookup[cid] for cid in wanted}


def _one_hot(preds: list[str], labels: list[str]) -> np.ndarray:
    idx = {label: i for i, label in enumerate(labels)}
    out = np.zeros((len(preds), len(labels)), dtype=np.float64)
    for i, label in enumerate(preds):
        out[i, idx[label]] = 1.0
    return out


def compare(
    preds_a: dict[int, str],
    preds_b: dict[int, str],
    id_to_true: dict[int, str],
    labels: list[str],
    *,
    receipts_a: str,
    receipts_b: str,
    split: str,
) -> dict:
    """Paired accuracy/macro_f1 deltas (A - B) with CIs + McNemar over the two arms.

    Fails loud if the two id sets differ (pairing requirement). Ids are processed in ascending
    ``complaint_id`` order so A and B (and y_true) are index-aligned.
    """
    ids_a, ids_b = set(preds_a), set(preds_b)
    if ids_a != ids_b:
        only_a = sorted(ids_a - ids_b)[:5]
        only_b = sorted(ids_b - ids_a)[:5]
        raise ValueError(
            "receipt id sets differ between arms (pairing requirement): "
            f"{len(ids_a - ids_b)} only in A (e.g. {only_a}), "
            f"{len(ids_b - ids_a)} only in B (e.g. {only_b})"
        )

    ordered_ids = sorted(ids_a)
    y_true = [id_to_true[i] for i in ordered_ids]
    pred_a = [preds_a[i] for i in ordered_ids]
    pred_b = [preds_b[i] for i in ordered_ids]
    probs_a = _one_hot(pred_a, labels)
    probs_b = _one_hot(pred_b, labels)

    deltas = {
        metric: harness.paired_bootstrap_delta(
            y_true, pred_a, pred_b, probs_a, probs_b, metric, labels
        )
        for metric in ("accuracy", "macro_f1")
    }
    mcnemar = harness.mcnemar(y_true, pred_a, pred_b, labels)

    return {
        "comparison": "A_minus_B",
        "arm_a_receipts": str(receipts_a),
        "arm_b_receipts": str(receipts_b),
        "arm_a_role": "few_shot (convention: first arg)",
        "arm_b_role": "zero_shot (convention: second arg)",
        "split": split,
        "n_examples": len(ordered_ids),
        "deltas": deltas,
        "mcnemar": mcnemar,
    }


def restrict_to_shared(preds_a: dict[int, str], preds_b: dict[int, str]) -> tuple[
        dict[int, str], dict[int, str], int, int]:
    """Both arms restricted to the ids they share, plus what each side lost.

    Equivalent to hand-filtering one arm's committed receipts down to the other's id set
    (the pre-2026-08-13 procedure) because `compare` sorts by complaint_id and the paired
    bootstrap's index stream depends only on n and the seed: same rows, same order, same
    draws, same numbers.
    """
    shared = set(preds_a) & set(preds_b)
    if not shared:
        raise ValueError("the two arms share no complaint_id; there is nothing to pair")
    return ({cid: preds_a[cid] for cid in shared},
            {cid: preds_b[cid] for cid in shared},
            len(set(preds_a) - shared), len(set(preds_b) - shared))


def resolve_run_id(receipts_path, records: list[dict]) -> dict:
    """The run record whose committed `extra.raw_log_path` IS this receipts file.

    Resolved rather than passed in: a run id typed on a command line is a claim, a run id
    matched against the log is a fact. A path outside the log (a hand-filtered temp copy,
    say) resolves to nulls, which is the honest answer and shows up in the artifact.
    """
    target = Path(receipts_path).resolve()
    for record in records:
        logged = (record.get("extra") or {}).get("raw_log_path")
        if not logged:
            continue
        if (harness.REPO_ROOT / logged).resolve() == target:
            return {
                "run_id": record["run_id"],
                "config_name": Path(record.get("config_path", "")).stem,
                "model_slug": (record.get("extra") or {}).get("model_slug"),
            }
    return {"run_id": None, "config_name": None, "model_slug": None}


def _rel(path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(harness.REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def build_artifact(report: dict, *, key: str, pair_on: str, dropped_a: int, dropped_b: int,
                   labels: list[str], fallback_label: str, prompt_version: str,
                   repro_command: str, records: list[dict]) -> dict:
    """The committed form of a stdout report: same numbers, plus provenance.

    Everything a reader needs to re-derive or challenge the comparison: which receipts,
    which runs, which rows survived pairing, the bootstrap protocol constants, and the
    command. `excludes_zero` is materialized next to each interval so a consumer never has
    to re-derive the one thing the claim actually turns on.
    """
    deltas = {
        metric: {**band, "excludes_zero": bool(band["ci_lo"] > 0 or band["ci_hi"] < 0)}
        for metric, band in report["deltas"].items()
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis": ANALYSIS,
        "key": key,
        "comparison": "A_minus_B",
        "split": report["split"],
        "pairing": pair_on,
        "n_examples": report["n_examples"],
        "arm_a": {
            "role": report["arm_a_role"],
            "receipts": _rel(report["arm_a_receipts"]),
            "n_dropped_unpaired": dropped_a,
            **resolve_run_id(report["arm_a_receipts"], records),
        },
        "arm_b": {
            "role": report["arm_b_role"],
            "receipts": _rel(report["arm_b_receipts"]),
            "n_dropped_unpaired": dropped_b,
            **resolve_run_id(report["arm_b_receipts"], records),
        },
        "deltas": deltas,
        "mcnemar": report["mcnemar"],
        "bootstrap": {
            "method": "percentile",
            "n_resamples": harness.N_RESAMPLES,
            "seed": harness.BOOTSTRAP_SEED,
            "ci_pct": [harness.CI_LOWER_PCT, harness.CI_UPPER_PCT],
            "pairing": "shared_index_vectors_per_replicate",
        },
        "class_labels": list(labels),
        "fallback_label": fallback_label,
        "prompt_version": prompt_version,
        "protocol_note": (
            "frozen protocol: a parse_failed call enters as the run's fallback label, "
            "exactly as the runner scored it; no row is dropped for failing to parse"
        ),
        "cost_note": COST_NOTE,
        "cost_usd": 0.0,
        "repro_command": repro_command,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "git_sha": harness._git_sha(),
    }


def write_artifact(artifact: dict, path) -> Path:
    """Deterministic atomic write: sorted keys, 2-space indent, trailing newline.

    Byte-identical across runs except `generated_at` and `git_sha`, the two fields every
    other analysis artifact in `results/` also stamps.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(artifact, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m triage_lab.tier_c_compare")
    parser.add_argument("receipts_a", help="arm A receipts calls.jsonl (few-shot by convention)")
    parser.add_argument("receipts_b", help="arm B receipts calls.jsonl (zero-shot by convention)")
    parser.add_argument("--split", default="cal", help="split whose gold labels to load")
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--prompt-version", default="v1")
    parser.add_argument("--pair-on", choices=(PAIR_EXACT, PAIR_SHARED), default=PAIR_EXACT,
                        help="exact (default): id sets must match. shared: intersect them.")
    parser.add_argument("--out", type=Path, default=None,
                        help="also write a committed derived artifact to this path")
    parser.add_argument("--key", default=None,
                        help="artifact key; defaults to --out's stem")
    parser.add_argument("--role-a", default=None,
                        help="what arm A actually is (default: the CAL-ablation convention)")
    parser.add_argument("--role-b", default=None)
    parser.add_argument("--results", type=Path, default=harness.DEFAULT_RESULTS_PATH)
    args = parser.parse_args(argv)

    bundle = load_prompt_bundle(args.prompt_version)
    labels = bundle.labels
    fallback_label = labels[0]

    preds_a = read_receipt_predictions(args.receipts_a, labels, fallback_label)
    preds_b = read_receipt_predictions(args.receipts_b, labels, fallback_label)
    dropped_a = dropped_b = 0
    if args.pair_on == PAIR_SHARED:
        preds_a, preds_b, dropped_a, dropped_b = restrict_to_shared(preds_a, preds_b)
    id_to_true = load_true_labels(args.split, list(preds_a), args.splits_dir)

    report = compare(
        preds_a, preds_b, id_to_true, labels,
        receipts_a=args.receipts_a, receipts_b=args.receipts_b, split=args.split,
    )
    if args.role_a is not None:
        report["arm_a_role"] = args.role_a
    if args.role_b is not None:
        report["arm_b_role"] = args.role_b
    print(json.dumps(report, sort_keys=True, indent=2))

    if args.out is not None:
        argv_shown = argv if argv is not None else sys.argv[1:]
        # shlex.quote so the recorded command is PASTEABLE: the role strings contain
        # spaces and parentheses, and a repro command you have to repair is not one.
        repro = "uv run python -m triage_lab.tier_c_compare " + " ".join(
            shlex.quote(str(a)) for a in argv_shown)
        artifact = build_artifact(
            report, key=args.key or Path(args.out).stem, pair_on=args.pair_on,
            dropped_a=dropped_a, dropped_b=dropped_b, labels=labels,
            fallback_label=fallback_label, prompt_version=args.prompt_version,
            repro_command=repro, records=predictions.load_records(args.results))
        path = write_artifact(artifact, args.out)
        print(f"wrote {_rel(path)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
