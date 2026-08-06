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

It is strictly read-only: it opens the receipt files and the split parquet for reading and
writes nothing to disk (results, logs, or otherwise). ``probs`` for the delta call are the
same degenerate one-hot the runner emits; for accuracy/macro_f1 the deltas depend on the
predictions only, so the one-hot is a formality that keeps the call signature consistent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from triage_lab import harness
from triage_lab.tier_c import DEFAULT_SPLITS_DIR, load_split_frame
from triage_lab.tier_c_prompt import load_prompt_bundle


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m triage_lab.tier_c_compare")
    parser.add_argument("receipts_a", help="arm A receipts calls.jsonl (few-shot by convention)")
    parser.add_argument("receipts_b", help="arm B receipts calls.jsonl (zero-shot by convention)")
    parser.add_argument("--split", default="cal", help="split whose gold labels to load")
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--prompt-version", default="v1")
    args = parser.parse_args(argv)

    bundle = load_prompt_bundle(args.prompt_version)
    labels = bundle.labels
    fallback_label = labels[0]

    preds_a = read_receipt_predictions(args.receipts_a, labels, fallback_label)
    preds_b = read_receipt_predictions(args.receipts_b, labels, fallback_label)
    id_to_true = load_true_labels(args.split, list(preds_a), args.splits_dir)

    report = compare(
        preds_a, preds_b, id_to_true, labels,
        receipts_a=args.receipts_a, receipts_b=args.receipts_b, split=args.split,
    )
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
