#!/usr/bin/env python
"""Export the B2 (DistilBERT) checkpoint to int8 ONNX and verify PyTorch parity.

Phase 2 deliverable (UPGRADE_PLAN.md §8): the deployment point runs int8-quantized in
the browser (transformers.js), so we must prove the quantized ONNX graph agrees with
the PyTorch model it was distilled from. Pipeline:

    checkpoint --torch.onnx.export--> model.onnx (fp32)
              --onnxruntime dynamic quant--> model.int8.onnx (weights int8)
    parity: argmax agreement + mean|Δprob| of the two models on a FIXED-SEED 5k
            subsample of CAL (never TEST-*), written to parity_report.json.

Dynamic quantization (`onnxruntime.quantization.quantize_dynamic`) is used rather than
`optimum` to keep the dependency surface small; it quantizes weights to int8 and keeps
activations fp32, which is the transformers.js-compatible path and enough to hit the
≥99% agreement bar for a topic classifier.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLITS_DIR = REPO_ROOT / "data" / "splits"
DEFAULT_SUBSAMPLE = 5000
SUBSAMPLE_SEED = 20260805
DEFAULT_OPSET = 17


def _load_cal_texts(splits_dir: Path, n: int, seed: int, text_col: str = "narrative"):
    import duckdb

    con = duckdb.connect()
    try:
        con.execute("SET threads=1")
        con.execute("SET preserve_insertion_order=true")
        rows = con.execute(
            f'SELECT "{text_col}" FROM read_parquet(\'{splits_dir / "cal.parquet"}\') '
            "ORDER BY complaint_id"
        ).fetchall()
    finally:
        con.close()
    texts = ["" if r[0] is None else str(r[0]) for r in rows]
    if n < len(texts):
        idx = np.sort(np.random.default_rng(seed).permutation(len(texts))[:n])
        texts = [texts[i] for i in idx]
    return texts


def _max_seq_length(ckpt: Path, fallback: int = 256) -> int:
    meta = ckpt / "training_meta.json"
    if meta.exists():
        return int(json.loads(meta.read_text()).get("max_seq_length", fallback))
    return fallback


def export_and_verify(ckpt: Path, out_dir: Path, splits_dir: Path,
                      n_samples: int, opset: int) -> dict:
    import torch
    from onnxruntime import InferenceSession
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    ckpt = Path(ckpt)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = out_dir / "model.onnx"
    int8_path = out_dir / "model.int8.onnx"

    max_len = _max_seq_length(ckpt)
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    model = AutoModelForSequenceClassification.from_pretrained(str(ckpt)).eval()

    # --- Export fp32 ONNX with dynamic batch + sequence axes -----------------
    dummy = tokenizer(
        ["placeholder narrative for tracing"], truncation=True, max_length=max_len,
        padding="max_length", return_tensors="pt",
    )
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "logits": {0: "batch"},
    }
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(fp32_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,  # legacy TorchScript exporter: honors dynamic_axes, stable graph
    )

    # --- Dynamic int8 quantization ------------------------------------------
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)

    # --- Parity on a fixed-seed CAL subsample --------------------------------
    texts = _load_cal_texts(splits_dir, n_samples, SUBSAMPLE_SEED)
    sess = InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])

    torch_probs, onnx_probs = [], []
    batch = 32
    with torch.no_grad():
        for start in range(0, len(texts), batch):
            chunk = texts[start : start + batch]
            enc = tokenizer(chunk, truncation=True, max_length=max_len, padding=True,
                            return_tensors="pt")
            t_logits = model(**enc).logits.numpy()
            torch_probs.append(_softmax(t_logits))
            o_logits = sess.run(
                ["logits"],
                {"input_ids": enc["input_ids"].numpy(),
                 "attention_mask": enc["attention_mask"].numpy()},
            )[0]
            onnx_probs.append(_softmax(o_logits))

    tp = np.concatenate(torch_probs, axis=0)
    op = np.concatenate(onnx_probs, axis=0)
    agreement = float((tp.argmax(1) == op.argmax(1)).mean())
    mean_abs_prob_delta = float(np.abs(tp - op).mean())

    report = {
        "checkpoint": str(ckpt),
        "n_samples": len(texts),
        "subsample_seed": SUBSAMPLE_SEED,
        "subsample_split": "cal",
        "max_seq_length": max_len,
        "opset": opset,
        "argmax_agreement": agreement,
        "mean_abs_prob_delta": mean_abs_prob_delta,
        "fp32_onnx_bytes": fp32_path.stat().st_size,
        "int8_onnx_bytes": int8_path.stat().st_size,
        "pytorch_weights_bytes": _weights_size(ckpt),
    }
    (out_dir / "parity_report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _weights_size(ckpt: Path) -> int | None:
    for name in ("model.safetensors", "pytorch_model.bin"):
        p = ckpt / name
        if p.exists():
            return p.stat().st_size
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="export_onnx_distilbert")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    p.add_argument("--n-samples", type=int, default=DEFAULT_SUBSAMPLE)
    p.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    args = p.parse_args(argv)

    report = export_and_verify(args.checkpoint, args.out, args.splits_dir,
                               args.n_samples, args.opset)
    print(f"ONNX parity for {args.checkpoint}:")
    print(f"  argmax_agreement   = {report['argmax_agreement']:.4f} "
          f"(n={report['n_samples']})")
    print(f"  mean_abs_prob_delta= {report['mean_abs_prob_delta']:.5f}")
    print(f"  sizes: fp32={report['fp32_onnx_bytes']//1024}KB "
          f"int8={report['int8_onnx_bytes']//1024}KB "
          f"pytorch={(report['pytorch_weights_bytes'] or 0)//1024}KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
