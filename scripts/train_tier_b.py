#!/usr/bin/env python
"""Self-contained Tier-B fine-tuning script (runs on Colab/GPU box, no repo needed).

One config YAML (the same file the eval harness later hashes) + the exported data kit
(`scripts/export_tier_b_data.py`) are the only inputs. Nothing here imports from the
`triage_lab` package, so the human can upload just this file, the config, the kit, and
`requirements-tierb-colab.txt` and run:

    python train_tier_b.py --config tier_b1_modernbert_sa.yaml --data-dir kit/ --out ckpt/

What it guarantees / produces (UPGRADE_PLAN.md §8 Phase 2, "training curves + configs
archived"):

* **Integrity gate first.** Re-hashes train.parquet / cal.parquet against the kit's
  manifest.json and fails loud on any mismatch before a single GPU cycle is spent.
* **Determinism.** Seeds python / numpy / torch and the Trainer's own data-shuffle seed
  from `config.seed`; the parquet is read in complaint_id order.
* **Precision auto-fallback.** bf16 where the GPU supports it (A10/A100), else fp16
  (T4 has no bf16), else fp32 (CPU/MPS smoke).
* **max_seq_length truncation is measured**, not assumed: the realized truncation rate
  on TRAIN is computed with the actual tokenizer and written to training_meta.json.
* Saves under `--out`: the checkpoint (safetensors + tokenizer + config with
  id2label/label2id), `training_log.jsonl` (the per-step loss/eval curve), and
  `training_meta.json` (hardware, wall-clock, effective batch size, key-lib versions,
  data-manifest sha, truncation rate, seed, label order).

Heavy deps (torch/transformers/pyarrow) are imported lazily inside functions so the
integrity/manifest helpers stay importable (and unit-testable) without them.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import yaml

CHUNK_SIZE = 1024 * 1024
TOKENIZE_CHUNK_SIZE = 5000  # tokenize TRAIN/CAL in ~5k-row chunks to bound CPU RAM
KEY_LIBS = ("torch", "transformers", "accelerate", "tokenizers", "safetensors", "numpy")


# ---------------------------------------------------------------------------
# Integrity: verify the uploaded data kit against its manifest before training.
# (Duplicated — deliberately — from export_tier_b_data.sha256_file so this script
# has zero repo imports and runs standalone on Colab.)
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(data_dir: Path) -> dict:
    """Re-hash every file the manifest lists; raise on any drift. Returns the manifest."""
    data_dir = Path(data_dir)
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no manifest.json in data dir {data_dir}")
    manifest = json.loads(manifest_path.read_text())
    for name, info in manifest["files"].items():
        path = data_dir / name
        if not path.exists():
            raise FileNotFoundError(f"manifest lists {name} but it is missing from {data_dir}")
        actual = sha256_file(path)
        if actual != info["sha256"]:
            raise ValueError(
                f"data integrity check failed for {name}: sha256 {actual} "
                f"!= manifest {info['sha256']}"
            )
    return manifest


def manifest_sha(data_dir: Path) -> str:
    return sha256_file(Path(data_dir) / "manifest.json")


# ---------------------------------------------------------------------------
# Config access (only the training-relevant keys are read here).
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if not isinstance(cfg, dict) or "model" not in cfg or "training" not in cfg:
        raise ValueError(f"config {path} missing model/training blocks")
    if "base" not in cfg["model"]:
        raise ValueError(f"config {path} missing model.base (HF model id to fine-tune)")
    return cfg


# ---------------------------------------------------------------------------
# Data loading (pyarrow, lazy import).
# ---------------------------------------------------------------------------

def read_split(data_dir: Path, split: str, text_col: str, label_col: str,
               cap: int | None, seed: int):
    """Read (texts, labels) from a kit parquet in file order, optional capped subsample.

    A `cap` (only used by the smoke config) takes a *seeded* subsample rather than the
    file head, so a few-thousand-row cap still covers all nine classes; real runs pass
    cap=None and read every row in complaint_id order. Selected indices are re-sorted so
    the training row order stays deterministic.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(Path(data_dir) / f"{split}.parquet", columns=[text_col, label_col])
    texts = table.column(text_col).to_pylist()
    labels = table.column(label_col).to_pylist()
    del table  # release the Arrow buffers immediately; downstream uses the Python lists
    gc.collect()
    texts = ["" if t is None else str(t) for t in texts]
    if cap is not None and cap < len(texts):
        import numpy as np

        idx = np.sort(np.random.default_rng(seed).permutation(len(texts))[:cap])
        texts = [texts[i] for i in idx]
        labels = [labels[i] for i in idx]
    return texts, labels


# ---------------------------------------------------------------------------
# Training core (torch/transformers, lazy import).
# ---------------------------------------------------------------------------

def _select_precision():
    """Return (bf16, fp16) flags for the current accelerator."""
    import torch

    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return True, False  # A10/A100 etc.
        return False, True  # T4: no bf16 -> fp16
    return False, False  # MPS / CPU smoke: full precision


def _truncation_rate(tokenizer, texts, max_len: int, sample: int = 20000) -> float:
    """Fraction of (sampled) narratives whose token length exceeds max_len."""
    subset = texts if len(texts) <= sample else texts[:sample]
    lengths = tokenizer(subset, add_special_tokens=True, truncation=False,
                        return_length=True, return_attention_mask=False)["length"]
    over = sum(1 for n in lengths if n > max_len)
    return over / len(subset) if subset else 0.0


def _rss_mb() -> float:
    """Resident set size in MB. Linux /proc/self/status first, resource.getrusage fallback."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0  # kB -> MB
    except OSError:
        pass
    import resource

    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux ru_maxrss is kB; macOS/BSD is bytes.
    return ru / 1024.0 if sys.platform.startswith("linux") else ru / (1024.0 * 1024.0)


def tokenize_chunked(tokenizer, texts, max_len, chunk_size=TOKENIZE_CHUNK_SIZE, tag="", log=None):
    """Tokenize `texts` in `chunk_size`-row chunks; keep each row as a NumPy int32 array.

    Numerically identical to one whole-list ``tokenizer(texts, ...)`` call — tokenization
    is per-text and order-independent, so chunking then concatenating yields the exact same
    id sequences (proven by `arrays_sha256`). The win is memory: only `chunk_size` texts are
    tokenized at once, and rows are stored as compact int32 arrays rather than lists of
    Python ints (~7x smaller). The int32 rows are cast to torch long per batch by the
    collator, so no training math changes. Emits flushed chunk-progress + RSS logs.
    """
    import numpy as np

    ids_rows: list = []
    mask_rows: list = []
    n = len(texts)
    for start in range(0, n, chunk_size):
        enc = tokenizer(
            texts[start : start + chunk_size], truncation=True, max_length=max_len, padding=False
        )
        ids_rows.extend(np.asarray(r, dtype=np.int32) for r in enc["input_ids"])
        mask_rows.extend(np.asarray(r, dtype=np.int32) for r in enc["attention_mask"])
        del enc
        if log is not None:
            log(f"[tokenize:{tag}] {min(start + chunk_size, n)}/{n} rows  rss={_rss_mb():.0f}MB",
                flush=True)
    return ids_rows, mask_rows


def arrays_sha256(ids_rows, mask_rows) -> str:
    """Canonical sha256 over ragged int32 rows — used to prove tokenization equivalence."""
    import numpy as np

    h = hashlib.sha256()
    for tag, rows in ((b"##ids##", ids_rows), (b"##mask##", mask_rows)):
        h.update(tag)
        for r in rows:
            h.update(np.asarray(r, dtype=np.int32).tobytes())
            h.update(b"|")
    return h.hexdigest()


def train(config: dict, data_dir: Path, out_dir: Path) -> dict:
    import numpy as np
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = verify_manifest(data_dir)

    model_cfg = config["model"]
    tr = config["training"]
    data_cfg = config.get("data", {})
    seed = int(config.get("seed", 20260805))
    base = model_cfg["base"]
    max_len = int(tr["max_seq_length"])
    text_col = data_cfg.get("text_column", "narrative")
    label_col = data_cfg.get("label_column", "class")
    train_split = data_cfg.get("train_split", "train")
    cal_split = data_cfg.get("cal_split", "cal")
    cap = tr.get("train_rows_cap")

    set_seed(seed)

    x_train, y_train = read_split(data_dir, train_split, text_col, label_col, cap, seed)
    x_cal, y_cal = read_split(data_dir, cal_split, text_col, label_col, None, seed)

    labels_sorted = sorted(set(y_train))
    label2id = {lab: i for i, lab in enumerate(labels_sorted)}
    id2label = {i: lab for lab, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(base)
    trunc_rate = _truncation_rate(tokenizer, x_train, max_len)

    # Chunked, memory-bounded tokenization (the OOM fix). Store int32 rows, then drop the
    # raw texts + labels and gc so the ~300k-string list does not coexist with the tokens.
    train_ids, train_mask = tokenize_chunked(tokenizer, x_train, max_len, tag="train", log=print)
    train_labels = np.asarray([label2id[v] for v in y_train], dtype=np.int64)
    n_train_rows = len(train_labels)
    del x_train, y_train
    gc.collect()

    cal_ids, cal_mask = tokenize_chunked(tokenizer, x_cal, max_len, tag="cal", log=print)
    cal_labels = np.asarray([label2id[v] for v in y_cal], dtype=np.int64)
    n_cal_rows = len(cal_labels)
    del x_cal, y_cal
    gc.collect()

    class TokenizedDataset(torch.utils.data.Dataset):
        """Holds int32 rows; __getitem__ returns the same list/int structure the collator
        saw before (input_ids/attention_mask as Python-int lists, label as int), so batch
        padding and casting to torch long are byte-for-byte unchanged."""

        def __init__(self, ids_rows, mask_rows, label_codes):
            self.ids = ids_rows
            self.mask = mask_rows
            self.labels = label_codes

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            return {
                "input_ids": self.ids[i].tolist(),
                "attention_mask": self.mask[i].tolist(),
                "labels": int(self.labels[i]),
            }

    train_ds = TokenizedDataset(train_ids, train_mask, train_labels)
    cal_ds = TokenizedDataset(cal_ids, cal_mask, cal_labels)

    from transformers import DataCollatorWithPadding

    collator = DataCollatorWithPadding(tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=len(labels_sorted), id2label=id2label, label2id=label2id
    )

    def compute_metrics(eval_pred):
        logits, gold = eval_pred
        pred = np.asarray(logits).argmax(axis=-1)
        gold = np.asarray(gold)
        k = len(labels_sorted)
        f1s = []
        for c in range(k):
            tp = int(np.sum((pred == c) & (gold == c)))
            fp = int(np.sum((pred == c) & (gold != c)))
            fn = int(np.sum((pred != c) & (gold == c)))
            denom = 2 * tp + fp + fn
            f1s.append(2 * tp / denom if denom else 0.0)
        return {"accuracy": float((pred == gold).mean()), "macro_f1": float(np.mean(f1s))}

    bf16, fp16 = _select_precision()
    per_device = int(tr["per_device_batch_size"])
    grad_accum = int(tr.get("grad_accum", 1))

    args = TrainingArguments(
        output_dir=str(out_dir / "hf_trainer"),
        num_train_epochs=float(tr["epochs"]),
        per_device_train_batch_size=per_device,
        per_device_eval_batch_size=int(config.get("inference", {}).get("batch_size", 64)),
        gradient_accumulation_steps=grad_accum,
        learning_rate=float(tr["learning_rate"]),
        warmup_ratio=float(tr.get("warmup_ratio", 0.06)),
        weight_decay=float(tr.get("weight_decay", 0.01)),
        logging_strategy="steps",
        logging_steps=int(tr.get("logging_steps", 50)),
        eval_strategy="epoch",
        save_strategy="no",
        seed=seed,
        data_seed=seed,
        bf16=bf16,
        fp16=fp16,
        report_to=[],
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=cal_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    t0 = time.perf_counter()
    trainer.train()
    wall = time.perf_counter() - t0

    # Persist the checkpoint (safetensors + config with id2label) + tokenizer.
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # Training curve artifact: one JSON object per Trainer log entry.
    with open(out_dir / "training_log.jsonl", "w", encoding="utf-8") as f:
        f.writelines(
            json.dumps(entry, sort_keys=True) + "\n" for entry in trainer.state.log_history
        )

    lib_versions = {}
    for lib in KEY_LIBS:
        try:
            lib_versions[lib] = __import__(lib).__version__
        except (ImportError, AttributeError):
            lib_versions[lib] = None

    device = (
        f"cuda:{torch.cuda.get_device_name(0)}"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    meta = {
        "base_model": base,
        "seed": seed,
        "max_seq_length": max_len,
        "truncation_rate": trunc_rate,
        "n_train_rows": n_train_rows,
        "n_cal_rows": n_cal_rows,
        "labels": labels_sorted,
        "epochs": float(tr["epochs"]),
        "learning_rate": float(tr["learning_rate"]),
        "per_device_batch_size": per_device,
        "grad_accum": grad_accum,
        "effective_batch_size": per_device * grad_accum * max(1, torch.cuda.device_count()),
        "precision": "bf16" if bf16 else ("fp16" if fp16 else "fp32"),
        "hardware": device,
        "platform": platform.platform(),
        "wall_clock_seconds": wall,
        "lib_versions": lib_versions,
        "data_manifest_sha256": manifest_sha(data_dir),
        "data_input_sha256": manifest.get("input_sha256"),
        "final_eval": trainer.state.log_history[-1] if trainer.state.log_history else {},
    }
    (out_dir / "training_meta.json").write_text(json.dumps(meta, sort_keys=True, indent=2) + "\n")
    return meta


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="train_tier_b")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    meta = train(cfg, args.data_dir, args.out)
    print(f"trained {meta['base_model']} seed={meta['seed']} -> {args.out}")
    print(f"  hardware={meta['hardware']} precision={meta['precision']} "
          f"wall={meta['wall_clock_seconds']:.1f}s trunc={meta['truncation_rate']:.3f}")
    fe = meta["final_eval"]
    if "eval_macro_f1" in fe:
        print(f"  final CAL macro_f1={fe['eval_macro_f1']:.4f} accuracy={fe.get('eval_accuracy'):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
