"""Tier B — fine-tuned transformer runner (UPGRADE_PLAN.md §4.2).

Training happens off-repo on a cloud GPU (see `scripts/train_tier_b.py` + the runbook);
this module is the *evaluation* half that runs locally through the harness. Given a
config pointing at a downloaded checkpoint dir it:

1. loads the checkpoint (tokenizer + `AutoModelForSequenceClassification`) and runs
   batched inference on the requested eval split (MPS if available, else CPU) to get
   raw logits — the class order is the checkpoint's own `id2label`;
2. optionally fits a single temperature scalar `T` by NLL on the CAL split's logits
   (LBFGS, `T > 0` via a log-parametrization) and divides eval logits by it —
   `calibration: "temperature" | "none"`, mirroring tier_a's `calibration` key;
3. returns a `RunnerResult` whose `probs` are the post-temperature softmax, plus an
   `extra` block recording the checkpoint content hash, fitted `T`, the training
   hardware/precision/truncation-rate (read back from `training_meta.json`), and the
   inference hardware — so a run record is self-describing.

Integrity: the eval + CAL split parquets are re-hashed against the frozen
splits_stats.yaml before inference (same fail-loud gate as tier_a). `max_seq_length`
is read from the checkpoint's `training_meta.json` so inference tokenization matches
training exactly; the config value is only a fallback.

Heavy transformers imports are lazy (inside the runner) so `fit_temperature` and the
config/checkpoint helpers stay importable — and unit-testable — with just torch.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import torch

from triage_lab.harness import RunnerResult, dataset_info, register_runner
from triage_lab.snapshot import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLITS_DIR = REPO_ROOT / "data" / "splits"

_DEFAULT_TEXT_COLUMN = "narrative"
_DEFAULT_LABEL_COLUMN = "class"
_DEFAULT_ORDER_COLUMN = "complaint_id"
_DEFAULT_CAL_SPLIT = "cal"
_DEFAULT_INFER_BATCH = 64
_DEFAULT_MAX_SEQ_LEN = 256


# ---------------------------------------------------------------------------
# Config / checkpoint helpers (torch-only, no transformers import)
# ---------------------------------------------------------------------------

def _splits_dir(config: dict) -> Path:
    return Path(config.get("data", {}).get("splits_dir", DEFAULT_SPLITS_DIR))


def checkpoint_dir(config: dict) -> Path:
    ckpt = config.get("model", {}).get("checkpoint")
    if not ckpt:
        raise ValueError("tier_b config missing model.checkpoint (path to trained dir)")
    return Path(ckpt)


def checkpoint_content_hash(ckpt_dir: Path) -> str:
    """sha256 of the weights file (safetensors preferred, else pytorch bin)."""
    ckpt_dir = Path(ckpt_dir)
    for name in ("model.safetensors", "pytorch_model.bin"):
        path = ckpt_dir / name
        if path.exists():
            return sha256_file(path)
    raise FileNotFoundError(f"no weights file (model.safetensors/pytorch_model.bin) in {ckpt_dir}")


def load_training_meta(ckpt_dir: Path) -> dict:
    path = Path(ckpt_dir) / "training_meta.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def resolve_max_seq_length(config: dict, meta: dict) -> int:
    """Prefer the length training actually used; fall back to config, then default."""
    if meta.get("max_seq_length"):
        return int(meta["max_seq_length"])
    tr = config.get("training", {})
    return int(tr.get("max_seq_length", _DEFAULT_MAX_SEQ_LEN))


def select_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Temperature scaling (single scalar T by NLL, LBFGS; testable in isolation)
# ---------------------------------------------------------------------------

def fit_temperature(logits, labels, max_iter: int = 100) -> float:
    """Fit T>0 minimizing NLL of softmax(logits / T). Parametrize log T to keep T>0."""
    z = torch.as_tensor(np.asarray(logits), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(labels), dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)  # T = exp(0) = 1
    nll = torch.nn.CrossEntropyLoss()
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = nll(z / torch.exp(log_t), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_t.detach()).item())


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Data loading + integrity (mirrors tier_a's gate)
# ---------------------------------------------------------------------------

def load_split_frame(path, text_column, label_column, order_column):
    con = duckdb.connect()
    try:
        con.execute("SET threads=1")
        con.execute("SET preserve_insertion_order=true")
        rows = con.execute(
            f'SELECT "{text_column}", "{label_column}" '
            f"FROM read_parquet('{path}') "
            f'ORDER BY "{order_column}"'
        ).fetchall()
    finally:
        con.close()
    texts = [("" if r[0] is None else str(r[0])) for r in rows]
    labels = np.array([r[1] for r in rows], dtype=object)
    return texts, labels


def _verify_integrity(split: str, path: Path, splits_stats_path: Path) -> None:
    info = dataset_info(split, splits_stats_path)
    actual = sha256_file(path)
    if actual != info["split_sha256"]:
        raise ValueError(
            f"integrity check failed for split {split!r}: parquet sha256 {actual} "
            f"!= frozen splits_stats.yaml {info['split_sha256']}"
        )


def _load_ids(path, order_column: str) -> np.ndarray:
    """Load the split's order column (complaint_id) in the same order as load_split_frame,
    so ids stay aligned to texts/labels for the per-example predictions artifact."""
    con = duckdb.connect()
    try:
        con.execute("SET threads=1")
        con.execute("SET preserve_insertion_order=true")
        rows = con.execute(
            f'SELECT "{order_column}" FROM read_parquet(\'{path}\') '
            f'ORDER BY "{order_column}"'
        ).fetchall()
    finally:
        con.close()
    return np.array([r[0] for r in rows], dtype=np.int64)


def subsample_indices(n: int, cap: int | None, seed: int) -> np.ndarray:
    """Deterministic re-sorted subsample index vector (shared by texts/labels/ids).

    `cap=None`/`cap>=n` -> full identity range; else a seeded `default_rng(seed)`
    permutation of the first `cap` rows, re-sorted so row order stays stable.
    """
    if cap is None or cap >= n:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).permutation(n)[:cap])


def subsample_eval(texts, labels, cap: int | None, seed: int):
    """Seeded in-memory subsample of an already-loaded eval split (frozen file untouched).

    `cap=None` (all shipped/real configs) is a no-op that returns the full split; a small
    `cap` (plumbing dry-runs) takes a deterministic seeded subset, re-sorted so row order
    stays stable. The full frozen parquet's sha256 is still verified upstream — only the
    in-memory view is thinned, never the file on disk.
    """
    idx = subsample_indices(len(texts), cap, seed)
    if len(idx) == len(texts) and np.array_equal(idx, np.arange(len(texts))):
        return texts, labels
    return [texts[i] for i in idx], labels[idx]


# ---------------------------------------------------------------------------
# Batched inference -> logits
# ---------------------------------------------------------------------------

def _infer_logits(model, tokenizer, texts, max_len, batch_size, device) -> np.ndarray:
    import torch as _torch  # local alias; module-level torch already imported

    out = []
    model.eval()
    with _torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = tokenizer(
                batch, truncation=True, max_length=max_len, padding=True, return_tensors="pt"
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits.float().cpu().numpy()
            out.append(logits)
    return np.concatenate(out, axis=0) if out else np.empty((0, 0))


# ---------------------------------------------------------------------------
# Registered runner
# ---------------------------------------------------------------------------

@register_runner("tier_b")
def tier_b_runner(config: dict) -> RunnerResult:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    data = config.get("data", {})
    text_col = data.get("text_column", _DEFAULT_TEXT_COLUMN)
    label_col = data.get("label_column", _DEFAULT_LABEL_COLUMN)
    order_col = data.get("order_column", _DEFAULT_ORDER_COLUMN)
    eval_split = data["split"]
    cal_split = data.get("cal_split", _DEFAULT_CAL_SPLIT)
    calibration = config.get("calibration", "none")
    batch_size = int(config.get("inference", {}).get("batch_size", _DEFAULT_INFER_BATCH))
    verify = data.get("verify_sha256", True)

    if calibration not in ("temperature", "none"):
        raise ValueError(f"unknown calibration {calibration!r}; choose 'temperature' or 'none'")

    ckpt = checkpoint_dir(config)
    meta = load_training_meta(ckpt)
    max_len = resolve_max_seq_length(config, meta)

    splits_dir = _splits_dir(config)
    stats_path = splits_dir / "splits_stats.yaml"
    eval_path = splits_dir / f"{eval_split}.parquet"
    if verify:
        _verify_integrity(eval_split, eval_path, stats_path)

    device = select_device()
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    model = AutoModelForSequenceClassification.from_pretrained(str(ckpt)).to(device)

    num_labels = model.config.num_labels
    id2label = model.config.id2label
    class_labels = [id2label[i] for i in range(num_labels)]

    eval_rows_cap = data.get("eval_rows_cap")
    seed = int(config.get("seed", 20260805))
    x_eval, y_eval = load_split_frame(eval_path, text_col, label_col, order_col)
    ids_eval = _load_ids(eval_path, order_col)
    sub_idx = subsample_indices(len(x_eval), eval_rows_cap, seed)
    x_eval = [x_eval[i] for i in sub_idx]
    y_eval = y_eval[sub_idx]
    ids_eval = ids_eval[sub_idx]
    eval_logits = _infer_logits(model, tokenizer, x_eval, max_len, batch_size, device)

    temperature = 1.0
    if calibration == "temperature":
        cal_path = splits_dir / f"{cal_split}.parquet"
        if verify:
            _verify_integrity(cal_split, cal_path, stats_path)
        x_cal, y_cal = load_split_frame(cal_path, text_col, label_col, order_col)
        cal_logits = _infer_logits(model, tokenizer, x_cal, max_len, batch_size, device)
        label2id = {lab: i for i, lab in enumerate(class_labels)}
        cal_codes = np.array([label2id[v] for v in y_cal], dtype=np.int64)
        temperature = fit_temperature(cal_logits, cal_codes)

    probs = softmax_np(eval_logits / temperature)
    y_pred = np.array([class_labels[i] for i in probs.argmax(axis=1)], dtype=object)

    dataset = dataset_info(eval_split, stats_path)
    extra = {
        "checkpoint": str(ckpt),
        "checkpoint_sha256": checkpoint_content_hash(ckpt),
        "temperature": temperature,
        "calibration": calibration,
        "max_seq_length": max_len,
        "num_labels": num_labels,
        "inference_hardware": device,
        "training_hardware": meta.get("hardware"),
        "training_precision": meta.get("precision"),
        "truncation_rate": meta.get("truncation_rate"),
        "base_model": meta.get("base_model"),
        "seed": meta.get("seed", config.get("seed")),
        "eval_rows_cap": eval_rows_cap,
        "eval_sample_size": len(y_eval),
        "run_type": config.get("run_type", "standard"),
    }
    return RunnerResult(
        y_true=np.asarray(y_eval, dtype=object),
        y_pred=y_pred,
        probs=np.asarray(probs, dtype=np.float64),
        class_labels=class_labels,
        dataset=dataset,
        cost_usd=None,
        extra=extra,
        ids=np.asarray(ids_eval, dtype=np.int64),
    )
