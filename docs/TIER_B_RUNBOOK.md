# Tier B Runbook — fine-tune off-repo, evaluate locally

Phase 2 trains four checkpoints on a **cloud GPU** (Colab T4/A10 or a rented box) and
evaluates them **locally** (Apple M4, MPS/CPU) through the existing harness. This is the
exact, ordered procedure. Nothing here touches TEST-* during iteration; every reported
number is one config → one JSONL record.

Models: **B1 = `answerdotai/ModernBERT-base`** (149M, headline, 3 seeds) and
**B2 = `distilbert-base-uncased`** (66M, deployment point, 1 seed).

**Frozen Tier-B seed list (never changes):** `a = 20260805`, `b = 20260806`,
`c = 20260807` — the Phase-0 base seed and its two successors. B2 uses `s0 = 20260805`.

---

## 0. Prereqs (local)

```bash
make setup                       # uv sync --frozen (includes nothing GPU-side)
uv sync --extra tierb            # local eval/ONNX deps: torch(MPS), transformers, onnxruntime
```

## 1. Export the training kit (local)

```bash
make tier-b-data                 # -> data/tier_b_kit/{train.parquet,cal.parquet,manifest.json}
```

`train.parquet` is the full ~300k TRAIN (only `complaint_id, narrative, class`),
`cal.parquet` is the full CAL. `manifest.json` records each file's sha256, row counts,
and the frozen split/source shas. The export is byte-deterministic. **Do not edit these
files** — the training script re-hashes them against the manifest and aborts on drift.

## 2. Upload to the GPU box

The upload set is packaged into a single reproducible **Colab bundle** so you upload one
file. It contains exactly what the box needs (the training script has **zero repo
imports**, so this is everything):

```bash
make tier-b-bundle    # -> data/tier_b_colab_bundle.tar.gz  (prints path, size, sha256)
```

Bundle contents (flat layout the runbook's commands assume — `tier_b_kit/`, the script,
the requirements, the four configs):

- `tier_b_kit/` — both parquets + `manifest.json` (from `make tier-b-data`)
- `train_tier_b.py`
- `requirements-tierb-colab.txt`
- `tier_b1_modernbert_sa.yaml`, `...sb.yaml`, `...sc.yaml`, `tier_b2_distilbert_s0.yaml`

Upload `data/tier_b_colab_bundle.tar.gz`, then on the box: `tar xzf tier_b_colab_bundle.tar.gz`.

## 3. Install deps (GPU box)

```bash
pip install -r requirements-tierb-colab.txt
# torch: use the box's preinstalled CUDA build (Colab already has it). Do NOT pip-install
# a CPU torch wheel over it.
```

The script auto-selects precision: **bf16** on A10/A100, **fp16** on **T4 (no bf16)**,
fp32 on CPU/MPS. No flag needed.

## 4. Train the four checkpoints (GPU box)

Always run with `python -u` (unbuffered) and `tee` the console to a logfile, so the
flushed chunk-tokenization + RSS-memory progress survives a disconnect and is available if
a run is killed (e.g. CPU-RAM OOM during data prep):

```bash
mkdir -p logs
python -u train_tier_b.py --config tier_b1_modernbert_sa.yaml --data-dir tier_b_kit --out ckpt/tier_b1_sa 2>&1 | tee logs/tier_b1_sa.log
python -u train_tier_b.py --config tier_b1_modernbert_sb.yaml --data-dir tier_b_kit --out ckpt/tier_b1_sb 2>&1 | tee logs/tier_b1_sb.log
python -u train_tier_b.py --config tier_b1_modernbert_sc.yaml --data-dir tier_b_kit --out ckpt/tier_b1_sc 2>&1 | tee logs/tier_b1_sc.log
python -u train_tier_b.py --config tier_b2_distilbert_s0.yaml --data-dir tier_b_kit --out ckpt/tier_b2_s0 2>&1 | tee logs/tier_b2_s0.log
```

Tokenization is chunked (~5k rows/chunk) and the raw texts are released + `gc`'d before
GPU training, so CPU RAM stays bounded; watch the `[tokenize:*] N/M rows rss=…MB` lines.
Each run prints hardware/precision/wall-clock and the final CAL macro-F1, and writes into
its `--out`:

- the checkpoint: `model.safetensors` + `config.json` (with `id2label`) + tokenizer files
- `training_log.jsonl` — per-step loss + per-epoch CAL metrics (**the training curve**)
- `training_meta.json` — hardware, wall-clock, effective batch, key-lib versions, data
  manifest sha, **measured truncation rate**, seed, label order

If ModernBERT OOMs on a T4: lower `per_device_batch_size` to 8 and raise `grad_accum` to
4 in the config (keeps effective batch 32) — but note that edits the config's hash, so
re-export is a new config identity; prefer a bigger GPU if you want the shipped configs
verbatim.

## 5. Download checkpoints back and place them (local)

Download each `ckpt/tier_b1_sa` … into the repo under `data/checkpoints/` at the paths
the configs point to (these dirs are gitignored):

```
data/checkpoints/tier_b1_sa/   <- ckpt/tier_b1_sa
data/checkpoints/tier_b1_sb/
data/checkpoints/tier_b1_sc/
data/checkpoints/tier_b2_s0/
```

Each dir must contain `model.safetensors`, `config.json`, `training_meta.json`, and the
tokenizer files.

## 6. Evaluate locally through the harness (local)

Each config evals on TEST-IID with temperature scaling fit on CAL — one record each:

```bash
uv run python -m triage_lab.harness configs/tier_b1_modernbert_sa.yaml
uv run python -m triage_lab.harness configs/tier_b1_modernbert_sb.yaml
uv run python -m triage_lab.harness configs/tier_b1_modernbert_sc.yaml
uv run python -m triage_lab.harness configs/tier_b2_distilbert_s0.yaml
```

Each appends one fully-provenanced record to `results/runs.jsonl` (metrics + 95% CIs,
git sha, config/split shas, wall-clock; the record's `extra` also carries checkpoint
sha, fitted temperature `T`, train/inference hardware, truncation rate).

Then, in `EXPERIMENT_LOG.md`, record the hypothesis → result → verdict, the seed
variance across s{a,b,c} (mean ± sd of macro-F1), the B1-vs-A and B1-vs-B2 **paired**
bootstrap deltas (via `harness.paired_bootstrap_delta`), and each run's reproduction
command.

## 7. Export the int8 ONNX deployment artifact + parity (local)

Off the B2 checkpoint only:

```bash
uv run python scripts/export_onnx_distilbert.py \
    --checkpoint data/checkpoints/tier_b2_s0 \
    --out data/onnx/tier_b2_s0
```

Writes `model.onnx` (fp32), `model.int8.onnx` (dynamic int8 weights), and
`parity_report.json` (argmax agreement + mean |Δprob| vs PyTorch on a fixed-seed 5k CAL
subsample, plus file sizes). **Acceptance: `argmax_agreement ≥ 0.99`.** Log the parity
numbers in `EXPERIMENT_LOG.md`.

---

## Local pipeline smoke test (no GPU, proves the wiring)

Before trusting the cloud round-trip you can prove the whole export→train→eval→ONNX chain
locally with a tiny DistilBERT run (garbage metrics by design; ~minutes on MPS/CPU;
one-time ~260 MB model download). Smoke artifacts stay under the gitignored `data/` tree
and the eval uses `--no-append` so `results/runs.jsonl` is untouched:

```bash
uv run python scripts/export_tier_b_data.py --out data/tier_b_kit_smoke
uv run --extra tierb python -u scripts/train_tier_b.py \
    --config configs/tier_b_smoke.yaml --data-dir data/tier_b_kit_smoke \
    --out data/checkpoints/tier_b_smoke 2>&1 | tee logs/tier_b_smoke.log
uv run --extra tierb python -m triage_lab.harness configs/tier_b_smoke.yaml --no-append
uv run --extra tierb python scripts/export_onnx_distilbert.py \
    --checkpoint data/checkpoints/tier_b_smoke --out data/onnx/tier_b_smoke --n-samples 300
```

## max_seq_length note

Both real configs use `max_seq_length = 256`. TRAIN narratives run long (word-count
p50≈135, p90≈419), so 256 subword tokens truncates the long tail — the exact realized
rate is measured per run and stored in `training_meta.json`. Rationale: product-class
signal is front-loaded in complaints, and the B2 deployment story (int8 in-browser)
favors short sequences for latency; keeping B1 and B2 at the same length also keeps the
B1-vs-B2 comparison clean. Revisit only if per-class truncation turns out to be skewed
toward a product that then underperforms.
