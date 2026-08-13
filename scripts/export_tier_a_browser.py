#!/usr/bin/env python
"""Export the calibrated Tier A model for in-browser reimplementation (Phase 6 live demo).

    uv run python scripts/export_tier_a_browser.py

Three files are written, all under ``demo/live/``:

1. ``tier_a/tier_a_live.json`` — the whole Tier A deployment point as data: both TF-IDF
   vocabularies + idf vectors, the LogReg coefficients/intercepts, and the nine fitted
   isotonic calibrators. The browser reimplements the *math*; it never re-fits anything.
2. ``tier_b2/live_config.json`` — the scalars the in-browser int8 DistilBERT needs
   (class order + the CAL-fitted temperature), with the parity numbers that license it.
3. ``tier_b2/python_int8_curated.json`` — a Python/onnxruntime int8 reference over the
   200 curated narratives, so a browser-vs-Python disagreement can be decomposed into
   "quantization" (already measured in results/onnx_parity) vs "kernel/runtime".

WHY A REFIT. The frozen Tier A run (``configs/tier_a_logreg_test_iid.yaml``, run
8e4d6345…) persisted per-sample predictions, not the estimator. The only honest way to
ship weights is to re-run the harness's own code path — ``tier_a.fit_predict`` — and then
PROVE the refit is the same model by replaying it against the frozen artifact. That is
step 2 below; a single label disagreement on the curated 200 is a hard failure.

The refit is captured, not reimplemented: ``tier_a.CalibratedClassifierCV`` is swapped for
a factory that hands back the real object and keeps a reference. Every line that builds
features, fits SAGA, or wires the isotonic calibration is still ``tier_a``'s, so this
script cannot drift from the harness by construction.

CALIBRATION SEMANTICS — READ THIS BEFORE WRITING THE JS. Verified against the installed
sklearn 1.9.0 source (``sklearn/calibration.py`` ``_CalibratedClassifier.predict_proba``,
line 797: ``_get_response_values(..., response_method=["decision_function",
"predict_proba"])``): each isotonic calibrator is fed the RAW DECISION FUNCTION column
(``tfidf @ coef[k] + intercept[k]``), **not** a softmax probability. ``decision_function``
comes first in that preference list and the frozen estimator (a Pipeline ending in
LogisticRegression) has it, so the softmax is never computed anywhere in this model. The
same list is used at fit time (``calibration.py`` line 659), so train and predict agree.
Recombination is then: clip into the calibrator's fitted x-range, linearly interpolate,
and divide the row by its own sum (uniform 1/n_classes if that sum is exactly 0 —
``calibration.py`` lines 825-832).

Isotonic serialization: sklearn stores ``X_thresholds_``/``y_thresholds_`` and rebuilds a
linear ``interp1d`` from them (``isotonic.py`` ``_build_f``). The calibrators are built as
``IsotonicRegression(out_of_bounds="clip")`` (``calibration.py`` line 731), and
``_transform`` clips to ``[X_min_, X_max_]``. ``_build_y``'s duplicate trimming always
keeps the first and last knot, so ``X_min_ == x[0]`` and ``X_max_ == x[-1]`` — the export
asserts this, which is what lets the JSON carry only ``x``/``y`` and the browser clamp to
the array endpoints.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from triage_lab import demo_build, harness, predictions, tier_a

SCHEMA_VERSION = "live-v1"

TIER_A_CONFIG = REPO_ROOT / "configs" / "tier_a_logreg_test_iid.yaml"
TIER_A_RUN_ID = "8e4d6345b849a186dcd0e34367641239c235a7b3a48cdbfb4a574e37318abea7"
REFIT_SEED = 20260805

CURATED_IDS_PATH = REPO_ROOT / "demo" / "data" / "curated_ids.json"
PREDS_DIR = harness.DEFAULT_PREDS_DIR

TIER_B2_CONFIG = REPO_ROOT / "configs" / "tier_b2_distilbert_s0.yaml"
TIER_B2_CHECKPOINT = REPO_ROOT / "data" / "checkpoints" / "tier_b2_s0"
TIER_B2_ONNX_INT8 = REPO_ROOT / "data" / "onnx" / "tier_b2_s0" / "model.int8.onnx"
TIER_B2_PARITY = REPO_ROOT / "results" / "onnx_parity" / "tier_b2_s0_parity.json"
TIER_B2_MAX_LENGTH = 256

OUT_TIER_A = REPO_ROOT / "demo" / "live" / "tier_a" / "tier_a_live.json"
OUT_B2_CONFIG = REPO_ROOT / "demo" / "live" / "tier_b2" / "live_config.json"
OUT_B2_INT8 = REPO_ROOT / "demo" / "live" / "tier_b2" / "python_int8_curated.json"

# Gates (task spec). Step 2 is fidelity of the refit vs the frozen artifact; step 4 is
# sufficiency of the exported JSON vs the in-memory sklearn model.
LABEL_MATCH_REQUIRED = True
REFIT_PROB_TOL = 1e-3
SERIALIZATION_TOL = 1e-5

CALIBRATION_SEMANTICS = (
    "for each class k: p_k = interp(clip(d_k, x_k[0], x_k[-1]), x_k, y_k) where "
    "d_k = tfidf @ coef[k] + intercept[k] is the RAW decision_function column (sklearn "
    "prefers decision_function over predict_proba, so no softmax is involved); then "
    "divide the row by its sum, or emit uniform 1/n_classes if that sum is exactly 0"
)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _f32_b64(values) -> str:
    """Standard base64 of a little-endian Float32Array's bytes (JS: new Float32Array(buf))."""
    return base64.b64encode(np.asarray(values, dtype="<f4").tobytes()).decode("ascii")


def _write_compact_json(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")
    return path


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _mb(path: Path) -> str:
    return f"{path.stat().st_size / 1e6:.2f} MB"


# ---------------------------------------------------------------------------
# 1. refit through the harness's own code path
# ---------------------------------------------------------------------------

def refit_via_harness(config: dict):
    """Run `tier_a.fit_predict` verbatim and keep the fitted CalibratedClassifierCV.

    The factory swap is the whole trick: `tier_a` still constructs, fits and calls the
    model; we only hold a reference to the object it made. Nothing about the pipeline,
    the seed, the split ordering or the calibration wiring is restated here.
    """
    original = tier_a.CalibratedClassifierCV
    captured: dict = {}

    def factory(estimator, **kwargs):
        model = original(estimator, **kwargs)
        captured["model"] = model
        return model

    tier_a.CalibratedClassifierCV = factory
    try:
        y_true, y_pred, probs, class_labels = tier_a.fit_predict(config)
    finally:
        tier_a.CalibratedClassifierCV = original

    if "model" not in captured:
        raise RuntimeError(
            "no CalibratedClassifierCV was constructed — the config is not "
            "`calibration: isotonic`, so there is nothing to export"
        )
    return captured["model"], y_true, y_pred, probs, class_labels


def unpack_model(model, class_labels: list[str]):
    """Pull the exportable parts out and check every ordering assumption we rely on."""
    if len(model.calibrated_classifiers_) != 1:
        raise RuntimeError(
            f"expected exactly one _CalibratedClassifier (frozen/prefit path), found "
            f"{len(model.calibrated_classifiers_)}"
        )
    calibrated = model.calibrated_classifiers_[0]
    pipe = calibrated.estimator.estimator  # FrozenEstimator -> Pipeline
    features = pipe.named_steps["features"]
    clf = pipe.named_steps["clf"]

    if list(model.classes_) != list(class_labels):
        raise RuntimeError("CalibratedClassifierCV.classes_ != returned class_labels")
    if list(clf.classes_) != list(class_labels):
        raise RuntimeError(
            "the base LogisticRegression's classes_ differ from the calibrated wrapper's; "
            "the per-class calibrator list would not be class-aligned"
        )
    # `_CalibratedClassifier.predict_proba` writes calibrator j into column
    # LabelEncoder().fit(self.classes).transform(estimator.classes_)[j]. The identity of
    # the two class arrays (checked above) makes that mapping the identity, which is what
    # lets us emit `per_class` in plain class order.
    if list(calibrated.classes) != list(class_labels):
        raise RuntimeError("_CalibratedClassifier.classes != class_labels")
    calibrators = list(calibrated.calibrators)
    if len(calibrators) != len(class_labels):
        raise RuntimeError(f"{len(calibrators)} calibrators for {len(class_labels)} classes")

    branches = dict(features.transformer_list)
    for name in ("word", "char"):
        if name not in branches:
            raise RuntimeError(f"the fitted FeatureUnion has no {name!r} branch")
    return features, branches["word"], branches["char"], clf, calibrators


def branch_vocab(vectorizer) -> list[str]:
    """Terms ordered so that `vocab[i]` is the term occupying output column `i`."""
    vocab = vectorizer.vocabulary_
    out: list[str | None] = [None] * len(vocab)
    for term, index in vocab.items():
        out[int(index)] = term
    if any(t is None for t in out):
        raise RuntimeError("vocabulary_ indices are not a contiguous 0..n-1 range")
    return out  # type: ignore[return-value]


def confirm_feature_order(features, word, char, word_vocab, char_vocab, probe_texts) -> str:
    """Prove the concatenated column order is [word columns…, char columns…].

    Two independent confirmations, because the coef_ layout depends on this and a silent
    swap would still produce plausible-looking probabilities:

    a) `FeatureUnion.transform` on real documents is bit-identical to
       `hstack([word.transform, char.transform])`;
    b) `FeatureUnion.get_feature_names_out()` equals ["word__" + t for word vocab] followed
       by ["char__" + t for char vocab], in the exact per-branch column order exported.
    """
    union = features.transform(probe_texts).tocsr()
    manual = sp.hstack([word.transform(probe_texts), char.transform(probe_texts)]).tocsr()
    if union.shape != manual.shape:
        raise RuntimeError(f"FeatureUnion shape {union.shape} != hstack shape {manual.shape}")
    diff = abs(union - manual)
    if diff.nnz != 0 and float(diff.max()) != 0.0:
        raise RuntimeError("FeatureUnion output is not word-then-char hstack")

    names = list(features.get_feature_names_out())
    expected = [f"word__{t}" for t in word_vocab] + [f"char__{t}" for t in char_vocab]
    if names != expected:
        first_bad = next(i for i, (a, b) in enumerate(zip(names, expected)) if a != b)
        raise RuntimeError(
            f"get_feature_names_out disagrees with the exported column order at index "
            f"{first_bad}: {names[first_bad]!r} != {expected[first_bad]!r}"
        )
    return (
        f"FeatureUnion.transform == hstack([word, char]) exactly on {len(probe_texts)} real "
        f"TEST-IID documents, and get_feature_names_out() == "
        f"['word__'+t for the {len(word_vocab)} word terms] + "
        f"['char__'+t for the {len(char_vocab)} char terms] in exported order"
    )


def isotonic_block(calibrator) -> dict:
    """`{"x": X_thresholds_, "y": y_thresholds_}` + the checks that make clipping safe."""
    x = np.asarray(calibrator.X_thresholds_, dtype=np.float64)
    y = np.asarray(calibrator.y_thresholds_, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or x.size == 0:
        raise RuntimeError(f"malformed isotonic thresholds: x{x.shape} y{y.shape}")
    if not np.all(np.diff(x) > 0):
        raise RuntimeError("X_thresholds_ is not strictly increasing; np.interp would be wrong")
    if calibrator.out_of_bounds != "clip":
        raise RuntimeError(f"calibrator out_of_bounds={calibrator.out_of_bounds!r}, expected 'clip'")
    if float(calibrator.X_min_) != float(x[0]) or float(calibrator.X_max_) != float(x[-1]):
        raise RuntimeError(
            "X_min_/X_max_ are not the first/last threshold; clamping to the exported array "
            "endpoints would not reproduce sklearn's out_of_bounds='clip'"
        )
    return {"x": [float(v) for v in x], "y": [float(v) for v in y]}


# ---------------------------------------------------------------------------
# JSON-only recombination (step 4): the reference implementation the JS mirrors
# ---------------------------------------------------------------------------

def recombine_from_json(payload: dict, feature_matrix) -> np.ndarray:
    """Calibrated probabilities using ONLY `payload` for the post-vectorization math."""
    n_classes = len(payload["class_labels"])
    coef = np.frombuffer(base64.b64decode(payload["coef_b64"]), dtype="<f4")
    coef = coef.reshape(n_classes, -1).astype(np.float64)
    intercept = np.asarray(payload["intercept"], dtype=np.float64)

    decision = np.asarray(feature_matrix @ coef.T) + intercept  # decision_function

    proba = np.zeros_like(decision)
    for k, block in enumerate(payload["calibration"]["per_class"]):
        x = np.asarray(block["x"], dtype=np.float64)
        y = np.asarray(block["y"], dtype=np.float64)
        proba[:, k] = np.interp(np.clip(decision[:, k], x[0], x[-1]), x, y)

    denominator = proba.sum(axis=1, keepdims=True)
    out = np.full_like(proba, 1.0 / n_classes)
    np.divide(proba, denominator, out=out, where=denominator != 0)
    return out


# ---------------------------------------------------------------------------
# Tier A export
# ---------------------------------------------------------------------------

def export_tier_a(out_path: Path) -> dict:
    config = harness.load_config(TIER_A_CONFIG)
    curated = json.loads(CURATED_IDS_PATH.read_text(encoding="utf-8"))
    curated_ids = [int(c) for c in curated["complaint_ids"]]

    print(f"[1/4] refitting {TIER_A_CONFIG.name} through tier_a.fit_predict "
          f"(TRAIN fit + isotonic on CAL + TEST-IID predict) — this takes a while…",
          flush=True)
    t0 = time.time()
    model, _, y_pred, probs, class_labels = refit_via_harness(config)
    print(f"      refit + eval done in {time.time() - t0:.1f}s", flush=True)

    eval_ids = tier_a.load_eval_ids(config)
    features, word, char, clf, calibrators = unpack_model(model, class_labels)

    # --- step 2: fidelity vs the frozen per-sample artifact -------------------------
    art = predictions.read_artifact(PREDS_DIR / f"{TIER_A_RUN_ID}.parquet")
    if list(art.class_labels) != list(class_labels):
        raise RuntimeError("frozen artifact class order differs from the refit's classes_")
    pos_refit = {int(cid): i for i, cid in enumerate(eval_ids)}
    pos_frozen = {int(cid): i for i, cid in enumerate(art.complaint_id)}
    rows_refit = np.array([pos_refit[c] for c in curated_ids], dtype=np.int64)
    rows_frozen = np.array([pos_frozen[c] for c in curated_ids], dtype=np.int64)

    refit_probs = np.asarray(probs, dtype=np.float64)[rows_refit]
    frozen_probs = art.probs[rows_frozen]
    label_exact = int(sum(str(a) == str(b) for a, b in
                          zip(y_pred[rows_refit], art.y_pred[rows_frozen])))
    d_prob = float(np.abs(refit_probs - frozen_probs).max())
    d_pmax = float(np.abs(refit_probs.max(axis=1) - art.p_max[rows_frozen]).max())
    print(f"[2/4] refit fidelity on the curated {len(curated_ids)}: labels "
          f"{label_exact}/{len(curated_ids)}, max|Δp_max|={d_pmax:.3e}, "
          f"max|Δprob|={d_prob:.3e}", flush=True)
    if LABEL_MATCH_REQUIRED and label_exact != len(curated_ids):
        raise RuntimeError(
            f"REFIT IS NOT THE FROZEN MODEL: {len(curated_ids) - label_exact} of "
            f"{len(curated_ids)} curated labels disagree with run {TIER_A_RUN_ID[:8]}"
        )
    if max(d_prob, d_pmax) >= REFIT_PROB_TOL:
        raise RuntimeError(
            f"refit probabilities drift from the frozen artifact by "
            f"{max(d_prob, d_pmax):.3e} >= {REFIT_PROB_TOL:g}"
        )

    # --- step 3: serialize -----------------------------------------------------------
    word_vocab = branch_vocab(word)
    char_vocab = branch_vocab(char)
    # Re-read the eval texts through tier_a's own loader (same path, same ordering as the
    # fit_predict call above), so the strings fed to the vectorizers in step 4 are exactly
    # the strings the frozen run scored — not a second rendering of them.
    data = config["data"]
    x_eval, _ = tier_a.load_split_frame(
        tier_a._splits_dir(config) / f"{data['split']}.parquet",
        data.get("text_column", "narrative"), data.get("label_column", "class"),
        data.get("order_column", "complaint_id"))
    curated_texts = [x_eval[i] for i in rows_refit]

    # …and cross-check them against the narratives the demo builder serves for the same
    # ids, so the live page cannot be scoring different text than this export verified.
    demo_rows = demo_build.load_split_rows(curated_ids, data["split"])
    mismatched = [c for c, t in zip(curated_ids, curated_texts) if demo_rows[c][0] != t]
    if mismatched:
        raise RuntimeError(
            f"{len(mismatched)} curated narrative(s) differ between tier_a.load_split_frame "
            f"and demo_build.load_split_rows, e.g. {mismatched[:5]}"
        )

    order_note = confirm_feature_order(features, word, char, word_vocab, char_vocab,
                                       curated_texts[:8])

    coef = np.asarray(clf.coef_, dtype=np.float64)
    n_features = len(word_vocab) + len(char_vocab)
    if coef.shape != (len(class_labels), n_features):
        raise RuntimeError(f"coef_ shape {coef.shape} != ({len(class_labels)}, {n_features})")

    # The `live-v1` schema carries ONE `sublinear_tf` flag and no `norm`/`smooth_idf`
    # fields, so it can only describe branches that agree on the first and use sklearn's
    # defaults for the rest. Assert that rather than silently exporting a model the schema
    # cannot express: a mismatch here means the JS would compute different features.
    if bool(word.sublinear_tf) != bool(char.sublinear_tf):
        raise RuntimeError(
            f"schema carries one sublinear_tf flag but the branches disagree "
            f"(word={word.sublinear_tf}, char={char.sublinear_tf})"
        )
    for name, vec, vocab in (("word", word, word_vocab), ("char", char, char_vocab)):
        if len(vec.idf_) != len(vocab):
            raise RuntimeError(f"{name}: idf_ has {len(vec.idf_)} entries for {len(vocab)} terms")
        if (vec.norm, vec.smooth_idf, vec.use_idf, vec.lowercase, vec.strip_accents,
                vec.binary) != ("l2", True, True, True, None, False):
            raise RuntimeError(
                f"{name} branch uses non-default text params the live-v1 schema cannot "
                f"express (norm={vec.norm!r}, smooth_idf={vec.smooth_idf}, "
                f"use_idf={vec.use_idf}, lowercase={vec.lowercase}, "
                f"strip_accents={vec.strip_accents!r}, binary={vec.binary})"
            )
    if word.token_pattern != r"(?u)\b\w\w+\b" or word.stop_words is not None:
        raise RuntimeError("word branch tokenization is not the sklearn default the JS assumes")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "provenance": {
            "kind": "tier_a_live_export",
            "source_run_id": TIER_A_RUN_ID,
            "source_config": "configs/tier_a_logreg_test_iid.yaml",
            "refit_seed": REFIT_SEED,
            "git_sha": harness._git_sha(),
            "verify": {
                "curated_n": len(curated_ids),
                "label_exact": label_exact,
                "max_abs_delta_p_max": d_pmax,
                "max_abs_delta_prob": d_prob,
            },
        },
        "class_labels": list(class_labels),
        "sublinear_tf": bool(word.sublinear_tf and char.sublinear_tf),
        "word": {
            "ngram_range": list(word.ngram_range),
            "vocab": word_vocab,
            "idf_b64": _f32_b64(word.idf_),
        },
        "char": {
            "ngram_range": list(char.ngram_range),
            "vocab": char_vocab,
            "idf_b64": _f32_b64(char.idf_),
        },
        "coef_b64": _f32_b64(coef.ravel(order="C")),
        "intercept": [float(v) for v in np.asarray(clf.intercept_, dtype=np.float64)],
        "calibration": {
            "method": "isotonic",
            "per_class": [isotonic_block(c) for c in calibrators],
            "semantics": CALIBRATION_SEMANTICS,
        },
    }
    _write_compact_json(payload, out_path)

    # --- step 4: is the JSON sufficient? ---------------------------------------------
    reloaded = json.loads(out_path.read_text(encoding="utf-8"))
    curated_features = features.transform(curated_texts)
    from_json = recombine_from_json(reloaded, curated_features)
    sklearn_probs = model.predict_proba(curated_texts)
    d_json = float(np.abs(from_json - np.asarray(sklearn_probs, dtype=np.float64)).max())
    json_labels_exact = int(sum(
        class_labels[int(i)] == str(lbl)
        for i, lbl in zip(from_json.argmax(axis=1), y_pred[rows_refit])))
    print(f"[3/4] JSON-only recombination vs sklearn predict_proba: "
          f"max|Δprob|={d_json:.3e}, labels {json_labels_exact}/{len(curated_ids)}",
          flush=True)
    if d_json >= SERIALIZATION_TOL:
        raise RuntimeError(
            f"the exported JSON does not reproduce sklearn: max|Δprob|={d_json:.3e} >= "
            f"{SERIALIZATION_TOL:g}; the documented semantics are insufficient"
        )

    return {
        "path": out_path,
        "class_labels": list(class_labels),
        "n_word": len(word_vocab),
        "n_char": len(char_vocab),
        "n_thresholds": [len(b["x"]) for b in payload["calibration"]["per_class"]],
        "label_exact": label_exact,
        "max_abs_delta_p_max": d_pmax,
        "max_abs_delta_prob": d_prob,
        "max_abs_delta_json": d_json,
        "json_labels_exact": json_labels_exact,
        "order_note": order_note,
        "curated_ids": curated_ids,
    }


# ---------------------------------------------------------------------------
# Tier B2 support files
# ---------------------------------------------------------------------------

def b2_test_iid_record() -> dict:
    """The B2 TEST-IID final, resolved from the results log by (config stem, split)."""
    records = predictions.load_records()
    resolved = demo_build.resolve_records(records)
    return demo_build.record_for(resolved, TIER_B2_CONFIG.stem, "test_iid")


def export_b2_live_config(out_path: Path, class_labels: list[str], record: dict) -> dict:
    temperature = float((record.get("extra") or {})["temperature"])
    parity = json.loads(TIER_B2_PARITY.read_text(encoding="utf-8"))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "class_labels": list(class_labels),
        "temperature": temperature,
        "provenance": {
            "checkpoint": "data/checkpoints/tier_b2_s0",
            "onnx_int8": "model.int8.onnx",
            "temperature_source_run_id": record["run_id"],
            "temperature_source": (
                "results/runs.jsonl -> the tier_b2_distilbert_s0.yaml TEST-IID record's "
                "extra.temperature (fit by NLL on CAL, tier_b.fit_temperature)"
            ),
            "parity": {
                "source": "results/onnx_parity/tier_b2_s0_parity.json",
                "n_samples": parity["n_samples"],
                "subsample_split": parity["subsample_split"],
                "argmax_agreement": parity["argmax_agreement"],
                "agreement": parity["agreement"],
                "mean_abs_prob_delta": parity["mean_abs_prob_delta"],
                "mean_abs_prob_delta_pairs": parity["mean_abs_prob_delta_pairs"],
                "macro_f1": parity["macro_f1"],
                "max_seq_length": parity["max_seq_length"],
                "model_int8_onnx_sha256": parity["provenance"]["model_int8_onnx_sha256"],
            },
        },
    }
    demo_build.write_json(payload, out_path)
    return {"path": out_path, "temperature": temperature}


def _softmax(logits: np.ndarray) -> np.ndarray:
    """fp64 max-shifted softmax — the same recipe tier_b/export_onnx_distilbert use."""
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _int8_probs(session, tokenizer, texts, temperature, *, batch_size, padding):
    out = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        enc = tokenizer(chunk, truncation=True, max_length=TIER_B2_MAX_LENGTH,
                        padding=padding, return_tensors="np")
        logits = session.run(["logits"], {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        })[0]
        out.append(_softmax(np.asarray(logits, dtype=np.float64) / temperature))
    return np.concatenate(out, axis=0)


def export_b2_int8_reference(out_path: Path, class_labels: list[str], curated_ids: list[int],
                             temperature: float, record: dict) -> dict:
    from onnxruntime import InferenceSession
    from onnxruntime import __version__ as ort_version
    from transformers import AutoTokenizer

    rows = demo_build.load_split_rows(curated_ids, "test_iid")
    texts = [rows[cid][0] for cid in curated_ids]

    tokenizer = AutoTokenizer.from_pretrained(str(TIER_B2_CHECKPOINT))
    session = InferenceSession(str(TIER_B2_ONNX_INT8), providers=["CPUExecutionProvider"])

    # batch_size=1 / no padding is the EMITTED reference: it is what a single-example
    # transformers.js call does, so the browser-vs-Python gap is not contaminated by a
    # batching difference.
    #
    # That choice is not cosmetic. `quantize_dynamic` leaves activations in fp32 and inserts
    # DynamicQuantizeLinear, which derives the activation scale from the ACTUAL input tensor
    # at run time — so batch composition and pad length change the quantization grid and
    # therefore the logits. The two alternative regimes are re-scored here and their
    # disagreement published, because it bounds how well any browser can ever agree with a
    # Python reference that batches differently.
    t0 = time.time()
    probs = _int8_probs(session, tokenizer, texts, temperature, batch_size=1, padding=False)
    single_seconds = time.time() - t0
    variants = {
        "batch32_padded": _int8_probs(session, tokenizer, texts, temperature,
                                      batch_size=32, padding=True),
        "batch1_pad_to_max_length": _int8_probs(session, tokenizer, texts, temperature,
                                                batch_size=1, padding="max_length"),
    }
    sensitivity = {
        name: {
            "max_abs_prob_delta": float(np.abs(probs - other).max()),
            "mean_abs_prob_delta": float(np.abs(probs - other).mean()),
            "label_disagreements": int((probs.argmax(1) != other.argmax(1)).sum()),
        }
        for name, other in variants.items()
    }

    order = probs.argmax(axis=1)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "class_labels": list(class_labels),
        "provenance": {
            "kind": "tier_b2_python_int8_reference",
            "purpose": (
                "Python/onnxruntime int8 reference over the curated 200, so a browser "
                "disagreement decomposes into quantization (results/onnx_parity) vs "
                "runtime/kernel"
            ),
            "onnx_int8": "data/onnx/tier_b2_s0/model.int8.onnx",
            "onnx_int8_sha256": _sha256_file(TIER_B2_ONNX_INT8),
            "checkpoint": "data/checkpoints/tier_b2_s0",
            "tokenizer": "data/checkpoints/tier_b2_s0",
            "max_length": TIER_B2_MAX_LENGTH,
            "truncation": True,
            "batching": "batch_size=1, no padding (matches a single-example browser call)",
            "batching_sensitivity": {
                "note": (
                    "onnxruntime dynamic int8 derives activation scales from the actual "
                    "input tensor, so the same row scores differently under a different "
                    "batch composition or pad length; these are the same 200 rows re-scored "
                    "against the emitted batch_size=1/no-padding reference"
                ),
                **sensitivity,
            },
            "temperature": temperature,
            "temperature_source_run_id": record["run_id"],
            "probs": "softmax(logits / temperature), fp64 max-shifted",
            "narrative_source": (
                "data/splits/test_iid.parquet (same frozen split demo_build.build_samples "
                "reads)"
            ),
            "curated_ids_source": "demo/data/curated_ids.json",
            "onnxruntime": ort_version,
            "providers": ["CPUExecutionProvider"],
            "session_options": "default",
            "git_sha": harness._git_sha(),
            "n": len(curated_ids),
            "wall_clock_seconds": single_seconds,
        },
        "predictions": [
            {"complaint_id": int(cid),
             "label": class_labels[int(k)],
             "p_max": float(probs[i, int(k)])}
            for i, (cid, k) in enumerate(zip(curated_ids, order))
        ],
    }
    demo_build.write_json(payload, out_path)
    return {"path": out_path, "sensitivity": sensitivity, "seconds": single_seconds}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-tier-b2", action="store_true",
                        help="export only tier_a_live.json (debugging aid)")
    args = parser.parse_args(argv)

    started = time.time()
    a = export_tier_a(OUT_TIER_A)

    b2_cfg = b2_int8 = None
    if not args.skip_tier_b2:
        record = b2_test_iid_record()
        b2_cfg = export_b2_live_config(OUT_B2_CONFIG, a["class_labels"], record)
        print(f"[4/4] running int8 ONNX over the curated {len(a['curated_ids'])}…",
              flush=True)
        b2_int8 = export_b2_int8_reference(OUT_B2_INT8, a["class_labels"], a["curated_ids"],
                                           b2_cfg["temperature"], record)

    print("\n" + "=" * 78)
    print("TIER A BROWSER EXPORT — verification report")
    print("=" * 78)
    print(f"source run            : {TIER_A_RUN_ID}")
    print(f"source config         : configs/tier_a_logreg_test_iid.yaml (seed {REFIT_SEED})")
    print(f"git sha               : {harness._git_sha()}")
    print(f"classes               : {', '.join(a['class_labels'])}")
    print(f"features              : word {a['n_word']} + char {a['n_char']} = "
          f"{a['n_word'] + a['n_char']}")
    print(f"feature order proof   : {a['order_note']}")
    print(f"isotonic knots/class  : {a['n_thresholds']}")
    print("-" * 78)
    print("step 2 — refit vs frozen artifact (curated 200):")
    print(f"  label exact match   : {a['label_exact']}/200  (gate: 200/200)")
    print(f"  max|Δp_max|         : {a['max_abs_delta_p_max']:.6e}  (gate < {REFIT_PROB_TOL:g})")
    print(f"  max|Δprob|          : {a['max_abs_delta_prob']:.6e}  (gate < {REFIT_PROB_TOL:g})")
    print("step 4 — JSON-only recombination vs sklearn predict_proba (curated 200):")
    print(f"  max|Δprob|          : {a['max_abs_delta_json']:.6e}  "
          f"(gate < {SERIALIZATION_TOL:g})")
    print(f"  label exact match   : {a['json_labels_exact']}/200")
    print("-" * 78)
    print("files:")
    print(f"  {a['path'].relative_to(REPO_ROOT)}  {_mb(a['path'])}")
    if b2_cfg is not None:
        print(f"  {b2_cfg['path'].relative_to(REPO_ROOT)}  {_mb(b2_cfg['path'])}  "
              f"(T={b2_cfg['temperature']!r})")
    if b2_int8 is not None:
        print(f"  {b2_int8['path'].relative_to(REPO_ROOT)}  {_mb(b2_int8['path'])}")
        print("int8 batching sensitivity (vs the emitted batch-1/no-pad reference):")
        for name, s in b2_int8["sensitivity"].items():
            print(f"  {name:<26}: max|Δprob|={s['max_abs_prob_delta']:.3e}  "
                  f"mean|Δprob|={s['mean_abs_prob_delta']:.3e}  "
                  f"labels differ {s['label_disagreements']}/200")
    print("-" * 78)
    print("reproduce: uv run python scripts/export_tier_a_browser.py")
    print(f"total wall clock: {time.time() - started:.1f}s")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
