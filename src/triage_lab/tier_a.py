"""Tier A — classical baseline runner (UPGRADE_PLAN.md §4.2).

TF-IDF (word 1-2-grams + char_wb 3-5-grams) -> a linear classifier
(`logreg`: SAGA logistic regression, class-weighted; or `complement_nb`:
Complement Naive Bayes), optionally calibrated with isotonic regression on the
calibration split.

The whole experiment is identified by the config YAML (whose raw bytes the
harness hashes). Every hyperparameter that matters — n-gram ranges, `min_df`,
`max_features`, `sublinear_tf`, the model params, the calibration mode, the
seed — is read from the config; the code-level defaults here are only a safety
net so partial configs (e.g. tests) still run. The shipped `configs/tier_a_*`
YAMLs spell out every value explicitly.

Determinism: the SAGA solver's only source of randomness is seeded from
`seed`; TF-IDF, ComplementNB, and isotonic calibration are deterministic. Reads
are single-threaded and ordered by `complaint_id`, so `same config -> same
predictions` holds byte-for-byte.

Provenance: the runner rebuilds the record's `dataset` block from
`splits_stats.yaml` via `harness.dataset_info`, and verifies that the on-disk
train/eval parquet sha256 matches the frozen stats before training (fail-loud
integrity gate, CLAUDE.md rule 2).

Robustness: an optional `data.perturbation` block (see `perturb.py`) rewrites the
EVAL narratives only, deterministically per complaint_id, after loading and before
featurization. TRAIN and CAL are never perturbed, so a perturbed run is the frozen
clean model measured on noisy input.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline

from triage_lab import perturb
from triage_lab.harness import RunnerResult, dataset_info, register_runner
from triage_lab.snapshot import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLITS_DIR = REPO_ROOT / "data" / "splits"

# Code-level defaults (safety net only; the shipped YAMLs set every value).
_DEFAULT_TEXT_COLUMN = "narrative"
_DEFAULT_LABEL_COLUMN = "class"
_DEFAULT_ORDER_COLUMN = "complaint_id"
_DEFAULT_TRAIN_SPLIT = "train"
_DEFAULT_SEED = 20260805

_DEFAULT_WORD = {
    "enabled": True,
    "ngram_range": [1, 2],
    "min_df": 5,
    "max_features": 150000,
    "sublinear_tf": True,
}
_DEFAULT_CHAR = {
    "enabled": True,
    "ngram_range": [3, 5],
    "min_df": 5,
    "max_features": 150000,
    "sublinear_tf": True,
}
_DEFAULT_LOGREG = {"C": 1.0, "max_iter": 200, "tol": 1e-3}
_DEFAULT_CNB = {"alpha": 0.3, "norm": False}


# ---------------------------------------------------------------------------
# Config access
# ---------------------------------------------------------------------------

def _merged(cfg_block, defaults: dict) -> dict:
    """Overlay a (possibly partial / missing) config block onto defaults."""
    out = dict(defaults)
    if isinstance(cfg_block, dict):
        out.update(cfg_block)
    return out


def _splits_dir(config: dict) -> Path:
    data = config.get("data", {})
    return Path(data.get("splits_dir", DEFAULT_SPLITS_DIR))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_split_frame(
    path, text_column: str, label_column: str, order_column: str
) -> tuple[list[str], np.ndarray]:
    """Read (text, label) from a split parquet, single-threaded and ordered.

    Ordering by `order_column` (complaint_id on real splits) makes the row
    order — and therefore SAGA's shuffle and the fitted model — reproducible.
    """
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


# ---------------------------------------------------------------------------
# Feature and estimator construction
# ---------------------------------------------------------------------------

def build_features(features_cfg: dict) -> FeatureUnion:
    """Build the TF-IDF FeatureUnion from the config `features` block.

    The `char` block can be disabled (`enabled: false`) for the single-variable
    word-only ablation; at least one block must stay enabled.
    """
    features_cfg = features_cfg or {}
    word = _merged(features_cfg.get("word"), _DEFAULT_WORD)
    char = _merged(features_cfg.get("char"), _DEFAULT_CHAR)

    transformers = []
    if word.get("enabled", True):
        transformers.append((
            "word",
            TfidfVectorizer(
                analyzer="word",
                ngram_range=tuple(word["ngram_range"]),
                min_df=word["min_df"],
                max_features=word["max_features"],
                sublinear_tf=word["sublinear_tf"],
                lowercase=True,
            ),
        ))
    if char.get("enabled", True):
        transformers.append((
            "char",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=tuple(char["ngram_range"]),
                min_df=char["min_df"],
                max_features=char["max_features"],
                sublinear_tf=char["sublinear_tf"],
                lowercase=True,
            ),
        ))
    if not transformers:
        raise ValueError("features: at least one of word/char must be enabled")
    return FeatureUnion(transformers)


def build_estimator(family: str, params: dict, seed: int):
    """Build the linear classifier selected by `model.family`."""
    if family == "logreg":
        p = _merged(params, _DEFAULT_LOGREG)
        # penalty/n_jobs intentionally omitted: sklearn>=1.8 defaults to L2 and
        # deprecated both kwargs. Determinism comes from random_state alone.
        return LogisticRegression(
            solver="saga",
            C=p["C"],
            class_weight="balanced",
            max_iter=p["max_iter"],
            tol=p["tol"],
            random_state=seed,
        )
    if family == "complement_nb":
        p = _merged(params, _DEFAULT_CNB)
        return ComplementNB(alpha=p["alpha"], norm=p["norm"])
    raise ValueError(f"unknown model.family {family!r}; choose 'logreg' or 'complement_nb'")


def build_pipeline(config: dict) -> Pipeline:
    """Feature union + classifier for the given config (uncalibrated, unfitted)."""
    model = config.get("model", {})
    seed = int(config.get("seed", _DEFAULT_SEED))
    return Pipeline([
        ("features", build_features(config.get("features", {}))),
        ("clf", build_estimator(model.get("family", "logreg"), model.get("params", {}), seed)),
    ])


# ---------------------------------------------------------------------------
# Fit / predict core
# ---------------------------------------------------------------------------

def _verify_integrity(split: str, path: Path, splits_stats_path: Path) -> None:
    """Fail loud if the on-disk parquet sha256 drifts from the frozen stats."""
    info = dataset_info(split, splits_stats_path)
    actual = sha256_file(path)
    if actual != info["split_sha256"]:
        raise ValueError(
            f"integrity check failed for split {split!r}: parquet sha256 {actual} "
            f"!= frozen splits_stats.yaml {info['split_sha256']}"
        )


def fit_predict(config: dict):
    """Train on the train split, evaluate on `data.split`, return arrays.

    Returns (y_true, y_pred, probs, class_labels). `probs` columns are aligned
    to `class_labels` by construction (they are the fitted estimator's
    `classes_`), which is exactly the ordering the harness metrics expect.
    """
    data = config.get("data", {})
    text_col = data.get("text_column", _DEFAULT_TEXT_COLUMN)
    label_col = data.get("label_column", _DEFAULT_LABEL_COLUMN)
    order_col = data.get("order_column", _DEFAULT_ORDER_COLUMN)
    train_split = data.get("train_split", _DEFAULT_TRAIN_SPLIT)
    eval_split = data["split"]
    calibration = config.get("calibration", "none")
    verify = data.get("verify_sha256", True)

    splits_dir = _splits_dir(config)
    stats_path = splits_dir / "splits_stats.yaml"
    train_path = splits_dir / f"{train_split}.parquet"
    eval_path = splits_dir / f"{eval_split}.parquet"

    if verify:
        _verify_integrity(train_split, train_path, stats_path)
        _verify_integrity(eval_split, eval_path, stats_path)

    x_train, y_train = load_split_frame(train_path, text_col, label_col, order_col)
    x_eval, y_eval = load_split_frame(eval_path, text_col, label_col, order_col)

    # Phase 5 §6.3.4: optional eval-text perturbation, applied here — after loading, before
    # featurization — and to the EVAL frame only. TRAIN (above) and CAL (below) are read
    # separately and stay clean by construction, so a perturbed run is the frozen clean
    # model scored on noisy input, never a model trained on noise. Keyed by complaint_id,
    # which is read only when a perturbation is actually configured.
    spec = perturb.spec_from_config(config)
    if spec is not None:
        x_eval = perturb.apply_spec(x_eval, load_eval_ids(config), spec)

    pipe = build_pipeline(config)
    pipe.fit(x_train, y_train)

    if calibration == "isotonic":
        cal_split = data.get("cal_split", "cal")
        cal_path = splits_dir / f"{cal_split}.parquet"
        if verify:
            _verify_integrity(cal_split, cal_path, stats_path)
        x_cal, y_cal = load_split_frame(cal_path, text_col, label_col, order_col)
        # Frozen/prefit calibration (sklearn >=1.6 path): fit isotonic on CAL
        # over the already-trained pipeline; never refits the base model.
        model = CalibratedClassifierCV(FrozenEstimator(pipe), method="isotonic")
        model.fit(x_cal, y_cal)
    elif calibration == "none":
        model = pipe
    else:
        raise ValueError(f"unknown calibration {calibration!r}; choose 'none' or 'isotonic'")

    class_labels = list(model.classes_)
    probs = model.predict_proba(x_eval)
    y_pred = model.predict(x_eval)
    return y_eval, y_pred, probs, class_labels


# ---------------------------------------------------------------------------
# Registered runner
# ---------------------------------------------------------------------------

def load_eval_ids(config: dict) -> np.ndarray:
    """Load the eval split's `order_column` (complaint_id) in the same order fit_predict
    evaluates, so ids are aligned to the returned y_true/y_pred/probs. A separate small
    read keeps `fit_predict`'s 4-tuple contract (and its tests) untouched."""
    data = config.get("data", {})
    order_col = data.get("order_column", _DEFAULT_ORDER_COLUMN)
    eval_split = data["split"]
    eval_path = _splits_dir(config) / f"{eval_split}.parquet"
    con = duckdb.connect()
    try:
        con.execute("SET threads=1")
        con.execute("SET preserve_insertion_order=true")
        rows = con.execute(
            f'SELECT "{order_col}" FROM read_parquet(\'{eval_path}\') '
            f'ORDER BY "{order_col}"'
        ).fetchall()
    finally:
        con.close()
    return np.array([r[0] for r in rows], dtype=np.int64)


@register_runner("tier_a")
def tier_a_runner(config: dict) -> RunnerResult:
    """Config-driven Tier A runner (see module docstring)."""
    y_true, y_pred, probs, class_labels = fit_predict(config)
    ids = load_eval_ids(config)
    eval_split = config["data"]["split"]
    stats_path = _splits_dir(config) / "splits_stats.yaml"
    dataset = dataset_info(eval_split, stats_path)
    # Echo the applied perturbation so the record carries it and the harness gate can
    # confirm this runner honoured the config (harness._check_perturbation_applied). Absent
    # perturbation -> empty extra -> the record schema of every pre-Phase-5 run is unchanged.
    spec = perturb.spec_from_config(config)
    extra = {} if spec is None else {perturb.CONFIG_KEY: spec.as_dict()}
    return RunnerResult(
        y_true=np.asarray(y_true, dtype=object),
        y_pred=np.asarray(y_pred, dtype=object),
        probs=np.asarray(probs, dtype=np.float64),
        class_labels=class_labels,
        dataset=dataset,
        cost_usd=None,
        extra=extra,
        ids=ids,
    )
