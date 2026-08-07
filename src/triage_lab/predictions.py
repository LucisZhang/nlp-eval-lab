"""Per-example prediction artifacts (Phase 4 task 1, part A).

Every eval run in this lab reduces a model to four aligned per-example arrays —
``complaint_id``, ``y_true``, ``y_pred``, and a probability row — plus the ordered
``class_labels`` those probabilities index. This module persists exactly that, once per
run, as a Parquet artifact at ``data/preds/<run_id>.parquet`` so the router phase, the
risk-coverage tables, and the demo all read a single frozen source of per-example truth
instead of re-deriving predictions from receipts or re-fitting models ad hoc.

Design decisions (auditable because downstream numbers depend on them):

- **Parquet I/O goes through DuckDB, never pyarrow.** DuckDB is a core dependency (it
  already reads every split); pyarrow lives only in the ``tierb`` extra and is absent
  under ``uv sync --frozen`` / CI. Writing via ``con.register(dict-of-numpy)`` + ``COPY``
  keeps the harness auto-persist path dependency-clean in every environment.
- **Provenance is bound into the file**, not a detachable sidecar: run_id, config_sha256,
  git_sha, split, split_sha256, dataset input_sha256, the tier_c prompt_bundle_sha256,
  and the ordered class_labels ride in the Parquet key-value metadata, so an artifact
  names every input that produced it even if separated from ``results/``. Keys that do
  not apply to a tier (prompt_bundle_sha256 outside tier_c) are written as the empty
  string rather than omitted, so the metadata schema is fixed-width per SCHEMA_VERSION.
- **Columns**: ``complaint_id`` (int64), ``y_true``/``y_pred`` (label strings), ``p_max``
  (max class probability), and one ``prob::<label>`` column per class in ``class_labels``
  order. Full per-class probabilities are kept so risk-coverage and calibration can be
  recomputed downstream without the model.

Backfill CLI (``python -m triage_lab.predictions``) regenerates artifacts for historical
runs and verifies them against the logged point metrics:

- **Inputs are hash-checked before anything is written.** The config is re-hashed from the
  file actually loaded and must equal ``record["config_sha256"]``; for tier_c the loaded
  prompt bundle must equal ``record["extra"]["prompt_bundle_sha256"]``. A mismatch is a
  hard failure — stamping a historical hash onto data regenerated from different inputs
  would launder a provenance break into a green artifact.
- **tier_a / tier_b**: re-invoke the registered runner (deterministic, offline, seeded);
  it appends nothing. The runner now returns ``ids`` so the artifact is a byte-faithful
  view of what the run computed.
- **tier_c**: no network. Reconstruct y_true/y_pred per example from the run's raw
  receipts (``extra.raw_log_path``) joined to the frozen split, REUSING tier_c's own
  subset selection (eval_rows_cap + cap_seed) and ``parse_label``/fallback code paths, so
  the result is bit-identical to what the runner produced. Probs are the degenerate
  one-hot (p_max = 1.0).

The verification gate (``--verify``, default on) has two layers, each reported as its own
✓/✗ row; any ✗ exits nonzero (honest deltas, never a silently-loosened tolerance):

1. **Structural, per example** — ids unique and non-null; every id a member of the frozen
   split the artifact declares; ``y_true`` equal to that split's label for the same id;
   ``p_max`` exactly ``probs.max(axis=1)``; and ``y_pred`` the argmax class (tier_a/b) or
   the one-hot column with ``p_max == 1.0`` (tier_c). These catch a wrong-id row mapping,
   which the aggregate layer structurally cannot: permuting ids leaves every aggregate
   metric untouched.
2. **Aggregate** — accuracy / macro_f1 / aurc / acc_at_cov::* recomputed from the artifact
   with ``metrics.py`` and compared to the record's point estimates at 1e-9.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import duckdb
import numpy as np

from triage_lab import harness

REPO_ROOT = harness.REPO_ROOT
DEFAULT_PREDS_DIR = harness.DEFAULT_PREDS_DIR
DEFAULT_RESULTS_PATH = harness.DEFAULT_RESULTS_PATH

# Bumped v1 -> v2 when the KV-metadata block gained git_sha / input_sha256 /
# prompt_bundle_sha256. The column schema is unchanged; the *provenance* schema is not,
# and a reader that keys on this constant must be able to tell the two apart. Artifacts
# are gitignored and regenerable, so the bump costs nothing but honesty.
SCHEMA_VERSION = "preds-v2"
PROB_PREFIX = "prob::"
_FIXED_COLUMNS = ("complaint_id", "y_true", "y_pred", "p_max")

# Split location + column defaults, mirroring the data-block defaults of tier_a/tier_b/
# tier_c. Duplicated (not imported) so the verify path needs no tier module at import.
DEFAULT_SPLITS_DIR = REPO_ROOT / "data" / "splits"
_DEFAULT_LABEL_COLUMN = "class"
_DEFAULT_ORDER_COLUMN = "complaint_id"

# How many offending ids a failed structural check names in its detail line.
MAX_OFFENDERS_SHOWN = 5

# Metrics the verification gate checks against the logged record's point estimates.
VERIFY_METRICS = (
    "accuracy",
    "macro_f1",
    "aurc",
    "acc_at_cov::0.50",
    "acc_at_cov::0.80",
    "acc_at_cov::0.90",
    "acc_at_cov::0.95",
)
VERIFY_TOL = 1e-9

# ---------------------------------------------------------------------------
# Registered configuration DOCUMENTATION corrections
# ---------------------------------------------------------------------------
# A run record pins the sha256 of the config bytes that produced it, and every gate in
# this repo re-hashes the file and refuses a mismatch (`load_config_checked`). That is the
# right default: a config edit after the fact means the artifact's provenance names inputs
# that did not produce it.
#
# Very occasionally the correct action is to fix PROSE in a config — a comment that states
# something false about the experiment. Deleting the comment is worse (the false claim
# stays in git history with nothing pointing at it) and re-running is worse still (it would
# spend money/compute to change nothing). So the exception is made explicit here rather
# than by loosening the gate:
#
#   - keyed by run_id, so it applies to exactly one run;
#   - pinned to BOTH hashes, so it applies to exactly one before/after pair — any third
#     hash (including a later edit of the same file) fails like any other mismatch;
#   - carrying a reason and an owner-approval date, auditable against EXPERIMENT_LOG.md;
#   - announced on stdout whenever it is exercised.
#
# A semantic change (model, features, split, seed, calibration) must NEVER be registered
# here; it is a new run.
#
# The public name is a read-only MappingProxyType view. A provenance exemption list that
# any imported module (or test) can append to at runtime is not an exemption list, it is a
# disabled gate: `CONFIG_DOC_CORRECTIONS[some_run] = {...}` would be a one-line, invisible
# way to make any config mismatch pass. Registering a correction is a source edit and a
# code review, by construction.
_CONFIG_DOC_CORRECTIONS: dict[str, dict[str, str]] = {
    # tier_a_logreg_test_iid: the header claimed tier_a_logreg_wordchar_cal was "the
    # winning CAL rung". results/runs.jsonl says the word-only rung won every logged CAL
    # metric; the wordchar rung is the FEATURE MATCH to this frozen final. Comment-only.
    "8e4d6345b849a186dcd0e34367641239c235a7b3a48cdbfb4a574e37318abea7": {
        "recorded_sha256":
            "b22be1e963760c2f277562c4afa89eb1014dd6bc0fd1cb86fd588c12d6f3b8c0",
        "corrected_sha256":
            "0813065e47c7152959a36e1b6c193a6332608e13337570ce61161526da643983",
        "reason": (
            "header-comment documentation fix, owner-approved 2026-08-07 "
            "(EXPERIMENT_LOG Phase 4 task 4)"
        ),
    },
}

CONFIG_DOC_CORRECTIONS: Mapping[str, dict[str, str]] = MappingProxyType(
    _CONFIG_DOC_CORRECTIONS
)


# ---------------------------------------------------------------------------
# Provenance + artifact container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ArtifactProvenance:
    """Provenance bound into an artifact's Parquet key-value metadata.

    Every input that can change a prediction is named here: the code (`git_sha`), the
    config bytes (`config_sha256`), the data (`split` + `split_sha256` + the snapshot
    `input_sha256` those splits were cut from) and, for tier_c, the frozen prompt
    (`prompt_bundle_sha256`). The last three default to "" so id-less/local tiers and
    older call sites stay valid; "" means "not applicable", never "unknown but fine".
    """

    run_id: str
    config_sha256: str
    split: str
    split_sha256: str
    class_labels: list
    git_sha: str = ""
    input_sha256: str = ""
    prompt_bundle_sha256: str = ""
    schema_version: str = SCHEMA_VERSION


@dataclass
class PredictionsArtifact:
    """In-memory view of a predictions artifact (arrays are id-aligned)."""

    complaint_id: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    p_max: np.ndarray
    probs: np.ndarray
    class_labels: list
    provenance: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.complaint_id)


# ---------------------------------------------------------------------------
# SQL literal helpers (DuckDB has no parameter binding for COPY/KV_METADATA)
# ---------------------------------------------------------------------------

def _sql_str(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _sql_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _prob_column(label: str) -> str:
    return f"{PROB_PREFIX}{label}"


# ---------------------------------------------------------------------------
# Writer / reader (DuckDB-backed Parquet, no pyarrow)
# ---------------------------------------------------------------------------

def write_artifact(
    path,
    *,
    ids,
    y_true,
    y_pred,
    probs,
    class_labels,
    provenance: ArtifactProvenance,
) -> Path:
    """Write one predictions artifact to `path` (Parquet + bound KV-metadata provenance).

    Columns: complaint_id, y_true, y_pred, p_max, then prob::<label> in class_labels order.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    class_labels = list(class_labels)
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != len(class_labels):
        raise ValueError(
            f"probs shape {probs.shape} inconsistent with {len(class_labels)} class_labels"
        )
    n = probs.shape[0]
    for arr, name in ((ids, "ids"), (y_true, "y_true"), (y_pred, "y_pred")):
        if len(arr) != n:
            raise ValueError(f"{name} length {len(arr)} != probs rows {n}")

    columns: dict[str, np.ndarray] = {
        "complaint_id": np.asarray(ids, dtype=np.int64),
        "y_true": np.asarray([str(v) for v in y_true], dtype=object),
        "y_pred": np.asarray([str(v) for v in y_pred], dtype=object),
        "p_max": probs.max(axis=1).astype(np.float64) if n else np.empty(0, np.float64),
    }
    for j, label in enumerate(class_labels):
        columns[_prob_column(label)] = np.ascontiguousarray(probs[:, j], dtype=np.float64)

    meta = {
        "run_id": provenance.run_id,
        "config_sha256": provenance.config_sha256,
        "git_sha": provenance.git_sha,
        "split": provenance.split,
        "split_sha256": provenance.split_sha256,
        "input_sha256": provenance.input_sha256,
        "prompt_bundle_sha256": provenance.prompt_bundle_sha256,
        "class_labels": json.dumps(class_labels),
        "schema_version": provenance.schema_version,
    }
    kv = ", ".join(f"{_sql_str(k)}: {_sql_str(v)}" for k, v in meta.items())
    select_cols = ", ".join(_sql_ident(c) for c in columns)

    con = duckdb.connect()
    try:
        con.execute("SET threads=1")
        con.register("_preds", columns)
        con.execute(
            f"COPY (SELECT {select_cols} FROM _preds) TO {_sql_str(str(path))} "
            f"(FORMAT parquet, KV_METADATA {{{kv}}})"
        )
    finally:
        con.close()
    return path


def read_artifact(path) -> PredictionsArtifact:
    """Read a predictions artifact back into aligned arrays + provenance dict."""
    path = Path(path)
    con = duckdb.connect()
    try:
        con.execute("SET threads=1")
        kv_rows = con.execute(
            f"SELECT key, decode(value) FROM parquet_kv_metadata({_sql_str(str(path))})"
        ).fetchall()
        provenance = {
            (k.decode() if isinstance(k, bytes) else k): v for k, v in kv_rows
        }
        class_labels = json.loads(provenance["class_labels"])
        prob_cols = [_prob_column(lbl) for lbl in class_labels]
        select = list(_FIXED_COLUMNS) + prob_cols
        sql_cols = ", ".join(_sql_ident(c) for c in select)
        rows = con.execute(
            f"SELECT {sql_cols} FROM read_parquet({_sql_str(str(path))})"
        ).fetchall()
    finally:
        con.close()

    if rows:
        cols = list(zip(*rows, strict=True))
    else:
        cols = [()] * len(select)
    complaint_id = np.asarray(cols[0], dtype=np.int64)
    y_true = np.asarray([str(v) for v in cols[1]], dtype=object)
    y_pred = np.asarray([str(v) for v in cols[2]], dtype=object)
    p_max = np.asarray(cols[3], dtype=np.float64)
    if prob_cols:
        probs = np.column_stack(
            [np.asarray(cols[4 + j], dtype=np.float64) for j in range(len(prob_cols))]
        )
    else:
        probs = np.empty((len(complaint_id), 0), dtype=np.float64)
    return PredictionsArtifact(
        complaint_id=complaint_id,
        y_true=y_true,
        y_pred=y_pred,
        p_max=p_max,
        probs=probs,
        class_labels=class_labels,
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# Reconstruction from historical runs
# ---------------------------------------------------------------------------

def _repo_path(rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    return p if p.is_absolute() else REPO_ROOT / p


def _splits_dir(config: dict) -> Path:
    """Frozen-splits directory for a run config (same key as tier_a/tier_b/tier_c)."""
    return Path((config.get("data") or {}).get("splits_dir", DEFAULT_SPLITS_DIR))


def load_receipts_by_id(raw_log_path) -> dict[int, object]:
    """Map complaint_id -> returned content string from a tier_c receipts jsonl.

    Duplicate complaint_id lines are a HARD ERROR naming the offending ids, not a
    last-write-wins overwrite: with `max_concurrency > 1` the receipt line order follows
    completion order, so silently keeping the last line would make the reconstructed
    labels depend on file order — exactly the non-determinism this artifact exists to
    rule out. A run writes one receipt per example into a fresh timestamped directory,
    so a duplicate means two runs' receipts were merged; that must be resolved by a
    human, not papered over here.
    """
    path = _repo_path(raw_log_path)
    out: dict[int, object] = {}
    duplicates: list[int] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = int(rec["complaint_id"])
            if cid in out:
                duplicates.append(cid)
                continue
            out[cid] = rec.get("content")
    if duplicates:
        shown = sorted(set(duplicates))
        raise ValueError(
            f"duplicate complaint_id line(s) in receipts log {path}: "
            f"{shown[:MAX_OFFENDERS_SHOWN]}"
            f"{' ...' if len(shown) > MAX_OFFENDERS_SHOWN else ''} "
            f"({len(shown)} distinct id(s) repeated); reconstruction must not depend on "
            "line order — resolve the merged/duplicated log before backfilling"
        )
    return out


def reconstruct_tier_c_labels(ids, receipts_by_id, labels, fallback_label):
    """Per-example y_pred by REUSING tier_c.parse_label + fallback, joined by complaint_id.

    Bit-identical to what the runner computed: same parser, same fallback, same order.
    """
    from triage_lab.tier_c import parse_label

    y_pred = []
    for cid in ids:
        cid = int(cid)
        if cid not in receipts_by_id:
            raise KeyError(f"receipt for complaint_id {cid} missing from raw log")
        label = parse_label(receipts_by_id[cid], labels)
        y_pred.append(label if label is not None else fallback_label)
    return y_pred


def _onehot(y_pred, labels) -> np.ndarray:
    idx = {lbl: i for i, lbl in enumerate(labels)}
    probs = np.zeros((len(y_pred), len(labels)), dtype=np.float64)
    for i, lbl in enumerate(y_pred):
        probs[i, idx[lbl]] = 1.0
    return probs


def _check_prompt_bundle(bundle, record: dict, version: str) -> None:
    """Fail loud unless the loaded prompt bundle IS the one the run recorded.

    Tier C prompts are versioned and content-hashed (CLAUDE.md rule 4). If the frozen
    bundle on disk no longer hashes to what the record logged, the receipts were produced
    by a different prompt than the one we just loaded, and reconstruction would attribute
    that run's labels to today's prompt. Refuse rather than stamp.
    """
    logged = (record.get("extra") or {}).get("prompt_bundle_sha256")
    if not logged:
        raise ValueError(
            f"tier_c record {record.get('run_id', '?')[:8]} carries no "
            "extra.prompt_bundle_sha256; cannot confirm which prompt produced its "
            "receipts, so reconstruction is refused"
        )
    if bundle.bundle_sha256 != logged:
        raise ValueError(
            f"prompt bundle hash mismatch for version {version!r}: loaded bundle hashes "
            f"{bundle.bundle_sha256} but run {record.get('run_id', '?')[:8]} logged "
            f"{logged}; the frozen prompt changed since the run"
        )


def reconstruct_tier_c(record: dict, config: dict):
    """Rebuild (ids, y_true, y_pred, probs, class_labels) for a tier_c run from receipts."""
    from triage_lab import tier_c

    data = config.get("data", {})
    text_col = data.get("text_column", tier_c._DEFAULT_TEXT_COLUMN)
    label_col = data.get("label_column", tier_c._DEFAULT_LABEL_COLUMN)
    order_col = data.get("order_column", tier_c._DEFAULT_ORDER_COLUMN)
    eval_split = data["split"]
    eval_rows_cap = data.get("eval_rows_cap")
    cap_seed = int(data.get("cap_seed", config.get("seed", tier_c._DEFAULT_SEED)))

    eval_path = _splits_dir(config) / f"{eval_split}.parquet"
    ids, texts, y_true = tier_c.load_split_frame(eval_path, text_col, label_col, order_col)
    ids, texts, y_true = tier_c.subsample_eval(ids, texts, y_true, eval_rows_cap, cap_seed)

    prompt_cfg = config.get("prompt", {}) or {}
    version = prompt_cfg.get("version", tier_c._DEFAULT_PROMPT_VERSION)
    bundle = tier_c.load_prompt_bundle(version)
    _check_prompt_bundle(bundle, record, version)
    labels = list(bundle.labels)
    fallback_label = labels[0]

    raw_log_path = record["extra"]["raw_log_path"]
    receipts = load_receipts_by_id(raw_log_path)
    y_pred = reconstruct_tier_c_labels(ids, receipts, labels, fallback_label)
    probs = _onehot(y_pred, labels)
    return (
        np.asarray(ids, dtype=np.int64),
        np.asarray([str(v) for v in y_true], dtype=object),
        np.asarray(y_pred, dtype=object),
        probs,
        labels,
    )


def reconstruct_via_runner(runner_name: str, config: dict):
    """Deterministic offline reconstruction for tier_a/tier_b by re-invoking the runner."""
    if runner_name not in harness.RUNNERS:
        harness._load_optional_runners()
    if runner_name not in harness.RUNNERS:
        raise ValueError(f"runner {runner_name!r} not available for reconstruction")
    result = harness.RUNNERS[runner_name](config)
    if result.ids is None:
        raise ValueError(f"runner {runner_name!r} returned no ids; cannot build artifact")
    return (
        np.asarray(result.ids, dtype=np.int64),
        np.asarray(result.y_true, dtype=object),
        np.asarray(result.y_pred, dtype=object),
        np.asarray(result.probs, dtype=np.float64),
        list(result.class_labels),
    )


def reconstruct(record: dict, config: dict):
    """Dispatch reconstruction on the config's runner. tier_c -> receipts; else -> runner."""
    runner_name = config["model"]["runner"]
    if runner_name == "tier_c":
        return reconstruct_tier_c(record, config)
    return reconstruct_via_runner(runner_name, config)


# ---------------------------------------------------------------------------
# Verification gate
# ---------------------------------------------------------------------------

def _structural_row(check: str, ok: bool, detail: str) -> dict:
    return {"check": check, "kind": "structural", "ok": bool(ok), "detail": detail}


def _offenders(values, total: int) -> str:
    shown = list(values)[:MAX_OFFENDERS_SHOWN]
    tail = " ..." if total > len(shown) else ""
    return f"{shown}{tail}"


def check_ids(art_path) -> dict:
    """complaint_id must be non-null and unique. Counted in SQL, over the file itself.

    Done against the parquet rather than the in-memory arrays because NULL is only
    representable in the file (``read_artifact`` casts the column to int64), and because
    the counts stay O(1) memory on a full TEST-IID artifact.
    """
    src = _sql_str(str(art_path))
    con = duckdb.connect()
    try:
        con.execute("SET threads=1")
        n, n_nonnull, n_distinct = con.execute(
            "SELECT count(*), count(complaint_id), count(DISTINCT complaint_id) "
            f"FROM read_parquet({src})"
        ).fetchone()
        if n_nonnull == n and n_distinct == n:
            return _structural_row("ids_unique_nonnull", True, f"{n} rows, all distinct")
        dups = [
            r[0] for r in con.execute(
                f"SELECT complaint_id FROM read_parquet({src}) GROUP BY 1 "
                f"HAVING count(*) > 1 ORDER BY 1 LIMIT {MAX_OFFENDERS_SHOWN + 1}"
            ).fetchall()
        ]
    finally:
        con.close()
    n_dup = n_nonnull - n_distinct
    detail = f"{n - n_nonnull} NULL, {n_dup} duplicate row(s)"
    if dups:
        detail += f", ids {_offenders(dups, n_dup)}"
    return _structural_row("ids_unique_nonnull", False, detail)


# LEFT JOIN of the artifact onto the frozen split it declares. Membership and label
# agreement are two failure modes of one join, so they come from one scan.
_JOIN_SQL = """
WITH s AS (
    SELECT {order_col} AS cid, CAST({label_col} AS VARCHAR) AS lbl
    FROM read_parquet({split})
)
SELECT {select}
FROM read_parquet({art}) a LEFT JOIN s ON a.complaint_id = s.cid
{where}
"""


def _join_query(art_path, split_path, order_col, label_col, select, where=""):
    return _JOIN_SQL.format(
        order_col=_sql_ident(order_col),
        label_col=_sql_ident(label_col),
        split=_sql_str(str(split_path)),
        art=_sql_str(str(art_path)),
        select=select,
        where=where,
    )


def check_against_split(art: PredictionsArtifact, art_path, record: dict,
                        config: dict) -> list[dict]:
    """Join every artifact row to the frozen split: id membership + y_true agreement.

    These are the checks the aggregate gate structurally cannot make. Permuting the id
    column leaves accuracy, macro_f1, aurc and acc_at_cov::* bit-identical, so a run whose
    ids were mapped onto the wrong rows passes the metric diff and then silently poisons
    every downstream join (router, drift-by-year, demo). Here a permuted id column shows
    up as y_true disagreeing with the split's label for that id.
    """
    names = ("ids_in_split", "y_true_matches_split")
    declared = art.provenance.get("split", "")
    logged = (record.get("dataset") or {}).get("split", "")
    if declared and logged and declared != logged:
        detail = f"artifact declares split {declared!r}, record logged {logged!r}"
        return [_structural_row(name, False, detail) for name in names]
    split = declared or logged
    if not split:
        return [_structural_row(n, False, "no split named by artifact or record") for n in names]

    data = config.get("data") or {}
    label_col = data.get("label_column", _DEFAULT_LABEL_COLUMN)
    order_col = data.get("order_column", _DEFAULT_ORDER_COLUMN)
    split_path = _splits_dir(config) / f"{split}.parquet"
    if not split_path.exists():
        detail = f"frozen split parquet not found at {split_path}"
        return [_structural_row(name, False, detail) for name in names]

    con = duckdb.connect()
    try:
        con.execute("SET threads=1")
        n_missing, n_mismatch = con.execute(
            _join_query(
                art_path, split_path, order_col, label_col,
                select="count(*) FILTER (WHERE s.cid IS NULL), "
                       "count(*) FILTER (WHERE s.cid IS NOT NULL "
                       "AND s.lbl IS DISTINCT FROM a.y_true)",
            )
        ).fetchone()
        missing_ids, mismatch_ids = [], []
        if n_missing:
            missing_ids = [
                r[0] for r in con.execute(
                    _join_query(
                        art_path, split_path, order_col, label_col,
                        select="a.complaint_id",
                        where=f"WHERE s.cid IS NULL ORDER BY 1 LIMIT {MAX_OFFENDERS_SHOWN}",
                    )
                ).fetchall()
            ]
        if n_mismatch:
            mismatch_ids = [
                f"{r[0]}({r[1]}!={r[2]})" for r in con.execute(
                    _join_query(
                        art_path, split_path, order_col, label_col,
                        select="a.complaint_id, a.y_true, s.lbl",
                        where="WHERE s.cid IS NOT NULL AND s.lbl IS DISTINCT FROM a.y_true "
                              f"ORDER BY 1 LIMIT {MAX_OFFENDERS_SHOWN}",
                    )
                ).fetchall()
            ]
    finally:
        con.close()

    rows = [
        _structural_row(
            "ids_in_split",
            n_missing == 0,
            f"all {len(art)} ids in {split}" if n_missing == 0
            else f"{n_missing} id(s) absent from {split}: {_offenders(missing_ids, n_missing)}",
        ),
        _structural_row(
            "y_true_matches_split",
            n_mismatch == 0,
            f"{len(art)} labels match {split}" if n_mismatch == 0
            else f"{n_mismatch} row(s) disagree with {split} "
                 f"(id(artifact!=split)): {_offenders(mismatch_ids, n_mismatch)}",
        ),
    ]
    return rows


def check_probs(art: PredictionsArtifact, runner_name: str) -> list[dict]:
    """p_max/probs/y_pred internal consistency, row by row.

    Shared: ``p_max`` must be exactly ``probs.max(axis=1)`` (float64 round-trips through
    parquet losslessly, so exact is the right tolerance). Then per tier:

    - **tier_a / tier_b**: ``y_pred`` must be an argmax class. "An" argmax, not
      ``argmax()``'s lowest-index winner: sklearn/torch both predict the lowest index on
      an exact tie, so the two agree in practice, but tolerating a tie row costs no
      detection power (the row still has to carry maximal probability) and avoids a
      false ✗ on a measure-zero coincidence. Ties are counted in the detail line.
    - **tier_c**: probs are the degenerate one-hot, so ``p_max`` must be exactly 1.0 and
      the hot column must be ``y_pred`` (row sums to 1.0).
    """
    n = len(art)
    if n == 0 or art.probs.shape[1] == 0:
        return [_structural_row("p_max_equals_probs_max", True, "0 rows / no prob columns")]

    recomputed = art.probs.max(axis=1)
    n_bad = int(np.count_nonzero(recomputed != art.p_max))
    rows = [_structural_row(
        "p_max_equals_probs_max",
        n_bad == 0,
        f"{n} rows exact" if n_bad == 0 else f"{n_bad}/{n} row(s) differ from probs.max",
    )]

    index = {str(lbl): j for j, lbl in enumerate(art.class_labels)}
    pred_idx = np.array([index.get(str(v), -1) for v in art.y_pred], dtype=np.int64)
    n_unknown = int(np.count_nonzero(pred_idx < 0))
    if n_unknown:
        rows.append(_structural_row(
            "y_pred_in_class_labels", False,
            f"{n_unknown}/{n} y_pred value(s) are not class_labels members",
        ))
        return rows
    p_at_pred = art.probs[np.arange(n), pred_idx]

    if runner_name == "tier_c":
        n_not_one = int(np.count_nonzero(art.p_max != 1.0))
        rows.append(_structural_row(
            "p_max_is_one", n_not_one == 0,
            f"{n} rows at 1.0" if n_not_one == 0
            else f"{n_not_one}/{n} degenerate row(s) with p_max != 1.0",
        ))
        n_off = int(np.count_nonzero((p_at_pred != 1.0) | (art.probs.sum(axis=1) != 1.0)))
        rows.append(_structural_row(
            "y_pred_matches_onehot", n_off == 0,
            f"{n} one-hot rows hot on y_pred" if n_off == 0
            else f"{n_off}/{n} row(s) whose hot column is not y_pred",
        ))
        return rows

    n_off = int(np.count_nonzero(p_at_pred != art.p_max))
    n_ties = int(np.count_nonzero((pred_idx != art.probs.argmax(axis=1)) & (p_at_pred == art.p_max)))
    rows.append(_structural_row(
        "y_pred_is_argmax", n_off == 0,
        f"{n} rows ({n_ties} exact tie(s))" if n_off == 0
        else f"{n_off}/{n} y_pred value(s) do not carry the row's max probability",
    ))
    return rows


def verify_metrics(art: PredictionsArtifact, record: dict) -> list[dict]:
    """Recompute VERIFY_METRICS from the artifact and diff against the record's points.

    Returns one row per metric: {check, kind, logged, recomputed, abs_delta, ok}.
    """
    points = harness.evaluate(art.y_true, art.y_pred, art.probs, art.class_labels)
    rows = []
    for name in VERIFY_METRICS:
        logged = record["metrics"][name]["point"]
        got = points[name]
        delta = abs(got - logged)
        rows.append({
            "check": name,
            "kind": "metric",
            "logged": float(logged),
            "recomputed": float(got),
            "abs_delta": float(delta),
            "ok": bool(delta <= VERIFY_TOL),
        })
    return rows


def verify_artifact(art: PredictionsArtifact, record: dict, config: dict, *,
                    art_path) -> list[dict]:
    """Full gate: structural per-example checks first, then the aggregate metric diff.

    Structural rows come first because a row-mapping fault explains the metric rows below
    it; every row carries `check`/`kind`/`ok`, and the caller fails the run if any `ok` is
    False. No check has a skip path — a check that cannot be performed (missing split
    file, split name disagreement) reports ✗, because "not verified" is not "verified".
    """
    runner_name = (config.get("model") or {}).get("runner", "")
    rows = [check_ids(art_path)]
    rows.extend(check_against_split(art, art_path, record, config))
    rows.extend(check_probs(art, runner_name))
    rows.extend(verify_metrics(art, record))
    return rows


# ---------------------------------------------------------------------------
# Records / run selection
# ---------------------------------------------------------------------------

def load_records(results_path=DEFAULT_RESULTS_PATH) -> list[dict]:
    records = []
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def records_by_config(results_path=DEFAULT_RESULTS_PATH) -> dict[str, dict]:
    """Run records keyed by config file stem, for callers that select runs by config.

    A duplicated config stem is a hard error rather than a last-one-wins pick: two runs of
    the same config differ in something the record does not name (code, environment), and
    silently choosing one would make every downstream number depend on file order.
    """
    out: dict[str, dict] = {}
    for record in load_records(results_path):
        name = Path(record.get("config_path", "")).stem
        if name in out:
            raise ValueError(
                f"two run records share config {name!r} ({out[name]['run_id'][:8]}, "
                f"{record['run_id'][:8]}); cannot pick one without a rule"
            )
        out[name] = record
    return out


def _is_smoke(record: dict) -> bool:
    return "smoke" in Path(record.get("config_path", "")).name.lower()


def select_records(records, selectors, *, select_all: bool) -> list[dict]:
    if select_all:
        return list(records)
    chosen = []
    for sel in selectors:
        matches = [r for r in records if r["run_id"].startswith(sel)]
        if not matches:
            raise ValueError(f"no run_id matches prefix {sel!r}")
        chosen.extend(matches)
    return chosen


def config_doc_correction(record: dict) -> dict | None:
    """The registered documentation correction for this run, if the file hash matches it.

    Returns the registry entry only when BOTH hashes line up — the record still logs
    `recorded_sha256` and the file on disk now hashes to `corrected_sha256`. Any other
    combination returns None and the caller hard-fails, so the exception cannot widen into
    "this run is allowed to drift".
    """
    entry = CONFIG_DOC_CORRECTIONS.get(record.get("run_id", ""))
    if not entry:
        return None
    path = _repo_path(record["config_path"])
    if (record.get("config_sha256", "") == entry["recorded_sha256"]
            and harness.config_sha256(path) == entry["corrected_sha256"]):
        return entry
    return None


def load_config_checked(record: dict) -> dict:
    """Load the record's config, refusing it unless the file still hashes as logged.

    The backfill stamps ``record["config_sha256"]`` into the artifact it regenerates. If
    the config file has changed since the run, that stamp would name inputs that did not
    produce this data — a provenance forgery, and the exact failure the artifact exists to
    make impossible. Hard fail, naming both hashes.

    The single exception is ``CONFIG_DOC_CORRECTIONS``: a comment-only edit, registered in
    code against one run id and one exact pair of hashes. It is announced on stdout every
    time it is used, because a provenance exception that nobody sees is indistinguishable
    from a provenance hole.
    """
    path = _repo_path(record["config_path"])
    config = harness.load_config(path)
    actual = harness.config_sha256(path)
    logged = record.get("config_sha256", "")
    if actual == logged:
        return config

    correction = config_doc_correction(record)
    if correction is None:
        raise ValueError(
            f"config hash mismatch for {path}: file hashes {actual} but run "
            f"{record.get('run_id', '?')[:8]} logged {logged}; the config changed since "
            "the run, so this reconstruction is not that run"
        )
    print(
        f"[{record['run_id'][:8]}] ACCEPTED REGISTERED DOCUMENTATION CORRECTION for "
        f"{path.name}: recorded {logged[:12]} -> file {actual[:12]}. "
        f"{correction['reason']}"
    )
    return config


# ---------------------------------------------------------------------------
# Backfill driver
# ---------------------------------------------------------------------------

def backfill(
    records,
    *,
    preds_dir=DEFAULT_PREDS_DIR,
    force: bool = False,
    verify: bool = True,
    only: str | None = None,
) -> dict:
    """Regenerate + (optionally) verify artifacts for the given records.

    Returns {"results": [per-run dicts], "ok": bool}. Skips smoke runs, honors --only,
    and skips regeneration of existing artifacts unless --force (still verifies them).
    Config loading is inside the try so a hash mismatch (`load_config_checked`) lands as a
    reported error row and a nonzero exit, like any other reconstruction failure.
    """
    preds_dir = Path(preds_dir)
    out_rows = []
    all_ok = True
    for record in records:
        run_id = record["run_id"]
        short = run_id[:8]
        config_path = record.get("config_path", "")
        if only is not None and only not in run_id and only not in config_path:
            continue
        if _is_smoke(record):
            out_rows.append({"run_id": run_id, "status": "skip-smoke", "verify": None})
            continue
        art_path = preds_dir / f"{run_id}.parquet"
        runner_name = "?"

        try:
            config = load_config_checked(record)
            runner_name = config["model"]["runner"]
            if art_path.exists() and not force:
                status = "exists"
            else:
                ids, y_true, y_pred, probs, class_labels = reconstruct(record, config)
                provenance = ArtifactProvenance(
                    run_id=run_id,
                    config_sha256=record["config_sha256"],
                    split=record["dataset"]["split"],
                    split_sha256=record["dataset"].get("split_sha256", ""),
                    class_labels=class_labels,
                    git_sha=record.get("git_sha", ""),
                    input_sha256=record["dataset"].get("input_sha256", ""),
                    prompt_bundle_sha256=(
                        record.get("extra") or {}).get("prompt_bundle_sha256", ""),
                )
                write_artifact(
                    art_path,
                    ids=ids,
                    y_true=y_true,
                    y_pred=y_pred,
                    probs=probs,
                    class_labels=class_labels,
                    provenance=provenance,
                )
                status = "written"
        except Exception as exc:  # noqa: BLE001 — reconstruction failure is a finding, report it
            all_ok = False
            out_rows.append({
                "run_id": run_id,
                "status": f"error: {type(exc).__name__}: {exc}",
                "verify": None,
            })
            print(f"[{short}] {runner_name:7s} ERROR {type(exc).__name__}: {exc}")
            continue

        verify_rows = None
        if verify:
            art = read_artifact(art_path)
            verify_rows = verify_artifact(art, record, config, art_path=art_path)
            run_ok = all(row["ok"] for row in verify_rows)
            all_ok = all_ok and run_ok
            _print_verify_table(short, runner_name, status, verify_rows, run_ok)
        else:
            print(f"[{short}] {runner_name:7s} {status} (no verify)")

        out_rows.append({"run_id": run_id, "status": status, "verify": verify_rows})

    return {"results": out_rows, "ok": all_ok}


def _print_verify_table(short, runner_name, status, verify_rows, run_ok) -> None:
    """One ✓/✗ line per check. Structural rows print their detail across the number columns."""
    banner = "✓ PASS" if run_ok else "✗ FAIL"
    print(f"[{short}] {runner_name:7s} {status:8s} {banner}")
    print(f"    {'check':22s} {'logged':>14s} {'recomputed':>14s} {'abs_delta':>12s}  ok")
    for row in verify_rows:
        mark = "✓" if row["ok"] else "✗"
        if row["kind"] == "structural":
            print(f"    {row['check']:22s} {row['detail']:<43s}  {mark}")
        else:
            print(
                f"    {row['check']:22s} {row['logged']:14.10f} {row['recomputed']:14.10f} "
                f"{row['abs_delta']:12.2e}  {mark}"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m triage_lab.predictions")
    parser.add_argument("run_id", nargs="*", help="run_id prefix(es) to backfill")
    parser.add_argument("--all", action="store_true", help="backfill every non-smoke run")
    parser.add_argument("--force", action="store_true", help="regenerate even if artifact exists")
    parser.add_argument(
        "--only",
        default=None,
        help="process only runs whose run_id or config_path contains this substring "
        "(e.g. isolate the slow LogReg re-fit for a separate background launch)",
    )
    parser.add_argument("--no-verify", dest="verify", action="store_false")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--preds-dir", type=Path, default=DEFAULT_PREDS_DIR)
    args = parser.parse_args(argv)

    if not args.all and not args.run_id:
        parser.error("give run_id prefix(es) or --all")

    records = load_records(args.results)
    selected = select_records(records, args.run_id, select_all=args.all)
    summary = backfill(
        selected,
        preds_dir=args.preds_dir,
        force=args.force,
        verify=args.verify,
        only=args.only,
    )
    print(f"\nbackfill: {len(summary['results'])} run(s) processed, "
          f"overall {'OK' if summary['ok'] else 'MISMATCH'}")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    import sys

    from triage_lab.predictions import main as _main

    sys.exit(_main())
