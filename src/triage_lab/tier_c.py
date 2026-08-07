"""Tier C — LLM (Claude via OpenRouter) few-shot runner (UPGRADE_PLAN.md §4.2).

Tier C classifies a complaint narrative into exactly one harmonized product class by
prompting a Claude model **through OpenRouter** (CLAUDE.md rule 6): an OpenAI-compatible
client pointed at ``https://openrouter.ai/api/v1`` with the ``OPENROUTER_API_KEY`` loaded
at runtime from the repo-root ``.env``. The prompt/schema/exemplars are the frozen,
content-hashed v1 bundle (``triage_lab.tier_c_prompt``); every run record carries the
``prompt_version`` and ``bundle_sha256`` so the exact prompt is reproducible.

Measurement contract (the reason this file is careful):

- **Cost is measured, never estimated** (CLAUDE.md rule 6). At run start we pull the
  published per-token prices for the configured model from OpenRouter's ``/models``
  endpoint (that endpoint *is* the price list) and fail loud if the slug is absent. The
  canonical ``cost_usd`` is ``Σ prompt_tokens·prompt_price + completion_tokens·completion_price``
  over the actual per-call token usage. We also ask OpenRouter to return its own
  ``usage.cost`` (``extra_body={"usage": {"include": true}}``) and record that separately
  as ``openrouter_reported_cost_usd`` — a cross-check, not the headline.
- **Every call leaves a raw receipt.** One JSON line per call is appended under
  ``results/tier_c_raw/<model.name>/<run_ts>/calls.jsonl`` with the id, upstream provider
  (OpenRouter's ``provider`` field), token counts, both cost figures, latency,
  finish_reason, the returned content string, and retry/error info. Never the API key,
  never the full prompt (reconstructible from the frozen bundle + example id).
- **probs is a degenerate one-hot.** An LLM label decision carries no class distribution,
  so ``probs`` is a one-hot row (confidence 1.0) on the predicted class. This is the honest
  representation and keeps every harness metric well-defined (see the module note below).

Determinism / integrity: the eval split parquet is re-hashed against the frozen
``splits_stats.yaml`` before any call (same fail-loud gate as tier_a/tier_b); the optional
``data.eval_rows_cap`` subsample is seeded by ``data.cap_seed`` and re-sorted so row order
is stable (same convention as ``tier_b.subsample_eval``).

Concurrency: ``model.params.max_concurrency`` (default 1 = sequential) fans the per-example
calls out across a ThreadPoolExecutor. Predictions are re-collected by input index, so
``y_true``/``y_pred`` stay id-aligned regardless of completion order; only the raw
receipt-line order follows completion order (each line still carries its ``complaint_id``).
The API's temperature-0 answers do not depend on call order, so this does not affect the
metrics — it only shortens wall-clock.

The ``openai`` client is imported lazily (inside the calling functions) so this module —
and its pure helpers (env loader, pricing math, parsing, subsample, receipts) — import and
unit-test with just numpy/duckdb, matching the harness's tolerant optional-runner loading.

Note on degenerate probs and the harness metrics (``triage_lab.metrics``): with every
confidence pinned at 1.0, ECE folds into the top bin (``|acc - 1|``), Brier is
``2·(1-accuracy)``, and the risk-coverage/AURC curve orders by ascending index (all ties).
None of these divide by zero or produce NaN on a one-hot ``(N, K)`` matrix, so
``bootstrap_ci`` is safe. What Tier C simply cannot report meaningfully is *selective*
prediction (there is no confidence to rank on) — the numbers are computed but should not be
read as calibration quality; the router phase supplies calibrated confidence elsewhere.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import numpy as np

from triage_lab.harness import RunnerResult, dataset_info, register_runner
from triage_lab.snapshot import sha256_file
from triage_lab.tier_c_prompt import build_messages, load_prompt_bundle

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLITS_DIR = REPO_ROOT / "data" / "splits"
RAW_LOG_ROOT = REPO_ROOT / "results" / "tier_c_raw"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODELS_URL = "https://openrouter.ai/api/v1/models"

_DEFAULT_TEXT_COLUMN = "narrative"
_DEFAULT_LABEL_COLUMN = "class"
_DEFAULT_ORDER_COLUMN = "complaint_id"
_DEFAULT_PROMPT_VERSION = "v1"
_DEFAULT_NUM_EXEMPLARS = 9
_DEFAULT_TEMPERATURE = 0.0
_DEFAULT_MAX_TOKENS = 64
_DEFAULT_REQUEST_TIMEOUT_S = 60.0
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_MAX_CONCURRENCY = 1
_DEFAULT_SEED = 20260806

# Retry backoff: sleep = min(base * 2**attempt, cap) seconds.
_BACKOFF_BASE_S = 0.5
_BACKOFF_CAP_S = 8.0


# ---------------------------------------------------------------------------
# .env loader (no python-dotenv dependency; only OPENROUTER_API_KEY)
# ---------------------------------------------------------------------------

def load_openrouter_key(env_path: Path | None = None, environ: dict | None = None) -> str:
    """Return OPENROUTER_API_KEY: process env wins, else parsed from repo-root .env.

    Parses simple ``KEY=VALUE`` lines (``#`` comments and blanks skipped, surrounding
    quotes stripped). Fails loud with an actionable message if the key is nowhere — the
    key is never logged or printed anywhere (CLAUDE.md rule 6).
    """
    environ = os.environ if environ is None else environ
    from_env = environ.get("OPENROUTER_API_KEY")
    if from_env:
        return from_env

    env_path = REPO_ROOT / ".env" if env_path is None else Path(env_path)
    if env_path.exists():
        for raw in env_path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == "OPENROUTER_API_KEY":
                return value.strip().strip('"').strip("'")

    raise RuntimeError(
        "OPENROUTER_API_KEY not found in the environment or in repo-root .env. "
        "Tier C goes through OpenRouter (CLAUDE.md rule 6): add OPENROUTER_API_KEY=... "
        "to .env (never commit it) or export it before running."
    )


# ---------------------------------------------------------------------------
# Pricing snapshot from OpenRouter /models (the published price list)
# ---------------------------------------------------------------------------

def fetch_models_payload(url: str = MODELS_URL, api_key: str | None = None,
                         timeout: float = 30.0) -> dict:
    """GET the OpenRouter /models payload (JSON). Auth is optional but sent if given."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def find_model_pricing(payload: dict, slug: str) -> dict:
    """Extract published per-token prices for ``slug`` from a /models payload.

    OpenRouter prices are per-token USD strings under ``pricing.{prompt,completion}``.
    Fails loud (listing same-provider slugs) if ``slug`` is absent, so a wrong/renamed
    slug is caught before spending a cent (CLAUDE.md rule 6).
    """
    data = payload.get("data", payload) if isinstance(payload, dict) else payload
    for model in data:
        if isinstance(model, dict) and model.get("id") == slug:
            pricing = model.get("pricing") or {}
            if "prompt" not in pricing or "completion" not in pricing:
                raise ValueError(
                    f"model {slug!r} carries no prompt/completion pricing in the "
                    "OpenRouter /models payload"
                )
            return {
                "prompt_usd_per_token": float(pricing["prompt"]),
                "completion_usd_per_token": float(pricing["completion"]),
            }
    provider_prefix = slug.split("/", 1)[0]
    same_provider = sorted(
        m.get("id") for m in data
        if isinstance(m, dict) and str(m.get("id", "")).startswith(provider_prefix + "/")
    )
    raise ValueError(
        f"model slug {slug!r} not found in OpenRouter /models. "
        f"Slugs under {provider_prefix!r}: {same_provider}"
    )


def compute_call_cost(prompt_tokens: int, completion_tokens: int, pricing: dict) -> float:
    """Canonical per-call cost: tokens × published per-token price (never estimated)."""
    return (
        prompt_tokens * pricing["prompt_usd_per_token"]
        + completion_tokens * pricing["completion_usd_per_token"]
    )


# ---------------------------------------------------------------------------
# Data loading + integrity + seeded subsample (mirrors tier_a / tier_b)
# ---------------------------------------------------------------------------

def load_split_frame(path, text_column: str, label_column: str, order_column: str):
    """Read (ids, texts, labels) from a split parquet, single-threaded and id-ordered."""
    con = duckdb.connect()
    try:
        con.execute("SET threads=1")
        con.execute("SET preserve_insertion_order=true")
        rows = con.execute(
            f'SELECT "{order_column}", "{text_column}", "{label_column}" '
            f"FROM read_parquet('{path}') "
            f'ORDER BY "{order_column}"'
        ).fetchall()
    finally:
        con.close()
    ids = [r[0] for r in rows]
    texts = [("" if r[1] is None else str(r[1])) for r in rows]
    labels = np.array([r[2] for r in rows], dtype=object)
    return ids, texts, labels


def _verify_integrity(split: str, path: Path, splits_stats_path: Path) -> None:
    info = dataset_info(split, splits_stats_path)
    actual = sha256_file(path)
    if actual != info["split_sha256"]:
        raise ValueError(
            f"integrity check failed for split {split!r}: parquet sha256 {actual} "
            f"!= frozen splits_stats.yaml {info['split_sha256']}"
        )


def subsample_eval(ids, texts, labels, cap: int | None, cap_seed: int):
    """Seeded subsample of an already-loaded eval split (frozen file untouched).

    Same convention as ``tier_b.subsample_eval``: ``cap=None`` (real runs) is a no-op; a
    small ``cap`` takes a deterministic ``default_rng(cap_seed)`` permutation of the first
    ``cap`` rows, re-sorted so id order stays stable. Only the in-memory view is thinned —
    the split's frozen sha256 is still verified upstream.
    """
    n = len(texts)
    if cap is None or cap >= n:
        return ids, texts, labels
    idx = np.sort(np.random.default_rng(cap_seed).permutation(n)[:cap])
    return ([ids[i] for i in idx], [texts[i] for i in idx], labels[idx])


# ---------------------------------------------------------------------------
# Response parsing + field extraction
# ---------------------------------------------------------------------------

def parse_label(content: str | None, labels: list[str]) -> str | None:
    """Parse the JSON content and return its ``label`` iff it is a valid enum member.

    Returns None on any failure (non-JSON, non-object, missing/unknown label) — the
    caller treats None as a parse/validation failure and substitutes the fallback.
    """
    if not content:
        return None
    try:
        obj = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    label = obj.get("label")
    return label if label in labels else None


def _extra_field(obj, name: str):
    """Read ``name`` from an object whether it is a plain attr, pydantic extra, or dict."""
    value = getattr(obj, name, None)
    if value is not None:
        return value
    model_extra = getattr(obj, "model_extra", None)
    if isinstance(model_extra, dict) and model_extra.get(name) is not None:
        return model_extra[name]
    if isinstance(obj, dict):
        return obj.get(name)
    return None


def extract_completion_fields(completion) -> dict:
    """Pull the fields we record from a chat-completion object (SDK- and dict-tolerant)."""
    choice = completion.choices[0]
    message = choice.message
    content = getattr(message, "content", None)
    finish_reason = getattr(choice, "finish_reason", None)

    usage = completion.usage
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens))
    openrouter_cost = _extra_field(usage, "cost")

    return {
        "content": content,
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "openrouter_cost": None if openrouter_cost is None else float(openrouter_cost),
        "provider": _extra_field(completion, "provider"),
    }


# ---------------------------------------------------------------------------
# OpenRouter client + retrying single call
# ---------------------------------------------------------------------------

def build_client(api_key: str, timeout: float):
    """Construct the OpenAI-compatible client pointed at OpenRouter (lazy openai import).

    ``max_retries=0``: we run our own retry loop so every attempt is recorded.
    """
    from openai import OpenAI

    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        timeout=timeout,
        max_retries=0,
    )


def _response_format(schema: dict) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {"name": "triage_label", "strict": True, "schema": schema},
    }


def _create_completion(client, *, slug, messages, schema, temperature, max_tokens):
    return client.chat.completions.create(
        model=slug,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=_response_format(schema),
        extra_body={"usage": {"include": True}},
    )


def _is_retryable(exc) -> bool:
    """Retry only transient transport failures: connection/timeout/429/5xx."""
    import openai

    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError)):
        return True
    if isinstance(exc, openai.APIStatusError):
        return int(getattr(exc, "status_code", 0) or 0) >= 500
    return False


def call_with_retries(client, request: dict, max_retries: int, *, sleep=time.sleep):
    """One classification call with exponential backoff on transient errors.

    Returns ``(fields, latency_ms, retries)``. Retries only transport/429/5xx (``_is_retryable``);
    a non-retryable 4xx (e.g. a 400 from a malformed request) is re-raised immediately, and a
    still-failing transient error after ``max_retries`` is re-raised too — both fail the run
    loud rather than masquerading as a per-example parse failure. Parse/validation of the
    *returned content* is the caller's job (that path uses the fallback label).
    """
    import openai

    attempt = 0
    while True:
        t0 = time.perf_counter()
        try:
            completion = _create_completion(client, **request)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return extract_completion_fields(completion), latency_ms, attempt
        except openai.APIError as exc:
            if _is_retryable(exc) and attempt < max_retries:
                sleep(min(_BACKOFF_BASE_S * (2**attempt), _BACKOFF_CAP_S))
                attempt += 1
                continue
            raise


# ---------------------------------------------------------------------------
# Raw receipt line
# ---------------------------------------------------------------------------

def build_receipt(
    *,
    complaint_id,
    slug: str,
    provider,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    computed_cost_usd: float,
    openrouter_cost_usd,
    latency_ms: float,
    finish_reason,
    content,
    retries: int,
    parse_failed: bool,
) -> dict:
    """One raw per-call receipt line. No API key, no prompt (reconstructible from bundle)."""
    return {
        "complaint_id": int(complaint_id),
        "slug": slug,
        "provider": provider,
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(total_tokens),
        "computed_cost_usd": float(computed_cost_usd),
        "openrouter_cost_usd": openrouter_cost_usd,
        "latency_ms": float(latency_ms),
        "finish_reason": finish_reason,
        "content": content,
        "retries": int(retries),
        "parse_failed": bool(parse_failed),
    }


# ---------------------------------------------------------------------------
# Per-example classification (optionally concurrent)
# ---------------------------------------------------------------------------

def classify_examples(
    client,
    ids,
    texts,
    *,
    bundle,
    num_exemplars: int,
    labels: list[str],
    fallback_label: str,
    slug: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    pricing: dict,
    max_concurrency: int,
    receipt_sink,
) -> list[dict]:
    """Classify every example and return per-example results in the input (id) order.

    Each example is handled by ``_work``; with ``max_concurrency > 1`` the work runs on a
    ThreadPoolExecutor (the openai/httpx client is thread-safe for concurrent requests).
    Results are placed into a pre-sized list by their input index, so the returned list is
    aligned to the id-ordered ``ids``/``texts`` regardless of completion order — the caller
    can zip it straight against ``y_true``. ``receipt_sink(receipt)`` is always invoked under
    a lock, so it is thread-safe; with concurrency the receipt *line order* in the sink
    follows completion order, but every line self-identifies via its ``complaint_id``.
    Per-call latency and retries are measured per worker (``call_with_retries``) and are
    unaffected by concurrency.
    """
    n = len(texts)
    results: list[dict | None] = [None] * n
    lock = threading.Lock()

    def _work(idx: int) -> dict:
        complaint_id = ids[idx]
        narrative = texts[idx]
        messages = build_messages(bundle, narrative, num_exemplars)
        request = {
            "slug": slug,
            "messages": messages,
            "schema": bundle.schema,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        fields, latency_ms, retries = call_with_retries(client, request, max_retries)
        prompt_tokens = fields["prompt_tokens"]
        completion_tokens = fields["completion_tokens"]
        call_cost = compute_call_cost(prompt_tokens, completion_tokens, pricing)

        label = parse_label(fields["content"], labels)
        parse_failed = label is None
        if parse_failed:
            label = fallback_label

        receipt = build_receipt(
            complaint_id=complaint_id,
            slug=slug,
            provider=fields["provider"],
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=fields["total_tokens"],
            computed_cost_usd=call_cost,
            openrouter_cost_usd=fields["openrouter_cost"],
            latency_ms=latency_ms,
            finish_reason=fields["finish_reason"],
            content=fields["content"],
            retries=retries,
            parse_failed=parse_failed,
        )
        with lock:
            receipt_sink(receipt)

        return {
            "idx": idx,
            "complaint_id": complaint_id,
            "label": label,
            "parse_failed": parse_failed,
            "provider": fields["provider"] or "unknown",
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "openrouter_cost": fields["openrouter_cost"],
            "call_cost": call_cost,
            "latency_ms": latency_ms,
        }

    workers = max(1, int(max_concurrency))
    if workers == 1:
        for idx in range(n):
            results[idx] = _work(idx)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_work, idx) for idx in range(n)]
            for future in futures:
                record = future.result()  # re-raises worker exceptions -> loud fail
                results[record["idx"]] = record

    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# Registered runner
# ---------------------------------------------------------------------------

def _percentile(values: list[float], q: float):
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


@register_runner("tier_c")
def tier_c_runner(config: dict) -> RunnerResult:
    """Config-driven Tier C runner (see module docstring). Makes live OpenRouter calls."""
    data = config.get("data", {})
    model_cfg = config.get("model", {})
    slug = model_cfg.get("slug")
    if not slug:
        raise ValueError("tier_c config missing model.slug (OpenRouter model id)")
    model_name = model_cfg.get("name", "tier_c")
    params = model_cfg.get("params", {}) or {}
    temperature = float(params.get("temperature", _DEFAULT_TEMPERATURE))
    max_tokens = int(params.get("max_tokens", _DEFAULT_MAX_TOKENS))
    request_timeout_s = float(params.get("request_timeout_s", _DEFAULT_REQUEST_TIMEOUT_S))
    max_retries = int(params.get("max_retries", _DEFAULT_MAX_RETRIES))
    max_concurrency = int(params.get("max_concurrency", _DEFAULT_MAX_CONCURRENCY))

    prompt_cfg = config.get("prompt", {}) or {}
    version = prompt_cfg.get("version", _DEFAULT_PROMPT_VERSION)
    num_exemplars = int(prompt_cfg.get("num_exemplars", _DEFAULT_NUM_EXEMPLARS))

    text_col = data.get("text_column", _DEFAULT_TEXT_COLUMN)
    label_col = data.get("label_column", _DEFAULT_LABEL_COLUMN)
    order_col = data.get("order_column", _DEFAULT_ORDER_COLUMN)
    eval_split = data["split"]
    verify = data.get("verify_sha256", True)
    eval_rows_cap = data.get("eval_rows_cap")
    cap_seed = int(data.get("cap_seed", config.get("seed", _DEFAULT_SEED)))

    bundle = load_prompt_bundle(version)
    labels = bundle.labels
    # Deterministic fallback for a call whose returned content fails to parse/validate:
    # the first enum member (sorted taxonomy -> "card"). Chosen over "most common TRAIN
    # class" because it needs no split read and stays byte-deterministic; failures are
    # counted separately in extra.parse_failures so they never masquerade as real answers.
    fallback_label = labels[0]

    splits_dir = Path(data.get("splits_dir", DEFAULT_SPLITS_DIR))
    stats_path = splits_dir / "splits_stats.yaml"
    eval_path = splits_dir / f"{eval_split}.parquet"
    if verify:
        _verify_integrity(eval_split, eval_path, stats_path)

    ids, texts, y_true = load_split_frame(eval_path, text_col, label_col, order_col)
    ids, texts, y_true = subsample_eval(ids, texts, y_true, eval_rows_cap, cap_seed)

    api_key = load_openrouter_key()
    pricing = find_model_pricing(fetch_models_payload(api_key=api_key), slug)
    pricing_snapshot = {
        "slug": slug,
        "source": MODELS_URL,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "prompt_usd_per_token": pricing["prompt_usd_per_token"],
        "completion_usd_per_token": pricing["completion_usd_per_token"],
    }

    client = build_client(api_key, request_timeout_s)

    run_ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    raw_dir = RAW_LOG_ROOT / model_name / run_ts
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "calls.jsonl"

    with open(raw_path, "a", encoding="utf-8") as raw_f:
        def _sink(receipt: dict) -> None:
            raw_f.write(json.dumps(receipt, sort_keys=True) + "\n")

        results = classify_examples(
            client,
            ids,
            texts,
            bundle=bundle,
            num_exemplars=num_exemplars,
            labels=labels,
            fallback_label=fallback_label,
            slug=slug,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            pricing=pricing,
            max_concurrency=max_concurrency,
            receipt_sink=_sink,
        )

    # Aggregate deterministically in id order (results is already id-aligned).
    y_pred: list[str] = []
    latencies: list[float] = []
    provider_hist: dict[str, int] = {}
    parse_failures = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    computed_cost = 0.0
    openrouter_cost_total = 0.0
    for r in results:
        y_pred.append(r["label"])
        latencies.append(r["latency_ms"])
        provider_hist[r["provider"]] = provider_hist.get(r["provider"], 0) + 1
        parse_failures += int(r["parse_failed"])
        total_prompt_tokens += r["prompt_tokens"]
        total_completion_tokens += r["completion_tokens"]
        computed_cost += r["call_cost"]
        if r["openrouter_cost"] is not None:
            openrouter_cost_total += r["openrouter_cost"]

    n = len(y_pred)
    label2idx = {label: i for i, label in enumerate(labels)}
    probs = np.zeros((n, len(labels)), dtype=np.float64)
    for i, label in enumerate(y_pred):
        probs[i, label2idx[label]] = 1.0

    dataset = dataset_info(eval_split, stats_path)
    extra = {
        "tier": "C",
        "prompt_version": version,
        "prompt_bundle_sha256": bundle.bundle_sha256,
        "num_exemplars": num_exemplars,
        "model_slug": slug,
        "pricing_snapshot": pricing_snapshot,
        "prompt_tokens_total": total_prompt_tokens,
        "completion_tokens_total": total_completion_tokens,
        "tokens_per_call_mean": (
            (total_prompt_tokens + total_completion_tokens) / n if n else None
        ),
        "latency_ms_p50": _percentile(latencies, 50),
        "latency_ms_p95": _percentile(latencies, 95),
        "provider_histogram": provider_hist,
        "parse_failures": parse_failures,
        "openrouter_reported_cost_usd": openrouter_cost_total,
        "n_examples": n,
        "eval_rows_cap": eval_rows_cap,
        "cap_seed": cap_seed,
        "raw_log_path": str(raw_path.relative_to(REPO_ROOT)),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "max_concurrency": max_concurrency,
        "fallback_label": fallback_label,
        "run_type": config.get("run_type", "standard"),
    }
    return RunnerResult(
        y_true=np.asarray(y_true, dtype=object),
        y_pred=np.asarray(y_pred, dtype=object),
        probs=probs,
        class_labels=labels,
        dataset=dataset,
        cost_usd=computed_cost,
        extra=extra,
        ids=np.asarray(ids, dtype=np.int64),
    )
