"""Tier C tests that run WITHOUT network or any live OpenRouter call.

Covered here (pure logic): the .env key loader precedence + failure, pricing extraction
and cost arithmetic against a /models-shaped payload, seeded-subsample determinism,
response parsing incl. malformed-JSON/unknown-label fallback, completion-field extraction
from SDK-shaped and dict-shaped objects, the retry loop's backoff/re-raise behaviour, and
the raw receipt line's schema (no API key, no prompt). Anything needing a real call is
proven by the smoke pipeline the orchestrator runs, never here.

Only the retry test needs the `openai` exception classes; it is importorskip-gated so the
rest of the module stays runnable in a light env, matching the tier_b test pattern.
"""

from __future__ import annotations

import json
import re
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from triage_lab import harness, tier_c
from triage_lab.tier_c_prompt import load_prompt_bundle

REPO_ROOT = tier_c.REPO_ROOT

LABELS = [
    "card",
    "credit_reporting",
    "debt_collection",
    "deposit_account",
    "money_service",
    "mortgage",
    "payday_personal_loan",
    "student_loan",
    "vehicle_loan",
]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_runner_registered():
    assert "tier_c" in harness.RUNNERS
    assert harness.RUNNERS["tier_c"] is tier_c.tier_c_runner


# ---------------------------------------------------------------------------
# .env loader precedence + failure
# ---------------------------------------------------------------------------

def test_env_loader_prefers_process_env(tmp_path):
    env = tmp_path / ".env"
    env.write_text("OPENROUTER_API_KEY=from-file\n")
    assert tier_c.load_openrouter_key(env, {"OPENROUTER_API_KEY": "from-env"}) == "from-env"


def test_env_loader_reads_file_when_env_absent(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "# a comment\n\nUNRELATED=x\nOPENROUTER_API_KEY = \"quoted-secret\"  \n"
    )
    assert tier_c.load_openrouter_key(env, {}) == "quoted-secret"


def test_env_loader_fails_loud_when_missing(tmp_path):
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY not found"):
        tier_c.load_openrouter_key(tmp_path / "nope.env", {})


# ---------------------------------------------------------------------------
# Pricing extraction + cost arithmetic
# ---------------------------------------------------------------------------

def _payload():
    return {
        "data": [
            {"id": "anthropic/claude-sonnet-4.5",
             "pricing": {"prompt": "0.000003", "completion": "0.000015"}},
            {"id": "anthropic/claude-haiku-4.5",
             "pricing": {"prompt": "0.000001", "completion": "0.000005"}},
            {"id": "openai/gpt-4o", "pricing": {"prompt": "0.0000025", "completion": "0.00001"}},
        ]
    }


def test_find_model_pricing_extracts_per_token_prices():
    pricing = tier_c.find_model_pricing(_payload(), "anthropic/claude-haiku-4.5")
    assert pricing == {
        "prompt_usd_per_token": 1e-6,
        "completion_usd_per_token": 5e-6,
    }


def test_find_model_pricing_missing_slug_lists_same_provider():
    with pytest.raises(ValueError, match="not found") as exc:
        tier_c.find_model_pricing(_payload(), "anthropic/claude-haiku-9")
    msg = str(exc.value)
    assert "anthropic/claude-haiku-4.5" in msg
    assert "anthropic/claude-sonnet-4.5" in msg
    assert "openai/gpt-4o" not in msg  # only same-provider hints


def test_compute_call_cost_is_tokens_times_price():
    pricing = {"prompt_usd_per_token": 1e-6, "completion_usd_per_token": 5e-6}
    # 1000 prompt tokens * 1e-6 + 40 completion * 5e-6 = 0.001 + 0.0002 = 0.0012
    assert tier_c.compute_call_cost(1000, 40, pricing) == pytest.approx(0.0012)
    assert tier_c.compute_call_cost(0, 0, pricing) == 0.0


# ---------------------------------------------------------------------------
# Seeded subsample determinism (mirrors tier_b convention)
# ---------------------------------------------------------------------------

def test_subsample_eval_is_seeded_sized_stable_and_noop():
    ids = list(range(100))
    texts = [f"t{i}" for i in ids]
    labels = np.array([f"c{i % 3}" for i in ids], dtype=object)

    i1, t1, l1 = tier_c.subsample_eval(ids, texts, labels, 10, 20260806)
    i2, t2, l2 = tier_c.subsample_eval(ids, texts, labels, 10, 20260806)
    assert len(t1) == 10
    assert i1 == i2 and t1 == t2 and np.array_equal(l1, l2)      # deterministic
    assert i1 == sorted(i1)                                       # stable id order preserved
    assert set(i1).issubset(set(ids))                            # real subset

    # A different cap_seed gives a different draw.
    i3, _, _ = tier_c.subsample_eval(ids, texts, labels, 10, 111)
    assert i3 != i1

    # cap None or >= n is a no-op returning the split unchanged.
    i0, t0, l0 = tier_c.subsample_eval(ids, texts, labels, None, 1)
    assert i0 is ids and t0 is texts and l0 is labels
    assert len(tier_c.subsample_eval(ids, texts, labels, 999, 1)[1]) == 100


# ---------------------------------------------------------------------------
# Response parsing incl. malformed / unknown-label fallback
# ---------------------------------------------------------------------------

def test_parse_label_accepts_valid_enum_member():
    assert tier_c.parse_label('{"label": "mortgage"}', LABELS) == "mortgage"


@pytest.mark.parametrize("content", [
    None,
    "",
    "not json at all",
    "[1,2,3]",                       # valid JSON, not an object
    '{"label": "not_a_class"}',      # unknown label
    '{"nope": "mortgage"}',          # missing key
    '{"label": "mortgage"',          # truncated (finish_reason=length shape)
])
def test_parse_label_returns_none_on_bad_content(content):
    assert tier_c.parse_label(content, LABELS) is None


def test_fallback_label_is_first_enum_member():
    # The runner substitutes labels[0] on parse failure; document the contract here.
    assert LABELS[0] == "card"


# ---------------------------------------------------------------------------
# Completion-field extraction (SDK-shaped and dict-shaped)
# ---------------------------------------------------------------------------

def _sdk_completion(*, content, provider, cost, finish="stop", pt=120, ct=8):
    usage = SimpleNamespace(prompt_tokens=pt, completion_tokens=ct,
                            total_tokens=pt + ct, cost=cost)
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message, finish_reason=finish)
    return SimpleNamespace(choices=[choice], usage=usage, provider=provider)


def test_extract_completion_fields_from_sdk_object():
    comp = _sdk_completion(content='{"label": "card"}', provider="Anthropic", cost=0.00012)
    fields = tier_c.extract_completion_fields(comp)
    assert fields["content"] == '{"label": "card"}'
    assert fields["provider"] == "Anthropic"
    assert fields["prompt_tokens"] == 120
    assert fields["completion_tokens"] == 8
    assert fields["total_tokens"] == 128
    assert fields["openrouter_cost"] == pytest.approx(0.00012)
    assert fields["finish_reason"] == "stop"


def test_extract_completion_fields_tolerates_missing_extras():
    # provider absent + usage.cost absent -> None, tokens still read.
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12)
    choice = SimpleNamespace(message=SimpleNamespace(content="x"), finish_reason="length")
    comp = SimpleNamespace(choices=[choice], usage=usage)
    fields = tier_c.extract_completion_fields(comp)
    assert fields["provider"] is None
    assert fields["openrouter_cost"] is None
    assert fields["total_tokens"] == 12


def test_extra_field_reads_pydantic_model_extra():
    obj = SimpleNamespace(model_extra={"provider": "DeepInfra"})
    assert tier_c._extra_field(obj, "provider") == "DeepInfra"


# ---------------------------------------------------------------------------
# Retry loop: backoff on transient, re-raise on non-retryable / exhausted
# ---------------------------------------------------------------------------

class _FakeCompletions:
    def __init__(self, script):
        self._script = list(script)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _FakeClient:
    def __init__(self, script):
        self.chat = SimpleNamespace(completions=_FakeCompletions(script))


def _request():
    return {"slug": "s", "messages": [], "schema": {}, "temperature": 0.0, "max_tokens": 8}


def test_call_with_retries_recovers_after_transient():
    openai = pytest.importorskip("openai")
    good = _sdk_completion(content='{"label": "card"}', provider="Anthropic", cost=1e-4)
    client = _FakeClient([
        openai.APITimeoutError(request=None),
        good,
    ])
    slept = []
    fields, latency_ms, retries = tier_c.call_with_retries(
        client, _request(), max_retries=3, sleep=slept.append
    )
    assert retries == 1
    assert fields["content"] == '{"label": "card"}'
    assert latency_ms >= 0.0
    assert len(slept) == 1                       # one backoff sleep before the retry
    assert client.chat.completions.calls == 2


def test_call_with_retries_reraises_after_exhausting():
    openai = pytest.importorskip("openai")
    client = _FakeClient([openai.APITimeoutError(request=None)] * 5)
    with pytest.raises(openai.APITimeoutError):
        tier_c.call_with_retries(client, _request(), max_retries=2, sleep=lambda _s: None)
    assert client.chat.completions.calls == 3     # initial + 2 retries


def test_call_with_retries_reraises_non_retryable_immediately():
    openai = pytest.importorskip("openai")
    import httpx

    response = httpx.Response(400, request=httpx.Request("POST", "https://openrouter.ai"))
    err = openai.BadRequestError(message="bad", response=response, body=None)
    client = _FakeClient([err])
    with pytest.raises(openai.BadRequestError):
        tier_c.call_with_retries(client, _request(), max_retries=3, sleep=lambda _s: None)
    assert client.chat.completions.calls == 1     # no retry on a 4xx


# ---------------------------------------------------------------------------
# Raw receipt line schema (no api key, no prompt)
# ---------------------------------------------------------------------------

def test_build_receipt_schema_and_no_secrets():
    receipt = tier_c.build_receipt(
        complaint_id=12345,
        slug="anthropic/claude-haiku-4.5",
        provider="Anthropic",
        prompt_tokens=120,
        completion_tokens=8,
        total_tokens=128,
        computed_cost_usd=0.00016,
        openrouter_cost_usd=0.00017,
        latency_ms=812.5,
        finish_reason="stop",
        content='{"label": "card"}',
        retries=1,
        parse_failed=False,
    )
    assert set(receipt) == {
        "complaint_id", "slug", "provider", "prompt_tokens", "completion_tokens",
        "total_tokens", "computed_cost_usd", "openrouter_cost_usd", "latency_ms",
        "finish_reason", "content", "retries", "parse_failed",
    }
    assert receipt["complaint_id"] == 12345 and isinstance(receipt["complaint_id"], int)
    # JSON-serializable and carries no api-key-ish field.
    line = json.dumps(receipt, sort_keys=True)
    lowered = line.lower()
    assert "api_key" not in lowered and "authorization" not in lowered
    assert "prompt" not in {k for k in receipt if k.endswith("_text")}


# ---------------------------------------------------------------------------
# Shipped smoke configs are well-formed + single-variable
# ---------------------------------------------------------------------------

def test_smoke_configs_wellformed_and_single_variable():
    cfgs = {}
    for name in ("tier_c_haiku_smoke_fewshot_cal", "tier_c_haiku_smoke_zeroshot_cal"):
        cfg = yaml.safe_load((REPO_ROOT / "configs" / f"{name}.yaml").read_text())
        assert cfg["model"]["runner"] == "tier_c"
        assert cfg["model"]["slug"]
        assert cfg["data"]["split"] == "cal"          # iteration only on CAL
        assert cfg["data"]["eval_rows_cap"] == 25
        assert cfg["prompt"]["version"] == "v1"
        cfgs[name] = cfg

    few = cfgs["tier_c_haiku_smoke_fewshot_cal"]
    zero = cfgs["tier_c_haiku_smoke_zeroshot_cal"]
    assert few["prompt"]["num_exemplars"] == 9
    assert zero["prompt"]["num_exemplars"] == 0
    # Single-variable discipline: only prompt.num_exemplars differs.
    few2 = json.loads(json.dumps(few))
    zero2 = json.loads(json.dumps(zero))
    few2["prompt"]["num_exemplars"] = None
    zero2["prompt"]["num_exemplars"] = None
    few2["model"]["name"] = zero2["model"]["name"] = "x"
    assert few2 == zero2


# ---------------------------------------------------------------------------
# Concurrency: predictions stay id-aligned regardless of completion order
# ---------------------------------------------------------------------------

class _ConcurrentMockCompletions:
    """Mock that answers with the label embedded in the query narrative, but completes in
    REVERSE id order (earlier ids sleep longer), to prove result re-collection by index."""

    def __init__(self, n, completion_order, order_lock):
        self._n = n
        self._completion_order = completion_order
        self._order_lock = order_lock

    def create(self, **kwargs):
        content = kwargs["messages"][-1]["content"]
        m = re.search(r"IDX=(\d+)\|LABEL=([a-z_]+)", content)
        idx, label = int(m.group(1)), m.group(2)
        time.sleep(0.003 * (self._n - idx))          # low idx finishes last
        with self._order_lock:
            self._completion_order.append(idx)
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=3, total_tokens=13, cost=1e-6)
        choice = SimpleNamespace(
            message=SimpleNamespace(content=json.dumps({"label": label})), finish_reason="stop"
        )
        return SimpleNamespace(choices=[choice], usage=usage, provider="Mock")


class _ConcurrentMockClient:
    def __init__(self, n):
        self.completion_order: list[int] = []
        comp = _ConcurrentMockCompletions(n, self.completion_order, threading.Lock())
        self.chat = SimpleNamespace(completions=comp)


def _run_classify(max_concurrency):
    bundle = load_prompt_bundle("v1")
    labels = bundle.labels
    n = 12
    ids = list(range(1000, 1000 + n))
    expected = [labels[i % len(labels)] for i in range(n)]
    texts = [f"IDX={i}|LABEL={expected[i]}" for i in range(n)]
    client = _ConcurrentMockClient(n)
    receipts: list[dict] = []
    results = tier_c.classify_examples(
        client, ids, texts,
        bundle=bundle, num_exemplars=0, labels=labels, fallback_label=labels[0],
        slug="anthropic/claude-haiku-4.5", temperature=0.0, max_tokens=8, max_retries=2,
        pricing={"prompt_usd_per_token": 1e-6, "completion_usd_per_token": 5e-6},
        max_concurrency=max_concurrency, receipt_sink=receipts.append,
    )
    return client, results, receipts, ids, expected, labels


def test_classify_examples_concurrent_is_id_aligned():
    client, results, receipts, ids, expected, _labels = _run_classify(max_concurrency=8)
    # Results and receipts are complete and predictions map to the correct id's narrative.
    assert [r["complaint_id"] for r in results] == ids       # id order preserved
    assert [r["label"] for r in results] == expected          # each label matches its id
    assert len(receipts) == len(ids)
    # The mock forced completion order to differ from id order -> alignment is non-trivial.
    assert client.completion_order != list(range(len(ids)))


def test_classify_examples_sequential_matches_concurrent():
    _c1, seq, _r1, ids, expected, _l = _run_classify(max_concurrency=1)
    assert [r["label"] for r in seq] == expected
    assert [r["complaint_id"] for r in seq] == ids
