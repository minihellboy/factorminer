"""Model-call failures and native-model crashes are typed, recorded, and contained."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from factorminer.agent.llm_interface import (
    AnthropicProvider,
    CascadeProvider,
    DeepSeekProvider,
    GoogleProvider,
    LLMProvider,
    MockProvider,
    TruncatedResponseError,
    create_provider,
)
from factorminer.agent.provider_config import (
    DRAFT,
    PRIMARY,
    GuardedProvider,
    ProviderCallError,
    classify_provider_error,
    describe_providers,
    resolve_credential,
)
from factorminer.benchmark.model_worker import ModelOutcome, run_isolated_selection
from factorminer.benchmark.statistics import (
    MethodResult,
    aggregate_method_results,
    selection_metric,
    unavailable_selections,
)

TARGETS = "factorminer.tests._model_worker_targets"


class RateLimitError(Exception):
    pass


class FailingProvider(LLMProvider):
    model = "broken-model"

    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def generate(self, system_prompt, user_prompt, temperature=0.8, max_tokens=4096, *,
                 cacheable_prefix=None):
        raise self.exc

    @property
    def provider_name(self) -> str:
        return "failing"


def test_roles_resolve_distinct_credentials(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-env")
    monkeypatch.delenv("FACTORMINER_DRAFT_API_KEY", raising=False)
    primary = resolve_credential(PRIMARY, "openai", {"api_key": "primary-secret"})
    assert (primary.source, primary.api_key) == ("config", "primary-secret")
    # A draft never inherits the primary's explicit key.
    draft = resolve_credential(DRAFT, "deepseek", {"api_key": "primary-secret"})
    assert (draft.source, draft.api_key) == ("env:DEEPSEEK_API_KEY", "deepseek-env")
    monkeypatch.setenv("FACTORMINER_DRAFT_API_KEY", "draft-env")
    assert resolve_credential(DRAFT, "deepseek", {}).api_key == "draft-env"
    assert resolve_credential(DRAFT, "deepseek", {"draft_api_key": "d"}).source == "config"
    assert resolve_credential(DRAFT, "local", {}).api_key == "local"
    assert "secret" not in json.dumps(primary.describe())
    assert primary.describe()["fingerprint"]


def test_hosted_draft_uses_its_own_credential_and_endpoint(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-env")
    monkeypatch.delenv("FACTORMINER_DRAFT_API_KEY", raising=False)
    provider = create_provider({
        "provider": "anthropic", "model": "frontier", "api_key": "primary-secret",
        "cascade": {"enabled": True, "draft_provider": "deepseek", "draft_model": "deepseek-flash"},
    })
    assert isinstance(provider, CascadeProvider)
    assert isinstance(provider.draft.inner, DeepSeekProvider)
    assert provider.draft.api_key == "deepseek-env"
    assert provider.draft.base_url == "https://api.deepseek.com"
    assert provider.frontier.api_key == "primary-secret"
    roles = {entry["role"]: entry for entry in describe_providers(provider)}
    assert roles["draft"]["credential"]["source"] == "env:DEEPSEEK_API_KEY"
    assert roles["primary"]["credential"]["source"] == "config"


def test_deepseek_effort_is_sent_as_reasoning_effort_and_named(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-env")
    provider = create_provider({"provider": "deepseek", "model": "deepseek-flash", "effort": "max"})
    assert provider.inner.effort == "max"
    assert provider.inner.build_request_kwargs("s", "u")["reasoning_effort"] == "max"
    assert provider.provider_name == "deepseek/deepseek-flash@max"
    default = DeepSeekProvider(api_key="key")
    assert "reasoning_effort" not in default.build_request_kwargs("s", "u")
    assert default.provider_name == "deepseek/deepseek-flash"
    with pytest.raises(ValueError):
        DeepSeekProvider(api_key="key", effort="ultra")


def test_cli_passes_llm_effort_to_the_provider(monkeypatch):
    from factorminer.cli.app import _create_llm_provider
    from factorminer.utils.config import load_config

    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-env")
    cfg = load_config(overrides={"llm": {"provider": "deepseek", "model": "deepseek-flash", "effort": "high"}})
    assert cfg.llm.effort == "high"
    assert _create_llm_provider(cfg, mock=False).inner.effort == "high"


def test_guarded_provider_types_and_records_failures():
    guarded = GuardedProvider(FailingProvider(RateLimitError("slow down")), role=PRIMARY)
    with pytest.raises(ProviderCallError) as caught:
        guarded.generate("s", "u")
    error = caught.value
    assert (error.role, error.provider, error.model, error.kind) == (
        "primary", "failing", "broken-model", "rate_limit"
    )
    assert error.retryable and isinstance(error.__cause__, RateLimitError)
    described = guarded.describe()
    assert described["failures"] == 1 and described["recent_failures"][0]["kind"] == "rate_limit"


def test_guarded_provider_redacts_credential_from_failures():
    from factorminer.agent.provider_config import ProviderCredential

    credential = ProviderCredential(PRIMARY, "failing", "config", "secret-key")
    guarded = GuardedProvider(
        FailingProvider(RuntimeError("request used secret-key")),
        role=PRIMARY, credential=credential,
    )
    with pytest.raises(ProviderCallError) as caught:
        guarded.generate("s", "u")
    assert "secret-key" not in str(caught.value)
    assert "secret-key" not in json.dumps(guarded.describe())


@pytest.mark.parametrize(
    "exc,kind",
    [
        (type("AuthenticationError", (Exception,), {})(), "auth"),
        (type("APITimeoutError", (Exception,), {})(), "timeout"),
        (ModuleNotFoundError("openai"), "missing_dependency"),
        (type("StatusError", (Exception,), {"status_code": 503})(), "server"),
        (ConnectionResetError(), "connection"),
        (TruncatedResponseError("out of tokens"), "truncated"),
        (ValueError("?"), "unknown"),
    ],
)
def test_error_classification(exc, kind):
    assert classify_provider_error(exc) == kind


def test_cascade_escalates_when_the_draft_call_fails():
    draft = GuardedProvider(FailingProvider(ConnectionError("engine down")), role=DRAFT)
    cascade = CascadeProvider(draft=draft, frontier=MockProvider())
    text = cascade.generate("sys", "Please generate 2 candidate factors.")
    assert "1." in text
    assert cascade.draft_failures == 1 and cascade.escalations == 1
    assert "connection" in cascade.last_draft_error
    assert describe_providers(cascade)[0]["recent_failures"][0]["kind"] == "connection"


def test_failed_primary_call_is_recorded_in_run_manifest(tmp_path):
    from factorminer.core.ralph_loop import RalphLoop
    from factorminer.tests.test_ralph_loop import _TestConfig

    rng = np.random.default_rng(0)
    provider = GuardedProvider(
        FailingProvider(type("AuthenticationError", (Exception,), {})("bad key")), role=PRIMARY
    )
    loop = RalphLoop(
        config=_TestConfig(output_dir=str(tmp_path)),
        data_tensor=rng.normal(size=(15, 60, 8)),
        returns=rng.normal(0, 0.02, (15, 60)),
        llm_provider=provider,
    )
    with pytest.raises(ProviderCallError, match="auth"):
        loop.run(max_iterations=1)
    manifest = json.loads((Path(tmp_path) / "run_manifest.json").read_text())
    failure = manifest["model_providers"][0]["recent_failures"][0]
    assert (failure["role"], failure["kind"]) == ("primary", "auth")
    assert "bad key" in failure["message"]


def _panel():
    rng = np.random.default_rng(1)
    return {1: rng.normal(size=(30, 6)), 2: rng.normal(size=(30, 6))}, rng.normal(size=(30, 6))


def test_worker_returns_rankings_and_contains_failures():
    signals, returns = _panel()
    ok = run_isolated_selection("demo", signals, returns, target=f"{TARGETS}:rank_by_id")
    assert ok.ok and ok.ranking == [(2, 2.0), (1, 1.0)]

    error = run_isolated_selection("demo", signals, returns, target=f"{TARGETS}:raise_error")
    assert error.status == "error" and "model rejected the panel" in error.cause

    crash = run_isolated_selection("demo", signals, returns, target=f"{TARGETS}:abort")
    assert crash.status == "crashed" and crash.exit_code < 0
    assert "SIGABRT" in crash.cause

    hung = run_isolated_selection("demo", signals, returns, target=f"{TARGETS}:hang",
                                  timeout_s=2)
    assert hung.status == "timeout"

    malformed = run_isolated_selection("demo", signals, returns,
                                       target=f"{TARGETS}:nonfinite")
    assert malformed.status == "error" and "non-finite" in malformed.cause


def test_unavailable_selection_is_never_reported_as_a_score():
    record = ModelOutcome("xgboost", "crashed", cause="terminated by signal SIGABRT",
                          exit_code=-6).unavailable_record()
    selections = {"lasso": {"ic": 0.03, "icir": 0.4}, "xgboost": record}
    assert selection_metric(selections, "lasso", "ic") == 0.03
    assert math.isnan(selection_metric(selections, "xgboost", "ic"))
    assert unavailable_selections(selections) == ["xgboost"]

    runs = [
        MethodResult(method="m", xgb_ic=0.02, run_id=0),
        MethodResult(method="m", xgb_ic=float("nan"), unavailable=["xgboost"], run_id=1),
    ]
    aggregate = aggregate_method_results(runs)
    assert math.isnan(aggregate.xgb_ic) and aggregate.unavailable == ["xgboost"]


def test_frozen_evaluation_reports_native_crash_as_unavailable(monkeypatch, tmp_path):
    from factorminer.benchmark import frozen_evaluation
    from factorminer.benchmark.runtime import run_table1_benchmark
    from factorminer.utils.config import load_config

    root = Path(__file__).resolve().parents[2]
    crashed = ModelOutcome("xgboost", "crashed", cause="terminated by signal SIGSEGV",
                           exit_code=-11)
    monkeypatch.setattr(frozen_evaluation, "run_isolated_selection",
                        lambda *args, **kwargs: crashed)
    cfg = load_config(root / "factorminer" / "configs" / "binance_sample.yaml")
    cfg.benchmark.baselines = ["alpha101_adapted"]
    cfg.benchmark.freeze_top_k = 5
    summary = run_table1_benchmark(
        cfg, tmp_path, data_path=str(root / "data" / "binance_crypto_5m.csv")
    )
    universe = summary["alpha101_adapted"]["universes"]["Binance"]
    assert universe["selections"]["xgboost"]["status"] == "unavailable"
    assert universe["selections"]["xgboost"]["failure"] == "crashed"
    assert any("xgboost unavailable" in warning for warning in universe["warnings"])

    from factorminer.benchmark.runtime import _method_result_from_runtime_payload

    result = _method_result_from_runtime_payload("alpha101_adapted", summary["alpha101_adapted"], cfg)
    assert math.isnan(result.xgb_ic) and result.unavailable == ["xgboost"]


def _completion(content, finish_reason):
    message = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish_reason)])


class _FakeClient:
    def __init__(self, response):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response))


def test_reasoning_model_out_of_tokens_is_a_typed_failure_not_an_empty_batch(monkeypatch):
    provider = DeepSeekProvider(api_key="key")
    monkeypatch.setattr(provider, "_get_client", lambda: _FakeClient(_completion(None, "length")))
    guarded = GuardedProvider(provider, role=PRIMARY)
    with pytest.raises(ProviderCallError) as caught:
        guarded.generate("s", "u", max_tokens=8192)
    assert caught.value.kind == "truncated" and not caught.value.retryable
    assert "llm.max_tokens" in str(caught.value)


def test_partially_truncated_answer_is_kept(monkeypatch, caplog):
    provider = DeepSeekProvider(api_key="key")
    monkeypatch.setattr(provider, "_get_client", lambda: _FakeClient(_completion("a: Neg($close)", "length")))
    assert provider.generate("s", "u") == "a: Neg($close)"
    assert "truncated" in caplog.text


def test_anthropic_thinking_out_of_tokens_is_a_typed_failure(monkeypatch):
    provider = AnthropicProvider(api_key="key")
    thinking_only = SimpleNamespace(content=[SimpleNamespace(type="thinking", thinking="...")],
                                    stop_reason="max_tokens")
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kwargs: thinking_only))
    monkeypatch.setattr(provider, "_get_client", lambda: client)
    with pytest.raises(TruncatedResponseError):
        provider.generate("s", "u", max_tokens=64)


def test_gemini_out_of_tokens_is_a_typed_failure(monkeypatch):
    provider = GoogleProvider(api_key="key")
    finish = SimpleNamespace(name="MAX_TOKENS")
    candidate = SimpleNamespace(finish_reason=finish, content=SimpleNamespace(parts=[]))

    def text():
        raise ValueError("no text part")

    response = type("R", (), {"candidates": [candidate], "text": property(lambda self: text())})()
    client = SimpleNamespace(generate_content=lambda *args, **kwargs: response)
    monkeypatch.setattr(provider, "_get_client", lambda: client)
    with pytest.raises(TruncatedResponseError):
        provider.generate("s", "u", max_tokens=64)


def test_token_usage_is_recorded_even_when_the_answer_is_truncated(monkeypatch):
    provider = DeepSeekProvider(api_key="key")
    response = _completion(None, "length")
    response.usage = SimpleNamespace(prompt_tokens=1200, completion_tokens=8192,
                                     prompt_cache_hit_tokens=1024)
    monkeypatch.setattr(provider, "_get_client", lambda: _FakeClient(response))
    guarded = GuardedProvider(provider, role=PRIMARY)
    with pytest.raises(ProviderCallError):
        guarded.generate("s", "u", max_tokens=8192)
    assert guarded.usage.totals() == (1200, 8192, 1024)
    assert guarded.usage is provider.usage


def test_anthropic_and_gemini_usage_include_cached_and_thinking_tokens(monkeypatch):
    anthropic = AnthropicProvider(api_key="key")
    message = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="a: Neg($close)")], stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=100, output_tokens=40, cache_read_input_tokens=900,
                              cache_creation_input_tokens=0),
    )
    monkeypatch.setattr(anthropic, "_get_client",
                        lambda: SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: message)))
    anthropic.generate("s", "u")
    assert anthropic.usage.totals() == (1000, 40, 900)

    gemini = GoogleProvider(api_key="key")
    part = SimpleNamespace(text="a: Neg($close)")
    candidate = SimpleNamespace(finish_reason=SimpleNamespace(name="STOP"),
                                content=SimpleNamespace(parts=[part]))
    response = SimpleNamespace(
        candidates=[candidate],
        usage_metadata=SimpleNamespace(prompt_token_count=300, candidates_token_count=20,
                                       thoughts_token_count=80, cached_content_token_count=0),
    )
    monkeypatch.setattr(gemini, "_get_client",
                        lambda: SimpleNamespace(generate_content=lambda *a, **kw: response))
    gemini.generate("s", "u")
    assert gemini.usage.totals() == (300, 100, 0)


def test_mining_budget_records_provider_tokens_per_iteration(tmp_path):
    from factorminer.core.ralph_loop import RalphLoop
    from factorminer.utils.config import load_config

    cfg = load_config()
    cfg.mining.target_library_size = 3
    cfg.mining.batch_size = 4
    provider = MockProvider()
    calls = {"n": 0}
    original = provider.generate

    def generate(*args, **kwargs):
        calls["n"] += 1
        provider.usage.record(500, 200)
        return original(*args, **kwargs)

    provider.generate = generate  # type: ignore[method-assign]
    rng = np.random.default_rng(0)
    data = rng.normal(size=(12, 60, 8))
    returns = rng.normal(size=(12, 60))
    loop = RalphLoop(config=cfg, data_tensor=data, returns=returns, llm_provider=provider)
    loop._loop_services.execute_iteration(4)
    assert calls["n"] >= 1
    assert loop.budget.llm_prompt_tokens == 500 * calls["n"]
    assert loop.budget.llm_completion_tokens == 200 * calls["n"]


def test_a_truncated_batch_is_skipped_but_a_run_of_them_aborts(monkeypatch):
    from factorminer.agent.factor_generator import MAX_CONSECUTIVE_TRUNCATIONS, FactorGenerator

    provider = DeepSeekProvider(api_key="key")
    responses = [_completion(None, "length"), _completion("1. a: Neg($close)", "stop")]
    responses += [_completion(None, "length")] * MAX_CONSECUTIVE_TRUNCATIONS
    monkeypatch.setattr(provider, "_get_client", lambda: _FakeClient(responses.pop(0)))
    generator = FactorGenerator(llm_provider=GuardedProvider(provider, role=PRIMARY))
    assert generator.generate_batch(batch_size=1) == []
    assert [c.formula for c in generator.generate_batch(batch_size=1)] == ["Neg($close)"]
    for _ in range(MAX_CONSECUTIVE_TRUNCATIONS - 1):
        assert generator.generate_batch(batch_size=1) == []
    with pytest.raises(ProviderCallError) as caught:
        generator.generate_batch(batch_size=1)
    assert caught.value.kind == "truncated"


def test_deepseek_thinking_calls_get_the_full_reasoning_budget(monkeypatch):
    from factorminer.agent.llm_interface import (
        DEEPSEEK_MAX_OUTPUT_TOKENS,
        DEEPSEEK_THINKING_TIMEOUT_S,
    )

    thinking = DeepSeekProvider(api_key="key", effort="max", timeout_s=300)
    assert thinking.build_request_kwargs("s", "u", max_tokens=128)["max_tokens"] == DEEPSEEK_MAX_OUTPUT_TOKENS
    assert thinking.build_request_kwargs("s", "u", max_tokens=1_000_000)["max_tokens"] == DEEPSEEK_MAX_OUTPUT_TOKENS
    assert thinking.timeout_s == DEEPSEEK_THINKING_TIMEOUT_S
    default = DeepSeekProvider(api_key="key")  # API default effort is high: thinking
    assert default.build_request_kwargs("s", "u", max_tokens=128)["max_tokens"] == DEEPSEEK_MAX_OUTPUT_TOKENS
    capped = DeepSeekProvider(api_key="key", reasoning_budget=50_000)
    assert capped.build_request_kwargs("s", "u", max_tokens=128)["max_tokens"] == 50_000
    plain = DeepSeekProvider(api_key="key", effort="none", timeout_s=60)
    assert plain.build_request_kwargs("s", "u", max_tokens=128)["max_tokens"] == 128
    assert plain.timeout_s == 60

    sent = {}

    def create(**kwargs):
        sent.update(kwargs)
        return _completion(None, "length")

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(thinking, "_get_client", lambda: client)
    with pytest.raises(TruncatedResponseError, match=str(DEEPSEEK_MAX_OUTPUT_TOKENS)):
        thinking.generate("s", "u", max_tokens=128)
    assert sent["max_tokens"] == DEEPSEEK_MAX_OUTPUT_TOKENS


def test_cli_passes_reasoning_budget(monkeypatch):
    from factorminer.cli.app import _create_llm_provider
    from factorminer.utils.config import load_config

    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-env")
    cfg = load_config(overrides={"llm": {"provider": "deepseek", "reasoning_budget": 100_000}})
    assert _create_llm_provider(cfg, mock=False).inner.reasoning_budget == 100_000
    cfg = load_config(overrides={"llm": {"provider": "deepseek"}})
    assert _create_llm_provider(cfg, mock=False).inner.reasoning_budget == 393_216
