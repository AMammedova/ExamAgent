"""Language selection: directive wiring, English no-op, and settings plumbing."""
from __future__ import annotations

import os

from examagent.services.llm import (
    LANGUAGE_NAMES,
    language_directive,
    system_with_language,
)


def test_english_is_a_no_op() -> None:
    assert language_directive("en") == ""
    assert system_with_language("BASE") == "BASE"


def test_unknown_language_is_a_no_op() -> None:
    assert language_directive("fr") == ""


def test_azerbaijani_directive_mentions_the_language_and_preserves_notation() -> None:
    directive = language_directive("az")
    assert directive.strip()
    assert "Azerbaijani" in directive
    assert "formulas" in directive.lower()
    assert "JSON keys" in directive


def test_system_with_language_appends_without_mutating_the_base() -> None:
    base = "BASE SYSTEM PROMPT"
    combined = system_with_language(base, "az")
    assert combined.startswith(base)
    assert combined != base
    assert "Azerbaijani" in combined


def test_falls_back_to_settings_language_when_unspecified(monkeypatch) -> None:
    from examagent.config import get_settings, reload_settings

    monkeypatch.setenv("LANGUAGE", "az")
    reload_settings()
    try:
        assert get_settings().language == "az"
        assert language_directive() != ""
    finally:
        monkeypatch.delenv("LANGUAGE", raising=False)
        reload_settings()
        assert get_settings().language == "en"


def test_language_setting_is_case_and_space_insensitive(monkeypatch) -> None:
    from examagent.config import get_settings, reload_settings

    monkeypatch.setenv("LANGUAGE", "  AZ  ")
    reload_settings()
    try:
        assert get_settings().language == "az"
    finally:
        monkeypatch.delenv("LANGUAGE", raising=False)
        reload_settings()


def test_language_names_cover_the_supported_set() -> None:
    assert LANGUAGE_NAMES["en"] == "English"
    assert "az" in LANGUAGE_NAMES
    assert "Az" in LANGUAGE_NAMES["az"] or "az" in LANGUAGE_NAMES["az"].lower()


def test_offline_evaluation_is_unaffected_by_language(monkeypatch, clean_db) -> None:
    """The deterministic paths must not change behaviour based on language."""
    from examagent.models.schemas import Question, QuestionType, Category, Priority
    from examagent.services.evaluator import evaluate

    q = Question(
        id="lang-test-calc", topic="backpropagation", category=Category.DL,
        question_type=QuestionType.ASSERTION_REASON, difficulty=5,
        priority=Priority.CRITICAL, prompt="p", correct_option="A",
        options=[{"key": k, "text": k} for k in "ABCDE"],
    )
    monkeypatch.setenv("LANGUAGE", "az")
    from examagent.config import reload_settings

    reload_settings()
    try:
        ev = evaluate(q, "A", use_llm=False)
        assert ev.score == 10.0
    finally:
        monkeypatch.delenv("LANGUAGE", raising=False)
        reload_settings()


# ------------------------------------------------- a dead account stops the flood
def _client_with_error(message: str):
    """An LLMClient whose provider always raises `message`.

    Built with explicit credentials rather than environment variables: the
    suite's own .env pins the provider to none, and `reload_settings()` would
    put it straight back.
    """
    from examagent.services import llm as llm_module

    calls = {"n": 0}

    class _Boom:
        class chat:  # noqa: N801 - mirrors the SDK's shape
            class completions:
                @staticmethod
                def create(**_kwargs):
                    calls["n"] += 1
                    raise RuntimeError(message)

    client = llm_module.LLMClient(provider="openai", api_key="sk-test",
                                  model="gpt-4o")
    client._client = _Boom()
    return client, calls


def test_a_transient_error_is_retried(monkeypatch) -> None:
    client, calls = _client_with_error("APITimeoutError: timed out")
    monkeypatch.setattr("time.sleep", lambda _s: None)
    resp = client.complete("hi", retries=2)
    assert not resp.ok
    assert calls["n"] == 3, "a timeout is worth another go"
    assert client.available, "a blip must not disable the client"


def test_exhausted_credit_is_not_retried_and_disables_the_client() -> None:
    """Observed live: with no credit left, every question retried three times,
    the SDK retried on top of that, and eight workers did it in parallel - the
    terminal filled with 429s and each paper crawled. The refusal is final, so
    it must be taken as final."""
    client, calls = _client_with_error(
        "RateLimitError: Error code: 429 - {'error': {'message': 'You have no "
        "credits remaining', 'code': 'credit_balance_exhausted'}}"
    )
    resp = client.complete("hi", retries=2)
    assert not resp.ok
    assert calls["n"] == 1, "a dead account must be asked once, not three times"
    assert "no credit" in (resp.error or "")
    assert not client.available, "the client must take itself out of service"

    again = client.complete("hi again")
    assert calls["n"] == 1, "later calls must not reach the provider at all"
    assert not again.ok


def test_a_rejected_key_is_also_treated_as_final() -> None:
    client, calls = _client_with_error("AuthenticationError: invalid_api_key")
    client.complete("hi", retries=2)
    assert calls["n"] == 1
    assert not client.available


def test_fatal_reasons_are_recognised_from_the_provider_message() -> None:
    from examagent.services.llm import _fatal_reason

    assert _fatal_reason("code: 'credit_balance_exhausted'")
    assert _fatal_reason("type: 'insufficient_quota'")
    assert not _fatal_reason("APIConnectionError: connection reset")
    assert not _fatal_reason("429 Too Many Requests"), (
        "a plain rate limit is transient and must stay retryable"
    )
