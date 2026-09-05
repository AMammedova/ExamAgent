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
