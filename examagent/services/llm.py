"""Provider-agnostic LLM client.

The whole app is designed to degrade gracefully: when no API key is configured
(or a call fails), `LLMClient.available` is False and every caller falls back to
the deterministic engines. Switching providers is a change to LLM_PROVIDER.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

from ..config import get_logger, get_settings

log = get_logger(__name__)


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    latency_ms: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text.strip())


class LLMClient:
    """Thin wrapper over Anthropic / OpenAI chat completion.

    By default everything comes from the global `.env`-backed settings (the
    right behaviour for a single-user local install). Pass `provider`/`api_key`/
    `model`/... to override any of that for one specific client instance without
    touching the shared settings - this is what per-session credentials (see
    `set_session_llm` below) build on, so one visitor's key on a shared
    deployment never becomes every visitor's key.
    """

    def __init__(
        self,
        *,
        provider: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> None:
        self.settings = get_settings()
        self.provider = (provider or self.settings.llm_provider).strip().lower()
        self.model = model or self.settings.model_name
        self._api_key = api_key if api_key is not None else self.settings.api_key
        self._default_max_tokens = max_tokens or self.settings.max_tokens
        self._default_temperature = (
            self.settings.temperature if temperature is None else temperature
        )
        self._client: Any = None
        self._init_error: str | None = None
        if self.available:
            try:
                self._client = self._build_client()
            except Exception as exc:  # pragma: no cover - import/credential issues
                self._init_error = str(exc)
                log.warning("LLM client init failed: %s", exc)

    # ---- capability ----
    @property
    def available(self) -> bool:
        return (bool(self._api_key.strip()) and self.provider in ("anthropic", "openai")
                and self._init_error is None)

    def status(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "available": self.available and self._client is not None,
            "error": self._init_error,
            "mode": "LLM" if (self.available and self._client is not None) else "OFFLINE",
        }

    def _build_client(self) -> Any:
        if self.provider == "anthropic":
            import anthropic

            return anthropic.Anthropic(api_key=self._api_key)
        if self.provider == "openai":
            import openai

            return openai.OpenAI(api_key=self._api_key)
        raise LLMError(f"unsupported provider: {self.provider}")

    # ---- calls ----
    def complete(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int | None = None,
        temperature: float | None = None,
        retries: int = 2,
    ) -> LLMResponse:
        if not self.available or self._client is None:
            return LLMResponse("", self.provider, self.model,
                               error="no LLM configured (offline mode)")

        max_tokens = max_tokens or self._default_max_tokens
        temperature = self._default_temperature if temperature is None else temperature
        started = time.time()
        last_err = ""

        for attempt in range(retries + 1):
            try:
                if self.provider == "anthropic":
                    kwargs: dict[str, Any] = dict(
                        model=self.model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    if system:
                        kwargs["system"] = system
                    resp = self._client.messages.create(**kwargs)
                    text = "".join(
                        block.text for block in resp.content if getattr(block, "type", "") == "text"
                    )
                else:
                    messages = []
                    if system:
                        messages.append({"role": "system", "content": system})
                    messages.append({"role": "user", "content": prompt})
                    resp = self._client.chat.completions.create(
                        model=self.model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        messages=messages,
                    )
                    text = resp.choices[0].message.content or ""
                return LLMResponse(
                    text.strip(), self.provider, self.model,
                    latency_ms=int((time.time() - started) * 1000),
                )
            except Exception as exc:
                last_err = f"{type(exc).__name__}: {exc}"
                log.warning("LLM call failed (attempt %d): %s", attempt + 1, last_err)
                if attempt < retries:
                    time.sleep(1.2 * (attempt + 1))

        return LLMResponse("", self.provider, self.model, error=last_err)

    def complete_json(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> tuple[dict[str, Any] | list[Any] | None, LLMResponse]:
        """Ask for JSON and parse it defensively (models like to add prose)."""
        guard = ("\n\nRespond with ONLY valid JSON. No markdown fences, no commentary "
                 "before or after the JSON.")
        resp = self.complete(prompt + guard, system=system, max_tokens=max_tokens,
                             temperature=temperature)
        if not resp.ok:
            return None, resp
        data = extract_json(resp.text)
        if data is None:
            resp.error = "could not parse JSON from model output"
        return data, resp


def extract_json(text: str) -> dict[str, Any] | list[Any] | None:
    """Pull the first JSON object/array out of a model response."""
    if not text:
        return None
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
    return None


_client: LLMClient | None = None

#: session_state key holding this visitor's own {provider, api_key, model,
#: max_tokens, temperature} - set by the Settings page's "only for my session"
#: option, never written to disk, invisible to every other visitor.
_SESSION_OVERRIDE_KEY = "_llm_session_override"
_SESSION_CLIENT_KEY = "_llm_session_client"


def _session_state() -> Any:
    """The running Streamlit session's state, or None outside a Streamlit app
    (tests, scripts) - callers must treat None as "no session, use the global
    client"."""
    try:
        import streamlit as st

        # st.session_state raises outside a running Streamlit script context
        return st.session_state
    except Exception:
        return None


def get_llm(force: bool = False) -> LLMClient:
    """The active LLM client: this browser session's own credentials if it has
    set any (see `set_session_llm`), otherwise the shared global client built
    from `.env`. A session override is only ever read from and written to
    `st.session_state` - it never touches the global settings or the client
    any other visitor gets."""
    state = _session_state()
    if state is not None and state.get(_SESSION_OVERRIDE_KEY):
        client = state.get(_SESSION_CLIENT_KEY)
        if client is None or force:
            client = LLMClient(**state[_SESSION_OVERRIDE_KEY])
            state[_SESSION_CLIENT_KEY] = client
        return client

    global _client
    if _client is None or force:
        _client = LLMClient()
    return _client


def reset_llm() -> None:
    """Reset the shared global client (built from `.env`). Does not touch any
    session-scoped override - use `clear_session_llm` for that."""
    global _client
    _client = None


def set_session_llm(
    provider: str,
    api_key: str,
    model: str,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> None:
    """Set LLM credentials for *this browser session only*. Never written to
    `.env`, never visible to or usable by any other visitor of a shared
    deployment - it lives entirely in this session's `st.session_state`."""
    state = _session_state()
    if state is None:
        raise LLMError("no active Streamlit session to attach credentials to")
    state[_SESSION_OVERRIDE_KEY] = {
        "provider": provider,
        "api_key": api_key,
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    state.pop(_SESSION_CLIENT_KEY, None)


def clear_session_llm() -> None:
    """Drop this session's own credentials; subsequent calls fall back to the
    shared global client again."""
    state = _session_state()
    if state is not None:
        state.pop(_SESSION_OVERRIDE_KEY, None)
        state.pop(_SESSION_CLIENT_KEY, None)


def session_llm_active() -> bool:
    """Whether this browser session currently has its own credentials set."""
    state = _session_state()
    return bool(state is not None and state.get(_SESSION_OVERRIDE_KEY))


# --------------------------------------------------------------- language
#: display names used in the directive sent to the model and in the UI
LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "az": "Azerbaijani (Azərbaycan dili)",
}


def language_directive(language: str | None = None) -> str:
    """Instruction appended to a system prompt to switch the model's output language.

    English is the baseline and needs no instruction. For any other configured
    language, technical notation (formulas, variable names, code) stays in its
    standard form while prose is translated - a formula written in Azerbaijani
    prose would be unreadable to the student in an exam context.
    """
    lang = (language or get_settings().language).strip().lower()
    if lang not in LANGUAGE_NAMES or lang == "en":
        return ""
    name = LANGUAGE_NAMES[lang]
    return (
        f"\n\nWrite your entire response in {name} EXCEPT every standard ML/DL "
        f"term of art, which must stay in its original English form - never coin "
        f"or use a {name} translation for it, even a natural-sounding one. This "
        f"covers, without limit: learning paradigms and algorithm names "
        f"(supervised learning, unsupervised learning, reinforcement learning, "
        f"gradient descent, backpropagation, k-means, PCA ...), loss and metric "
        f"names (training loss, validation loss, cross-entropy, accuracy, "
        f"precision, recall, F1 ...), architectures and components (CNN, RNN, "
        f"LSTM, GRU, transformer, encoder, decoder, attention, dropout, batch "
        f"normalisation ...), phenomena (overfitting, underfitting, vanishing "
        f"gradient ...), plus all mathematical notation, formulas, variable "
        f"names and code. Only the connecting prose around these terms - the "
        f"sentences that explain, compare and reason - is written in {name}; "
        f"the terms themselves sit inside that prose unchanged, exactly as they "
        f"would appear on the student's own exam paper (which is itself "
        f"bilingual this way). When genuinely unsure whether something counts "
        f"as a term of art, leave it in English rather than guessing a "
        f"translation. JSON string values follow the same rule; JSON keys "
        f"themselves must stay exactly as specified in the schema."
    )


def system_with_language(base_system: str, language: str | None = None) -> str:
    """Append the language directive to a system prompt."""
    return base_system + language_directive(language)


# --------------------------------------------------------------- prompts
EXAMINER_SYSTEM = """You are a strict university examiner and tutor for a combined
Machine Learning and Deep Learning final exam. You are NOT a friendly chatbot.

Rules you never break:
- You never reveal an answer before the student has attempted the question.
- You reward precise, technical, examiner-friendly phrasing and penalise vagueness.
- You keep explanations short and dense. The student has 7 days; do not lecture.
- When given SOURCE MATERIAL, ground course-specific claims in it and cite [S1], [S2].
  If the sources do not establish something, say so explicitly instead of inventing it.
- University material outranks Udemy material for what the exam expects.
- You prefer active recall: ask, evaluate, then correct only what was actually wrong."""

TUTOR_SYSTEM = """You are an exam-preparation tutor for a university Machine Learning
and Deep Learning final exam that is 7 days away. Be concise, technical and
practical. Prioritise what is examinable. Never pad. Never repeat an explanation
the student has already demonstrated they understand - target the exact gap."""
