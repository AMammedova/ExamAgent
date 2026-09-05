"""Chat: persistent tutor conversation with slash commands."""
from __future__ import annotations

from typing import Any

import streamlit as st

from ..models.db import kv_get, kv_set, session_scope
from ..services import progress, tutor
from ..services.evaluator import evaluate
from .common import llm_badge, render_citations, render_evaluation, render_question

HISTORY_KEY = "chat_history"
MAX_STORED = 60


def _load_history() -> list[dict[str, Any]]:
    if HISTORY_KEY in st.session_state:
        return st.session_state[HISTORY_KEY]
    with session_scope() as s:
        stored = kv_get(s, "chat_history", []) or []
    st.session_state[HISTORY_KEY] = stored
    return stored


def _save_history(history: list[dict[str, Any]]) -> None:
    st.session_state[HISTORY_KEY] = history[-MAX_STORED:]
    with session_scope() as s:
        kv_set(s, "chat_history", [
            {"role": m["role"], "text": m.get("text", "")}
            for m in history[-MAX_STORED:]
        ])


def render() -> None:
    st.session_state.pop("nav_payload", None)
    st.markdown("### Tutor chat")
    llm_badge()

    with st.expander("Commands", expanded=False):
        st.markdown("\n".join(f"- `{k}` — {v}" for k, v in tutor.COMMANDS.items()))
        st.caption("Plain English works too: “teach me PCA”, “quiz me on CNN”, "
                   "“give me a backpropagation calculation”, “what should I study now”, "
                   "“show my weakest topics”, “give me a 30 minute session”.")

    history = _load_history()

    for i, msg in enumerate(history):
        with st.chat_message(msg["role"]):
            if msg.get("text"):
                st.markdown(msg["text"])

    # an active question from a previous command
    pending = st.session_state.get("chat_question")
    if pending is not None:
        st.divider()
        _render_pending(pending)

    prompt = st.chat_input("Ask, or type a command…")
    if not prompt:
        return

    history.append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = tutor.route_command(
                prompt, use_llm=bool(st.session_state.get("use_llm", True))
            )
        if result.text:
            st.markdown(result.text)
        citations = (result.payload or {}).get("citations")
        if citations:
            render_citations(citations)
        if result.kind == "question" and result.question is not None:
            st.session_state["chat_question"] = {"question": result.question}
        if result.navigate:
            st.session_state["pending_nav"] = (result.navigate, result.payload or {})
            st.caption(f"↪ Opening **{result.navigate}**")

    history.append({"role": "assistant", "text": result.text})
    _save_history(history)

    if result.navigate:
        target, payload = st.session_state.pop("pending_nav")
        st.session_state["nav_target"] = target
        st.session_state["nav_payload"] = payload
    st.rerun()


def _render_pending(pending: dict[str, Any]) -> None:
    q = pending["question"]
    st.markdown("#### Answer this")
    if "evaluation" not in pending:
        answer = render_question(q, "chat_q")
        c1, c2 = st.columns([1, 3])
        if c1.button("Submit", type="primary"):
            filled = (any(str(v).strip() for v in answer.values())
                      if isinstance(answer, dict) else bool(str(answer).strip()))
            if not filled:
                st.error("Answer first — I evaluate afterwards.")
            else:
                with st.spinner("Marking…"):
                    ev = evaluate(q, answer,
                                  use_llm=bool(st.session_state.get("use_llm", True)))
                text = (answer if isinstance(answer, str)
                        else "; ".join(f"{k}={v}" for k, v in answer.items()))
                progress.record_attempt(q, ev, student_answer=text, context="chat",
                                        seconds=120)
                pending["evaluation"] = ev
                st.rerun()
        if c2.button("Dismiss"):
            st.session_state.pop("chat_question", None)
            st.rerun()
    else:
        render_question(q, "chat_q", disabled=True)
        st.divider()
        render_evaluation(pending["evaluation"], q)
        c1, c2 = st.columns([1, 3])
        if c1.button("Another question", type="primary"):
            nxt = tutor.followup_question(
                q.topic, q, pending["evaluation"].score,
                use_llm=bool(st.session_state.get("use_llm", True)),
            )
            st.session_state["chat_question"] = {"question": nxt}
            st.rerun()
        if c2.button("Done"):
            st.session_state.pop("chat_question", None)
            st.rerun()


def clear_history() -> None:
    st.session_state[HISTORY_KEY] = []
    with session_scope() as s:
        kv_set(s, "chat_history", [])
