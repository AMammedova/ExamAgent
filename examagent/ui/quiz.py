"""Quiz page: targeted practice with immediate strict feedback."""
from __future__ import annotations

import time
from typing import Any

import streamlit as st

from ..models.db import all_topics, session_scope
from ..models.schemas import Question, QuestionType
from ..services import planner, progress
from ..services.evaluator import evaluate
from ..services.question_gen import DIMENSION_TYPES, generate_question
from .common import (
    empty_state,
    llm_badge,
    render_evaluation,
    render_question,
)

STATE = "quiz"

TYPE_CHOICES = [
    ("Mixed (recommended)", None),
    ("Assertion & Reason", QuestionType.ASSERTION_REASON),
    ("Calculation", QuestionType.CALCULATION),
    ("Conceptual reasoning", QuestionType.CONCEPTUAL),
    ("What happens if…", QuestionType.WHAT_IF),
    ("Comparison", QuestionType.COMPARISON),
    ("Scenario", QuestionType.SCENARIO),
    ("Architecture interpretation", QuestionType.DIAGRAM),
    ("Graph interpretation", QuestionType.GRAPH),
]


def _state() -> dict[str, Any]:
    return st.session_state.setdefault(STATE, {})


def render() -> None:
    state = _state()
    payload = st.session_state.pop("nav_payload", None) or {}
    if payload.get("topic_id"):
        state["preset_topic"] = payload["topic_id"]
        state.pop("question", None)
        state.pop("evaluation", None)

    st.markdown("### Quiz")
    llm_badge()

    with st.expander("Question settings", expanded="question" not in state):
        _render_controls(state)

    if "question" not in state:
        empty_state("Pick a topic and press **Next question**.")
        _stats_row()
        return

    q: Question = state["question"]
    key = f"quiz_{state.get('serial', 0)}"

    st.divider()
    if "evaluation" not in state:
        answer = render_question(q, key)
        c1, c2 = st.columns([1, 1])
        if c1.button("Submit answer", type="primary", use_container_width=True):
            filled = (any(str(v).strip() for v in answer.values())
                      if isinstance(answer, dict) else bool(str(answer).strip()))
            if not filled:
                st.error("Answer first — I will evaluate it afterwards.")
            else:
                with st.spinner("Marking…"):
                    ev = evaluate(q, answer, use_llm=_use_llm())
                text = (answer if isinstance(answer, str)
                        else "; ".join(f"{k}={v}" for k, v in answer.items()))
                progress.record_attempt(
                    q, ev, student_answer=text, context="quiz",
                    seconds=int(time.time() - state.get("shown_at", time.time())),
                )
                state["evaluation"] = ev
                state["scores"] = state.get("scores", []) + [ev.score]
                st.rerun()
        if c2.button("Skip", use_container_width=True):
            _new_question(state)
            st.rerun()
    else:
        render_question(q, key, disabled=True)
        st.divider()
        render_evaluation(state["evaluation"], q)
        st.divider()
        c1, c2, c3 = st.columns(3)
        if c1.button("Next question", type="primary", use_container_width=True):
            _new_question(state)
            st.rerun()
        if c2.button("Same topic, harder", use_container_width=True):
            state["difficulty"] = min(6, state.get("difficulty", 4) + 1)
            _new_question(state)
            st.rerun()
        if c3.button("Same topic, easier", use_container_width=True):
            state["difficulty"] = max(1, state.get("difficulty", 4) - 1)
            _new_question(state)
            st.rerun()

    _stats_row()


def _render_controls(state: dict[str, Any]) -> None:
    with session_scope() as s:
        topics = sorted(all_topics(s), key=lambda t: (t.category, t.name))
    options = [(t.id, f"{t.name} · {t.category.split()[0]}") for t in topics]
    labels = ["Adaptive — target my weakest topic"] + [lbl for _, lbl in options]

    preset = state.get("preset_topic")
    index = 0
    if preset:
        for i, (tid, _) in enumerate(options):
            if tid == preset:
                index = i + 1
                break

    c1, c2, c3 = st.columns([2, 1.4, 1])
    with c1:
        pick = st.selectbox("Topic", labels, index=index, key="quiz_topic")
    with c2:
        type_label = st.selectbox("Question type", [t for t, _ in TYPE_CHOICES],
                                  key="quiz_type")
    with c3:
        difficulty = st.slider("Difficulty", 1, 6, state.get("difficulty", 4),
                               help="3-6 is exam territory")

    state["topic_id"] = None if pick == labels[0] else options[labels.index(pick) - 1][0]
    state["qtype"] = dict(TYPE_CHOICES)[type_label]
    state["difficulty"] = difficulty

    if st.button("Next question", type="primary"):
        _new_question(state)
        st.rerun()


def _new_question(state: dict[str, Any]) -> None:
    topic_id = state.get("topic_id")
    if not topic_id:
        nxt = planner.next_topic()
        topic_id = nxt["topic_id"] if nxt else "backpropagation"
        # target the dimension the student is weakest in
        if state.get("qtype") is None and nxt:
            focus = nxt.get("focus", "concept")
            types = DIMENSION_TYPES.get(focus)
            if types:
                state["auto_type"] = types[0]
    qtype = state.get("qtype") or state.pop("auto_type", None)

    with st.spinner("Building question…"):
        q = generate_question(
            topic_id, qtype, state.get("difficulty", 4), use_llm=_use_llm(),
            exclude_ids=set(state.get("seen", [])),
        )
    state["question"] = q
    state["seen"] = (state.get("seen", []) + [q.id])[-40:]
    state["serial"] = state.get("serial", 0) + 1
    state["count"] = state.get("count", 0) + 1
    state["shown_at"] = time.time()
    state.pop("evaluation", None)


def _stats_row() -> None:
    state = _state()
    scores = state.get("scores", [])
    if not scores:
        return
    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Answered this session", len(scores))
    c2.metric("Mean score", f"{sum(scores)/len(scores):.1f}/10")
    c3.metric("Correct (≥7)", f"{sum(1 for s in scores if s >= 7)}/{len(scores)}")


def _use_llm() -> bool:
    return bool(st.session_state.get("use_llm", True))
