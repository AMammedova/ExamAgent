"""Learning Path: one ordered queue of topics. Press Continue, learn the
topic, answer three questions on it, move to the next. No mode to pick, no
topic to search for - the whole page is one button and one list.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from ..models.db import all_topics, session_scope
from ..services import learning_path as lp
from ..services import progress, tutor
from ..services.evaluator import evaluate
from ..services.question_gen import generate_question
from .common import (
    chip,
    priority_color,
    render_citations,
    render_evaluation,
    render_question,
    score_bar,
)

STATE = "lp_active"

STATUS_ICON = {"done": "✅", "skipped": "⏭️", "current": "▶", "upcoming": "○"}


def _state() -> dict[str, Any]:
    return st.session_state.setdefault(STATE, {})


def _use_llm() -> bool:
    return bool(st.session_state.get("use_llm", True))


def _open(topic_id: str) -> None:
    st.session_state[STATE] = {"topic_id": topic_id}


def render() -> None:
    st.session_state.pop("nav_payload", None)
    st.markdown("### Learning Path")
    st.caption("One ordered pass through the syllabus: learn a topic, answer three "
               "questions on it, move on. Prerequisites first, CRITICAL topics first.")

    over = lp.summary()
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(f"**{over['done']}/{over['total']} topics done**"
                    + (f" · {over['skipped']} skipped" if over["skipped"] else ""))
        st.markdown(score_bar(over["fraction"], "#2da44e" if over["fraction"] >= 0.999
                              else "#539bf5"), unsafe_allow_html=True)
    with c2:
        if st.button("Restart path", use_container_width=True):
            lp.reset()
            st.session_state.pop(STATE, None)
            st.rerun()

    if over["finished"]:
        st.success(
            "Every topic has been learned or skipped. Time for a **Mock Exam** — "
            "use the sidebar to open it."
        )
        with st.expander("Go through it again"):
            _list_view()
        return

    state = _state()
    topic_id = state.get("topic_id") or lp.current_topic_id()
    if topic_id is None:
        st.info("Nothing left in the path.")
        return
    if "topic_id" not in state:
        state["topic_id"] = topic_id

    _topic_view(topic_id)
    st.divider()
    with st.expander("Full path", expanded=False):
        _list_view()


def _list_view() -> None:
    for row in lp.rows():
        t = row.topic
        icon = STATUS_ICON[row.status]
        mat = " 📄" if t.has_material else ""
        c1, c2, c3 = st.columns([0.4, 3, 1])
        c1.markdown(icon)
        with c2:
            st.markdown(f"{t.name}{mat}")
            st.caption(f"{t.category} · {t.priority}"
                      + (f" · {t.overall():.0%}" if t.attempt_count else ""))
        if row.status in ("upcoming", "skipped") and c3.button(
            "Jump", key=f"jump_{t.id}", use_container_width=True
        ):
            _open(t.id)
            st.rerun()
        elif row.status == "done" and c3.button(
            "Redo", key=f"redo_{t.id}", use_container_width=True
        ):
            lp.reopen(t.id)
            _open(t.id)
            st.rerun()


def _topic_view(topic_id: str) -> None:
    state = _state()
    with session_scope() as s:
        topic = next((t for t in all_topics(s) if t.id == topic_id), None)
    if topic is None:
        st.error("Unknown topic.")
        return

    c1, c2 = st.columns([4, 1])
    with c1:
        st.markdown(
            f"#### {topic.name} "
            + chip(topic.category, "#8b949e") + " "
            + chip(topic.priority, priority_color(topic.priority)),
            unsafe_allow_html=True,
        )
    if c2.button("Skip topic", use_container_width=True,
                help="Already know this one — move on without quizzing"):
        lp.mark_skip(topic_id)
        st.session_state.pop(STATE, None)
        st.rerun()

    # ---- learn
    if state.get("lesson_for") != topic_id:
        with st.spinner("Preparing the lesson…"):
            state["lesson"] = tutor.build_lesson(topic_id, use_llm=_use_llm())
        state["lesson_for"] = topic_id
    lesson = state["lesson"]
    with st.container(border=True):
        st.markdown(lesson.as_markdown())
        if lesson.citations:
            render_citations(lesson.citations)
        elif not lesson.grounded:
            st.caption("⚠️ Not grounded in uploaded material — upload lecture slides on "
                      "the Materials page for course-specific, citable explanations.")

    # ---- quiz (3 questions, fixed difficulty, no adaptive escalation)
    st.divider()
    scores: list[float] = state.setdefault("scores", [])
    idx = len(scores)

    if idx >= lp.QUESTIONS_PER_TOPIC:
        _topic_done(topic_id, scores)
        return

    st.markdown(f"**Question {idx + 1} of {lp.QUESTIONS_PER_TOPIC}**")
    if state.get("q_index") != idx:
        with st.spinner("Writing a question…"):
            state["q"] = generate_question(
                topic_id, difficulty=4, use_llm=_use_llm(),
                exclude_ids=set(state.get("asked", [])),
                avoid_prompts=state.get("asked_prompts"),
            )
        state["q_index"] = idx
        state.setdefault("asked", []).append(state["q"].id)
        state.setdefault("asked_prompts", []).append(state["q"].prompt)
        state.pop("eval", None)

    q = state["q"]
    key = f"lp_q_{topic_id}_{idx}"

    if "eval" not in state:
        answer = render_question(q, key)
        if st.button("Submit answer", type="primary"):
            filled = (any(str(v).strip() for v in answer.values())
                     if isinstance(answer, dict) else bool(str(answer).strip()))
            if not filled:
                st.error("Write an answer first — a blank answer scores zero in the "
                         "exam too.")
            else:
                with st.spinner("Marking…"):
                    ev = evaluate(q, answer, use_llm=_use_llm())
                text = (answer if isinstance(answer, str)
                       else "; ".join(f"{k}={v}" for k, v in answer.items()))
                progress.record_attempt(q, ev, student_answer=text,
                                        context="learning_path", seconds=100)
                state["eval"] = ev
                st.rerun()
        return

    render_question(q, key, disabled=True)
    st.divider()
    render_evaluation(state["eval"], q)
    st.divider()
    next_label = "Next question" if idx + 1 < lp.QUESTIONS_PER_TOPIC else "Finish topic"
    if st.button(next_label, type="primary"):
        scores.append(state["eval"].score)
        st.rerun()


def _topic_done(topic_id: str, scores: list[float]) -> None:
    mean = sum(scores) / len(scores) if scores else 0.0
    lp.mark_complete(topic_id)
    st.success(f"**Topic done** — mean {mean:.1f}/10 across {len(scores)} questions.")
    nxt = lp.current_topic_id()
    if nxt is None:
        if st.button("See summary", type="primary"):
            st.session_state.pop(STATE, None)
            st.rerun()
        return
    if st.button("Next topic ▶", type="primary"):
        st.session_state.pop(STATE, None)
        st.rerun()
