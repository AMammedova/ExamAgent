"""Study page: the 10-step teaching flow with enforced active recall.

State machine (kept in st.session_state["study"]):
    lesson  -> the explanation block, shown once
    recall  -> student explains it back, evaluated
    practice-> a queue of questions, adaptive after each answer
    done    -> summary + what changed in the profile
"""
from __future__ import annotations

import time
from typing import Any

import streamlit as st

from ..models.db import all_topics, session_scope
from ..models.schemas import Question, QuestionType, SessionMode
from ..services import planner, progress, tutor
from ..services.evaluator import evaluate
from ..services.question_gen import generate_question
from .common import (
    chip,
    llm_badge,
    priority_color,
    render_citations,
    render_evaluation,
    render_question,
    score_bar,
)

STATE = "study"


def _state() -> dict[str, Any]:
    return st.session_state.setdefault(STATE, {})


def _reset() -> None:
    st.session_state[STATE] = {}


def _topic_options() -> list[tuple[str, str]]:
    with session_scope() as s:
        rows = sorted(all_topics(s), key=lambda t: (t.category, t.subtopic, t.name))
        return [(t.id, f"{t.name}  ·  {t.category.split()[0]}") for t in rows]


def render() -> None:
    state = _state()
    payload = st.session_state.pop("nav_payload", None) or {}
    if payload.get("topic_id") and payload["topic_id"] != state.get("topic_id"):
        _reset()
        state = _state()
        state["topic_id"] = payload["topic_id"]
        state["mode"] = payload.get("mode", SessionMode.THIRTY.value)
    elif payload.get("mode") and not state.get("topic_id"):
        state["requested_mode"] = payload["mode"]

    if not state.get("topic_id"):
        _render_setup(state)
        return

    _render_session(state)


# --------------------------------------------------------------- setup
def _render_setup(state: dict[str, Any]) -> None:
    st.markdown("### Study")
    llm_badge()

    suggestion = planner.next_topic()
    if suggestion:
        st.markdown(
            f"<div class='ea-card'><b>Recommended:</b> {suggestion['topic']} "
            f"{chip(suggestion['focus'] + ' focus', '#539bf5')}"
            f"<div class='ea-muted'>{suggestion['reason']}</div></div>",
            unsafe_allow_html=True,
        )

    modes = [m.value for m in SessionMode]
    default_mode = state.get("requested_mode", SessionMode.THIRTY.value)
    c1, c2 = st.columns([1, 1])
    with c1:
        mode = st.selectbox("Session mode", modes,
                            index=modes.index(default_mode) if default_mode in modes else 1)
        st.caption(f"About {SessionMode(mode).minutes} minutes")
    with c2:
        options = _topic_options()
        labels = ["Let the planner choose"] + [label for _, label in options]
        pick = st.selectbox("Topic", labels, index=0)

    plan = planner.plan_session(mode)
    if plan["topic_names"]:
        st.caption("Planner would cover: " + ", ".join(plan["topic_names"][:5]))

    skip_lesson = st.checkbox(
        "Skip the explanation and go straight to questions", value=False,
        help="Recommended when you already know the topic and want retrieval practice only.",
    )

    if st.button("Start session", type="primary"):
        if pick == "Let the planner choose":
            topic_id = (suggestion["topic_id"] if suggestion
                        else (plan["topic_ids"][0] if plan["topic_ids"] else None))
        else:
            topic_id = options[labels.index(pick) - 1][0]
        if not topic_id:
            st.error("No topic available.")
            return
        _reset()
        s = _state()
        s.update({
            "topic_id": topic_id,
            "mode": mode,
            "step": "practice" if skip_lesson else "lesson",
            "started": time.time(),
            "answered": [],
            "session_id": progress.start_session(mode, topic_id),
            "queue_index": 0,
            "n_target": planner.MODE_QUESTIONS.get(mode, 7),
        })
        st.rerun()


# --------------------------------------------------------------- session
def _render_session(state: dict[str, Any]) -> None:
    topic_id = state["topic_id"]
    report = progress.topic_report(topic_id)
    name = report.get("name", topic_id)

    head, right = st.columns([3, 1])
    with head:
        st.markdown(f"### {name}")
        t = report.get("topic", {})
        st.markdown(
            chip(state.get("mode", "Study"), "#539bf5")
            + chip(t.get("priority", "HIGH"), priority_color(t.get("priority", "HIGH")))
            + chip(f"{report.get('mastery', 'Not tested')}", "#8b949e")
            + chip(f"weakest: {report.get('weak_dimension', '-')}", "#d4a72c"),
            unsafe_allow_html=True,
        )
    with right:
        elapsed = int(time.time() - state.get("started", time.time()))
        st.metric("Elapsed", f"{elapsed // 60}m {elapsed % 60:02d}s")
        if st.button("End session", use_container_width=True):
            _finish(state)
            return

    _dimension_strip(report)
    st.divider()

    step = state.get("step", "lesson")
    if step == "lesson":
        _step_lesson(state, topic_id, name)
    elif step == "recall":
        _step_recall(state, topic_id, name)
    elif step == "practice":
        _step_practice(state, topic_id, name)
    else:
        _step_done(state, topic_id, name)


def _dimension_strip(report: dict[str, Any]) -> None:
    t = report.get("topic", {})
    dims = [("Concept", "concept_score"), ("Calculation", "calculation_score"),
            ("Reasoning", "reasoning_score"), ("Comparison", "comparison_score"),
            ("Application", "application_score")]
    cols = st.columns(len(dims))
    for col, (label, field) in zip(cols, dims):
        value = float(t.get(field, 0.0))
        with col:
            st.markdown(f"<div class='ea-muted'>{label}</div>", unsafe_allow_html=True)
            st.markdown(
                score_bar(value, "#2da44e" if value >= 0.7 else
                          "#d4a72c" if value >= 0.45 else
                          "#cf222e" if value > 0 else "#484f58"),
                unsafe_allow_html=True,
            )
            st.markdown(f"<div class='ea-muted'>{value:.0%}</div>" if value > 0
                        else "<div class='ea-muted'>untested</div>", unsafe_allow_html=True)


# ---- step 1-5: lesson
def _step_lesson(state: dict[str, Any], topic_id: str, name: str) -> None:
    if "lesson" not in state:
        with st.spinner("Preparing the lesson…"):
            state["lesson"] = tutor.build_lesson(topic_id, use_llm=_use_llm())
        progress.mark_taught(topic_id)

    lesson = state["lesson"]
    st.markdown(lesson.as_markdown())
    if lesson.citations:
        render_citations(lesson.citations)
    elif not lesson.grounded:
        st.caption("⚠️ Not grounded in uploaded course material — upload your lecture "
                   "slides on the Materials page for course-specific explanations.")

    st.divider()
    st.markdown("**Now close the notes.** The next step is retrieval, not reading.")
    c1, c2 = st.columns([1, 1])
    if c1.button("I have read it — test me", type="primary", use_container_width=True):
        state["step"] = "recall"
        st.rerun()
    if c2.button("Skip to practice questions", use_container_width=True):
        state["step"] = "practice"
        st.rerun()


# ---- step 6-7: explain it back
def _step_recall(state: dict[str, Any], topic_id: str, name: str) -> None:
    st.markdown("#### Explain it in your own words")
    st.caption("Answer first. I will evaluate it afterwards — do not scroll back to the lesson.")

    if "recall_q" not in state:
        state["recall_q"] = generate_question(
            topic_id, QuestionType.CONCEPTUAL, 4, use_llm=_use_llm()
        )
    q: Question = state["recall_q"]
    st.markdown(f"> {q.prompt}")

    if "recall_eval" not in state:
        answer = st.text_area("Your explanation", height=180, key="recall_answer",
                              placeholder="Mechanism first, then the consequence.")
        if st.button("Submit", type="primary", disabled=not answer.strip()):
            with st.spinner("Marking…"):
                ev = evaluate(q, answer, use_llm=_use_llm())
            progress.record_attempt(q, ev, student_answer=answer, context="study", seconds=90)
            state["recall_eval"] = ev
            state["recall_answer_text"] = answer
            state.setdefault("answered", []).append(ev.score)
            st.rerun()
    else:
        render_evaluation(state["recall_eval"], q)
        st.divider()
        if st.button("Continue to practice questions", type="primary"):
            state["step"] = "practice"
            st.rerun()


# ---- step 8-10: adaptive practice
def _step_practice(state: dict[str, Any], topic_id: str, name: str) -> None:
    answered = state.get("answered", [])
    target = state.get("n_target", 7)
    idx = state.get("queue_index", 0)

    st.markdown(f"#### Practice · question {idx + 1} of ~{target}")
    st.progress(min(1.0, idx / max(1, target)))

    if idx >= target:
        state["step"] = "done"
        st.rerun()
        return

    if "current_q" not in state:
        with st.spinner("Building the next question…"):
            state["current_q"] = _next_question(state, topic_id)
    q: Question = state["current_q"]

    key = f"study_q_{idx}"
    if "current_eval" not in state:
        answer = render_question(q, key)
        submitted = st.button("Submit answer", type="primary")
        c1, c2 = st.columns([1, 1])
        if c2.button("Skip this question", use_container_width=True):
            state.pop("current_q", None)
            state["queue_index"] = idx + 1
            st.rerun()
        if submitted:
            filled = (any(str(v).strip() for v in answer.values())
                      if isinstance(answer, dict) else bool(str(answer).strip()))
            if not filled:
                st.error("Write an answer first — the evaluator marks what you produce, "
                         "and a blank answer scores zero in the exam too.")
            else:
                with st.spinner("Marking…"):
                    ev = evaluate(q, answer, use_llm=_use_llm())
                text = (answer if isinstance(answer, str)
                        else "; ".join(f"{k}={v}" for k, v in answer.items()))
                progress.record_attempt(q, ev, student_answer=text, context="study",
                                        seconds=int(q.estimated_time * 0.8))
                state["current_eval"] = ev
                answered.append(ev.score)
                state["answered"] = answered
                st.rerun()
    else:
        ev = state["current_eval"]
        render_question(q, key, disabled=True)
        st.divider()
        render_evaluation(ev, q)
        st.divider()
        nxt_label = ("Harder question" if ev.score >= 8
                     else "Similar question" if ev.score >= 5
                     else "Easier question on the same idea")
        c1, c2 = st.columns([1, 1])
        if c1.button(f"Next → {nxt_label}", type="primary", use_container_width=True):
            state["last_q"] = q
            state["last_score"] = ev.score
            state.pop("current_q", None)
            state.pop("current_eval", None)
            state["queue_index"] = idx + 1
            st.rerun()
        if c2.button("End session", use_container_width=True):
            _finish(state)


def _next_question(state: dict[str, Any], topic_id: str) -> Question:
    last: Question | None = state.get("last_q")
    if last is not None:
        return tutor.followup_question(topic_id, last, state.get("last_score", 5),
                                       use_llm=_use_llm())
    # first practice question targets the weakest dimension
    report = progress.topic_report(topic_id)
    dim = report.get("weak_dimension", "concept")
    from ..services.question_gen import DIMENSION_TYPES

    qtype = DIMENSION_TYPES.get(dim, [QuestionType.CONCEPTUAL])[0]
    difficulty = 4 if report.get("topic", {}).get("overall", 0) < 0.6 else 5
    return generate_question(topic_id, qtype, difficulty, use_llm=_use_llm())


# ---- summary
def _step_done(state: dict[str, Any], topic_id: str, name: str) -> None:
    _finish(state, show=True)


def _finish(state: dict[str, Any], show: bool = True) -> None:
    scores = state.get("answered", [])
    elapsed = int(time.time() - state.get("started", time.time()))
    if state.get("session_id"):
        progress.end_session(
            state["session_id"], elapsed, len(scores),
            round(sum(scores) / len(scores), 2) if scores else 0.0,
        )
        state["session_id"] = None

    topic_id = state.get("topic_id", "")
    report = progress.topic_report(topic_id) if topic_id else {}

    st.markdown("### Session complete")
    c1, c2, c3 = st.columns(3)
    c1.metric("Questions answered", len(scores))
    c2.metric("Mean score", f"{sum(scores)/len(scores):.1f}/10" if scores else "—")
    c3.metric("Time", f"{elapsed // 60}m")

    if report:
        st.markdown(f"**{report['name']}** is now "
                    f"**{report['mastery']}** ({report['topic']['overall']:.0%}).")
        st.info(report["recommended_action"])
        _dimension_strip(report)

    st.divider()
    nxt = planner.next_topic()
    if nxt:
        st.markdown(f"**Next highest-value topic:** {nxt['topic']}")
        st.caption(nxt["reason"])
    c1, c2, c3 = st.columns(3)
    if c1.button("Study the next topic", type="primary", use_container_width=True):
        _reset()
        if nxt:
            st.session_state[STATE] = {"topic_id": nxt["topic_id"],
                                       "mode": nxt.get("mode", "Quick Study")}
        st.rerun()
    if c2.button("Back to dashboard", use_container_width=True):
        _reset()
        st.session_state["nav_target"] = "Dashboard"
        st.rerun()
    if c3.button("New session", use_container_width=True):
        _reset()
        st.rerun()


def _use_llm() -> bool:
    return bool(st.session_state.get("use_llm", True))
