"""Mock exam page: timed, no hints, no feedback until submission."""
from __future__ import annotations

import time
from typing import Any

import streamlit as st

from ..models.schemas import MockExamReport, Question
from ..services import learning_path as lp
from ..services import mock_exam
from .common import (
    TYPE_LABEL,
    chip,
    llm_badge,
    mastery_color,
    render_question,
    score_bar,
)

STATE = "mock"


def _state() -> dict[str, Any]:
    return st.session_state.setdefault(STATE, {})


def _quick_mock_from_learning_path(state: dict[str, Any]) -> None:
    """One-click exam scoped to only the topics done in the Learning Path -
    for testing yourself on what you've actually covered, not the full
    syllabus, without configuring anything."""
    done_ids = lp.completed_topic_ids()
    if not done_ids:
        return

    n = min(18, max(6, len(done_ids) * 2))
    minutes = max(15, round(n * 1.8))
    with st.container(border=True):
        st.markdown(f"📍 **Quick Mock — Learning Path** ({len(done_ids)} topic"
                   + ("s" if len(done_ids) != 1 else "") + " done)")
        st.caption(f"Only from topics you've already learned. ~{n} questions, "
                  f"~{minutes} min.")
        if st.button("Generate quick mock", type="primary", key="quick_mock_lp"):
            with st.spinner("Building the paper…"):
                exam = mock_exam.build_exam(
                    n_questions=n, duration_minutes=minutes,
                    label="Quick Mock — Learning Path",
                    use_llm=bool(st.session_state.get("use_llm", True)),
                    balance_ml_dl=True, topic_ids=done_ids,
                )
            state["exam"] = exam
            state["answers"] = {}
            state["started_at"] = time.time()
            state["index"] = 0
            st.rerun()


def render() -> None:
    state = _state()
    st.session_state.pop("nav_payload", None)

    if state.get("report"):
        _render_report(state)
    elif state.get("exam"):
        _render_exam(state)
    else:
        _render_setup(state)


# --------------------------------------------------------------- setup
def _render_setup(state: dict[str, Any]) -> None:
    st.markdown("### Mock Exam")
    st.caption("Real conditions: a clock, no hints, no feedback until you submit.")
    llm_badge()

    _quick_mock_from_learning_path(state)

    st.divider()
    st.markdown("#### Custom exam")
    c1, c2, c3 = st.columns(3)
    with c1:
        n = st.slider("Questions", 6, 30, 18)
    with c2:
        minutes = st.slider("Time limit (minutes)", 15, 180, 75)
    with c3:
        label = st.text_input("Label", value="Mock Exam")

    balance = st.checkbox("Balance Machine Learning and Deep Learning", value=True)
    st.caption(
        "Blueprint follows the university exam samples: assertion-reason, calculation, "
        "conceptual reasoning, what-happens-if, comparison, scenario and architecture "
        "interpretation, at difficulty 4-6."
    )

    if st.button("Generate exam", type="primary"):
        with st.spinner("Building the paper… (calculation problems are generated fresh)"):
            exam = mock_exam.build_exam(
                n_questions=n, duration_minutes=minutes, label=label,
                use_llm=bool(st.session_state.get("use_llm", True)),
                balance_ml_dl=balance,
            )
        state["exam"] = exam
        state["answers"] = {}
        state["started_at"] = time.time()
        state["index"] = 0
        st.rerun()

    history = mock_exam.exam_history(limit=6)
    if history:
        st.divider()
        st.markdown("#### Previous exams")
        for h in history:
            c1, c2, c3, c4 = st.columns([2.4, 1, 1, 1])
            c1.markdown(f"**{h['label']}** · {h['started']}")
            c2.markdown(f"{h['n_questions']} questions")
            c3.markdown(f"**{h['percentage']:.0f}%**" if h["completed"] else "_incomplete_")
            if h["completed"] and c4.button("View report", key=f"rep_{h['id']}"):
                try:
                    state["report"] = MockExamReport(**h["report"])
                    state["exam"] = mock_exam.load_exam(h["id"])
                    state["answers"] = state["exam"].get("answers", {})
                    st.rerun()
                except (TypeError, ValueError):
                    st.error("That report could not be loaded.")


# --------------------------------------------------------------- exam
def _render_exam(state: dict[str, Any]) -> None:
    exam = state["exam"]
    questions: list[Question] = exam["questions"]
    limit = exam["duration_minutes"] * 60
    elapsed = int(time.time() - state.get("started_at", time.time()))
    remaining = max(0, limit - elapsed)

    top1, top2, top3 = st.columns([2, 1, 1])
    with top1:
        st.markdown(f"### {exam.get('label', 'Mock Exam')}")
        st.caption(f"{len(questions)} questions · exam conditions")
    with top2:
        colour = "#cf222e" if remaining < 300 else "#d4a72c" if remaining < 900 else "#2da44e"
        st.markdown(
            f"<div style='text-align:right;font-size:1.9rem;font-weight:700;color:{colour}'>"
            f"{remaining // 60}:{remaining % 60:02d}</div>"
            f"<div class='ea-muted' style='text-align:right'>remaining</div>",
            unsafe_allow_html=True,
        )
    with top3:
        answered = sum(1 for q in questions if _has_answer(state["answers"].get(q.id)))
        st.metric("Answered", f"{answered}/{len(questions)}")

    if remaining <= 0:
        st.error("Time is up. Submitting your paper.")
        _submit(state)
        return

    st.progress(answered / max(1, len(questions)))

    view = st.radio("View", ["One question at a time", "All questions"],
                    horizontal=True, label_visibility="collapsed")

    if view == "One question at a time":
        idx = state.get("index", 0)
        idx = max(0, min(len(questions) - 1, idx))
        q = questions[idx]
        st.divider()
        st.markdown(f"**Question {idx + 1} of {len(questions)}** "
                    + chip(TYPE_LABEL.get(q.question_type, ""), "#539bf5")
                    + chip(q.category.value.split()[0], "#8b949e"),
                    unsafe_allow_html=True)
        answer = render_question(q, f"mock_{q.id}", show_meta=False)
        state["answers"][q.id] = answer

        n1, n2, n3 = st.columns([1, 1, 2])
        if n1.button("← Previous", disabled=idx == 0, use_container_width=True):
            state["index"] = idx - 1
            st.rerun()
        if n2.button("Next →", disabled=idx >= len(questions) - 1, use_container_width=True):
            state["index"] = idx + 1
            st.rerun()
        with n3:
            jump = st.selectbox(
                "Jump to", [f"{i+1}. {'✓' if _has_answer(state['answers'].get(qq.id)) else '○'} "
                            f"{TYPE_LABEL.get(qq.question_type, '')}"
                            for i, qq in enumerate(questions)],
                index=idx, label_visibility="collapsed",
            )
            new_idx = int(jump.split(".")[0]) - 1
            if new_idx != idx:
                state["index"] = new_idx
                st.rerun()
    else:
        for i, q in enumerate(questions, 1):
            st.divider()
            st.markdown(f"**Question {i}** "
                        + chip(TYPE_LABEL.get(q.question_type, ""), "#539bf5")
                        + chip(q.category.value.split()[0], "#8b949e"),
                        unsafe_allow_html=True)
            state["answers"][q.id] = render_question(q, f"mockall_{q.id}", show_meta=False)

    st.divider()
    c1, c2 = st.columns([1, 3])
    if c1.button("Submit paper", type="primary", use_container_width=True):
        _submit(state)
        return
    unanswered = len(questions) - answered
    if unanswered:
        c2.caption(f"{unanswered} question(s) unanswered — blank answers score zero, "
                   "and partially correct methods still earn marks.")
    if st.button("Abandon exam"):
        st.session_state[STATE] = {}
        st.rerun()


def _has_answer(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        return any(str(v).strip() for v in value.values())
    return bool(str(value).strip())


def _submit(state: dict[str, Any]) -> None:
    exam = state["exam"]
    elapsed = int(time.time() - state.get("started_at", time.time()))
    with st.spinner("Marking the paper…"):
        report = mock_exam.submit_exam(
            exam["exam_id"], state["answers"], duration_seconds=elapsed,
            use_llm=bool(st.session_state.get("use_llm", True)),
        )
    state["report"] = report
    st.rerun()


# --------------------------------------------------------------- report
def _render_report(state: dict[str, Any]) -> None:
    report: MockExamReport = state["report"]
    st.markdown("### Exam report")

    colour = ("#2da44e" if report.percentage >= 70
              else "#d4a72c" if report.percentage >= 50 else "#cf222e")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"<div style='font-size:2.6rem;font-weight:700;color:{colour}'>"
            f"{report.percentage:.0f}%</div><div class='ea-muted'>overall</div>",
            unsafe_allow_html=True,
        )
    c2.metric("Raw score", f"{report.total_score:.1f}/{report.max_score:.0f}")
    c3.metric("Machine Learning", f"{report.ml_score:.0f}%")
    c4.metric("Deep Learning", f"{report.dl_score:.0f}%")

    if report.duration_seconds:
        st.caption(f"Completed in {report.duration_seconds // 60} minutes")

    st.divider()
    left, right = st.columns(2)
    with left:
        st.markdown("#### By dimension")
        if not report.by_dimension:
            st.caption("—")
        for dim, val in sorted(report.by_dimension.items(), key=lambda kv: kv[1]):
            st.markdown(f"<div style='margin-bottom:8px'>{dim.title()} "
                        f"<span class='ea-muted'>{val:.0f}%</span>"
                        f"{score_bar(val / 100, mastery_color(val / 100))}</div>",
                        unsafe_allow_html=True)
    with right:
        st.markdown("#### By question type")
        for qtype, val in sorted(report.by_question_type.items(), key=lambda kv: kv[1]):
            st.markdown(f"<div style='margin-bottom:8px'>{qtype.replace('_', ' ').title()} "
                        f"<span class='ea-muted'>{val:.0f}%</span>"
                        f"{score_bar(val / 100, mastery_color(val / 100))}</div>",
                        unsafe_allow_html=True)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Top 5 weaknesses")
        if report.top_weaknesses:
            for w in report.top_weaknesses:
                st.markdown(f"- {w}")
        else:
            st.caption("No topic scored below 7/10.")
    with c2:
        st.markdown("#### Top 5 strengths")
        if report.top_strengths:
            for s in report.top_strengths:
                st.markdown(f"- {s}")
        else:
            st.caption("No topic scored 7/10 or above yet.")

    if report.dangerous_gaps:
        st.error("**Most dangerous knowledge gaps** — critical, high-relevance topics you failed:\n\n"
                 + "\n".join(f"- {g}" for g in report.dangerous_gaps))

    if report.revision_plan:
        st.markdown("#### Revision plan")
        for line in report.revision_plan:
            st.markdown(f"- {line}")

    st.divider()
    with st.expander("Question-by-question review", expanded=False):
        _render_review(state)

    c1, c2, c3 = st.columns(3)
    if c1.button("New mock exam", type="primary", use_container_width=True):
        st.session_state[STATE] = {}
        st.rerun()
    if c2.button("Repair these weaknesses", use_container_width=True):
        st.session_state[STATE] = {}
        st.session_state["nav_target"] = "Study"
        st.session_state["nav_payload"] = {"mode": "Weakness Repair"}
        st.rerun()
    if c3.button("Back to dashboard", use_container_width=True):
        st.session_state[STATE] = {}
        st.session_state["nav_target"] = "Dashboard"
        st.rerun()


def _render_review(state: dict[str, Any]) -> None:
    """Show each question with the student's answer and the correct treatment."""
    from ..services.evaluator import evaluate

    exam = state.get("exam") or {}
    questions: list[Question] = exam.get("questions", [])
    answers = state.get("answers", {})
    if not questions:
        st.caption("No question data available for this exam.")
        return
    for i, q in enumerate(questions, 1):
        st.markdown(f"**Q{i}. {TYPE_LABEL.get(q.question_type, '')}** · "
                    f"{q.topic.replace('_', ' ').title()}")
        st.markdown(q.prompt)
        ans = answers.get(q.id)
        st.markdown(f"*Your answer:* {ans if ans else '_(blank)_'}")
        ev = evaluate(q, ans if ans is not None else "", use_llm=False)
        st.markdown(f"*Score:* {ev.score:.1f}/10")
        if q.model_answer or ev.model_answer:
            with st.expander("Model answer", expanded=False):
                st.markdown(ev.model_answer or q.model_answer)
        st.divider()
