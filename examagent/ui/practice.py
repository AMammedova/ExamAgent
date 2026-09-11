"""Practice paper: every topic, the real paper's formats, marked as you go.

Deliberately the plainest page in the app. No clock, no configuration, no
planner: generate a sweep over the whole syllabus, answer a question, see
immediately whether it was right, and at the end get the paper's total and a
button to take a fresh one.

Mock Exam is the opposite of this by design - exam conditions, no feedback
until you submit. This is for grinding through questions.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from ..models.schemas import Question, points_for
from ..services import mock_exam
from ..services.evaluator import evaluate
from .common import TYPE_LABEL, chip, llm_badge, render_question, score_bar

STATE = "practice"


def _state() -> dict[str, Any]:
    return st.session_state.setdefault(STATE, {})


def render() -> None:
    st.session_state.pop("nav_payload", None)
    state = _state()

    st.markdown("### Practice paper")
    st.caption("Every topic, in the real paper's formats. Marked as you go - answer, "
               "see if it was right, keep moving. No clock.")

    if state.get("report"):
        _render_report(state)
    elif state.get("exam"):
        _render_paper(state)
    else:
        _render_intro(state)


def _generate(state: dict[str, Any]) -> None:
    bar = st.progress(0.0, text="Building the paper…")

    def tick(done: int, total: int) -> None:
        bar.progress(done / total, text=f"Building the paper… {done}/{total}")

    exam = mock_exam.build_topic_sweep(
        use_llm=bool(st.session_state.get("use_llm", True)),
        label="Practice paper — all topics",
        on_progress=tick,
    )
    bar.empty()
    mock_exam.clear_progress()
    st.session_state[STATE] = {"exam": exam, "answers": {}, "marks": {}}
    st.rerun()


def _resume(saved: dict[str, Any]) -> None:
    """Pick a saved paper back up. Only the answers were stored; the marks are
    recomputed, which is exact for every format on this page."""
    exam, answers = saved["exam"], saved["answers"]
    by_id = {q.id: q for q in exam["questions"]}
    marks = {qid: evaluate(by_id[qid], answer, use_llm=False)
             for qid, answer in answers.items() if qid in by_id}
    st.session_state[STATE] = {"exam": exam, "answers": answers, "marks": marks}
    st.rerun()


def _render_intro(state: dict[str, Any]) -> None:
    llm_badge()

    saved = mock_exam.load_progress()
    if saved:
        answered = len(saved["answers"])
        total = len(saved["exam"]["questions"])
        with st.container(border=True):
            st.markdown(f"↩️ **You have a paper in progress** — {answered}/{total} "
                        "answered.")
            c1, c2 = st.columns(2)
            if c1.button("Resume it", type="primary", use_container_width=True):
                _resume(saved)
            if c2.button("Discard and start fresh", use_container_width=True):
                mock_exam.clear_progress()
                st.rerun()
        return

    with st.container(border=True):
        st.markdown("**One question on every topic in the syllabus**, in the three "
                    "formats the paper uses:")
        st.markdown("- True / False — 1 mark\n"
                    "- Multiple choice, single best answer — 3 marks\n"
                    "- Multiple response, all-or-nothing — 4 marks")
        st.caption("Building it takes a minute or so. You can answer in any order, and "
                   "each answer is marked the moment you submit it. Your answers are "
                   "saved as you go, so closing the app does not lose the paper.")
        if st.button("Build my practice paper", type="primary"):
            _generate(state)


def _render_paper(state: dict[str, Any]) -> None:
    questions: list[Question] = state["exam"]["questions"]
    marks: dict[str, Any] = state["marks"]

    right = sum(1 for ev in marks.values() if ev.correct)
    earned = sum(points_for(q.question_type) for q in questions
                 if marks.get(q.id) is not None and marks[q.id].correct)
    available = sum(points_for(q.question_type) for q in questions)

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.markdown(f"**{len(marks)}/{len(questions)} answered** · {right} right · "
                    f"{earned}/{available} marks so far")
        st.markdown(score_bar(len(marks) / max(1, len(questions)), "#539bf5"),
                    unsafe_allow_html=True)
    if c2.button("Submit paper", type="primary", use_container_width=True,
                 disabled=not marks):
        _submit(state)
        return
    if c3.button("Start over", use_container_width=True):
        mock_exam.clear_progress()
        st.session_state[STATE] = {}
        st.rerun()

    for i, q in enumerate(questions, 1):
        st.divider()
        st.markdown(f"**Q{i}.** " + chip(TYPE_LABEL.get(q.question_type, ""), "#539bf5")
                    + chip(f"{points_for(q.question_type)} pt", "#8b949e")
                    + chip(q.topic.replace("_", " ").title(), "#6e7681"),
                    unsafe_allow_html=True)

        already = marks.get(q.id)
        answer = render_question(q, f"practice_{q.id}", disabled=already is not None,
                                 show_meta=False)
        if already is None:
            if st.button("Submit answer", key=f"check_{q.id}"):
                if not str(answer).strip():
                    st.warning("Pick an answer first.")
                else:
                    state["answers"][q.id] = answer
                    # every format here marks exactly, so this needs no LLM and
                    # comes back instantly
                    marks[q.id] = evaluate(q, answer, use_llm=False)
                    mock_exam.save_progress(state["exam"]["exam_id"], state["answers"])
                    st.rerun()
        else:
            _verdict(q, already)


def _verdict(question: Question, evaluation) -> None:
    if evaluation.correct:
        st.success(f"✅ Correct — {points_for(question.question_type)} marks.")
        return
    correct_text = next((o.text for o in question.options
                         if o.key == question.correct_option), "")
    st.error(f"❌ Wrong — correct answer: **{question.correct_option}**"
             + (f" ({correct_text})" if correct_text
                and correct_text != question.correct_option else ""))
    if question.model_answer:
        st.caption(question.model_answer)


def _submit(state: dict[str, Any]) -> None:
    questions: list[Question] = state["exam"]["questions"]
    answers = {q.id: state["answers"].get(q.id, "") for q in questions}
    state["report"] = mock_exam.submit_exam(
        state["exam"]["exam_id"], answers, duration_seconds=0, use_llm=False,
    )
    mock_exam.clear_progress()
    st.rerun()


def _render_report(state: dict[str, Any]) -> None:
    report = state["report"]
    questions: list[Question] = state["exam"]["questions"]
    marks: dict[str, Any] = state["marks"]

    colour = ("#2da44e" if report.percentage >= 70
              else "#d4a72c" if report.percentage >= 50 else "#cf222e")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"<div style='font-size:2.6rem;font-weight:700;color:{colour}'>"
            f"{report.percentage:.0f}%</div><div class='ea-muted'>overall</div>",
            unsafe_allow_html=True,
        )
    c2.metric("Marks", f"{report.total_score:.0f}/{report.max_score:.0f}")
    answered = sum(1 for q in questions if marks.get(q.id) is not None)
    c3.metric("Answered", f"{answered}/{len(questions)}")

    if report.pass_mark:
        if report.passed:
            st.success(f"Above the 60% line ({report.pass_mark:.0f} marks).")
        else:
            st.error(f"{report.pass_mark - report.total_score:.0f} marks short of the "
                     f"60% line ({report.pass_mark:.0f}).")

    wrong = [q for q in questions
             if marks.get(q.id) is not None and not marks[q.id].correct]
    if wrong:
        st.markdown(f"#### The {len(wrong)} you got wrong")
        for q in wrong:
            with st.container(border=True):
                st.markdown(f"**{q.topic.replace('_', ' ').title()}** — {q.prompt}")
                for n, statement in enumerate(q.statements, 1):
                    st.markdown(f"{n}. {statement}")
                st.markdown(f"Correct answer: **{q.correct_option}**")
                if q.model_answer:
                    st.caption(q.model_answer)

    unanswered = len(questions) - answered
    if unanswered:
        st.caption(f"{unanswered} question(s) left blank — they scored zero.")

    st.divider()
    c1, c2 = st.columns(2)
    if c1.button("Take a fresh paper", type="primary", use_container_width=True):
        mock_exam.clear_progress()
        st.session_state[STATE] = {}
        _generate(_state())
    if c2.button("Back to my answers", use_container_width=True):
        state.pop("report", None)
        st.rerun()
