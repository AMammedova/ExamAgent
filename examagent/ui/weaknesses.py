"""Weaknesses page: what is weak, what is dangerous, what went wrong."""
from __future__ import annotations

import streamlit as st

from ..models.schemas import QuestionType
from ..services import progress, weakness
from ..services.evaluator import evaluate
from ..services.question_gen import DIMENSION_TYPES
from .common import (
    chip,
    empty_state,
    go_to,
    mastery_color,
    priority_color,
    render_evaluation,
    render_question,
    score_bar,
)


def render() -> None:
    st.session_state.pop("nav_payload", None)
    st.markdown("### Weaknesses")

    report = weakness.weakness_report(limit=12)
    tabs = st.tabs(["Weakest topics", "Dimension gaps", "Error log", "Retry queue",
                    "Dangerous gaps"])

    with tabs[0]:
        _weakest(report)
    with tabs[1]:
        _gaps(report)
    with tabs[2]:
        _error_log()
    with tabs[3]:
        _retry_queue()
    with tabs[4]:
        _dangerous()


def _weakest(report: dict) -> None:
    rows = report["weakest"]
    if not rows:
        empty_state("Nothing recorded yet. Answer some questions first.")
        return
    st.caption("Ranked by exam damage: (1 − score) × priority × exam relevance. "
               "Untested topics count as risk, not as zero.")
    for r in rows:
        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(
                    f"**{r['name']}** "
                    + chip(r["priority"], priority_color(r["priority"]))
                    + chip(r["mastery"], mastery_color(r["effective"], r["attempts"] > 0))
                    + chip(f"weakest: {r['weak_dimension']}", "#d4a72c"),
                    unsafe_allow_html=True,
                )
                st.markdown(score_bar(r["effective"],
                                      mastery_color(r["effective"], r["attempts"] > 0)),
                            unsafe_allow_html=True)
                st.markdown(f"<div class='ea-muted'>{r['action']}</div>",
                            unsafe_allow_html=True)
                dims = " · ".join(
                    f"{k} {v:.0%}" for k, v in r["dimensions"].items() if v > 0
                ) or "no dimension tested"
                st.markdown(f"<div class='ea-muted'>{dims} · {r['attempts']} attempts · "
                            f"{r['mistakes']} mistakes</div>", unsafe_allow_html=True)
            with c2:
                if st.button("Study", key=f"w_study_{r['id']}", use_container_width=True):
                    go_to("Study", topic_id=r["id"])
                if st.button("Drill", key=f"w_quiz_{r['id']}", use_container_width=True,
                             type="primary"):
                    go_to("Quiz", topic_id=r["id"])


def _gaps(report: dict) -> None:
    gaps = report["dimension_gaps"]
    if not gaps:
        empty_state("No dimension gaps detected yet — answer both conceptual and "
                    "calculation questions on the same topic so the system can compare them.")
        return
    st.caption("A topic where one dimension is far ahead of another. This is the single "
               "most actionable signal: it tells you what NOT to study.")
    for g in gaps:
        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"**{g['name']}**")
                st.markdown(
                    f"<span class='ea-muted'>{g['strong']} "
                    f"<b style='color:#2da44e'>{g['strong_score']:.0%}</b> vs "
                    f"{g['weak']} <b style='color:#cf222e'>{g['weak_score']:.0%}</b> "
                    f"(gap {g['gap']:.0%})</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(g["advice"])
            with c2:
                if st.button(f"Drill {g['weak']}", key=f"gap_{g['id']}",
                             type="primary", use_container_width=True):
                    types = DIMENSION_TYPES.get(g["weak"], [QuestionType.CONCEPTUAL])
                    st.session_state.setdefault("quiz", {})["qtype"] = types[0]
                    go_to("Quiz", topic_id=g["id"])


def _error_log() -> None:
    c1, c2 = st.columns([1, 3])
    with c1:
        unresolved = st.checkbox("Unresolved only", value=True)
    errors = weakness.error_log(limit=60, unresolved_only=unresolved)
    if not errors:
        empty_state("No mistakes logged. Take a mock exam to find your real gaps.")
        return

    counts = weakness.weakness_report()["mistake_types"]
    if counts:
        st.markdown(" ".join(
            chip(f"{k}: {v}", "#cf222e" if k in ("Conceptual", "Reasoning", "Formula")
                 else "#d4a72c")
            for k, v in counts.items()
        ), unsafe_allow_html=True)

    for e in errors:
        with st.expander(
            f"{'🔴' if e['severity'] == 'High' else '🟡'} {e['topic']} · "
            f"{e['mistake_type']} · {e['date']}"
            + ("" if e["retry_required"] else " · resolved"),
            expanded=False,
        ):
            st.markdown(f"**Question**\n\n{e['question'][:1200]}")
            st.markdown(f"**Your answer**\n\n{e['student_answer'][:800] or '_(blank)_'}")
            if e["correct_concept"]:
                st.markdown(f"**Correct treatment**\n\n{e['correct_concept'][:1500]}")
            c1, c2 = st.columns([1, 3])
            if c1.button("Retry this topic", key=f"retry_{e['id']}", type="primary"):
                go_to("Quiz", topic_id=e["topic_id"])
            if e["retry_required"] and c2.button("Mark as understood", key=f"res_{e['id']}"):
                progress.retry_recorded(e["id"], success=True)
                st.rerun()


def _retry_queue() -> None:
    """Spaced-repetition queue: mistakes that are due to be retried now."""
    state = st.session_state.setdefault("repair", {})
    due = weakness.due_retries(limit=15)

    if not due:
        empty_state("Nothing due for retry right now. Mistakes come back on a schedule — "
                    "critical ones within a few hours.")
        return

    st.caption(f"{len(due)} mistake(s) due. Each retry generates a NEW question on the "
               "same topic — repeating the identical question would test memory, not learning.")

    if "question" not in state:
        target = due[0]
        st.markdown(f"**Next up: {target['topic']}** "
                    + chip(target["mistake_type"], "#cf222e"), unsafe_allow_html=True)
        st.caption(f"Original mistake: {target['question'][:220]}")
        if st.button("Start retry", type="primary"):
            from ..services.question_gen import generate_question as gq

            with st.spinner("Building a fresh question…"):
                state["question"] = gq(
                    target["topic_id"], None, 4,
                    use_llm=bool(st.session_state.get("use_llm", True)),
                )
            state["mistake_id"] = target["id"]
            st.rerun()
        return

    q = state["question"]
    if "evaluation" not in state:
        answer = render_question(q, "repair_q")
        if st.button("Submit", type="primary"):
            with st.spinner("Marking…"):
                ev = evaluate(q, answer, use_llm=bool(st.session_state.get("use_llm", True)))
            text = (answer if isinstance(answer, str)
                    else "; ".join(f"{k}={v}" for k, v in answer.items()))
            progress.record_attempt(q, ev, student_answer=text, context="study", seconds=120)
            progress.retry_recorded(state["mistake_id"], success=ev.score >= 7)
            state["evaluation"] = ev
            st.rerun()
    else:
        render_question(q, "repair_q", disabled=True)
        st.divider()
        ev = state["evaluation"]
        render_evaluation(ev, q)
        if ev.score >= 7:
            st.success("Mistake cleared from the retry queue.")
        else:
            st.warning("Still not right — this stays in the queue and will come back sooner.")
        if st.button("Next retry", type="primary"):
            st.session_state["repair"] = {}
            st.rerun()


def _dangerous() -> None:
    gaps = weakness.dangerous_gaps(limit=8)
    if not gaps:
        empty_state("No prerequisite risks detected.")
        return
    st.caption("Weak topics that other topics are built on. A gap here silently damages "
               "everything downstream, which is why they are worth fixing first.")
    for g in gaps:
        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"**{g['name']}** "
                            + chip(f"blocks {g['blocks']} topics", "#cf222e")
                            + ("" if g["tested"] else chip("never tested", "#8b949e")),
                            unsafe_allow_html=True)
                st.markdown(score_bar(g["score"], mastery_color(g["score"], g["tested"])),
                            unsafe_allow_html=True)
                st.markdown(f"<div class='ea-muted'>Downstream: "
                            f"{', '.join(g['dependents'])}</div>", unsafe_allow_html=True)
            with c2:
                if st.button("Fix now", key=f"dang_{g['id']}", type="primary",
                             use_container_width=True):
                    go_to("Study", topic_id=g["id"])
