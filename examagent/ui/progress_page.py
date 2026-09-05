"""Progress page: readiness breakdown, history, plan and dimension profile."""
from __future__ import annotations

import streamlit as st

from ..services import mock_exam, planner, progress, weakness
from .common import (
    chip,
    empty_state,
    mastery_color,
    priority_color,
    readiness_block,
    score_bar,
)


def render() -> None:
    st.session_state.pop("nav_payload", None)
    st.markdown("### Progress")

    readiness = progress.compute_readiness()
    left, right = st.columns([1, 1.5])

    with left:
        readiness_block(readiness)
    with right:
        st.markdown("#### Dimension profile")
        dims = progress.dimension_profile()
        for name, value in dims.items():
            c1, c2, c3 = st.columns([1.2, 3, 0.7])
            c1.markdown(f"<span style='font-size:0.88rem'>{name.title()}</span>",
                        unsafe_allow_html=True)
            c2.markdown(score_bar(value, mastery_color(value, value > 0)),
                        unsafe_allow_html=True)
            c3.markdown(f"<span class='ea-muted'>{value:.0%}</span>" if value > 0
                        else "<span class='ea-muted'>—</span>", unsafe_allow_html=True)
        st.caption("Calculation and reasoning carry the most weight in the readiness score "
                   "because that is what this exam tests.")

        cats = weakness.category_breakdown()
        st.markdown("#### Coverage")
        for cat, data in cats.items():
            st.markdown(
                f"**{cat}** · {data['tested']}/{data['topics']} topics tested "
                f"· mean {data['mean_score']:.0%} "
                + (chip(f"{data['critical_weak']} critical weak", "#cf222e")
                   if data["critical_weak"] else ""),
                unsafe_allow_html=True,
            )
            st.markdown(score_bar(data["coverage"], "#539bf5"), unsafe_allow_html=True)

    st.divider()
    tabs = st.tabs(["Study plan", "Activity", "Mock exams", "Mistake profile"])

    with tabs[0]:
        _plan()
    with tabs[1]:
        _activity()
    with tabs[2]:
        _exams()
    with tabs[3]:
        _mistakes()


def _plan() -> None:
    minutes = st.slider("Available study minutes per day", 60, 600,
                        st.session_state.get("minutes_per_day", 240), step=30)
    st.session_state["minutes_per_day"] = minutes

    summary = planner.plan_summary()
    st.caption(f"{summary['days_remaining']} days remaining · "
               f"{summary['total_minutes']} minutes planned in total. "
               "The plan is re-derived from your live scores every time you open it.")

    for day in summary["days"]:
        with st.expander(
            f"Day {day.day_number} · {day.date} · {day.theme} "
            f"({day.total_minutes} min)" + ("  ·  MOCK EXAM" if day.mock_exam else ""),
            expanded=day is summary["days"][0],
        ):
            if not day.blocks:
                st.caption("Nothing scheduled — everything at this priority is strong enough.")
            for b in day.blocks:
                c1, c2, c3 = st.columns([2.6, 1, 1])
                c1.markdown(f"**{b.topic}**  "
                            + chip(b.focus, "#539bf5")
                            + chip(b.priority.value, priority_color(b.priority.value)),
                            unsafe_allow_html=True)
                c2.markdown(f"<div class='ea-muted'>{b.minutes} min</div>",
                            unsafe_allow_html=True)
                c3.markdown(f"<div class='ea-muted'>{b.reason}</div>", unsafe_allow_html=True)

    if summary["top_time"]:
        st.markdown("#### Where your time goes")
        top = summary["top_time"]
        peak = max(v for _, v in top) or 1
        for topic, mins in top:
            c1, c2, c3 = st.columns([2, 3, 0.6])
            c1.markdown(f"<span style='font-size:0.86rem'>{topic}</span>",
                        unsafe_allow_html=True)
            c2.markdown(score_bar(mins / peak, "#539bf5"), unsafe_allow_html=True)
            c3.markdown(f"<span class='ea-muted'>{mins}m</span>", unsafe_allow_html=True)


def _activity() -> None:
    history = progress.progress_history(days=14)
    if not history:
        empty_state("No activity yet.")
        return
    st.markdown("#### Questions answered per day")
    st.bar_chart({h["date"]: h["questions"] for h in history})
    st.markdown("#### Mean score per day")
    st.line_chart({h["date"]: h["mean_score"] for h in history})
    st.dataframe(history, hide_index=True, use_container_width=True)


def _exams() -> None:
    history = mock_exam.exam_history(limit=10)
    completed = [h for h in history if h["completed"]]
    if not completed:
        empty_state("No completed mock exams yet.",
                    "A mock exam is the single most informative thing you can do — "
                    "it maps every topic at once.")
        return
    st.line_chart({h["started"]: h["percentage"] for h in reversed(completed)})
    for h in completed:
        with st.expander(f"{h['label']} · {h['started']} · {h['percentage']:.0f}%"):
            rep = h.get("report") or {}
            c1, c2, c3 = st.columns(3)
            c1.metric("ML", f"{rep.get('ml_score', 0):.0f}%")
            c2.metric("DL", f"{rep.get('dl_score', 0):.0f}%")
            c3.metric("Questions", h["n_questions"])
            if rep.get("by_dimension"):
                st.markdown("**By dimension:** " + " · ".join(
                    f"{k} {v:.0f}%" for k, v in rep["by_dimension"].items()))
            for line in rep.get("revision_plan", [])[:4]:
                st.markdown(f"- {line}")


def _mistakes() -> None:
    profile = progress.mistake_profile()
    if not profile:
        empty_state("No mistakes recorded.")
        return
    st.markdown("#### Mistakes by type")
    st.bar_chart(profile)
    total = sum(profile.values())
    worst = max(profile, key=lambda k: profile[k])
    advice = {
        "Arithmetic": "Your method is sound — slow down and carry more decimal places.",
        "Formula": "Write the formula before substituting numbers. Build a formula sheet.",
        "Conceptual": "The underlying mechanism is not secure. Re-learn, then self-test.",
        "Dimension": "Track tensor shapes explicitly at every layer — examiners penalise this hard.",
        "Reasoning": "You know the facts but not the causal chain. Drill assertion-reason "
                     "and 'what happens if' questions.",
        "Terminology": "Use the precise technical term; examiners mark for it.",
        "Incomplete": "You are leaving parts blank. Attempt every part — partial credit is real.",
    }.get(worst, "")
    st.info(f"**Most common: {worst}** ({profile[worst]}/{total}). {advice}")
