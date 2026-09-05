"""Dashboard: what to study now, why, and how ready you are."""
from __future__ import annotations

import streamlit as st

from ..services import mock_exam, planner, progress
from .common import (
    chip,
    empty_state,
    go_to,
    mastery_color,
    priority_color,
    readiness_block,
    score_bar,
)


def render() -> None:
    snap = progress.dashboard_snapshot()
    readiness = snap["readiness"]
    days = snap["days_remaining"]

    # ---------------------------------------------------------- header
    urgency = "#cf222e" if days <= 2 else "#e8804a" if days <= 4 else "#539bf5"
    st.markdown(
        f"<h2 style='margin-bottom:0'>EXAM IN <span style='color:{urgency}'>{days} "
        f"DAY{'S' if days != 1 else ''}</span></h2>"
        f"<div class='ea-muted'>Day {snap['day_number']} of study · exam on {snap['exam_date']}</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    top_left, top_right = st.columns([1, 2])

    with top_left:
        readiness_block(readiness, compact=True)
        st.write("")
        c1, c2 = st.columns(2)
        c1.metric("ML", f"{readiness.ml_score:.0%}")
        c2.metric("DL", f"{readiness.dl_score:.0%}")

    with top_right:
        nxt = planner.next_topic()
        if nxt:
            st.markdown("#### What to study now")
            st.markdown(
                f"<div class='ea-card'>"
                f"<div style='font-size:1.25rem;font-weight:700'>{nxt['topic']}</div>"
                f"<div style='margin:6px 0'>{chip(nxt['priority'], priority_color(nxt['priority']))}"
                f"{chip(nxt['focus'] + ' focus', '#539bf5')}</div>"
                f"<div class='ea-muted'>{nxt['reason']}</div></div>",
                unsafe_allow_html=True,
            )
            b1, b2, b3 = st.columns(3)
            if b1.button("Study this topic", type="primary", use_container_width=True):
                go_to("Study", topic_id=nxt["topic_id"])
            if b2.button("Quiz me on it", use_container_width=True):
                go_to("Quiz", topic_id=nxt["topic_id"])
            if b3.button("Repair weaknesses", use_container_width=True):
                go_to("Study", mode="Weakness Repair")
        else:
            empty_state("No topics loaded yet.")

    st.divider()

    # ---------------------------------------------------------- today
    left, right = st.columns([1.35, 1])

    with left:
        st.markdown("#### Today's plan")
        day = planner.today_plan(minutes_per_day=st.session_state.get("minutes_per_day", 240))
        st.caption(f"{day.theme} · {day.total_minutes} min planned")
        if not day.blocks:
            empty_state("Nothing scheduled - everything at this priority is already strong.")
        for i, block in enumerate(day.blocks[:7], 1):
            bc1, bc2 = st.columns([4, 1])
            with bc1:
                st.markdown(
                    f"**{i}. {block.topic}** "
                    + chip(block.focus, "#539bf5")
                    + chip(block.priority.value, priority_color(block.priority.value)),
                    unsafe_allow_html=True,
                )
                st.markdown(f"<div class='ea-muted'>{block.reason}</div>",
                            unsafe_allow_html=True)
            bc2.markdown(f"<div style='text-align:right;font-weight:600'>{block.minutes} min</div>",
                         unsafe_allow_html=True)
        if day.mock_exam:
            st.info("A **mock exam** is scheduled for today.")

    with right:
        st.markdown("#### Critical weaknesses")
        gaps = snap["critical_gaps"]
        if not gaps:
            empty_state("No critical weaknesses recorded yet — answer some questions "
                        "so the system can find them.")
        for g in gaps:
            st.markdown(
                f"<div style='margin-bottom:9px'><b>{g['name']}</b> "
                f"<span class='ea-muted'>· weakest: {g['weak_dimension']}</span>"
                f"{score_bar(g['score'], mastery_color(g['score']))}</div>",
                unsafe_allow_html=True,
            )

        st.markdown("#### Today")
        m1, m2 = st.columns(2)
        m1.metric("Questions", snap["questions_today"])
        m2.metric("Mean score", f"{snap['mean_score_today']:.1f}/10"
                  if snap["questions_today"] else "—")
        m3, m4 = st.columns(2)
        m3.metric("Open mistakes", snap["open_mistakes"])
        m4.metric("Topics due", snap["due_count"])

    st.divider()

    # ---------------------------------------------------------- session modes
    st.markdown("#### Start a session")
    modes = [
        ("Quick Study", "15 min · highest-priority weakness only"),
        ("30 Minute Study", "30 min · mixed practice"),
        ("Rapid Revision", "20 min · fast active recall"),
        ("Weakness Repair", "45 min · previously failed concepts"),
    ]
    cols = st.columns(len(modes))
    for col, (mode, desc) in zip(cols, modes):
        with col:
            if st.button(mode, use_container_width=True, key=f"mode_{mode}"):
                go_to("Study", mode=mode)
            st.markdown(f"<div class='ea-muted'>{desc}</div>", unsafe_allow_html=True)

    e1, e2 = st.columns(2)
    with e1:
        if st.button("Take a full mock exam", type="primary", use_container_width=True):
            go_to("Mock Exam")
    with e2:
        if st.button("Review my mistakes", use_container_width=True):
            go_to("Weaknesses")

    # ---------------------------------------------------------- last exam
    latest = mock_exam.latest_report()
    if latest:
        st.divider()
        st.markdown("#### Last mock exam")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Score", f"{latest.percentage:.0f}%")
        c2.metric("ML", f"{latest.ml_score:.0f}%")
        c3.metric("DL", f"{latest.dl_score:.0f}%")
        worst = (min(latest.by_dimension, key=lambda k: latest.by_dimension[k])
                 if latest.by_dimension else None)
        c4.metric("Weakest dimension",
                  f"{worst} {latest.by_dimension[worst]:.0f}%" if worst else "—")
        for line in latest.revision_plan[:3]:
            st.markdown(f"- {line}")

    # ---------------------------------------------------------- coverage warning
    if snap["untouched_critical"]:
        st.divider()
        st.warning(
            f"**{snap['untouched_critical']} CRITICAL topics have never been tested.** "
            f"Untested topics are invisible risk: the readiness score cannot account for "
            f"what you have not attempted. Run a mock exam or a broad quiz to map them."
        )
