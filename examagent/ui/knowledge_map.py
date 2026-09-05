"""Knowledge map: visual topic grid coloured by mastery, drill-down per topic."""
from __future__ import annotations

import streamlit as st

from ..services import progress, weakness
from .common import chip, go_to, mastery_color, priority_color, score_bar

LEGEND = [
    ("Mastered", "#1a7f37"), ("Strong", "#2da44e"), ("Medium", "#d4a72c"),
    ("Weak", "#e8804a"), ("Critical weakness", "#cf222e"), ("Not tested", "#8b949e"),
]


def render() -> None:
    payload = st.session_state.pop("nav_payload", None) or {}
    if payload.get("topic_id"):
        st.session_state["km_selected"] = payload["topic_id"]

    st.markdown("### Knowledge Map")
    st.markdown(" ".join(chip(name, colour) for name, colour in LEGEND),
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        show = st.selectbox("Show", ["All topics", "CRITICAL only", "CRITICAL + HIGH",
                                     "Weak or untested only"])
    with c2:
        category_filter = st.selectbox("Category", ["Both", "Machine Learning", "Deep Learning"])
    with c3:
        sort_by = st.selectbox("Sort by", ["Subtopic", "Weakest first", "Exam relevance"])

    data = weakness.knowledge_map()
    selected = st.session_state.get("km_selected")

    for category, rows in data.items():
        if category_filter != "Both" and category != category_filter:
            continue
        rows = _filter(rows, show)
        if not rows:
            continue
        rows = _sort(rows, sort_by)

        st.markdown(f"## {category}")
        tested = [r for r in rows if r["attempts"] > 0]
        st.caption(f"{len(rows)} topics shown · {len(tested)} tested · "
                   f"mean {sum(r['score'] for r in tested)/len(tested):.0%}"
                   if tested else f"{len(rows)} topics shown · none tested yet")

        if sort_by == "Subtopic":
            groups: dict[str, list] = {}
            for r in rows:
                groups.setdefault(r["subtopic"] or "Other", []).append(r)
            for group, items in groups.items():
                st.markdown(f"**{group}**")
                _grid(items)
        else:
            _grid(rows)
        st.write("")

    if selected:
        st.divider()
        _detail(selected)


def _filter(rows: list[dict], show: str) -> list[dict]:
    if show == "CRITICAL only":
        return [r for r in rows if r["priority"] == "CRITICAL"]
    if show == "CRITICAL + HIGH":
        return [r for r in rows if r["priority"] in ("CRITICAL", "HIGH")]
    if show == "Weak or untested only":
        return [r for r in rows if r["attempts"] == 0 or r["score"] < 0.55]
    return rows


def _sort(rows: list[dict], sort_by: str) -> list[dict]:
    if sort_by == "Weakest first":
        return sorted(rows, key=lambda r: (r["score"] if r["attempts"] else -1))
    if sort_by == "Exam relevance":
        return sorted(rows, key=lambda r: -r["exam_relevance"])
    return sorted(rows, key=lambda r: (r["subtopic"], r["name"]))


def _grid(rows: list[dict], per_row: int = 3) -> None:
    for i in range(0, len(rows), per_row):
        cols = st.columns(per_row)
        for col, r in zip(cols, rows[i:i + per_row]):
            with col:
                colour = r["color"]
                pct = f" · {r['score']:.0%}" if r["attempts"] else ""
                caption = f"{r['mastery']}{pct} · {r['priority'].title()}"
                bar = score_bar(r["score"], colour, height=6)
                st.markdown(
                    "<div class='ea-card' style='border-left:4px solid " + colour + ";"
                    "padding:9px 11px;margin-bottom:7px'>"
                    "<div style='font-weight:600;font-size:0.92rem'>" + r["name"] + "</div>"
                    "<div style='margin:4px 0'>" + bar + "</div>"
                    "<div class='ea-muted' style='font-size:0.74rem'>" + caption + "</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )
                if st.button("Details", key=f"km_{r['id']}", use_container_width=True):
                    st.session_state["km_selected"] = r["id"]
                    st.rerun()


def _detail(topic_id: str) -> None:
    report = progress.topic_report(topic_id)
    if not report:
        return
    t = report["topic"]
    st.markdown(f"## {report['name']}")
    st.markdown(
        chip(t["priority"], priority_color(t["priority"]))
        + chip(report["mastery"], mastery_color(report["effective"], t["attempt_count"] > 0))
        + chip(f"exam relevance {t['exam_relevance']:.0%}", "#539bf5")
        + chip(f"weakest: {report['weak_dimension']}", "#d4a72c"),
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1.4, 1])
    with c1:
        st.markdown("#### Dimension scores")
        for dim in ("concept", "calculation", "reasoning", "comparison", "application"):
            value = t.get(f"{dim}_score", 0.0)
            a, b, c = st.columns([1.2, 3, 0.7])
            a.markdown(f"<span style='font-size:0.86rem'>{dim.title()}</span>",
                       unsafe_allow_html=True)
            b.markdown(score_bar(value, mastery_color(value, value > 0)),
                       unsafe_allow_html=True)
            c.markdown(f"<span class='ea-muted'>{value:.0%}</span>" if value > 0
                       else "<span class='ea-muted'>untested</span>", unsafe_allow_html=True)

        st.markdown("#### Recommended next action")
        st.info(report["recommended_action"])

        if report["prerequisites"]:
            st.markdown("#### Prerequisites")
            for p in report["prerequisites"]:
                st.markdown(f"- **{p['name']}** — {p['score']:.0%}"
                            + ("  ⚠️ weak prerequisite" if p["score"] < 0.45 else ""))
        if report["dependents"]:
            st.markdown(f"#### Topics that depend on this\n{', '.join(report['dependents'])}")

    with c2:
        st.metric("Attempts", t["attempt_count"])
        st.metric("Mistakes", t["mistake_count"])
        st.metric("Confidence", f"{t['confidence']:.0%}")
        if st.button("Study this", type="primary", use_container_width=True):
            go_to("Study", topic_id=topic_id)
        if st.button("Quiz me", use_container_width=True):
            go_to("Quiz", topic_id=topic_id)

    if report["attempts"]:
        st.markdown("#### Recent attempts")
        st.dataframe(report["attempts"], hide_index=True, use_container_width=True)

    if report["mistakes"]:
        st.markdown("#### Previous mistakes")
        for m in report["mistakes"]:
            with st.expander(f"{m['type']} · {m['severity']} · {m['date']}"):
                st.markdown(m["question"])
                st.markdown(f"**Correct treatment:** {m['correct_concept']}")
