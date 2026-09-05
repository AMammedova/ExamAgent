"""ExamAgent — ML/DL exam preparation system.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from examagent.config import get_logger, get_settings
from examagent.models.db import kv_get, session_scope
from examagent.services import materials, planner, progress
from examagent.ui import (
    chat,
    dashboard,
    knowledge_map,
    materials as materials_ui,
    mock,
    progress_page,
    quiz,
    settings_page,
    study,
    weaknesses,
)
from examagent.ui.common import inject_css, llm_badge, score_bar

log = get_logger(__name__)

PAGES = {
    "Dashboard": dashboard.render,
    "Study": study.render,
    "Quiz": quiz.render,
    "Mock Exam": mock.render,
    "Chat": chat.render,
    "Weaknesses": weaknesses.render,
    "Knowledge Map": knowledge_map.render,
    "Progress": progress_page.render,
    "Materials": materials_ui.render,
    "Settings": settings_page.render,
}

PAGE_ICONS = {
    "Dashboard": "🎯", "Study": "📖", "Quiz": "✍️", "Mock Exam": "⏱️", "Chat": "💬",
    "Weaknesses": "🔴", "Knowledge Map": "🗺️", "Progress": "📈",
    "Materials": "📚", "Settings": "⚙️",
}


def bootstrap() -> None:
    """Idempotent startup: create the schema and seed the topic graph."""
    if st.session_state.get("_bootstrapped"):
        return
    progress.initialize()
    st.session_state["_bootstrapped"] = True
    st.session_state.setdefault("use_llm", True)
    st.session_state.setdefault("minutes_per_day", 240)


def sidebar() -> str:
    settings = get_settings()
    days = settings.days_remaining()

    with st.sidebar:
        st.markdown("## ExamAgent")
        colour = "#cf222e" if days <= 2 else "#e8804a" if days <= 4 else "#539bf5"
        st.markdown(
            f"<div style='font-size:1.5rem;font-weight:700;color:{colour}'>"
            f"{days} day{'s' if days != 1 else ''} left</div>"
            f"<div class='ea-muted'>ML + DL final · {settings.exam_day.isoformat()}</div>",
            unsafe_allow_html=True,
        )

        readiness = progress.compute_readiness()
        st.markdown(
            f"<div style='margin-top:10px'>Readiness "
            f"<b>{readiness.overall:.0%}</b></div>"
            + score_bar(readiness.overall,
                        "#2da44e" if readiness.overall >= 0.7
                        else "#d4a72c" if readiness.overall >= 0.45 else "#cf222e"),
            unsafe_allow_html=True,
        )
        st.write("")

        current = st.session_state.get("page", "Dashboard")
        target = st.session_state.pop("nav_target", None)
        if target in PAGES:
            current = target
            st.session_state["page"] = target

        options = list(PAGES)
        choice = st.radio(
            "Navigation",
            options,
            index=options.index(current) if current in options else 0,
            format_func=lambda p: f"{PAGE_ICONS.get(p, '•')}  {p}",
            label_visibility="collapsed",
            key="nav_radio",
        )
        st.session_state["page"] = choice

        st.divider()
        st.session_state["use_llm"] = st.toggle(
            "Use LLM when available", value=st.session_state.get("use_llm", True),
            help="Off = deterministic engines only (faster, free, fully offline).",
        )
        llm_badge()

        nxt = planner.next_topic()
        if nxt:
            st.markdown("---")
            st.markdown("<div class='ea-muted'>Next best action</div>", unsafe_allow_html=True)
            st.markdown(f"**{nxt['topic']}**")
            st.caption(nxt["focus"] + " focus")
            if st.button("Go", use_container_width=True, key="sidebar_go"):
                st.session_state["nav_target"] = "Study"
                st.session_state["nav_payload"] = {"topic_id": nxt["topic_id"]}
                st.rerun()

    return st.session_state["page"]


def first_run_gate() -> bool:
    """Show the onboarding screen until the student dismisses it. True = handled."""
    with session_scope() as s:
        done = kv_get(s, "first_run_complete", False)
    if done:
        return False

    settings = get_settings()
    st.markdown(f"# You have {settings.days_remaining()} days.")
    st.markdown(
        "This is not a chatbot. It is an examiner: it will ask before it explains, mark "
        "you strictly, and spend your remaining time on whatever is most likely to cost "
        "you marks."
    )

    status = materials.library_status()
    lib = progress.dashboard_snapshot()

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Topics tracked", lib["total_topics"])
    with session_scope() as s:
        from examagent.models.db import all_topics

        topics = all_topics(s)
    c2.metric("Critical", sum(1 for t in topics if t.priority == "CRITICAL"))
    c3.metric("High priority", sum(1 for t in topics if t.priority == "HIGH"))
    c4.metric("Material chunks", status["chunks"])

    ml = [t for t in topics if t.category == "Machine Learning"]
    dl = [t for t in topics if t.category == "Deep Learning"]
    st.markdown(f"**Machine Learning:** {len(ml)} topics  ·  **Deep Learning:** {len(dl)} topics")

    st.divider()
    st.markdown("### 1. Upload your course materials")
    st.caption("Optional but recommended: it makes explanations course-specific and citable. "
               "Everything else works without it.")
    label_map = materials_ui.SOURCE_LABELS
    c1, c2 = st.columns([1, 2])
    with c1:
        category = st.selectbox("Category", list(label_map), key="fr_cat")
    with c2:
        files = st.file_uploader(
            "Files", type=["pdf", "txt", "md", "docx", "pptx", "tex"],
            accept_multiple_files=True, key="fr_files",
        )
    if files and st.button("Ingest", type="primary"):
        bar = st.progress(0.0)
        for i, f in enumerate(files, 1):
            path = materials.save_upload(f.getvalue(), f.name)
            r = materials.ingest_file(path, label_map[category], source_name=f.name)
            st.write(f"**{f.name}** — {r['status']}"
                     + (f" · {r['chunks']} chunks" if r.get("chunks") else "")
                     + (f" · {r['error']}" if r.get("error") else ""))
            bar.progress(i / len(files))
        st.rerun()

    if status["chunks"]:
        st.success(f"{status['chunks']} chunks indexed across "
                   f"{status['topics_with_material']} topics.")
        if status["missing_critical"]:
            names = ", ".join(m["name"] for m in status["missing_critical"][:8])
            st.warning(f"**Missing material for critical topics:** {names}"
                       + (" …" if len(status["missing_critical"]) > 8 else ""))

    st.divider()
    st.markdown("### 2. Your 7-day plan")
    plans = planner.build_plan(minutes_per_day=st.session_state.get("minutes_per_day", 240))
    for day in plans[:7]:
        topics_line = ", ".join(b.topic for b in day.blocks[:5]) or "adaptive"
        st.markdown(f"**Day {day.day_number}** · {day.theme}")
        st.caption(f"{topics_line}" + ("  ·  MOCK EXAM" if day.mock_exam else ""))

    st.divider()
    st.markdown("### 3. Start")
    nxt = planner.next_topic()
    if nxt:
        st.markdown(f"Recommended first session: **{nxt['topic']}** — {nxt['reason']}")
    c1, c2 = st.columns(2)
    if c1.button("Start my first session", type="primary", use_container_width=True):
        progress.mark_first_run_complete()
        st.session_state["nav_target"] = "Study"
        if nxt:
            st.session_state["nav_payload"] = {"topic_id": nxt["topic_id"]}
        st.rerun()
    if c2.button("Skip to the dashboard", use_container_width=True):
        progress.mark_first_run_complete()
        st.session_state["nav_target"] = "Dashboard"
        st.rerun()
    return True


def main() -> None:
    st.set_page_config(
        page_title="ExamAgent — ML/DL Exam Prep",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    bootstrap()

    if first_run_gate():
        return

    page = sidebar()
    renderer = PAGES.get(page, dashboard.render)
    try:
        renderer()
    except Exception as exc:  # a page error must not lose the student's session
        log.exception("page %s failed", page)
        st.error(f"Something went wrong on this page: {type(exc).__name__}: {exc}")
        with st.expander("Details"):
            import traceback

            st.code(traceback.format_exc())
        if st.button("Reset this page's state"):
            for key in ("study", "quiz", "mock", "repair"):
                st.session_state.pop(key, None)
            st.rerun()


if __name__ == "__main__":
    main()
