"""Yaddaş kitabçası - the written revision booklet, in one page.

Reading material, not an exercise: numbered claims to memorise, grouped the
way the syllabus is taught. Nothing here is generated at load time, so it
reads the same every time - which is the point of something you revise from -
and it needs no API credit.
"""
from __future__ import annotations

import streamlit as st

from ..data import memory_book_az as book

STATE = "memory_book"


def render() -> None:
    st.session_state.pop("nav_payload", None)
    st.markdown("### Yaddaş kitabçası")
    st.caption("İmtahanda lazım olan hər şey, nömrələnmiş bəndlərlə. Texniki terminlər "
               "ingiliscə saxlanılıb — kağızda da belə olacaq.")

    counts = book.stats()
    c1, c2, c3 = st.columns(3)
    c1.metric("Bənd", counts["points"])
    c2.metric("Bölmə", counts["sections"])
    c3.metric("Düstur", counts["formulas"])

    query = st.text_input(
        "Axtar", placeholder="məsələn: dropout, recall, attention, padding…",
        label_visibility="collapsed",
    ).strip()

    if query:
        _render_search(query)
        return

    tabs = st.tabs(["Machine Learning", "Deep Learning", "Bütün düsturlar"])
    for tab, (_category, sections) in zip(tabs, book.all_sections()):
        with tab:
            _render_sections(sections)
    with tabs[2]:
        _render_formulas()


def _render_sections(sections: list[book.Section]) -> None:
    titles = [s.title for s in sections]
    chosen = st.radio("Bölmə", titles, label_visibility="collapsed", horizontal=False,
                      key=f"mb_{titles[0][:12]}")
    section = next(s for s in sections if s.title == chosen)

    st.markdown(f"#### {section.title}")
    for i, point in enumerate(section.points, 1):
        st.markdown(f"**{i}.** {point}")
    if section.formulas:
        st.markdown("**Düsturlar**")
        for formula in section.formulas:
            st.code(formula, language=None)


def _render_formulas() -> None:
    st.caption("İmtahanda yadda saxlanılması lazım olan düsturlar, bir yerdə.")
    for _category, sections in book.all_sections():
        for section in sections:
            if not section.formulas:
                continue
            st.markdown(f"**{section.title}**")
            for formula in section.formulas:
                st.code(formula, language=None)


def _render_search(query: str) -> None:
    needle = query.lower()
    hits = 0
    for category, sections in book.all_sections():
        for section in sections:
            matched = [(i, p) for i, p in enumerate(section.points, 1)
                       if needle in p.lower()]
            formulas = [f for f in section.formulas if needle in f.lower()]
            if not matched and not formulas:
                continue
            hits += len(matched) + len(formulas)
            st.markdown(f"**{section.title}** · {category}")
            for i, point in matched:
                st.markdown(f"**{i}.** {point}")
            for formula in formulas:
                st.code(formula, language=None)
            st.divider()

    if not hits:
        st.info(f"'{query}' üzrə heç nə tapılmadı. Başqa söz sınayın — məsələn "
                "terminin ingiliscə adı (dropout, recall, stride).")
