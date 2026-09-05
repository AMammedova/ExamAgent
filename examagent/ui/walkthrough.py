"""Walk an uploaded document section by section, being tested on each one.

The Materials page indexes what you upload; this is where you actually work
through it. One section at a time: what it says, then a question you answer
before you are allowed to see whether you were right.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from ..services import materials, progress, walkthrough
from ..services.evaluator import evaluate
from ..services.question_gen import TOPIC_INDEX
from ..services.walkthrough import Section
from .common import (
    empty_state,
    render_citations,
    render_evaluation,
    render_question,
    score_bar,
)

STATE_KEY = "walk"


def _state() -> dict[str, Any]:
    return st.session_state.setdefault(STATE_KEY, {})


def _use_llm() -> bool:
    return bool(st.session_state.get("use_llm", True))


def _open(filename: str, index: int) -> None:
    """Move the walk to one section, dropping whatever was on screen for the
    previous one."""
    st.session_state[STATE_KEY] = {"filename": filename, "index": index}


def render() -> None:
    docs = walkthrough.documents_with_progress(materials.documents())
    if not docs:
        empty_state(
            "Nothing to walk through yet. Upload a lecture, a set of slides or your "
            "own notes on the Upload tab — then this is where you work through it, "
            "section by section, answering as you go."
        )
        return

    state = _state()

    def _label(d: dict[str, Any]) -> str:
        p = d["progress"]
        mark = "✅" if p["remaining"] == 0 else f"{p['done']}/{p['total']}"
        return f"{d['source_name'] or d['filename']} · {mark}"

    names = [d["filename"] for d in docs]
    current = state.get("filename") if state.get("filename") in names else names[0]
    chosen = st.selectbox(
        "Document", names, index=names.index(current),
        format_func=lambda fn: _label(next(d for d in docs if d["filename"] == fn)),
    )
    if chosen != state.get("filename"):
        _open(chosen, walkthrough.next_index(chosen))
        state = _state()

    secs = walkthrough.sections(chosen)
    if not secs:
        empty_state("This document produced no readable sections.")
        return

    over = walkthrough.overview(chosen, secs)
    _progress_header(chosen, over)

    idx = min(state.get("index", 0), len(secs) - 1)
    section = secs[idx]
    _section_view(chosen, section, secs, over)


def _progress_header(filename: str, over: dict[str, Any]) -> None:
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.markdown(
            f"**{over['done']}/{over['total']} sections** · "
            + (f"mean {over['mean_score']:.1f}/10" if over["done"] else "not started")
            + (f" · ~{over['minutes_left']} min left" if over["remaining"] else " · complete")
        )
        st.markdown(score_bar(over["fraction"], "#2da44e" if over["fraction"] >= 0.999
                              else "#539bf5"), unsafe_allow_html=True)
    if c2.button("Continue", use_container_width=True,
                 help="Jump to the first section you have not answered"):
        _open(filename, walkthrough.next_index(filename))
        st.rerun()
    if c3.button("Restart", use_container_width=True,
                 help="Clear the marks for this document and start again"):
        walkthrough.reset(filename)
        _open(filename, 0)
        st.rerun()

    if over["weak_sections"] or over["flagged"]:
        with st.expander(f"Needs another pass — {len(over['weak_sections'])} scored under 6, "
                         f"{len(over['flagged'])} flagged"):
            seen: set[str] = set()
            for s in over["weak_sections"] + over["flagged"]:
                if s.chunk_id in seen:
                    continue
                seen.add(s.chunk_id)
                c1, c2 = st.columns([4, 1])
                c1.markdown(f"· {s.heading}")
                if c2.button("Open", key=f"weak_{s.chunk_id}", use_container_width=True):
                    _open(filename, s.index)
                    st.rerun()
    st.divider()


def _section_view(filename: str, section: Section, secs: list[Section],
                  over: dict[str, Any]) -> None:
    state = _state()
    scores = walkthrough.load_state(filename)["scores"]
    done_mark = ""
    if section.chunk_id in scores:
        done_mark = f" · answered {scores[section.chunk_id]:.0f}/10"

    st.markdown(f"#### Section {section.index + 1} of {len(secs)} — {section.heading}"
                f"<span class='ea-muted'>{done_mark}</span>", unsafe_allow_html=True)

    nav1, nav2, nav3 = st.columns([1, 1, 1])
    if nav1.button("◀ Previous", disabled=section.index == 0, use_container_width=True):
        _open(filename, section.index - 1)
        st.rerun()
    flagged = section.chunk_id in walkthrough.load_state(filename)["flagged"]
    if nav2.button("🚩 Flagged" if flagged else "🚩 Flag this",
                   use_container_width=True,
                   help="Mark it to come back to before the exam"):
        walkthrough.toggle_flag(filename, section.chunk_id)
        st.rerun()
    if nav3.button("Next ▶", disabled=section.index >= len(secs) - 1,
                   use_container_width=True):
        _open(filename, section.index + 1)
        st.rerun()

    # ---- what the section says
    if state.get("brief_for") != section.chunk_id:
        with st.spinner("Reading the section…"):
            state["brief"] = walkthrough.brief(section, use_llm=_use_llm())
        state["brief_for"] = section.chunk_id

    brief = state.get("brief") or {"points": [], "exam_angle": "", "verbatim": True}
    with st.container(border=True):
        st.markdown("**What this section establishes**")
        for point in brief["points"]:
            st.markdown(f"- {point}")
        if brief.get("exam_angle"):
            st.markdown(f"**Exam angle.** {brief['exam_angle']}")
        if brief.get("verbatim"):
            st.caption("Extracted from the section verbatim — no LLM configured, so "
                       "nothing here is paraphrased or added.")
    if section.citation:
        render_citations([section.citation])
    with st.expander("Show the original text"):
        st.markdown(section.text)
    if section.topics:
        st.caption("Topics: " + ", ".join(
            TOPIC_INDEX.get(t, {}).get("name", t) for t in section.topics))

    # ---- answer it
    st.divider()
    st.markdown("**Close the material and answer.** Reading is not remembering.")

    if state.get("q_for") != section.chunk_id:
        with st.spinner("Writing a question from this section…"):
            state["q"] = walkthrough.question_for(section, use_llm=_use_llm())
        state["q_for"] = section.chunk_id
        state.pop("eval", None)

    question = state["q"]
    key = f"walk_q_{section.chunk_id}"

    if "eval" not in state:
        answer = render_question(question, key)
        if st.button("Submit answer", type="primary"):
            filled = (any(str(v).strip() for v in answer.values())
                      if isinstance(answer, dict) else bool(str(answer).strip()))
            if not filled:
                st.error("Write an answer first — a blank answer scores zero in the exam too.")
            else:
                with st.spinner("Marking…"):
                    ev = evaluate(question, answer, use_llm=_use_llm())
                text = (answer if isinstance(answer, str)
                        else "; ".join(f"{k}={v}" for k, v in answer.items()))
                walkthrough.record(filename, section.chunk_id, ev.score)
                if question.topic in TOPIC_INDEX:
                    progress.record_attempt(question, ev, student_answer=text,
                                            context="walkthrough", seconds=150)
                state["eval"] = ev
                st.rerun()
        return

    render_question(question, key, disabled=True)
    st.divider()
    render_evaluation(state["eval"], question)
    st.divider()

    last = section.index >= len(secs) - 1
    c1, c2 = st.columns([1, 1])
    if c1.button("Finish document" if last else "Next section ▶",
                 type="primary", use_container_width=True):
        _open(filename, section.index if last else section.index + 1)
        st.rerun()
    if c2.button("Answer this one again", use_container_width=True):
        state.pop("eval", None)
        state.pop("q_for", None)
        st.rerun()

    if last:
        fresh = walkthrough.overview(filename, secs)
        if fresh["remaining"] == 0:
            st.success(
                f"Document complete — {fresh['total']} sections, mean "
                f"{fresh['mean_score']:.1f}/10."
                + (f" {len(fresh['weak_sections'])} still under 6: work those before "
                   "the exam." if fresh["weak_sections"] else "")
            )
