"""Shared UI widgets.

One rule enforced here: `render_question` never renders anything that reveals
the answer. Model answers, worked solutions and truth flags live exclusively in
`render_evaluation`, which is only called after submission.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from ..models.schemas import Citation, Evaluation, Mastery, Question, QuestionType

DIFFICULTY_LABEL = {
    1: "1 · recognition", 2: "2 · understanding", 3: "3 · application",
    4: "4 · reasoning", 5: "5 · exam level", 6: "6 · hard exam level",
}

TYPE_LABEL = {
    QuestionType.MCQ: "Multiple choice",
    QuestionType.ASSERTION_REASON: "Assertion & Reason",
    QuestionType.SHORT_ANSWER: "Short answer",
    QuestionType.CALCULATION: "Calculation",
    QuestionType.CONCEPTUAL: "Conceptual reasoning",
    QuestionType.COMPARISON: "Comparison",
    QuestionType.SCENARIO: "Scenario",
    QuestionType.DIAGRAM: "Architecture interpretation",
    QuestionType.WHAT_IF: "What happens if…",
    QuestionType.GRAPH: "Graph interpretation",
}


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px;}
        div[data-testid="stMetricValue"] {font-size: 1.6rem;}
        .ea-chip {display:inline-block; padding:2px 9px; border-radius:11px;
                  font-size:0.72rem; font-weight:600; margin-right:5px; margin-bottom:3px;
                  border:1px solid rgba(128,128,128,.35);}
        .ea-card {border:1px solid rgba(128,128,128,.25); border-radius:10px;
                  padding:14px 16px; margin-bottom:10px;}
        .ea-bar-bg {background:rgba(128,128,128,.18); border-radius:5px; height:9px;
                    width:100%; overflow:hidden;}
        .ea-bar-fg {height:9px; border-radius:5px;}
        .ea-muted {color:#8b949e; font-size:0.82rem;}
        .ea-src {font-size:0.74rem; color:#8b949e; border-left:2px solid rgba(128,128,128,.4);
                 padding-left:8px; margin-top:2px;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def chip(text: str, color: str = "#8b949e") -> str:
    return f'<span class="ea-chip" style="color:{color};">{text}</span>'


def score_bar(value: float, color: str = "#2da44e", height: int = 9) -> str:
    pct = max(0.0, min(1.0, value)) * 100
    return (f'<div class="ea-bar-bg" style="height:{height}px;">'
            f'<div class="ea-bar-fg" style="width:{pct:.1f}%;background:{color};'
            f'height:{height}px;"></div></div>')


def mastery_color(score: float, tested: bool = True) -> str:
    if not tested:
        return "#8b949e"
    return Mastery.from_score(score).color


def priority_color(priority: str) -> str:
    return {"CRITICAL": "#cf222e", "HIGH": "#e8804a",
            "MEDIUM": "#d4a72c", "LOW": "#8b949e"}.get(priority, "#8b949e")


def render_citations(citations: list[Citation], label: str = "Source") -> None:
    if not citations:
        return
    lines = "".join(
        f'<div class="ea-src">{label}: {c.label()}</div>' for c in citations[:4]
    )
    st.markdown(lines, unsafe_allow_html=True)


def question_header(question: Question, index: int | None = None,
                    total: int | None = None) -> None:
    bits = []
    if index is not None:
        bits.append(f"**Q{index}" + (f"/{total}" if total else "") + "**")
    bits.append(chip(TYPE_LABEL.get(question.question_type, question.question_type.value),
                     "#539bf5"))
    bits.append(chip(DIFFICULTY_LABEL.get(question.difficulty, str(question.difficulty)),
                     "#d4a72c"))
    bits.append(chip(question.priority.value, priority_color(question.priority.value)))
    bits.append(chip(question.topic.replace("_", " ").title(), "#8b949e"))
    st.markdown(" ".join(bits), unsafe_allow_html=True)


def render_question(question: Question, key_prefix: str,
                    disabled: bool = False, show_meta: bool = True) -> Any:
    """Render a question and return the student's answer.

    Returns a str for open/choice questions and a dict for calculations.
    Never displays the answer, the model answer or the worked solution.
    """
    if show_meta:
        question_header(question)

    st.markdown(question.prompt)

    if question.question_type == QuestionType.CALCULATION and question.calc_spec:
        return _render_calculation(question, key_prefix, disabled)

    if question.options:
        labels = [f"{o.key}. {o.text}" for o in question.options]
        choice = st.radio(
            "Your answer", labels, index=None, key=f"{key_prefix}_choice",
            disabled=disabled, label_visibility="collapsed",
        )
        return choice.split(".")[0].strip() if choice else ""

    placeholder = {
        QuestionType.COMPARISON: "Structure it: axis by axis, then when each is preferred.",
        QuestionType.SCENARIO: "Diagnosis → mechanism → intervention → what would not help.",
        QuestionType.WHAT_IF: "State the consequence, then the mechanism that causes it.",
        QuestionType.DIAGRAM: "Component by component: what it computes, shapes, what breaks.",
    }.get(question.question_type,
          "Answer as you would in the exam: mechanism first, then consequence.")

    return st.text_area(
        "Your answer", key=f"{key_prefix}_text", height=190,
        placeholder=placeholder, disabled=disabled, label_visibility="collapsed",
    )


def _render_calculation(question: Question, key_prefix: str, disabled: bool) -> dict[str, str]:
    spec = question.calc_spec or {}
    parts = spec.get("parts", [])
    st.caption(f"{len(parts)} parts · each is marked separately, so attempt every one "
               "(partial credit applies)")
    answers: dict[str, str] = {}
    cols_per_row = 2
    for i in range(0, len(parts), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, part in zip(cols, parts[i:i + cols_per_row]):
            with col:
                if part.get("kind") == "choice" and part.get("choices"):
                    val = st.selectbox(
                        part["label"], ["—"] + list(part["choices"]),
                        key=f"{key_prefix}_{part['key']}", disabled=disabled,
                    )
                    answers[part["key"]] = "" if val == "—" else val
                else:
                    answers[part["key"]] = st.text_input(
                        part["label"], key=f"{key_prefix}_{part['key']}",
                        disabled=disabled, placeholder="number",
                    )
    return answers


def render_evaluation(evaluation: Evaluation, question: Question | None = None,
                      show_model_answer: bool = True) -> None:
    """Full marking feedback. Only call this after the student has submitted."""
    color = ("#2da44e" if evaluation.score >= 7
             else "#d4a72c" if evaluation.score >= 4 else "#cf222e")
    verdict = ("Correct" if evaluation.correct
               else "Partially correct" if evaluation.partial else "Incorrect")

    c1, c2, c3 = st.columns([1.1, 1.1, 2.2])
    c1.metric("Score", f"{evaluation.score:.1f}/10")
    c2.metric("Verdict", verdict)
    with c3:
        st.markdown(f"**Marked by:** {evaluation.evaluator}")
        if evaluation.mistake_type.value != "None":
            st.markdown(
                chip(f"{evaluation.mistake_type.value} error", color)
                + chip(f"{evaluation.severity} severity", "#8b949e"),
                unsafe_allow_html=True,
            )
    st.markdown(score_bar(evaluation.score / 10, color, height=7), unsafe_allow_html=True)

    if evaluation.sub_scores:
        st.markdown("**Step-by-step marking**")
        rows = []
        for s in evaluation.sub_scores:
            mark = "✅" if s.correct else "❌"
            rows.append({
                "": mark,
                "Part": s.label,
                "Your answer": s.student or "—",
                "Expected": s.expected,
                "Note": s.note[:150] if s.note else "",
            })
        st.dataframe(rows, hide_index=True, use_container_width=True)

    left, right = st.columns(2)
    with left:
        if evaluation.got_right:
            st.markdown("**What you got right**")
            for item in evaluation.got_right:
                st.markdown(f"- {item}")
    with right:
        if evaluation.missed:
            st.markdown("**What you missed**")
            for item in evaluation.missed:
                st.markdown(f"- {item}")

    if evaluation.incorrect:
        st.markdown("**What was incorrect**")
        for item in evaluation.incorrect:
            st.markdown(f"- {item}")

    if evaluation.examiner_expects:
        st.info(f"**What the examiner expects:** {evaluation.examiner_expects}")

    if evaluation.improvement:
        st.warning(f"**Turn this into an exam answer:** {evaluation.improvement}")

    if show_model_answer and (evaluation.model_answer or (question and question.model_answer)):
        with st.expander("Model answer / worked solution", expanded=not evaluation.correct):
            st.markdown(evaluation.model_answer or (question.model_answer if question else ""))

    if question and question.citations:
        render_citations(question.citations, "Based on")


def readiness_block(readiness, compact: bool = False) -> None:
    pct = readiness.overall
    color = "#2da44e" if pct >= 0.7 else "#d4a72c" if pct >= 0.45 else "#cf222e"
    st.markdown(
        f"<div style='font-size:2.6rem;font-weight:700;color:{color};line-height:1.1'>"
        f"{pct:.0%}</div><div class='ea-muted'>exam readiness</div>",
        unsafe_allow_html=True,
    )
    st.markdown(score_bar(pct, color, height=11), unsafe_allow_html=True)
    if compact:
        return
    rows = [
        ("Critical topic mastery", readiness.critical_mastery, readiness.weights.get("critical", 0)),
        ("Calculation ability", readiness.calculation, readiness.weights.get("calculation", 0)),
        ("Reasoning ability", readiness.reasoning, readiness.weights.get("reasoning", 0)),
        ("Exam question performance", readiness.exam_performance, readiness.weights.get("exam", 0)),
        ("Coverage", readiness.coverage, readiness.weights.get("coverage", 0)),
        ("Confidence", readiness.confidence, readiness.weights.get("confidence", 0)),
    ]
    for label, value, weight in rows:
        c1, c2, c3 = st.columns([2.4, 3, 0.9])
        c1.markdown(f"<span style='font-size:0.86rem'>{label}</span>", unsafe_allow_html=True)
        c2.markdown(score_bar(value, mastery_color(value)), unsafe_allow_html=True)
        c3.markdown(f"<span class='ea-muted'>{value:.0%} · w{weight:.0%}</span>",
                    unsafe_allow_html=True)


def llm_badge() -> None:
    from ..config import get_settings
    from ..services.llm import get_llm

    status = get_llm().status()
    lang = get_settings().language
    lang_tag = f" · {lang.upper()}" if lang != "en" else ""
    if status["mode"] == "LLM":
        st.caption(f"🟢 {status['provider']} · {status['model']}{lang_tag}")
    else:
        note = ("⚪ Offline mode — deterministic engines "
               "(add an API key in Settings for generated questions)")
        if lang != "en":
            note += ". Offline content stays in English; language switching needs an LLM."
        st.caption(note)


def empty_state(message: str, action: str = "") -> None:
    st.markdown(
        f"<div class='ea-card'><div class='ea-muted'>{message}</div>"
        + (f"<div style='margin-top:6px'>{action}</div>" if action else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def go_to(page: str, **payload: Any) -> None:
    """Queue a navigation for the next rerun."""
    st.session_state["nav_target"] = page
    st.session_state["nav_payload"] = payload
    st.rerun()
