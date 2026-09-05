"""Settings: LLM provider, exam date, readiness weights, data management."""
from __future__ import annotations

from datetime import date

import streamlit as st

from ..config import PROJECT_ROOT, get_settings, reload_settings
from ..services.assertion_engine import bank_stats
from ..services.calc_engine import GENERATORS
from ..services.llm import LANGUAGE_NAMES, get_llm, reset_llm
from ..services.vectorstore import reset_vector_store

ENV_PATH = PROJECT_ROOT / ".env"


def render() -> None:
    st.session_state.pop("nav_payload", None)
    st.markdown("### Settings")
    settings = get_settings()

    tabs = st.tabs(["LLM provider", "Exam & scoring", "Engines", "Data"])
    with tabs[0]:
        _llm(settings)
    with tabs[1]:
        _exam(settings)
    with tabs[2]:
        _engines()
    with tabs[3]:
        _data(settings)


def _llm(settings) -> None:
    status = get_llm().status()
    if status["mode"] == "LLM":
        st.success(f"Connected: **{status['provider']}** · `{status['model']}`")
    else:
        st.info(
            "**Offline mode.** Everything still works: calculation problems, "
            "assertion-reason questions, the seed bank, scoring, the planner and mock "
            "exams are all deterministic. Adding an API key additionally enables "
            "LLM-generated questions, LLM marking of open answers, and generated lessons."
        )
    if status["error"]:
        st.error(f"Client error: {status['error']}")

    with st.form("llm_form"):
        provider = st.selectbox(
            "Provider", ["anthropic", "openai", "none"],
            index=["anthropic", "openai", "none"].index(settings.llm_provider)
            if settings.llm_provider in ("anthropic", "openai", "none") else 0,
        )
        key = st.text_input(
            "API key", type="password",
            value="",
            placeholder="leave blank to keep the existing key",
            help="Stored in the local .env file. Never sent anywhere except the provider.",
        )
        model = st.text_input("Model name", value=settings.model_name)
        c1, c2 = st.columns(2)
        max_tokens = c1.number_input("Max tokens", 256, 8192, settings.max_tokens, step=128)
        temperature = c2.slider("Temperature", 0.0, 1.0, settings.temperature, 0.05)

        st.markdown("**Language of explanations**")
        lang_codes = list(LANGUAGE_NAMES)
        language = st.selectbox(
            "Language", lang_codes,
            index=lang_codes.index(settings.language) if settings.language in lang_codes else 0,
            format_func=lambda code: LANGUAGE_NAMES[code],
            label_visibility="collapsed",
            help="Applies to LLM-generated lessons, chat explanations, generated "
                 "questions and open-answer feedback. The deterministic offline content "
                 "(calculation problems, the assertion-reason bank, the seed question "
                 "bank) is fixed English text and is not translated.",
        )
        if language != "en":
            st.caption(
                "Riyazi düsturlar, dəyişən adları və standart termin (ReLU, softmax və s.) "
                "orijinal formada qalır - yalnız izahlar tərcümə olunur. Offline rejimdə "
                "(kalkulyasiya mühərriki, Assertion-Reason bankı, seed suallar) mətnlər "
                "ingilis dilində qalır, çünki onlar sabit pedaqoji mətndir."
            )

        if st.form_submit_button("Save", type="primary"):
            updates = {
                "LLM_PROVIDER": provider,
                "MODEL_NAME": model,
                "MAX_TOKENS": str(int(max_tokens)),
                "TEMPERATURE": str(temperature),
                "LANGUAGE": language,
            }
            if key.strip():
                updates["ANTHROPIC_API_KEY" if provider == "anthropic"
                        else "OPENAI_API_KEY"] = key.strip()
            _write_env(updates)
            reload_settings()
            reset_llm()
            st.success("Saved.")
            st.rerun()

    st.caption("Model suggestions — Anthropic: `claude-sonnet-4-5`, `claude-opus-4-1`; "
               "OpenAI: `gpt-4o`, `gpt-4o-mini`.")

    st.divider()
    if st.button("Test the connection"):
        with st.spinner("Calling the provider…"):
            resp = get_llm().complete("Reply with exactly: OK", max_tokens=16)
        if resp.ok:
            st.success(f"Response in {resp.latency_ms} ms: {resp.text[:120]}")
        else:
            st.error(f"Failed: {resp.error}")


def _exam(settings) -> None:
    with st.form("exam_form"):
        c1, c2 = st.columns(2)
        with c1:
            exam_date = st.date_input("Exam date", value=settings.exam_day)
        with c2:
            study_days = st.number_input("Study days in the plan", 1, 21,
                                         settings.study_days)

        st.markdown("#### Exam readiness weights")
        st.caption("These decide what 'ready' means. They must be meaningful, not a flat "
                   "average — calculation and reasoning dominate this exam.")
        w = settings.readiness_weights()
        w1, w2, w3 = st.columns(3)
        crit = w1.slider("Critical topic mastery", 0.0, 0.6, w["critical"], 0.05)
        calc = w2.slider("Calculation", 0.0, 0.6, w["calculation"], 0.05)
        reas = w3.slider("Reasoning", 0.0, 0.6, w["reasoning"], 0.05)
        w4, w5, w6 = st.columns(3)
        exam_w = w4.slider("Exam performance", 0.0, 0.6, w["exam"], 0.05)
        cov = w5.slider("Coverage", 0.0, 0.6, w["coverage"], 0.05)
        conf = w6.slider("Confidence", 0.0, 0.6, w["confidence"], 0.05)
        total = crit + calc + reas + exam_w + cov + conf
        st.caption(f"Total {total:.2f} — weights are normalised automatically.")

        if st.form_submit_button("Save", type="primary"):
            _write_env({
                "EXAM_DATE": exam_date.isoformat(),
                "STUDY_DAYS": str(int(study_days)),
                "READINESS_W_CRITICAL": f"{crit:.3f}",
                "READINESS_W_CALCULATION": f"{calc:.3f}",
                "READINESS_W_REASONING": f"{reas:.3f}",
                "READINESS_W_EXAM": f"{exam_w:.3f}",
                "READINESS_W_COVERAGE": f"{cov:.3f}",
                "READINESS_W_CONFIDENCE": f"{conf:.3f}",
            })
            reload_settings()
            st.success(f"Saved. {max(0, (exam_date - date.today()).days)} days remaining.")
            st.rerun()

    st.divider()
    st.markdown("#### Session defaults")
    minutes = st.slider("Available study minutes per day", 60, 600,
                        st.session_state.get("minutes_per_day", 240), step=30)
    st.session_state["minutes_per_day"] = minutes
    st.session_state["use_llm"] = st.checkbox(
        "Use the LLM when available", value=st.session_state.get("use_llm", True),
        help="Turn off to force the deterministic engines (faster, no API cost).",
    )


def _engines() -> None:
    st.markdown("#### Deterministic engines")
    st.caption("These run without an API key and produce exactly-gradable questions.")

    from ..data.seed_questions import SEED_QUESTIONS, seed_question_count
    from ..services.calc_engine import TOPIC_GENERATORS

    c1, c2, c3 = st.columns(3)
    c1.metric("Calculation generators", len(GENERATORS))
    c2.metric("Assertion-Reason bank", bank_stats()["total"])
    c3.metric("Seed questions", len(SEED_QUESTIONS))

    st.markdown("**Calculation problem types**")
    st.markdown(", ".join(sorted(GENERATORS)))
    st.caption(f"Covering {len(TOPIC_GENERATORS)} topics with fresh randomised values "
               "on every generation.")

    st.markdown("**Assertion-Reason answer distribution in the bank**")
    st.json(bank_stats()["by_answer"])

    st.markdown("**Seed questions by type**")
    st.json(seed_question_count())


def _data(settings) -> None:
    st.markdown("#### Storage")
    st.code(f"Database:     {settings.data_path / 'examagent.db'}\n"
            f"Uploads:      {settings.upload_path}\n"
            f"Vector store: {settings.data_path / 'vectorstore'}\n"
            f"Config:       {ENV_PATH}", language="text")

    st.markdown("#### Retrieval")
    with st.form("rag_form"):
        c1, c2, c3 = st.columns(3)
        backend = c1.selectbox("Vector backend", ["local", "chroma"],
                               index=0 if settings.vector_backend == "local" else 1,
                               help="ChromaDB is used if installed; otherwise the app "
                                    "falls back to the local TF-IDF store automatically.")
        chunk = c2.number_input("Chunk size", 300, 4000, settings.chunk_size, step=50)
        top_k = c3.number_input("Retrieval top-k", 1, 20, settings.retrieval_top_k)
        if st.form_submit_button("Save"):
            _write_env({
                "VECTOR_BACKEND": backend,
                "CHUNK_SIZE": str(int(chunk)),
                "RETRIEVAL_TOP_K": str(int(top_k)),
            })
            reload_settings()
            reset_vector_store()
            st.success("Saved. Re-ingest materials for a new chunk size to take effect.")

    st.divider()
    st.markdown("#### Reset")
    st.caption("Progress and the knowledge base are separate — clearing one keeps the other.")
    c1, c2 = st.columns(2)
    with c1:
        confirm = st.text_input("Type RESET to confirm", key="reset_confirm")
        if st.button("Delete all study progress", disabled=confirm != "RESET"):
            _reset_progress()
            st.success("Progress cleared. Topics re-seeded.")
            st.rerun()
    with c2:
        st.markdown("This deletes attempts, mistakes, mock exams, sessions and all topic "
                    "scores. Uploaded materials are kept.")


def _reset_progress() -> None:
    from ..models.db import (
        Attempt, KeyValue, Mistake, MockExam, QuestionRecord, StudySession, Topic,
        session_scope,
    )
    from ..services.progress import ensure_topics

    with session_scope() as s:
        for model in (Attempt, Mistake, MockExam, StudySession, QuestionRecord, Topic, KeyValue):
            s.query(model).delete()
    with session_scope() as s:
        ensure_topics(s)


def _write_env(updates: dict[str, str]) -> None:
    """Merge key=value pairs into the local .env file, preserving other keys."""
    lines: list[str] = []
    if ENV_PATH.exists():
        lines = ENV_PATH.read_text(encoding="utf-8").splitlines()

    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            out.append(line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in updates:
            out.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            out.append(line)
    for key, value in updates.items():
        if key not in seen:
            out.append(f"{key}={value}")
    ENV_PATH.write_text("\n".join(out) + "\n", encoding="utf-8")
