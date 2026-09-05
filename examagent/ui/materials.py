"""Materials page: upload, ingest and inspect the course knowledge base."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from ..models.schemas import SourceType
from ..services import materials, rag
from .common import chip, empty_state, score_bar

SOURCE_LABELS = {
    "University ML": SourceType.UNIVERSITY_ML.value,
    "University DL": SourceType.UNIVERSITY_DL.value,
    "Udemy ML": SourceType.UDEMY_ML.value,
    "Udemy DL": SourceType.UDEMY_DL.value,
    "Exam sample / past paper": SourceType.EXAM_SAMPLES.value,
    "My notes": SourceType.STUDENT_NOTES.value,
}

SOURCE_COLORS = {
    "UNIVERSITY_ML": "#2da44e", "UNIVERSITY_DL": "#2da44e",
    "UDEMY_ML": "#539bf5", "UDEMY_DL": "#539bf5",
    "EXAM_SAMPLES": "#cf222e", "STUDENT_NOTES": "#d4a72c",
}


def render() -> None:
    st.session_state.pop("nav_payload", None)
    st.markdown("### Materials")
    st.caption("University material outranks Udemy material when the app answers questions "
               "about what the exam expects. Exam samples are used for question style.")

    tabs = st.tabs(["Upload", "Library", "Coverage", "Search"])
    with tabs[0]:
        _upload()
    with tabs[1]:
        _library()
    with tabs[2]:
        _coverage()
    with tabs[3]:
        _search()


def _upload() -> None:
    c1, c2 = st.columns([1, 1])
    with c1:
        label = st.selectbox("Source category", list(SOURCE_LABELS))
    with c2:
        lecture = st.text_input("Lecture / section (optional)",
                                placeholder="e.g. Lecture 3 — Model Validation")

    files = st.file_uploader(
        "Course materials",
        type=["pdf", "txt", "md", "markdown", "docx", "pptx", "tex", "csv",
              "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="PDF, TXT, MD, DOCX, PPTX are extracted directly. Images need an OCR engine "
             "(pytesseract + Pillow) to contribute text.",
    )

    if files and st.button("Ingest into knowledge base", type="primary"):
        source_type = SOURCE_LABELS[label]
        bar = st.progress(0.0)
        status = st.empty()
        results = []
        for i, f in enumerate(files, 1):
            status.markdown(f"Processing **{f.name}** ({i}/{len(files)})…")
            path = materials.save_upload(f.getvalue(), f.name)
            results.append(materials.ingest_file(
                path, source_type, source_name=Path(f.name).stem, lecture=lecture))
            bar.progress(i / len(files))
        status.empty()
        bar.empty()

        for r in results:
            if r["status"] == "indexed":
                st.success(f"**{r['filename']}** — {r['chunks']} chunks indexed"
                           + (f" · topics detected: {', '.join(r.get('topics', [])[:5])}"
                              if r.get("topics") else ""))
            elif r["status"] == "duplicate":
                st.info(f"**{r['filename']}** — skipped: {r['error']}")
            elif r["status"] == "empty":
                st.warning(f"**{r['filename']}** — {r['error']}")
            else:
                st.error(f"**{r['filename']}** — {r['status']}: {r['error']}")
        st.rerun()

    st.divider()
    st.markdown("#### Add notes directly")
    note_label = st.selectbox("Category", list(SOURCE_LABELS), index=5, key="note_cat")
    note_title = st.text_input("Title", placeholder="e.g. Backpropagation summary")
    note_body = st.text_area("Text", height=180,
                             placeholder="Paste lecture notes, a summary or a past question…")
    if st.button("Add to knowledge base", disabled=not note_body.strip()):
        safe = "".join(ch for ch in (note_title or "note") if ch.isalnum() or ch in " -_")[:60]
        path = materials.save_upload(note_body.encode("utf-8"), f"{safe or 'note'}.md")
        r = materials.ingest_file(path, SOURCE_LABELS[note_label],
                                  source_name=note_title or "Note")
        if r["status"] == "indexed":
            st.success(f"Added — {r['chunks']} chunks.")
        else:
            st.warning(f"{r['status']}: {r['error']}")
        st.rerun()


def _library() -> None:
    docs = materials.documents()
    status = materials.library_status()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documents", len(docs))
    c2.metric("Chunks", status["chunks"])
    c3.metric("Topics covered", status["topics_with_material"])
    c4.metric("Backend", status["backend"] or "—")

    if not docs:
        empty_state(
            "No materials uploaded yet. The app works fully without them — the "
            "calculation engine, assertion-reason bank and seed questions are all offline — "
            "but uploading your lecture slides is what makes explanations course-specific "
            "and citable."
        )
        return

    by_source = status["by_source_type"]
    if by_source:
        st.markdown(" ".join(
            chip(f"{k.replace('_', ' ').title()}: {v}", SOURCE_COLORS.get(k, "#8b949e"))
            for k, v in by_source.items()
        ), unsafe_allow_html=True)

    st.divider()
    for d in docs:
        c1, c2, c3 = st.columns([3, 1.4, 0.7])
        with c1:
            icon = {"indexed": "✅", "duplicate": "↔️", "empty": "⚠️",
                    "failed": "❌"}.get(d["status"], "•")
            st.markdown(f"{icon} **{d['filename']}**")
            meta = f"{d['source_type'].replace('_', ' ').title()}"
            if d["lecture"]:
                meta += f" · {d['lecture']}"
            meta += f" · {d['date']}"
            st.markdown(f"<div class='ea-muted'>{meta}</div>", unsafe_allow_html=True)
            if d["error"]:
                st.markdown(f"<div class='ea-muted'>{d['error']}</div>",
                            unsafe_allow_html=True)
        c2.markdown(f"<div class='ea-muted'>{d['chunks']} chunks · "
                    f"{d['chars']:,} chars</div>", unsafe_allow_html=True)
        if c3.button("Remove", key=f"del_{d['id']}"):
            materials.delete_document(d["id"])
            st.rerun()

    st.divider()
    with st.expander("Danger zone"):
        if st.button("Delete the entire knowledge base"):
            materials.reset_knowledge_base()
            st.success("Knowledge base cleared.")
            st.rerun()


def _coverage() -> None:
    status = materials.library_status()
    if not status["chunks"]:
        empty_state("Upload materials to see coverage.")
        return

    st.markdown("#### Coverage by category")
    for cat, data in status["by_category"].items():
        pct = data["covered"] / data["total"] if data["total"] else 0
        st.markdown(f"**{cat}** — {data['covered']}/{data['total']} topics have material")
        st.markdown(score_bar(pct, "#539bf5"), unsafe_allow_html=True)

    if status["missing_sources"]:
        st.warning("**No material uploaded for:** "
                   + ", ".join(s.replace("_", " ").title()
                               for s in status["missing_sources"])
                   + ". The app will say so rather than invent course-specific claims.")

    missing = status["missing_critical"]
    if missing:
        st.markdown("#### CRITICAL topics with no uploaded material")
        st.caption("These are the highest-value gaps in your knowledge base. The app will "
                   "still quiz you on them using its built-in engines, but explanations "
                   "will not be course-specific or citable.")
        for cat in ("Machine Learning", "Deep Learning"):
            rows = [m for m in missing if m["category"] == cat]
            if rows:
                st.markdown(f"**{cat}:** " + ", ".join(m["name"] for m in rows))
    else:
        st.success("Every CRITICAL topic has some supporting material.")


def _search() -> None:
    st.markdown("#### Search the knowledge base")
    query = st.text_input("Query", placeholder="e.g. why does batch normalization help")
    c1, c2 = st.columns([1, 1])
    with c1:
        source_filter = st.multiselect(
            "Restrict to sources", [s.value for s in SourceType],
            format_func=lambda s: s.replace("_", " ").title(),
        )
    with c2:
        k = st.slider("Results", 1, 12, 5)

    if query:
        result = rag.retrieve(query, k=k, source_types=source_filter or None)
        if not result.chunks:
            st.warning("Nothing found. The source material does not establish an answer "
                       "to this query.")
            return
        if not result.grounded:
            st.caption("⚠️ Weak matches only — treat these as loosely related.")
        for c in result.chunks:
            with st.container(border=True):
                st.markdown(f"<div class='ea-muted'>{c.citation.label()} · "
                            f"score {c.score:.3f}</div>", unsafe_allow_html=True)
                st.markdown(c.text[:1400])

        st.divider()
        st.markdown("#### University vs Udemy on this query")
        cmp = __import__("examagent.services.tutor", fromlist=["compare_sources_answer"]) \
            .compare_sources_answer(query)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**University**")
            for item in cmp["university"][:2]:
                st.caption(item["citation"])
                st.markdown(item["text"][:600])
            if not cmp["university"]:
                st.caption("_no university material_")
        with c2:
            st.markdown("**Udemy**")
            for item in cmp["udemy"][:2]:
                st.caption(item["citation"])
                st.markdown(item["text"][:600])
            if not cmp["udemy"]:
                st.caption("_no Udemy material_")
        st.info(cmp["conflict_note"])
