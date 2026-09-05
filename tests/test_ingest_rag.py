"""Document ingestion, chunking, retrieval and source-priority tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from examagent.models.schemas import SourceType
from examagent.services import materials, rag
from examagent.services.ingest import (
    RawSection,
    chunk_sections,
    clean_text,
    detect_topics,
    file_hash,
    load_document,
    prepare_document,
)


# ---------------------------------------------------------------- loading
def test_load_markdown_splits_on_headings(tmp_docs: Path) -> None:
    sections = load_document(tmp_docs / "lecture3.md")
    assert len(sections) >= 2
    headings = [s.section for s in sections if s.section]
    assert any("Cross Validation" in h for h in headings)


def test_load_plain_text(tmp_docs: Path) -> None:
    sections = load_document(tmp_docs / "udemy_regression.txt")
    assert sections
    assert "least squares" in sections[0].text.lower()


def test_unsupported_extension_raises(tmp_path: Path) -> None:
    bad = tmp_path / "x.exe"
    bad.write_bytes(b"\x00\x01")
    with pytest.raises(ValueError):
        load_document(bad)


def test_clean_text_dehyphenates_and_normalises() -> None:
    raw = "regu-\nlarization   is    useful\n\n\n\nreally"
    out = clean_text(raw)
    assert "regularization" in out
    assert "    " not in out
    assert "\n\n\n" not in out


# ---------------------------------------------------------------- chunking
def test_chunking_respects_size_and_keeps_metadata() -> None:
    body = ("Backpropagation applies the chain rule in reverse. " * 60)
    sections = [RawSection(body, page=4, section="Lecture 2")]
    chunks = chunk_sections(sections, {"source_type": "UNIVERSITY_DL",
                                       "source_name": "dl2"},
                            chunk_size=400, overlap=60)
    assert len(chunks) > 1
    for c in chunks:
        assert c.metadata["page"] == 4
        assert c.metadata["section"] == "Lecture 2"
        assert c.metadata["source_type"] == "UNIVERSITY_DL"
        assert len(c.text) < 1200


def test_chunks_never_span_pages() -> None:
    sections = [
        RawSection("Page one content about convolution and stride. " * 8, page=1),
        RawSection("Page two content about attention and softmax. " * 8, page=2),
    ]
    chunks = chunk_sections(sections, {"source_type": "UNIVERSITY_DL",
                                       "source_name": "x"}, chunk_size=2000, overlap=50)
    for c in chunks:
        text = c.text.lower()
        if c.metadata["page"] == 1:
            assert "attention" not in text
        if c.metadata["page"] == 2:
            assert "convolution" not in text


def test_chunk_ids_are_unique() -> None:
    sections = [RawSection("Gradient descent updates weights. " * 40, page=1)]
    chunks = chunk_sections(sections, {"source_type": "UNIVERSITY_DL",
                                       "source_name": "x"}, chunk_size=200, overlap=20)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_empty_and_tiny_sections_are_dropped() -> None:
    chunks = chunk_sections([RawSection("   "), RawSection("short")],
                            {"source_type": "STUDENT_NOTES", "source_name": "n"})
    assert chunks == []


# ---------------------------------------------------------------- tagging
def test_detect_topics_finds_the_right_topics() -> None:
    topics = detect_topics(
        "The convolution kernel slides with a stride and padding to produce feature maps."
    )
    assert any(t in topics for t in ("convolution", "cnn_basics", "stride_padding"))


def test_detect_topics_empty_for_unrelated_text() -> None:
    assert detect_topics("The weather today is pleasant and mild.") == []


def test_file_hash_is_content_based(tmp_path: Path) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("identical", encoding="utf-8")
    b.write_text("identical", encoding="utf-8")
    assert file_hash(a) == file_hash(b)


def test_prepare_document_returns_summary(tmp_docs: Path) -> None:
    chunks, summary = prepare_document(tmp_docs / "lecture3.md", "UNIVERSITY_ML",
                                       lecture="Lecture 3")
    assert chunks
    assert summary["n_chunks"] == len(chunks)
    assert summary["n_chars"] > 0
    assert summary["hash"]
    assert chunks[0].metadata["lecture"] == "Lecture 3"


# ---------------------------------------------------------------- retrieval
def test_ingest_then_retrieve(clean_db, clean_vectorstore, tmp_docs: Path) -> None:
    r = materials.ingest_file(tmp_docs / "lecture3.md", SourceType.UNIVERSITY_ML.value,
                              lecture="Lecture 3")
    assert r["status"] == "indexed"
    assert r["chunks"] > 0

    result = rag.retrieve("k fold cross validation data leakage", k=3)
    assert result.chunks
    assert result.grounded
    top = result.chunks[0]
    assert "fold" in top.text.lower()
    assert top.citation.source_type == "UNIVERSITY_ML"
    assert top.citation.lecture == "Lecture 3"
    assert "Lecture 3" in top.citation.label()


def test_duplicate_ingestion_is_skipped(clean_db, clean_vectorstore, tmp_docs: Path) -> None:
    first = materials.ingest_file(tmp_docs / "lecture3.md", SourceType.UNIVERSITY_ML.value)
    second = materials.ingest_file(tmp_docs / "lecture3.md", SourceType.UNIVERSITY_ML.value)
    assert first["status"] == "indexed"
    assert second["status"] == "duplicate"
    assert clean_vectorstore.count() == first["chunks"]


def test_university_material_outranks_udemy(clean_db, clean_vectorstore,
                                            tmp_path: Path) -> None:
    """Same content from two sources: the university copy must rank first."""
    uni = tmp_path / "uni_reg.md"
    ude = tmp_path / "udemy_reg.md"
    text = ("Linear regression fits a line by ordinary least squares, minimising the "
            "sum of squared residuals between predictions and targets. " * 3)
    uni.write_text(text, encoding="utf-8")
    ude.write_text(text + "\nPractical note.", encoding="utf-8")

    materials.ingest_file(ude, SourceType.UDEMY_ML.value)
    materials.ingest_file(uni, SourceType.UNIVERSITY_ML.value)

    result = rag.retrieve("ordinary least squares residuals", k=4)
    assert result.chunks
    assert result.chunks[0].citation.source_type == "UNIVERSITY_ML"


def test_source_filtering(clean_db, clean_vectorstore, tmp_docs: Path) -> None:
    materials.ingest_file(tmp_docs / "lecture3.md", SourceType.UNIVERSITY_ML.value)
    materials.ingest_file(tmp_docs / "udemy_regression.txt", SourceType.UDEMY_ML.value)

    only_udemy = rag.retrieve("regression least squares", k=5,
                              source_types=[SourceType.UDEMY_ML.value])
    assert only_udemy.chunks
    assert all(c.citation.source_type == "UDEMY_ML" for c in only_udemy.chunks)

    only_exam = rag.retrieve("regression", k=5,
                             source_types=[SourceType.EXAM_SAMPLES.value])
    assert not only_exam.chunks or all(
        c.citation.source_type == "EXAM_SAMPLES" for c in only_exam.chunks)


def test_empty_knowledge_base_is_not_grounded(clean_db, clean_vectorstore) -> None:
    result = rag.retrieve("anything at all", k=3)
    assert result.chunks == []
    assert not result.grounded


def test_irrelevant_query_is_not_grounded(clean_db, clean_vectorstore,
                                          tmp_docs: Path) -> None:
    materials.ingest_file(tmp_docs / "lecture3.md", SourceType.UNIVERSITY_ML.value)
    result = rag.retrieve("medieval french poetry and cheese", k=3)
    assert not result.grounded


def test_compare_sources_flags_the_precedence(clean_db, clean_vectorstore,
                                              tmp_docs: Path) -> None:
    from examagent.services.tutor import compare_sources_answer

    materials.ingest_file(tmp_docs / "lecture3.md", SourceType.UNIVERSITY_ML.value)
    materials.ingest_file(tmp_docs / "udemy_regression.txt", SourceType.UDEMY_ML.value)
    out = compare_sources_answer("linear regression least squares")
    assert out["conflict_note"]
    assert "university" in out["conflict_note"].lower()


def test_coverage_and_library_status(clean_db, clean_vectorstore, tmp_docs: Path) -> None:
    materials.ingest_file(tmp_docs / "lecture3.md", SourceType.UNIVERSITY_ML.value)
    status = materials.library_status()
    assert status["chunks"] > 0
    assert status["topics_with_material"] >= 1
    assert isinstance(status["missing_critical"], list)
    # every source type that was never uploaded should be reported as missing
    assert SourceType.UDEMY_DL.value in status["missing_sources"]


def test_document_delete_removes_chunks(clean_db, clean_vectorstore,
                                        tmp_docs: Path) -> None:
    materials.ingest_file(tmp_docs / "lecture3.md", SourceType.UNIVERSITY_ML.value)
    before = clean_vectorstore.count()
    assert before > 0
    doc_id = materials.documents()[0]["id"]
    assert materials.delete_document(doc_id)
    assert clean_vectorstore.count() < before
    assert materials.documents() == []


def test_context_block_and_citations(clean_db, clean_vectorstore, tmp_docs: Path) -> None:
    materials.ingest_file(tmp_docs / "exam_sample.md", SourceType.EXAM_SAMPLES.value)
    result = rag.retrieve_exam_style("MLP backpropagation", k=3)
    assert result.chunks
    block = result.context_block(2000)
    assert "[S1]" in block
    assert len(block) <= 2200
    assert result.citations()[0].source_type == "EXAM_SAMPLES"
