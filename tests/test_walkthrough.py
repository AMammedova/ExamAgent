"""Document walkthrough: ordering, per-document progress, offline brief and question."""
from __future__ import annotations

from pathlib import Path

import pytest

from examagent.models.schemas import QuestionType, SourceType
from examagent.services import materials, walkthrough
from examagent.services.evaluator import evaluate


@pytest.fixture()
def ingested(clean_db, clean_vectorstore, tmp_docs: Path) -> str:
    """One indexed lecture; returns its filename."""
    result = materials.ingest_file(
        tmp_docs / "lecture3.md", SourceType.UNIVERSITY_ML.value,
        source_name="Lecture 3", lecture="Lecture 3 — Model Validation",
    )
    assert result["status"] == "indexed", result
    return "lecture3.md"


# ---------------------------------------------------------------- sections
def test_sections_are_returned_in_document_order(ingested: str) -> None:
    secs = walkthrough.sections(ingested)
    assert secs, "an indexed document must yield sections"
    assert [s.index for s in secs] == list(range(len(secs)))
    # the markdown's first heading precedes the second in the document
    joined = " ".join(s.text for s in secs)
    assert joined.index("training, validation and test") < joined.index("K-fold")


def test_sections_carry_a_citation_and_label(ingested: str) -> None:
    section = walkthrough.sections(ingested)[0]
    assert section.citation is not None
    assert section.citation.source_type == SourceType.UNIVERSITY_ML.value
    assert section.label
    assert section.heading


def test_unknown_document_has_no_sections(clean_db, clean_vectorstore) -> None:
    assert walkthrough.sections("nothing_here.md") == []


# ---------------------------------------------------------------- grouping
def _slide(n: int, chars: int) -> walkthrough.Section:
    return walkthrough.Section(
        chunk_id=f"c{n}", text="word " * (chars // 5), label=f"Slide {n}", page=n,
        topics=[f"t{n}"],
    )


def test_short_slides_are_merged_into_answerable_steps() -> None:
    steps = walkthrough._group([_slide(n, 200) for n in range(1, 13)])
    assert len(steps) < 12, "one question per bullet is not a study session"
    for s in steps:
        assert len(s.text) >= walkthrough.MIN_STEP_CHARS * 0.5


def test_a_merged_step_reports_its_page_range_and_keeps_the_first_id() -> None:
    parts = [_slide(n, 200) for n in range(1, 6)]
    step = walkthrough._merge(parts)
    assert step.label == "Slides 1-5"
    assert step.heading == "Slides 1-5 · p.1-5"
    assert step.chunk_id == "c1", "progress must key on a stable id"
    assert step.topics == ["t1", "t2", "t3", "t4"]


def test_a_long_section_stands_on_its_own_step() -> None:
    steps = walkthrough._group([
        _slide(1, walkthrough.MIN_STEP_CHARS + 500),
        _slide(2, walkthrough.MIN_STEP_CHARS + 500),
    ])
    assert [s.label for s in steps] == ["Slide 1", "Slide 2"]


def test_a_stub_tail_joins_the_previous_step_instead_of_standing_alone() -> None:
    steps = walkthrough._group([
        _slide(1, walkthrough.MIN_STEP_CHARS + 500),
        _slide(2, 150),
    ])
    assert len(steps) == 1, "a 150-character orphan is not worth its own question"
    assert steps[0].chunk_id == "c1"
    assert steps[0].heading == "Slides 1-2 · p.1-2"


def test_grouping_never_exceeds_the_ceiling_mid_document() -> None:
    steps = walkthrough._group([_slide(n, 800) for n in range(1, 10)])
    assert all(len(s.text) <= walkthrough.MAX_STEP_CHARS + 900 for s in steps)


# ---------------------------------------------------------------- progress
def test_recording_a_score_advances_the_overview(ingested: str) -> None:
    secs = walkthrough.sections(ingested)
    before = walkthrough.overview(ingested, secs)
    assert before["done"] == 0
    assert before["remaining"] == before["total"]

    walkthrough.record(ingested, secs[0].chunk_id, 8.0)
    after = walkthrough.overview(ingested, secs)
    assert after["done"] == 1
    assert after["mean_score"] == pytest.approx(8.0)
    assert after["fraction"] == pytest.approx(1 / len(secs))


def test_progress_survives_a_fresh_read(ingested: str) -> None:
    secs = walkthrough.sections(ingested)
    walkthrough.record(ingested, secs[0].chunk_id, 6.5)
    assert walkthrough.load_state(ingested)["scores"][secs[0].chunk_id] == 6.5


def test_reanswering_overwrites_rather_than_appends(ingested: str) -> None:
    secs = walkthrough.sections(ingested)
    walkthrough.record(ingested, secs[0].chunk_id, 3.0)
    walkthrough.record(ingested, secs[0].chunk_id, 9.0)
    over = walkthrough.overview(ingested, secs)
    assert over["done"] == 1
    assert over["mean_score"] == pytest.approx(9.0)
    assert over["weak_sections"] == []


def test_weak_sections_are_the_ones_under_six(ingested: str) -> None:
    secs = walkthrough.sections(ingested)
    walkthrough.record(ingested, secs[0].chunk_id, 4.0)
    weak = walkthrough.overview(ingested, secs)["weak_sections"]
    assert [s.chunk_id for s in weak] == [secs[0].chunk_id]


def test_next_index_skips_answered_sections(ingested: str) -> None:
    secs = walkthrough.sections(ingested)
    assert walkthrough.next_index(ingested, secs) == 0
    walkthrough.record(ingested, secs[0].chunk_id, 7.0)
    assert walkthrough.next_index(ingested, secs) == min(1, len(secs) - 1)


def test_flagging_toggles_and_a_good_score_clears_it(ingested: str) -> None:
    secs = walkthrough.sections(ingested)
    cid = secs[0].chunk_id
    assert walkthrough.toggle_flag(ingested, cid) is True
    assert cid in walkthrough.load_state(ingested)["flagged"]
    assert walkthrough.toggle_flag(ingested, cid) is False

    walkthrough.toggle_flag(ingested, cid)
    walkthrough.record(ingested, cid, 8.0)
    assert cid not in walkthrough.load_state(ingested)["flagged"]


def test_reset_clears_the_document(ingested: str) -> None:
    secs = walkthrough.sections(ingested)
    walkthrough.record(ingested, secs[0].chunk_id, 8.0)
    walkthrough.reset(ingested)
    assert walkthrough.overview(ingested, secs)["done"] == 0


def test_documents_with_progress_lists_only_indexed_documents(ingested: str) -> None:
    rows = walkthrough.documents_with_progress(materials.documents())
    assert [r["filename"] for r in rows] == [ingested]
    assert rows[0]["progress"]["total"] > 0


# ---------------------------------------------------------------- brief
def test_offline_brief_is_extracted_verbatim(ingested: str) -> None:
    section = walkthrough.sections(ingested)[0]
    brief = walkthrough.brief(section, use_llm=False)

    assert brief["verbatim"] is True
    assert brief["source"] == "extract"
    assert brief["points"], "a readable section must yield at least one point"
    for point in brief["points"]:
        assert point in section.text, "offline briefs must never paraphrase"


def test_key_sentences_prefer_definitions_and_causes() -> None:
    text = (
        "This slide is about several things. "
        "Cross validation is a resampling procedure used to estimate generalisation. "
        "It lowers variance because every fold serves as validation exactly once."
    )
    picked = walkthrough.key_sentences(text, limit=2)
    assert any("is a resampling procedure" in p for p in picked)
    assert any("because" in p for p in picked)


def test_concepts_are_drawn_from_the_topic_registry(ingested: str) -> None:
    section = next(s for s in walkthrough.sections(ingested)
                   if "cross validation" in s.text.lower())
    concepts = walkthrough.concepts_in(section.text)
    assert concepts
    assert any("cross validation" in c.lower() or "validation" in c.lower()
               for c in concepts)


# ---------------------------------------------------------------- questions
def test_offline_question_is_about_this_section_and_is_markable(ingested: str) -> None:
    section = walkthrough.sections(ingested)[0]
    q = walkthrough.question_for(section, use_llm=False)

    assert q.question_type == QuestionType.CONCEPTUAL
    assert section.heading in q.prompt
    assert q.expected_concepts, "the offline evaluator needs a rubric to grade against"
    assert q.source_basis == "walkthrough"

    ev = evaluate(q, "A vague non-answer.", use_llm=False)
    assert 0.0 <= ev.score <= 10.0


def test_a_section_matching_no_topic_still_produces_a_question() -> None:
    section = walkthrough.Section(
        chunk_id="c1",
        text="Administrative notes about the timetable and the room number for the class.",
        label="Admin",
    )
    q = walkthrough.question_for(section, use_llm=False)
    assert q.prompt
    assert q.topic == "", "no registry topic should be claimed for unrelated material"
