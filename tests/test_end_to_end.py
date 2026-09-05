"""End-to-end journey covering the project's definition of done.

Walks the whole system the way the student would: upload material, see the map,
study, answer, get marked, have weaknesses updated, get a recommendation, take
assertion-reason and calculation questions, sit a timed mock exam, read the
report, and resume where they left off.
"""
from __future__ import annotations

import pytest

from examagent.models.db import get_topic, session_scope
from examagent.models.schemas import Category, QuestionType, SessionMode, SourceType
from examagent.services import (
    materials,
    mock_exam,
    planner,
    progress,
    rag,
    tutor,
    weakness,
)
from examagent.services.evaluator import evaluate
from examagent.services.question_gen import generate_question


def test_full_student_journey(clean_db, clean_vectorstore, tmp_docs) -> None:
    # 1-3. upload and ingest course materials -----------------------------
    r1 = materials.ingest_file(tmp_docs / "lecture3.md",
                               SourceType.UNIVERSITY_ML.value, lecture="Lecture 3")
    r2 = materials.ingest_file(tmp_docs / "udemy_regression.txt",
                               SourceType.UDEMY_ML.value)
    r3 = materials.ingest_file(tmp_docs / "exam_sample.md",
                               SourceType.EXAM_SAMPLES.value)
    assert all(r["status"] == "indexed" for r in (r1, r2, r3))
    assert materials.library_status()["chunks"] > 0

    # 4. the topic map exists and is honest about what is untested --------
    kmap = weakness.knowledge_map()
    assert set(kmap) == {"Machine Learning", "Deep Learning"}
    assert all(r["mastery"] == "Not tested" for rows in kmap.values() for r in rows)

    # 5-6. start a study session and receive an explanation ---------------
    suggestion = planner.next_topic()
    assert suggestion is not None
    topic_id = suggestion["topic_id"]
    lesson = tutor.build_lesson(topic_id, use_llm=False)
    assert lesson.as_markdown().strip()
    assert lesson.topic_name

    session_id = progress.start_session(SessionMode.THIRTY.value, topic_id)

    # 7-10. answer questions, get marked, weaknesses update ---------------
    q = generate_question(topic_id, QuestionType.CONCEPTUAL, 4, use_llm=False, seed=1)
    weak_answer = "It is a thing that helps the model be better."
    ev = evaluate(q, weak_answer, use_llm=False)
    assert ev.score < 5
    assert ev.improvement and ev.examiner_expects
    progress.record_attempt(q, ev, student_answer=weak_answer, context="study", seconds=60)

    with session_scope() as s:
        topic = get_topic(s, topic_id)
        assert topic.attempt_count == 1
        assert topic.mistake_count == 1
    assert weakness.error_log(topic_id=topic_id), "the mistake must be logged"

    # 11. a recommendation exists and is justified ------------------------
    report = progress.topic_report(topic_id)
    assert report["recommended_action"].strip()
    nxt = planner.next_topic()
    assert nxt and nxt["reason"].strip()

    # 12. assertion-reason -------------------------------------------------
    ar = generate_question("dropout", QuestionType.ASSERTION_REASON, 5,
                           use_llm=False, seed=2)
    assert ar.correct_option in {"A", "B", "C", "D", "E"}
    assert len(ar.options) == 5
    assert "true" not in ar.prompt.lower().split("assertion")[0]
    ar_ev = evaluate(ar, ar.correct_option, use_llm=False)
    assert ar_ev.score == 10.0
    progress.record_attempt(ar, ar_ev, student_answer=ar.correct_option, context="study")

    # 13. calculation problems --------------------------------------------
    calc = generate_question("backpropagation", QuestionType.CALCULATION, 6,
                             use_llm=False, seed=3)
    assert calc.calc_spec
    parts = calc.calc_spec["parts"]
    # answer half correctly -> partial credit, and a specific diagnosis
    answers = {p["key"]: str(p["answer"]) for p in parts[: len(parts) // 2]}
    calc_ev = evaluate(calc, answers, use_llm=False)
    assert 0 < calc_ev.score < 10
    assert calc_ev.sub_scores
    assert any(not s.correct for s in calc_ev.sub_scores)
    progress.record_attempt(calc, calc_ev, student_answer=str(answers), context="study")

    with session_scope() as s:
        bp = get_topic(s, "backpropagation")
        assert bp.calculation_score > 0
        assert bp.calculation_score < 1.0

    progress.end_session(session_id, seconds=1800, questions=3, mean_score=6.0)

    # 14-15. timed mock exam and a detailed report -------------------------
    exam = mock_exam.build_exam(n_questions=12, duration_minutes=45,
                                use_llm=False, seed=4)
    assert len(exam["questions"]) == 12
    exam_answers = {}
    for i, eq in enumerate(exam["questions"]):
        if eq.question_type == QuestionType.CALCULATION and eq.calc_spec:
            ps = eq.calc_spec["parts"]
            exam_answers[eq.id] = {p["key"]: str(p["answer"])
                                   for p in (ps if i % 2 == 0 else ps[:1])}
        elif eq.question_type == QuestionType.ASSERTION_REASON:
            exam_answers[eq.id] = eq.correct_option if i % 2 == 0 else "E"
        else:
            exam_answers[eq.id] = eq.model_answer if i % 2 == 0 else ""
    exam_report = mock_exam.submit_exam(exam["exam_id"], exam_answers,
                                        duration_seconds=2400, use_llm=False)
    assert 0 < exam_report.percentage < 100
    assert exam_report.by_dimension and exam_report.by_question_type
    assert exam_report.revision_plan

    # 16. strongest and weakest topics are visible -------------------------
    wr = weakness.weakness_report(limit=8)
    assert wr["weakest"]
    assert all("action" in row for row in wr["weakest"])

    # 17. the session resumes: state survives and readiness reflects it ----
    readiness = progress.compute_readiness()
    assert readiness.overall > 0
    assert readiness.exam_performance > 0
    assert readiness.coverage > 0

    snapshot = progress.dashboard_snapshot()
    assert snapshot["questions_today"] >= 3
    assert snapshot["open_mistakes"] >= 1
    assert snapshot["days_remaining"] >= 0

    plan = planner.build_plan()
    assert plan and plan[0].blocks


def test_app_is_fully_usable_without_an_llm(clean_db) -> None:
    """The offline path must cover every question type the exam uses."""
    from examagent.services.llm import get_llm

    assert not get_llm().available, "this test must run in offline mode"

    produced = {}
    for qtype in (QuestionType.CALCULATION, QuestionType.ASSERTION_REASON,
                  QuestionType.CONCEPTUAL, QuestionType.WHAT_IF,
                  QuestionType.COMPARISON, QuestionType.SCENARIO,
                  QuestionType.DIAGRAM, QuestionType.GRAPH):
        q = generate_question("backpropagation", qtype, 5, use_llm=True, seed=1)
        assert q.prompt.strip(), f"{qtype.value} produced an empty prompt"
        produced[qtype] = q.source_basis
    # the deterministic engines must have handled the two exam-critical formats
    assert produced[QuestionType.CALCULATION] == "calculation engine"
    assert produced[QuestionType.ASSERTION_REASON] == "bank"


def test_chat_commands_drive_the_whole_app(clean_db) -> None:
    checks = [
        ("/help", "text", None),
        ("/study", "navigate", "Study"),
        ("/weakness", "text", None),
        ("/progress", "text", None),
        ("/mock", "navigate", "Mock Exam"),
        ("/calculate backpropagation", "question", None),
        ("/assertion dropout", "question", None),
        ("/review", "text", None),
    ]
    for text, kind, nav in checks:
        result = tutor.route_command(text, use_llm=False)
        assert result.kind == kind, f"{text} -> {result.kind}"
        if nav:
            assert result.navigate == nav
        if kind == "question":
            assert result.question is not None
            assert result.question.prompt.strip()


def test_natural_language_requests_are_understood(clean_db) -> None:
    cases = {
        "teach me PCA": ("navigate", "Study"),
        "quiz me on CNN": ("question", None),
        "give me a backpropagation calculation": ("question", None),
        "give me a mock exam": ("navigate", "Mock Exam"),
        "show my weakest topics": ("text", None),
        "what should I study now": ("navigate", "Study"),
        "give me a 30 minute session": ("navigate", "Study"),
    }
    for text, (kind, nav) in cases.items():
        result = tutor.route_command(text, use_llm=False)
        assert result.kind == kind, f"{text!r} -> {result.kind}"
        if nav:
            assert result.navigate == nav


def test_no_hallucination_without_material(clean_db, clean_vectorstore) -> None:
    """With no sources and no LLM, the app must say so rather than invent facts."""
    answer = tutor.explain("What does lecture 7 of this course say about GMMs?",
                           use_llm=False)
    assert not answer["grounded"]
    assert answer["citations"] == []
    text = answer["answer"].lower()
    assert "does not establish" in text or "no llm" in text or "upload" in text


def test_citations_are_real_and_never_invented(clean_db, clean_vectorstore,
                                               tmp_docs) -> None:
    materials.ingest_file(tmp_docs / "lecture3.md", SourceType.UNIVERSITY_ML.value,
                          lecture="Lecture 3")
    result = rag.retrieve("cross validation folds", k=3)
    assert result.grounded
    for citation in result.citations():
        assert citation.source_type == "UNIVERSITY_ML"
        assert citation.source_name
        assert citation.lecture == "Lecture 3"
        # the cited text must actually exist in the store
        assert any(c.citation.label() == citation.label() for c in result.chunks)


def test_progress_is_durable_across_engine_restarts(clean_db) -> None:
    from examagent.models.db import reset_engine
    from examagent.models.schemas import Evaluation

    q = generate_question("attention", QuestionType.CONCEPTUAL, 5, use_llm=False, seed=1)
    progress.record_attempt(q, Evaluation(score=8.0, correct=True),
                            student_answer="ok", context="quiz")
    with session_scope() as s:
        before = get_topic(s, "attention").reasoning_score

    reset_engine()  # simulate closing and reopening the app

    with session_scope() as s:
        assert get_topic(s, "attention").reasoning_score == before
        assert get_topic(s, "attention").attempt_count == 1
