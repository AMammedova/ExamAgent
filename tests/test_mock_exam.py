"""Mock exam construction, scoring and reporting."""
from __future__ import annotations

import pytest

from examagent.models.db import Attempt, MockExam, get_topic, session_scope
from examagent.models.schemas import Category, QuestionType
from examagent.services import mock_exam, progress


def _answer_sheet(questions, quality: str = "perfect") -> dict:
    """Build an answer sheet of a given quality."""
    answers: dict = {}
    for i, q in enumerate(questions):
        if q.question_type == QuestionType.CALCULATION and q.calc_spec:
            parts = q.calc_spec["parts"]
            if quality == "perfect":
                answers[q.id] = {p["key"]: str(p["answer"]) for p in parts}
            elif quality == "blank":
                answers[q.id] = {}
            else:
                answers[q.id] = {p["key"]: str(p["answer"]) for p in parts[:1]}
        elif q.question_type == QuestionType.ASSERTION_REASON:
            if quality == "perfect":
                answers[q.id] = q.correct_option
            elif quality == "blank":
                answers[q.id] = ""
            else:
                answers[q.id] = q.correct_option if i % 2 == 0 else "E"
        else:
            if quality == "perfect":
                answers[q.id] = q.model_answer or "x" * 200
            elif quality == "blank":
                answers[q.id] = ""
            else:
                answers[q.id] = "It helps the model work better overall."
    return answers


def test_exam_follows_the_blueprint(clean_db) -> None:
    exam = mock_exam.build_exam(n_questions=18, duration_minutes=75,
                                use_llm=False, seed=1)
    questions = exam["questions"]
    assert len(questions) == 18

    types = {q.question_type for q in questions}
    assert QuestionType.ASSERTION_REASON in types
    assert QuestionType.CALCULATION in types
    assert len(types) >= 4, "the paper must mix question formats"

    # difficulty must be exam level
    assert all(q.difficulty >= 4 for q in questions)

    # ML and DL must both appear
    cats = {q.category for q in questions}
    assert cats == {Category.ML, Category.DL}

    assert len({q.id for q in questions}) == len(questions), "no duplicate questions"


def test_short_exam_still_mixes_formats(clean_db) -> None:
    exam = mock_exam.build_exam(n_questions=8, duration_minutes=30, use_llm=False, seed=2)
    assert len(exam["questions"]) == 8
    assert len({q.question_type for q in exam["questions"]}) >= 3


def test_exam_questions_never_leak_answers(clean_db) -> None:
    exam = mock_exam.build_exam(n_questions=12, use_llm=False, seed=3)
    for q in exam["questions"]:
        low = q.prompt.lower()
        assert "model answer" not in low
        assert "correct answer is" not in low
        if q.question_type == QuestionType.CALCULATION:
            # the worked solution must live in model_answer, not the prompt
            assert q.model_answer
            assert q.model_answer not in q.prompt


def test_exam_is_persisted_and_reloadable(clean_db) -> None:
    exam = mock_exam.build_exam(n_questions=6, use_llm=False, seed=4)
    loaded = mock_exam.load_exam(exam["exam_id"])
    assert loaded is not None
    assert len(loaded["questions"]) == len(exam["questions"])
    assert loaded["questions"][0].id == exam["questions"][0].id
    assert not loaded["completed"]


def test_perfect_paper_scores_near_full_marks(clean_db) -> None:
    exam = mock_exam.build_exam(n_questions=10, use_llm=False, seed=5)
    answers = _answer_sheet(exam["questions"], "perfect")
    report = mock_exam.submit_exam(exam["exam_id"], answers, duration_seconds=1800,
                                   use_llm=False)
    assert report.percentage > 75
    assert report.total_score <= report.max_score
    assert report.top_strengths


def test_blank_paper_scores_zero(clean_db) -> None:
    exam = mock_exam.build_exam(n_questions=8, use_llm=False, seed=6)
    report = mock_exam.submit_exam(exam["exam_id"],
                                   _answer_sheet(exam["questions"], "blank"),
                                   duration_seconds=600, use_llm=False)
    assert report.percentage == 0.0
    assert report.top_weaknesses
    assert report.immediate_revision


def test_report_breaks_performance_down(clean_db) -> None:
    exam = mock_exam.build_exam(n_questions=14, use_llm=False, seed=7)
    report = mock_exam.submit_exam(exam["exam_id"],
                                   _answer_sheet(exam["questions"], "mixed"),
                                   duration_seconds=2400, use_llm=False)
    assert 0 <= report.percentage <= 100
    assert report.by_dimension
    assert report.by_question_type
    assert all(0 <= v <= 100 for v in report.by_dimension.values())
    assert all(0 <= v <= 100 for v in report.by_question_type.values())
    assert report.revision_plan, "a report must always tell the student what to do next"
    assert len(report.top_weaknesses) <= 5
    assert len(report.top_strengths) <= 5


def test_submission_updates_the_knowledge_profile(clean_db) -> None:
    exam = mock_exam.build_exam(n_questions=8, use_llm=False, seed=8)
    topics = {q.topic for q in exam["questions"]}
    with session_scope() as s:
        before = {t: get_topic(s, t).attempt_count for t in topics}

    mock_exam.submit_exam(exam["exam_id"], _answer_sheet(exam["questions"], "perfect"),
                          duration_seconds=1200, use_llm=False)

    with session_scope() as s:
        for t in topics:
            assert get_topic(s, t).attempt_count > before[t]
        recorded = s.query(Attempt).filter(Attempt.context == "mock").count()
    assert recorded == len(exam["questions"])


def test_exam_attempts_are_linked_to_the_exam(clean_db) -> None:
    exam = mock_exam.build_exam(n_questions=6, use_llm=False, seed=9)
    mock_exam.submit_exam(exam["exam_id"], _answer_sheet(exam["questions"], "mixed"),
                          use_llm=False)
    with session_scope() as s:
        rows = s.query(Attempt).filter(Attempt.exam_id == exam["exam_id"]).all()
    assert len(rows) == 6


def test_completed_exam_is_marked_and_stored(clean_db) -> None:
    exam = mock_exam.build_exam(n_questions=6, use_llm=False, seed=10)
    report = mock_exam.submit_exam(exam["exam_id"],
                                   _answer_sheet(exam["questions"], "mixed"),
                                   use_llm=False)
    with session_scope() as s:
        row = s.get(MockExam, exam["exam_id"])
        assert row.completed
        assert row.finished_at is not None
        assert row.percentage == report.percentage

    latest = mock_exam.latest_report()
    assert latest is not None
    assert latest.exam_id == exam["exam_id"]

    history = mock_exam.exam_history()
    assert history and history[0]["completed"]


def test_exam_performance_feeds_the_readiness_score(clean_db) -> None:
    exam = mock_exam.build_exam(n_questions=10, use_llm=False, seed=11)
    mock_exam.submit_exam(exam["exam_id"], _answer_sheet(exam["questions"], "perfect"),
                          use_llm=False)
    readiness = progress.compute_readiness()
    assert readiness.exam_performance > 0.5


def test_dangerous_gaps_are_critical_topics_only(clean_db) -> None:
    exam = mock_exam.build_exam(n_questions=12, use_llm=False, seed=12)
    report = mock_exam.submit_exam(exam["exam_id"],
                                   _answer_sheet(exam["questions"], "blank"),
                                   use_llm=False)
    for gap in report.dangerous_gaps:
        assert "CRITICAL" in gap


def test_ml_dl_scores_are_reported_separately(clean_db) -> None:
    exam = mock_exam.build_exam(n_questions=12, use_llm=False, seed=13,
                                balance_ml_dl=True)
    answers = {}
    for q in exam["questions"]:
        # answer the ML questions well and the DL questions not at all
        good = q.category == Category.ML
        if q.question_type == QuestionType.CALCULATION and q.calc_spec:
            answers[q.id] = ({p["key"]: str(p["answer"]) for p in q.calc_spec["parts"]}
                             if good else {})
        elif q.question_type == QuestionType.ASSERTION_REASON:
            answers[q.id] = q.correct_option if good else ""
        else:
            answers[q.id] = (q.model_answer or "x" * 200) if good else ""
    report = mock_exam.submit_exam(exam["exam_id"], answers, use_llm=False)
    assert report.ml_score > report.dl_score
    assert any("Deep Learning" in line for line in report.revision_plan)


def test_unknown_exam_id_raises(clean_db) -> None:
    with pytest.raises(ValueError):
        mock_exam.submit_exam(999999, {}, use_llm=False)
