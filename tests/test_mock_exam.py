"""Mock exam construction, scoring and reporting."""
from __future__ import annotations

import pytest

from collections import Counter

from examagent.models.db import Attempt, MockExam, get_topic, session_scope
from examagent.models.schemas import Category, QuestionType, points_for
from examagent.services import mock_exam, progress
from examagent.services.question_gen import EXAM_TYPES


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

    # The paper is closed-form: True/False, single-best MCQ, multiple response.
    # Offline the bank can run dry for a pooled topic and degrade to a written
    # question; that is allowed, but must stay marginal - with an LLM every
    # item is generated in the requested format.
    closed = [q for q in questions if q.question_type in EXAM_TYPES]
    assert len(closed) >= 0.85 * len(questions), (
        f"the paper drifted off the real format: "
        f"{Counter(q.question_type.value for q in questions)}"
    )
    assert len({q.question_type for q in closed}) >= 2, "must mix the closed formats"

    # difficulty must be exam level
    assert all(q.difficulty >= 4 for q in questions)

    # ML and DL must both appear
    cats = {q.category for q in questions}
    assert cats == {Category.ML, Category.DL}

    assert len({q.id for q in questions}) == len(questions), "no duplicate questions"


def test_the_full_paper_matches_the_real_one(clean_db) -> None:
    """The announced paper: 60 questions in two 30-question parts, each part
    12 True/False and 18 single-best multiple choice. No multiple response -
    the practice PDF had a Section C, the sat paper does not."""
    exam = mock_exam.build_exam(
        n_questions=mock_exam.FULL_EXAM_QUESTIONS,
        duration_minutes=mock_exam.FULL_EXAM_MINUTES,
        use_llm=False, seed=13,
    )
    questions = exam["questions"]
    assert len(questions) == 60

    counts = Counter(q.question_type for q in questions)
    assert counts[QuestionType.TRUE_FALSE] == 24
    assert counts[QuestionType.MCQ] == 36
    assert QuestionType.MULTIPLE_RESPONSE not in counts

    assert Counter(q.category for q in questions)[Category.ML] == 30
    for category in (Category.ML, Category.DL):
        part = [q for q in questions if q.category == category]
        by_type = Counter(q.question_type for q in part)
        assert by_type[QuestionType.TRUE_FALSE] == 12, f"{category}: {by_type}"
        assert by_type[QuestionType.MCQ] == 18, f"{category}: {by_type}"

    report = mock_exam.submit_exam(
        exam["exam_id"], {q.id: q.correct_option for q in questions},
        duration_seconds=9000, use_llm=False,
    )
    expected = sum(points_for(q.question_type) for q in questions)
    assert report.max_score == expected
    assert report.total_score == expected
    assert report.pass_mark == pytest.approx(expected * 0.6)
    assert report.passed is True


def test_a_failing_paper_is_reported_as_below_the_pass_mark(clean_db) -> None:
    exam = mock_exam.build_exam(n_questions=12, use_llm=False, seed=21)
    blank = {q.id: "" for q in exam["questions"]}
    report = mock_exam.submit_exam(exam["exam_id"], blank, duration_seconds=600,
                                   use_llm=False)
    assert report.total_score == 0
    assert report.passed is False
    assert report.pass_mark == pytest.approx(report.max_score * 0.6)


def test_topic_ids_restricts_the_paper_to_that_pool(clean_db) -> None:
    allowed = ["backpropagation", "pca", "knn", "attention", "cnn_basics"]
    exam = mock_exam.build_exam(n_questions=10, use_llm=False, seed=5,
                                topic_ids=allowed)
    topics = {q.topic for q in exam["questions"]}
    assert topics, "the scoped pool must still produce questions"
    assert topics <= set(allowed), (
        f"question topics {topics} must all come from the allowed set {allowed}"
    )


def test_topic_ids_with_a_single_topic_still_builds_a_paper(clean_db) -> None:
    exam = mock_exam.build_exam(n_questions=6, use_llm=False, seed=6,
                                topic_ids=["backpropagation"])
    assert exam["questions"]
    assert {q.topic for q in exam["questions"]} == {"backpropagation"}


def test_short_exam_still_mixes_formats(clean_db) -> None:
    exam = mock_exam.build_exam(n_questions=8, duration_minutes=30, use_llm=False, seed=2)
    assert len(exam["questions"]) == 8
    assert len({q.question_type for q in exam["questions"]}) >= 2


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


def test_the_paper_runs_in_the_printed_order(clean_db) -> None:
    """Part 1 is Machine Learning, Part 2 Deep Learning, and inside each part
    the sections run A (True/False), B (multiple choice), C (multiple
    response) - not shuffled. Sitting them in the paper's order is part of
    practising the paper."""
    exam = mock_exam.build_exam(n_questions=60, duration_minutes=150,
                                use_llm=False, seed=13)
    questions = exam["questions"]

    # the running order, collapsed to one entry per (part, section) block
    blocks = []
    for q in questions:
        key = (q.category, q.question_type)
        if not blocks or blocks[-1] != key:
            blocks.append(key)

    assert blocks == [
        (Category.ML, QuestionType.TRUE_FALSE),
        (Category.ML, QuestionType.MCQ),
        (Category.DL, QuestionType.TRUE_FALSE),
        (Category.DL, QuestionType.MCQ),
    ], f"sections are out of order: {blocks}"

    ml = [q for q in questions if q.category == Category.ML]
    assert len(ml) == 30, "Part 1 is half the paper"
    assert questions[:30] == ml, "all of Part 1 comes before Part 2"


def test_a_short_paper_keeps_the_section_structure(clean_db) -> None:
    exam = mock_exam.build_exam(n_questions=12, use_llm=False, seed=4)
    questions = exam["questions"]
    assert len(questions) == 12

    blocks = []
    for q in questions:
        key = (q.category, q.question_type)
        if not blocks or blocks[-1] != key:
            blocks.append(key)
    assert len(blocks) == len(set(blocks)), (
        f"a section must appear once, not be revisited later: {blocks}"
    )


# ---------------------------------------------------------------- topic sweep
def test_a_sweep_covers_every_topic_it_is_given(clean_db) -> None:
    """The sweep exists because `build_exam` trades topics for format fidelity:
    it walks to a neighbouring topic when the planned section format does not
    come out. A practice sweep inverts that - the topic is the point."""
    from examagent.services.exam_formats import facts_for

    wanted = [t for t in ("pca", "backpropagation", "dropout", "knn", "attention")
              if facts_for(t)]
    sweep = mock_exam.build_topic_sweep(topic_ids=wanted, use_llm=False,
                                        label="sweep test")
    covered = {q.topic for q in sweep["questions"]}
    assert covered == set(wanted), f"missing {set(wanted) - covered}"


def test_a_sweep_only_carries_questions_that_mark_exactly(clean_db) -> None:
    """It is marked question by question, so a written item nobody can tick or
    cross has no place on it."""
    sweep = mock_exam.build_topic_sweep(use_llm=False, label="sweep formats")
    questions = sweep["questions"]
    assert questions
    assert all(q.question_type in EXAM_TYPES for q in questions)
    assert all(q.correct_option for q in questions)


def test_a_sweep_carries_no_clock(clean_db) -> None:
    sweep = mock_exam.build_topic_sweep(topic_ids=["pca"], use_llm=False)
    assert sweep["duration_minutes"] == 0


def test_a_sweep_is_persisted_and_can_be_submitted(clean_db) -> None:
    wanted = ["pca", "backpropagation", "dropout"]
    sweep = mock_exam.build_topic_sweep(topic_ids=wanted, use_llm=False,
                                        label="sweep submit")
    questions = sweep["questions"]
    report = mock_exam.submit_exam(
        sweep["exam_id"], {q.id: q.correct_option for q in questions},
        duration_seconds=0, use_llm=False,
    )
    assert report.percentage == 100.0
    assert report.max_score == sum(points_for(q.question_type) for q in questions)


# ------------------------------------------------------- work in progress
def test_no_saved_paper_on_a_clean_profile(clean_db) -> None:
    assert mock_exam.load_progress() is None


def test_an_unfinished_paper_survives_and_comes_back_whole(clean_db) -> None:
    """The point of saving it: a refresh or an app restart must not throw away
    a paper that took minutes to build and longer to answer."""
    sweep = mock_exam.build_topic_sweep(topic_ids=["pca", "dropout"], use_llm=False,
                                        label="resume test")
    questions = sweep["questions"]
    answers = {questions[0].id: questions[0].correct_option}
    mock_exam.save_progress(sweep["exam_id"], answers)

    # nothing in memory survives a restart - only what reached the database
    saved = mock_exam.load_progress()
    assert saved is not None
    assert saved["answers"] == answers
    assert [q.id for q in saved["exam"]["questions"]] == [q.id for q in questions]


def test_saving_again_replaces_the_previous_answers(clean_db) -> None:
    sweep = mock_exam.build_topic_sweep(topic_ids=["pca"], use_llm=False)
    first, second = {"a": "A"}, {"a": "A", "b": "B"}
    mock_exam.save_progress(sweep["exam_id"], first)
    mock_exam.save_progress(sweep["exam_id"], second)
    assert mock_exam.load_progress()["answers"] == second


def test_a_submitted_paper_is_no_longer_in_progress(clean_db) -> None:
    """Resuming a finished paper would put the student back into a paper they
    have already been marked on."""
    sweep = mock_exam.build_topic_sweep(topic_ids=["pca", "dropout"], use_llm=False)
    questions = sweep["questions"]
    mock_exam.save_progress(sweep["exam_id"], {q.id: q.correct_option for q in questions})
    mock_exam.submit_exam(sweep["exam_id"], {q.id: q.correct_option for q in questions},
                          duration_seconds=0, use_llm=False)
    assert mock_exam.load_progress() is None


def test_clear_progress_forgets_the_paper(clean_db) -> None:
    sweep = mock_exam.build_topic_sweep(topic_ids=["pca"], use_llm=False)
    mock_exam.save_progress(sweep["exam_id"], {"a": "A"})
    mock_exam.clear_progress()
    assert mock_exam.load_progress() is None


def test_progress_pointing_at_a_deleted_exam_is_dropped(clean_db) -> None:
    mock_exam.save_progress(999999, {"a": "A"})
    assert mock_exam.load_progress() is None
