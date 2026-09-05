"""Score updates, weakness detection, spaced repetition, planning, persistence."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from examagent.models.db import (
    Attempt,
    Mistake,
    Topic,
    all_topics,
    get_topic,
    session_scope,
)
from examagent.models.schemas import (
    Category,
    Evaluation,
    Mastery,
    MistakeType,
    Priority,
    Question,
    QuestionType,
    SessionMode,
)
from examagent.services import planner, progress, weakness
from examagent.services.question_gen import generate_question


def _question(topic: str, qtype: QuestionType, difficulty: int = 4) -> Question:
    return Question(
        id=f"test:{topic}:{qtype.value}:{difficulty}",
        topic=topic,
        category=Category.DL,
        question_type=qtype,
        difficulty=difficulty,
        priority=Priority.CRITICAL,
        prompt="test prompt",
    )


def _record(topic: str, qtype: QuestionType, score: float, difficulty: int = 4,
            context: str = "quiz") -> None:
    q = _question(topic, qtype, difficulty)
    ev = Evaluation(score=score, correct=score >= 7, partial=3 <= score < 7,
                    mistake_type=MistakeType.NONE if score >= 7 else MistakeType.CONCEPTUAL,
                    severity="Low" if score >= 7 else "High")
    progress.record_attempt(q, ev, student_answer="answer", context=context, seconds=60)


# ---------------------------------------------------------------- seeding
def test_topics_are_seeded_once(clean_db) -> None:
    with session_scope() as s:
        before = len(all_topics(s))
        added = progress.ensure_topics(s)
    assert added == 0, "ensure_topics must be idempotent"
    assert before >= 90


def test_topic_graph_has_no_dangling_prerequisites(clean_db) -> None:
    with session_scope() as s:
        topics = all_topics(s)
        ids = {t.id for t in topics}
        for t in topics:
            for p in t.prereq_ids:
                assert p in ids, f"{t.id} depends on unknown topic {p}"


# ---------------------------------------------------------------- scoring
def test_attempt_updates_the_right_dimension(clean_db) -> None:
    _record("backpropagation", QuestionType.CALCULATION, 9.0)
    with session_scope() as s:
        t = get_topic(s, "backpropagation")
        assert t.calculation_score > 0.6
        assert t.concept_score == 0.0, "only the tested dimension may move"
        assert t.attempt_count == 1
        assert t.mistake_count == 0
        assert t.last_reviewed is not None
        assert t.next_review is not None


def test_wrong_answers_increment_the_mistake_count(clean_db) -> None:
    _record("pca", QuestionType.CONCEPTUAL, 2.0)
    with session_scope() as s:
        t = get_topic(s, "pca")
        assert t.mistake_count == 1
        assert t.reasoning_score < 0.35
    assert len(weakness.error_log(topic_id="pca")) == 1


def test_scores_are_a_moving_average_not_a_replacement(clean_db) -> None:
    _record("attention", QuestionType.CONCEPTUAL, 10.0)
    with session_scope() as s:
        first = get_topic(s, "attention").reasoning_score
    _record("attention", QuestionType.CONCEPTUAL, 0.0)
    with session_scope() as s:
        second = get_topic(s, "attention").reasoning_score
    assert 0 < second < first, "one bad answer must lower but not erase the estimate"


def test_harder_questions_carry_more_weight(clean_db) -> None:
    _record("lstm", QuestionType.CONCEPTUAL, 8.0, difficulty=6)
    _record("gru", QuestionType.CONCEPTUAL, 8.0, difficulty=2)
    with session_scope() as s:
        hard = get_topic(s, "lstm").reasoning_score
        easy = get_topic(s, "gru").reasoning_score
    assert hard > easy


def test_overall_ignores_untested_dimensions(clean_db) -> None:
    _record("kmeans", QuestionType.CALCULATION, 9.0)
    with session_scope() as s:
        t = get_topic(s, "kmeans")
        assert t.overall() > 0.6, "an untested dimension must not count as zero"
        assert t.tested_dimensions() == 1
        assert t.weakest_dimension() == "calculation"
        assert "concept" in t.untested_dimensions()


def test_mastery_thresholds() -> None:
    assert Mastery.from_score(0.95) == Mastery.MASTERED
    assert Mastery.from_score(0.75) == Mastery.STRONG
    assert Mastery.from_score(0.60) == Mastery.MEDIUM
    assert Mastery.from_score(0.40) == Mastery.WEAK
    assert Mastery.from_score(0.10) == Mastery.CRITICAL_WEAKNESS


# ---------------------------------------------------------------- spacing
def test_weak_topics_return_sooner_than_strong_ones(clean_db) -> None:
    _record("dropout", QuestionType.CONCEPTUAL, 10.0)
    _record("rnn", QuestionType.CONCEPTUAL, 1.0)
    with session_scope() as s:
        strong = get_topic(s, "dropout").next_review
        weak = get_topic(s, "rnn").next_review
    assert weak < strong


def test_critical_topics_return_sooner_than_low_priority_ones(clean_db) -> None:
    _record("backpropagation", QuestionType.CONCEPTUAL, 6.0)   # CRITICAL
    _record("eclat", QuestionType.CONCEPTUAL, 6.0)             # LOW
    with session_scope() as s:
        crit = get_topic(s, "backpropagation").next_review
        low = get_topic(s, "eclat").next_review
    assert crit < low


def test_reviews_are_never_scheduled_past_the_exam(clean_db) -> None:
    from examagent.config import get_settings

    _record("dropout", QuestionType.CONCEPTUAL, 10.0)
    with session_scope() as s:
        nxt = get_topic(s, "dropout").next_review
    assert nxt <= datetime.utcnow() + timedelta(
        days=get_settings().days_remaining() + 1)


def test_due_topics_are_returned(clean_db) -> None:
    _record("pca", QuestionType.CONCEPTUAL, 3.0)
    with session_scope() as s:
        get_topic(s, "pca").next_review = datetime.utcnow() - timedelta(hours=1)
    with session_scope() as s:
        due = [t.id for t in progress.due_for_review(s)]
    assert "pca" in due


# ---------------------------------------------------------------- readiness
def test_readiness_is_zero_on_a_fresh_profile(clean_db) -> None:
    r = progress.compute_readiness()
    assert r.overall == 0.0
    assert r.coverage == 0.0


def test_readiness_rises_with_performance(clean_db) -> None:
    before = progress.compute_readiness().overall
    for topic in ["backpropagation", "pca", "attention", "cnn_basics", "knn"]:
        _record(topic, QuestionType.CALCULATION, 9.0)
        _record(topic, QuestionType.CONCEPTUAL, 9.0)
    after = progress.compute_readiness().overall
    assert after > before


def test_readiness_weights_are_normalised_and_applied(clean_db) -> None:
    r = progress.compute_readiness()
    assert abs(sum(r.weights.values()) - 1.0) < 1e-6
    assert r.weights["critical"] >= r.weights["confidence"]


def test_readiness_is_not_a_flat_average(clean_db) -> None:
    """Strength on LOW priority topics must not inflate readiness like CRITICAL does."""
    for topic in ["eclat", "apriori", "advanced_multimodal", "kernel_pca"]:
        _record(topic, QuestionType.CONCEPTUAL, 10.0)
        _record(topic, QuestionType.CALCULATION, 10.0)
    low_only = progress.compute_readiness().overall

    for topic in ["backpropagation", "attention", "cnn_basics"]:
        _record(topic, QuestionType.CONCEPTUAL, 10.0)
        _record(topic, QuestionType.CALCULATION, 10.0)
    with_critical = progress.compute_readiness().overall
    assert with_critical - low_only > 0.05


def test_weak_prerequisites_drag_down_dependent_topics(clean_db) -> None:
    """backpropagation depends on forward_propagation, loss_functions, chain_rule."""
    _record("backpropagation", QuestionType.CONCEPTUAL, 10.0)
    with session_scope() as s:
        lookup = {t.id: t for t in all_topics(s)}
        clean = progress.effective_score(lookup["backpropagation"], lookup)

    _record("chain_rule", QuestionType.CONCEPTUAL, 1.0)
    with session_scope() as s:
        lookup = {t.id: t for t in all_topics(s)}
        dragged = progress.effective_score(lookup["backpropagation"], lookup)
    assert dragged < clean


def test_ml_and_dl_are_scored_separately(clean_db) -> None:
    for topic in ["attention", "cnn_basics", "lstm", "backpropagation"]:
        _record(topic, QuestionType.CONCEPTUAL, 10.0)
        _record(topic, QuestionType.CALCULATION, 10.0)
    r = progress.compute_readiness()
    assert r.dl_score > r.ml_score


# ---------------------------------------------------------------- weakness
def test_weakest_topics_are_ranked_by_exam_damage(clean_db) -> None:
    _record("backpropagation", QuestionType.CONCEPTUAL, 1.0)   # CRITICAL, relevance 1.0
    _record("eclat", QuestionType.CONCEPTUAL, 1.0)             # LOW, relevance 0.2
    with session_scope() as s:
        ranked = [t.id for t in progress.weakest_topics(
            s, limit=200, only_priorities=("CRITICAL", "HIGH", "MEDIUM", "LOW"))]
    assert ranked.index("backpropagation") < ranked.index("eclat")


def test_dimension_gap_detection_finds_concept_vs_calculation(clean_db) -> None:
    """The brief's headline example: concept 85%, calculation 31%."""
    _record("backpropagation", QuestionType.CONCEPTUAL, 9.0)
    _record("backpropagation", QuestionType.CONCEPTUAL, 9.0)
    _record("backpropagation", QuestionType.CALCULATION, 2.0)
    _record("backpropagation", QuestionType.CALCULATION, 3.0)

    gaps = weakness.weakness_report()["dimension_gaps"]
    found = next((g for g in gaps if g["id"] == "backpropagation"), None)
    assert found is not None
    assert found["weak"] == "calculation"
    assert found["strong_score"] > found["weak_score"]
    assert "calculation" in found["advice"].lower()


def test_untested_calculation_is_reported_as_a_gap(clean_db) -> None:
    _record("cnn_parameter_count", QuestionType.SHORT_ANSWER, 9.0)
    gaps = weakness.weakness_report()["dimension_gaps"]
    found = next((g for g in gaps if g["id"] == "cnn_parameter_count"), None)
    assert found is not None
    assert found["weak"] == "calculation"


def test_recommendation_targets_the_weak_dimension(clean_db) -> None:
    _record("backpropagation", QuestionType.CONCEPTUAL, 9.0)
    _record("backpropagation", QuestionType.CALCULATION, 2.0)
    report = progress.topic_report("backpropagation")
    assert report["weak_dimension"] == "calculation"
    assert "calculation" in report["recommended_action"].lower()


def test_dangerous_gaps_are_prerequisite_blockers(clean_db) -> None:
    gaps = weakness.dangerous_gaps(limit=5)
    assert gaps
    assert all(g["blocks"] > 0 for g in gaps)


def test_correct_answer_resolves_the_open_mistake(clean_db) -> None:
    q = _question("dropout", QuestionType.CONCEPTUAL)
    progress.record_attempt(q, Evaluation(score=2.0, mistake_type=MistakeType.CONCEPTUAL),
                            context="quiz")
    assert len(weakness.error_log(topic_id="dropout")) == 1
    progress.record_attempt(q, Evaluation(score=9.0, correct=True), context="quiz")
    assert weakness.error_log(topic_id="dropout") == []


def test_knowledge_map_covers_both_categories(clean_db) -> None:
    kmap = weakness.knowledge_map()
    assert set(kmap) == {"Machine Learning", "Deep Learning"}
    for rows in kmap.values():
        for r in rows:
            assert r["color"].startswith("#")
            assert r["mastery"]


# ---------------------------------------------------------------- planner
def test_plan_covers_the_remaining_days(clean_db) -> None:
    from examagent.config import get_settings

    plans = planner.build_plan()
    assert 1 <= len(plans) <= get_settings().study_days
    assert plans[0].blocks
    assert plans[-1].mock_exam, "the last day must include a mock exam"


def test_plan_respects_the_daily_time_budget(clean_db) -> None:
    plans = planner.build_plan(minutes_per_day=120)
    for day in plans:
        assert day.total_minutes <= 120


def test_plan_does_not_repeat_a_topic_on_consecutive_days(clean_db) -> None:
    plans = planner.build_plan()
    for earlier, later in zip(plans, plans[1:]):
        overlap = {b.topic for b in earlier.blocks} & {b.topic for b in later.blocks}
        assert not overlap, f"repeated on consecutive days: {overlap}"


def test_mastered_topics_are_dropped_from_the_plan(clean_db) -> None:
    for _ in range(4):
        for qtype in (QuestionType.CONCEPTUAL, QuestionType.CALCULATION,
                      QuestionType.COMPARISON):
            _record("pca", qtype, 10.0, difficulty=6)
    plans = planner.build_plan()
    scheduled = {b.topic for day in plans for b in day.blocks}
    assert "Principal Component Analysis (PCA)" not in scheduled


def test_planner_prioritises_the_weak_dimension(clean_db) -> None:
    """Once a topic is mapped, its weakest dimension decides the practice type."""
    # give every critical topic a baseline so untested unknowns do not dominate
    for t_id in ["mlp", "activation_functions", "forward_propagation", "loss_functions",
                 "chain_rule", "gradient_descent", "learning_rate", "optimizers",
                 "weight_initialization", "regularization_dl", "dropout"]:
        _record(t_id, QuestionType.CONCEPTUAL, 8.5)
    _record("backpropagation", QuestionType.CONCEPTUAL, 9.5)
    _record("backpropagation", QuestionType.CONCEPTUAL, 9.5)
    _record("backpropagation", QuestionType.CALCULATION, 1.0)

    plans = planner.build_plan()
    block = next((b for day in plans for b in day.blocks
                  if b.topic == "Backpropagation"), None)
    assert block is not None, "a known severe calculation weakness must be scheduled"
    assert block.focus == "calculation"
    assert "calculation" in block.reason.lower()


def test_a_known_severe_weakness_outranks_a_merely_average_topic(clean_db) -> None:
    """A catastrophic single dimension must not be hidden by a good average."""
    from examagent.services.planner import _damage

    _record("backpropagation", QuestionType.CONCEPTUAL, 9.5)
    _record("backpropagation", QuestionType.CALCULATION, 0.5)
    _record("attention", QuestionType.CONCEPTUAL, 6.0)
    _record("attention", QuestionType.CALCULATION, 6.0)
    with session_scope() as s:
        lookup = {t.id: t for t in all_topics(s)}
        assert _damage(lookup["backpropagation"], lookup) > _damage(lookup["attention"], lookup)


def test_next_topic_prefers_an_unresolved_high_severity_mistake(clean_db) -> None:
    _record("lstm", QuestionType.CONCEPTUAL, 1.0)
    nxt = planner.next_topic()
    assert nxt is not None
    assert nxt["topic_id"] == "lstm"
    assert nxt["mode"] == SessionMode.REPAIR.value
    assert "mistake" in nxt["reason"].lower()


def test_next_topic_redirects_to_a_weak_prerequisite(clean_db) -> None:
    """If the biggest gap depends on a broken prerequisite, fix the prerequisite."""
    with session_scope() as s:
        for t in all_topics(s):
            t.next_review = datetime.utcnow() + timedelta(days=30)
    # make chain_rule (a prerequisite of backpropagation) badly weak but resolved
    _record("chain_rule", QuestionType.CONCEPTUAL, 1.0)
    with session_scope() as s:
        for m in s.query(Mistake).all():
            m.resolved = True
            m.retry_required = False
    nxt = planner.next_topic()
    assert nxt is not None
    # either it targets the prerequisite directly, or it explains the dependency
    assert nxt["topic_id"] == "chain_rule" or "prerequisite" in nxt["reason"].lower()


def test_session_plan_matches_the_mode(clean_db) -> None:
    for mode in (SessionMode.QUICK, SessionMode.RAPID, SessionMode.EXAM_SIM,
                 SessionMode.REPAIR, SessionMode.SIXTY):
        plan = planner.plan_session(mode.value)
        assert plan["topic_ids"], f"{mode.value} produced no topics"
        assert plan["n_questions"] > 0
        assert plan["mix"]
        low, high = plan["difficulty_range"]
        assert 1 <= low <= high <= 6


def test_difficulty_adapts_to_the_student_level(clean_db) -> None:
    weak_plan = planner.plan_session("Quick Study")
    for topic in [t for t in weak_plan["topic_ids"]]:
        for _ in range(3):
            _record(topic, QuestionType.CONCEPTUAL, 10.0, difficulty=6)
            _record(topic, QuestionType.CALCULATION, 10.0, difficulty=6)
    strong_plan = planner.plan_session("Quick Study", topic_ids=weak_plan["topic_ids"])
    assert strong_plan["difficulty_range"][1] >= weak_plan["difficulty_range"][1]


# ---------------------------------------------------------------- persistence
def test_state_survives_a_new_session(clean_db) -> None:
    _record("attention", QuestionType.CALCULATION, 8.0)
    with session_scope() as s:
        stored = get_topic(s, "attention").calculation_score
    # a completely new session object must read the same value
    with session_scope() as s2:
        assert get_topic(s2, "attention").calculation_score == stored
        assert s2.query(Attempt).filter(Attempt.topic_id == "attention").count() == 1


def test_study_sessions_are_recorded(clean_db) -> None:
    sid = progress.start_session(SessionMode.QUICK.value, "pca")
    progress.end_session(sid, seconds=900, questions=5, mean_score=7.5)
    from examagent.models.db import StudySession

    with session_scope() as s:
        row = s.get(StudySession, sid)
        assert row.completed
        assert row.seconds == 900
        assert row.questions_answered == 5
        assert get_topic(s, "pca").study_seconds == 900


def test_progress_history_and_profiles(clean_db) -> None:
    _record("pca", QuestionType.CONCEPTUAL, 7.0)
    _record("knn", QuestionType.CALCULATION, 4.0)
    history = progress.progress_history(days=7)
    assert history and history[-1]["questions"] >= 2
    dims = progress.dimension_profile()
    assert dims["reasoning"] > 0 and dims["calculation"] > 0
    assert progress.mistake_profile()


def test_dashboard_snapshot_is_complete(clean_db) -> None:
    _record("backpropagation", QuestionType.CALCULATION, 3.0)
    snap = progress.dashboard_snapshot()
    for key in ("days_remaining", "readiness", "questions_today", "open_mistakes",
                "critical_gaps", "total_topics", "untouched_critical"):
        assert key in snap
    assert snap["questions_today"] >= 1
    assert snap["open_mistakes"] >= 1
