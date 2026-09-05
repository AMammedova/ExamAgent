"""Student state: score updates, spaced repetition, readiness scoring.

Design decisions worth knowing:

* Scores are per-dimension exponential moving averages, weighted by question
  difficulty - doing well on a level-6 question moves the needle more than on a
  level-2 one.
* Spaced repetition works in **hours**, not days: with a 7-day horizon a
  classic SM-2 schedule would never bring a topic back.
* Readiness is not a flat average. Critical topics, calculation and reasoning
  dominate, and a weak prerequisite drags down everything that depends on it.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from typing import Any, Iterable

from sqlalchemy.orm import Session

from ..config import get_logger, get_settings
from ..data.topics import DAY_THEMES, TOPIC_SEEDS, dependents_of
from ..models.db import (
    Attempt,
    Mistake,
    MockExam,
    StudySession,
    Topic,
    all_topics,
    attempts_today,
    get_topic,
    kv_get,
    kv_set,
    session_scope,
)
from ..models.schemas import (
    DIMENSIONS,
    Category,
    Evaluation,
    Mastery,
    Priority,
    Question,
    ReadinessBreakdown,
)

log = get_logger(__name__)


# --------------------------------------------------------------- bootstrap
def ensure_topics(session: Session) -> int:
    """Insert any topic seeds that are not in the database yet."""
    existing = {t.id for t in all_topics(session)}
    added = 0
    for seed in TOPIC_SEEDS:
        if seed["id"] in existing:
            continue
        session.add(Topic(
            id=seed["id"],
            name=seed["name"],
            category=seed["category"],
            subtopic=seed.get("subtopic", ""),
            source=seed.get("source", ""),
            priority=seed.get("priority", "MEDIUM"),
            difficulty=int(seed.get("difficulty", 3)),
            exam_relevance=float(seed.get("exam_relevance", 0.5)),
            prerequisites=json.dumps(seed.get("prereqs", [])),
            keywords=json.dumps(seed.get("keywords", [])),
        ))
        added += 1
    if added:
        log.info("seeded %d topics", added)
    return added


def initialize(force: bool = False) -> dict[str, Any]:
    """Idempotent first-run setup."""
    with session_scope() as s:
        added = ensure_topics(s)
        first_run = kv_get(s, "first_run_complete", False)
        if not first_run or force:
            kv_set(s, "started_at", datetime.utcnow().isoformat())
        return {"topics_added": added, "first_run_complete": bool(first_run)}


def mark_first_run_complete() -> None:
    with session_scope() as s:
        kv_set(s, "first_run_complete", True)


# --------------------------------------------------------------- scoring
#: how strongly a new result overwrites the running estimate
def _alpha(previous: float, attempts: int) -> float:
    if previous <= 0:
        return 0.85           # first evidence in this dimension dominates
    return 0.45 if attempts < 4 else 0.32


def _difficulty_weight(difficulty: int) -> float:
    """A hard question is worth more evidence than an easy one."""
    return min(1.15, 0.72 + 0.09 * max(1, min(6, difficulty)))


def update_topic_from_attempt(
    session: Session,
    topic_id: str,
    dimension: str,
    score01: float,
    difficulty: int,
    correct: bool,
) -> Topic | None:
    topic = get_topic(session, topic_id)
    if topic is None:
        return None
    if dimension not in DIMENSIONS:
        dimension = "concept"

    field = f"{dimension}_score"
    previous = float(getattr(topic, field, 0.0))
    adjusted = min(1.0, score01 * _difficulty_weight(difficulty))
    a = _alpha(previous, topic.attempt_count)
    new = (1 - a) * previous + a * adjusted
    setattr(topic, field, round(max(0.0, min(1.0, new)), 4))

    topic.attempt_count += 1
    if not correct:
        topic.mistake_count += 1
    topic.last_reviewed = datetime.utcnow()
    topic.confidence = round(_confidence(topic, score01), 4)
    topic.next_review = schedule_next_review(topic, score01)
    return topic


def _confidence(topic: Topic, latest: float) -> float:
    """Confidence grows with evidence, not just with a single good answer."""
    evidence = 1 - math.exp(-topic.attempt_count / 3.0)
    breadth = topic.tested_dimensions() / 3.0
    overall = topic.overall()
    return max(0.0, min(1.0, overall * (0.55 + 0.30 * evidence + 0.15 * min(1.0, breadth))))


def schedule_next_review(topic: Topic, score01: float) -> datetime:
    """Lightweight adaptive spacing, measured in hours."""
    if score01 >= 0.9:
        hours = 48.0
    elif score01 >= 0.75:
        hours = 24.0
    elif score01 >= 0.55:
        hours = 12.0
    elif score01 >= 0.35:
        hours = 6.0
    else:
        hours = 3.0

    # critical topics come back sooner; low-priority ones later
    priority_factor = {"CRITICAL": 0.65, "HIGH": 0.85, "MEDIUM": 1.3, "LOW": 2.0}
    hours *= priority_factor.get(topic.priority, 1.0)

    # a history of mistakes pulls the topic forward
    if topic.mistake_count:
        hours *= max(0.4, 1.0 - 0.12 * min(5, topic.mistake_count))

    # never schedule past the exam
    settings = get_settings()
    remaining_hours = max(2.0, settings.days_remaining() * 24 * 0.55)
    hours = min(hours, remaining_hours)
    return datetime.utcnow() + timedelta(hours=hours)


# --------------------------------------------------------------- recording
def record_attempt(
    question: Question,
    evaluation: Evaluation,
    student_answer: str = "",
    context: str = "quiz",
    seconds: int = 0,
    exam_id: int | None = None,
) -> dict[str, Any]:
    """Persist an answered question and update the knowledge profile."""
    with session_scope() as s:
        attempt = Attempt(
            topic_id=question.topic,
            question_id=question.id,
            question_type=question.question_type.value,
            dimension=question.dimension,
            difficulty=question.difficulty,
            prompt=question.prompt[:4000],
            student_answer=student_answer[:6000],
            score=evaluation.score,
            correct=evaluation.correct,
            evaluation_json=json.dumps(evaluation.model_dump(mode="json"), default=str),
            context=context,
            exam_id=exam_id,
            seconds_spent=seconds,
        )
        s.add(attempt)

        topic = update_topic_from_attempt(
            s, question.topic, question.dimension,
            evaluation.score01, question.difficulty, evaluation.correct,
        )

        mistake_id = None
        if evaluation.score < 7.0:
            mistake = Mistake(
                topic_id=question.topic,
                question_id=question.id,
                question=question.prompt[:3000],
                question_type=question.question_type.value,
                mistake_type=evaluation.mistake_type.value,
                severity=evaluation.severity,
                student_answer=student_answer[:3000],
                correct_concept=(evaluation.model_answer or question.model_answer)[:3000],
                retry_required=evaluation.score < 7.0,
                next_retry=datetime.utcnow() + timedelta(
                    hours=3 if evaluation.score < 4 else 8
                ),
            )
            s.add(mistake)
            s.flush()
            mistake_id = mistake.id
        else:
            _resolve_open_mistakes(s, question.topic, question.id)

        return {
            "topic": topic.name if topic else question.topic,
            "topic_overall": round(topic.overall(), 3) if topic else 0.0,
            "mastery": Mastery.from_score(topic.overall()).value if topic else "",
            "mistake_id": mistake_id,
            "next_review": topic.next_review.isoformat() if topic and topic.next_review else None,
        }


def _resolve_open_mistakes(session: Session, topic_id: str, question_id: str) -> int:
    """A later correct answer on the same question/topic closes the error log entry."""
    rows = session.query(Mistake).filter(
        Mistake.topic_id == topic_id,
        Mistake.resolved.is_(False),
    ).all()
    closed = 0
    for m in rows:
        if m.question_id == question_id:
            m.resolved = True
            m.retry_required = False
            closed += 1
    return closed


def retry_recorded(mistake_id: int, success: bool) -> None:
    with session_scope() as s:
        m = s.get(Mistake, mistake_id)
        if m is None:
            return
        m.retry_count += 1
        if success:
            m.resolved = True
            m.retry_required = False
        else:
            m.next_retry = datetime.utcnow() + timedelta(hours=4)


# --------------------------------------------------------------- sessions
def start_session(mode: str, topic_id: str = "") -> int:
    with session_scope() as s:
        row = StudySession(mode=mode, topic_id=topic_id)
        s.add(row)
        s.flush()
        return int(row.id)


def end_session(session_id: int, seconds: int, questions: int, mean_score: float) -> None:
    with session_scope() as s:
        row = s.get(StudySession, session_id)
        if row is None:
            return
        row.ended_at = datetime.utcnow()
        row.seconds = seconds
        row.questions_answered = questions
        row.mean_score = mean_score
        row.completed = True
        topic = get_topic(s, row.topic_id) if row.topic_id else None
        if topic:
            topic.study_seconds += seconds


def mark_taught(topic_id: str) -> None:
    with session_scope() as s:
        t = get_topic(s, topic_id)
        if t:
            t.taught_count += 1
            t.last_reviewed = datetime.utcnow()


# --------------------------------------------------------------- readiness
def effective_score(topic: Topic, lookup: dict[str, Topic]) -> float:
    """Topic score discounted by the state of its prerequisites."""
    base = topic.overall()
    prereqs = [lookup[p] for p in topic.prereq_ids if p in lookup]
    tested = [p for p in prereqs if p.attempt_count > 0]
    if not tested:
        return base
    weakest = min(p.overall() for p in tested)
    # a shaky foundation caps what you can really do with the dependent topic
    return round(base * (0.72 + 0.28 * weakest), 4)


def _weighted_mean(pairs: Iterable[tuple[float, float]]) -> float:
    pairs = list(pairs)
    tw = sum(w for _, w in pairs)
    if tw <= 0:
        return 0.0
    return sum(v * w for v, w in pairs) / tw


def compute_readiness(session: Session | None = None) -> ReadinessBreakdown:
    def _compute(s: Session) -> ReadinessBreakdown:
        settings = get_settings()
        weights = settings.readiness_weights()
        topics = all_topics(s)
        if not topics:
            return ReadinessBreakdown(weights=weights)
        lookup = {t.id: t for t in topics}

        critical = [t for t in topics if t.priority == Priority.CRITICAL.value]
        high = [t for t in topics if t.priority == Priority.HIGH.value]
        focus = critical + high

        critical_mastery = _weighted_mean(
            (effective_score(t, lookup), t.exam_relevance) for t in critical
        )

        calc_topics = [t for t in focus if t.calculation_score > 0 or t.attempt_count > 0]
        calculation = _weighted_mean(
            (t.calculation_score, Priority(t.priority).weight * t.exam_relevance)
            for t in calc_topics
        ) if calc_topics else 0.0

        reasoning = _weighted_mean(
            (t.reasoning_score, Priority(t.priority).weight * t.exam_relevance)
            for t in focus if t.attempt_count > 0
        ) if any(t.attempt_count for t in focus) else 0.0

        exam_performance = _exam_performance(s)

        covered = [t for t in focus if t.attempt_count > 0]
        coverage = len(covered) / len(focus) if focus else 0.0

        confidence = _weighted_mean(
            (t.confidence, Priority(t.priority).weight) for t in focus
        )

        overall = (
            weights["critical"] * critical_mastery
            + weights["calculation"] * calculation
            + weights["reasoning"] * reasoning
            + weights["exam"] * exam_performance
            + weights["coverage"] * coverage
            + weights["confidence"] * confidence
        )

        ml = _weighted_mean(
            (effective_score(t, lookup), Priority(t.priority).weight * t.exam_relevance)
            for t in topics if t.category == Category.ML.value
        )
        dl = _weighted_mean(
            (effective_score(t, lookup), Priority(t.priority).weight * t.exam_relevance)
            for t in topics if t.category == Category.DL.value
        )

        return ReadinessBreakdown(
            overall=round(overall, 4),
            critical_mastery=round(critical_mastery, 4),
            calculation=round(calculation, 4),
            reasoning=round(reasoning, 4),
            exam_performance=round(exam_performance, 4),
            coverage=round(coverage, 4),
            confidence=round(confidence, 4),
            ml_score=round(ml, 4),
            dl_score=round(dl, 4),
            weights=weights,
        )

    if session is not None:
        return _compute(session)
    with session_scope() as s:
        return _compute(s)


def _exam_performance(session: Session) -> float:
    """Recent mock-exam percentage, else performance on exam-level questions."""
    exams = session.query(MockExam).filter(MockExam.completed.is_(True)).order_by(
        MockExam.finished_at.desc()
    ).limit(3).all()
    if exams:
        # most recent counts most
        weights = [0.6, 0.25, 0.15][: len(exams)]
        total = sum(weights)
        return sum(e.percentage / 100.0 * w for e, w in zip(exams, weights)) / total

    hard = session.query(Attempt).filter(Attempt.difficulty >= 5).order_by(
        Attempt.created_at.desc()
    ).limit(25).all()
    if not hard:
        return 0.0
    return sum(a.score for a in hard) / (10.0 * len(hard))


# --------------------------------------------------------------- dashboard
def dashboard_snapshot() -> dict[str, Any]:
    settings = get_settings()
    with session_scope() as s:
        topics = all_topics(s)
        lookup = {t.id: t for t in topics}
        readiness = compute_readiness(s)
        today = attempts_today(s)
        open_mistakes = s.query(Mistake).filter(Mistake.resolved.is_(False)).count()
        critical_gaps = weakest_topics(s, limit=5, only_priorities=("CRITICAL",))
        due = [t for t in topics
               if t.next_review is not None and t.next_review <= datetime.utcnow()]
        never_touched = [t for t in topics
                         if t.attempt_count == 0 and t.priority == "CRITICAL"]
        sessions_today = s.query(StudySession).filter(
            StudySession.started_at >= datetime.utcnow().replace(
                hour=0, minute=0, second=0, microsecond=0)
        ).all()
        return {
            "days_remaining": settings.days_remaining(),
            "exam_date": settings.exam_day.isoformat(),
            "day_number": max(1, settings.study_days - settings.days_remaining() + 1),
            "readiness": readiness,
            "questions_today": len(today),
            "mean_score_today": round(
                sum(a.score for a in today) / len(today), 2) if today else 0.0,
            "study_seconds_today": sum(x.seconds for x in sessions_today),
            "open_mistakes": open_mistakes,
            "critical_gaps": [
                {"id": t.id, "name": t.name, "score": round(effective_score(t, lookup), 3),
                 "weak_dimension": t.weakest_dimension()}
                for t in critical_gaps
            ],
            "due_count": len(due),
            "untouched_critical": len(never_touched),
            "total_topics": len(topics),
            "topics_started": sum(1 for t in topics if t.attempt_count > 0),
        }


def weakest_topics(
    session: Session,
    limit: int = 10,
    only_priorities: tuple[str, ...] = ("CRITICAL", "HIGH"),
    include_untested: bool = True,
) -> list[Topic]:
    """Rank topics by exam damage: low score x high priority x high relevance."""
    topics = [t for t in all_topics(session) if t.priority in only_priorities]
    lookup = {t.id: t for t in all_topics(session)}
    if not include_untested:
        topics = [t for t in topics if t.attempt_count > 0]

    def risk(t: Topic) -> float:
        score = effective_score(t, lookup) if t.attempt_count else 0.25
        unknown_penalty = 0.0 if t.attempt_count else 0.15
        return (1.0 - score + unknown_penalty) * Priority(t.priority).weight * (
            0.4 + 0.6 * t.exam_relevance)

    return sorted(topics, key=lambda t: -risk(t))[:limit]


def strongest_topics(session: Session, limit: int = 5) -> list[Topic]:
    topics = [t for t in all_topics(session) if t.attempt_count > 0]
    return sorted(topics, key=lambda t: -t.overall())[:limit]


def due_for_review(session: Session, limit: int = 12) -> list[Topic]:
    now = datetime.utcnow()
    topics = [t for t in all_topics(session)
              if t.attempt_count > 0 and (t.next_review is None or t.next_review <= now)]
    return sorted(topics, key=lambda t: (t.next_review or now))[:limit]


def prerequisite_risks(session: Session) -> list[dict[str, Any]]:
    """Weak topics that many other topics depend on - the dangerous gaps."""
    topics = all_topics(session)
    lookup = {t.id: t for t in topics}
    out: list[dict[str, Any]] = []
    for t in topics:
        deps = dependents_of(t.id)
        if not deps:
            continue
        score = effective_score(t, lookup) if t.attempt_count else 0.0
        if score >= 0.6 and t.attempt_count > 0:
            continue
        out.append({
            "id": t.id,
            "name": t.name,
            "score": round(score, 3),
            "blocks": len(deps),
            "dependents": [lookup[d].name for d in deps if d in lookup][:6],
            "tested": t.attempt_count > 0,
        })
    return sorted(out, key=lambda d: (-d["blocks"], d["score"]))[:8]


def topic_report(topic_id: str) -> dict[str, Any]:
    with session_scope() as s:
        t = get_topic(s, topic_id)
        if t is None:
            return {}
        lookup = {x.id: x for x in all_topics(s)}
        attempts = s.query(Attempt).filter(Attempt.topic_id == topic_id).order_by(
            Attempt.created_at.desc()).limit(20).all()
        mistakes = s.query(Mistake).filter(
            Mistake.topic_id == topic_id, Mistake.resolved.is_(False)
        ).order_by(Mistake.created_at.desc()).limit(10).all()
        return {
            "topic": t.as_dict(),
            "name": t.name,
            "effective": round(effective_score(t, lookup), 3),
            "mastery": Mastery.from_score(t.overall()).value,
            "weak_dimension": t.weakest_dimension(),
            "prerequisites": [
                {"id": p, "name": lookup[p].name, "score": round(lookup[p].overall(), 3)}
                for p in t.prereq_ids if p in lookup
            ],
            "dependents": [lookup[d].name for d in dependents_of(t.id) if d in lookup],
            "attempts": [
                {"date": a.created_at.strftime("%d %b %H:%M"), "type": a.question_type,
                 "dimension": a.dimension, "score": a.score, "difficulty": a.difficulty}
                for a in attempts
            ],
            "mistakes": [
                {"id": m.id, "type": m.mistake_type, "severity": m.severity,
                 "question": m.question[:300], "date": m.created_at.strftime("%d %b"),
                 "correct_concept": m.correct_concept[:600]}
                for m in mistakes
            ],
            "recommended_action": recommend_action(t),
        }


def recommend_action(topic: Topic) -> str:
    from .calc_engine import topic_has_calculation

    overall = topic.overall()
    if topic.attempt_count == 0:
        return f"Not yet tested. Start a study session on {topic.name} to establish a baseline."
    dim = topic.weakest_dimension()
    if overall >= 0.88:
        return "Mastered - do not spend more time here. Revisit once before the exam."
    if dim == "calculation" and topic_has_calculation(topic.id):
        return (f"Concept is ahead of calculation. Drill calculation problems on {topic.name} "
                "until you can complete one without hints.")
    if dim == "reasoning":
        return (f"Practise assertion-reason and 'what happens if' questions on {topic.name} - "
                "you can state the facts but not the causal chain.")
    if dim == "concept":
        return f"Re-learn the core mechanism of {topic.name}, then immediately self-test."
    if dim == "comparison":
        return f"Practise contrasting {topic.name} with its nearest alternative along explicit axes."
    return f"Apply {topic.name} to a scenario question."


# --------------------------------------------------------------- history
def progress_history(days: int = 14) -> list[dict[str, Any]]:
    with session_scope() as s:
        start = datetime.utcnow() - timedelta(days=days)
        rows = s.query(Attempt).filter(Attempt.created_at >= start).order_by(
            Attempt.created_at).all()
        buckets: dict[str, list[float]] = {}
        for a in rows:
            key = a.created_at.strftime("%Y-%m-%d")
            buckets.setdefault(key, []).append(a.score)
        return [
            {"date": k, "questions": len(v), "mean_score": round(sum(v) / len(v), 2)}
            for k, v in sorted(buckets.items())
        ]


def dimension_profile() -> dict[str, float]:
    with session_scope() as s:
        topics = [t for t in all_topics(s) if t.attempt_count > 0]
        if not topics:
            return {d: 0.0 for d in DIMENSIONS}
        out: dict[str, float] = {}
        for d in DIMENSIONS:
            vals = [getattr(t, f"{d}_score") for t in topics
                    if getattr(t, f"{d}_score") > 0]
            out[d] = round(sum(vals) / len(vals), 3) if vals else 0.0
        return out


def mistake_profile() -> dict[str, int]:
    with session_scope() as s:
        rows = s.query(Mistake).all()
        out: dict[str, int] = {}
        for m in rows:
            out[m.mistake_type] = out.get(m.mistake_type, 0) + 1
        return out


def day_theme(day_number: int) -> dict[str, Any]:
    idx = max(1, min(len(DAY_THEMES), day_number)) - 1
    return DAY_THEMES[idx]
