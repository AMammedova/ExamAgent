"""Timed mock exam: build, run, submit and report.

Exam conditions mean: no hints, no per-question feedback, a clock, and mixed ML
and DL with the same question-type distribution as the real paper. Everything is
scored only after submission.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from ..config import get_logger
from ..models.db import MockExam, Topic, all_topics, session_scope
from ..models.schemas import (
    DIMENSIONS,
    Category,
    Evaluation,
    MockExamReport,
    Priority,
    Question,
    QuestionType,
)
from .evaluator import evaluate
from .progress import record_attempt, weakest_topics
from .question_gen import generate_question

log = get_logger(__name__)

#: Question-type distribution modelled on the university exam samples.
EXAM_BLUEPRINT: list[tuple[QuestionType, int]] = [
    (QuestionType.ASSERTION_REASON, 5),
    (QuestionType.CALCULATION, 4),
    (QuestionType.CONCEPTUAL, 3),
    (QuestionType.WHAT_IF, 2),
    (QuestionType.COMPARISON, 2),
    (QuestionType.SCENARIO, 1),
    (QuestionType.DIAGRAM, 1),
]

SHORT_BLUEPRINT: list[tuple[QuestionType, int]] = [
    (QuestionType.ASSERTION_REASON, 3),
    (QuestionType.CALCULATION, 2),
    (QuestionType.CONCEPTUAL, 2),
    (QuestionType.WHAT_IF, 1),
]


def _select_topics(session, n: int, category: str | None = None,
                   allowed_ids: set[str] | None = None) -> list[Topic]:
    """Mix the student's weak topics with high-relevance topics they may know.

    `allowed_ids`, when given, restricts the whole pool to those topics -
    e.g. only what has actually been covered in the Learning Path so far,
    for a mock exam scoped to material the student has actually seen.
    """
    topics = [t for t in all_topics(session)
              if (category is None or t.category == category)
              and (allowed_ids is None or t.id in allowed_ids)]
    weak = [t for t in weakest_topics(session, limit=max(n, len(allowed_ids or [])))
            if (category is None or t.category == category)
            and (allowed_ids is None or t.id in allowed_ids)]
    relevant = sorted(
        [t for t in topics if t.priority in ("CRITICAL", "HIGH")],
        key=lambda t: -(t.exam_relevance * Priority(t.priority).weight),
    )
    if allowed_ids is not None and not relevant:
        # a small allowed set may be entirely MEDIUM/LOW priority - still usable
        relevant = sorted(topics, key=lambda t: -(t.exam_relevance))
    out: list[Topic] = []
    seen: set[str] = set()
    # alternate weak / high-relevance so the paper is not purely a weakness drill
    for i in range(max(len(weak), len(relevant))):
        for src in (weak, relevant):
            if i < len(src) and src[i].id not in seen:
                out.append(src[i])
                seen.add(src[i].id)
            if len(out) >= n:
                return out
    return out[:n] or topics[:n]


def build_exam(
    n_questions: int = 18,
    duration_minutes: int = 75,
    label: str = "Mock Exam",
    use_llm: bool = True,
    balance_ml_dl: bool = True,
    seed: int | None = None,
    topic_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Generate a full paper and persist it.

    `topic_ids`, when given, restricts every question to that set of topics -
    e.g. a quick mock scoped to only what the Learning Path has covered so
    far, rather than the full syllabus.
    """
    import random

    rng = random.Random(seed)
    blueprint = EXAM_BLUEPRINT if n_questions >= 14 else SHORT_BLUEPRINT
    allowed_ids = set(topic_ids) if topic_ids else None

    # scale the blueprint to the requested length
    total_bp = sum(c for _, c in blueprint)
    plan: list[QuestionType] = []
    for qtype, count in blueprint:
        plan.extend([qtype] * max(1, round(count * n_questions / total_bp)))
    plan = plan[:n_questions]
    while len(plan) < n_questions:
        plan.append(QuestionType.CONCEPTUAL)
    rng.shuffle(plan)

    with session_scope() as s:
        if balance_ml_dl:
            ml_topics = _select_topics(s, n_questions, Category.ML.value, allowed_ids)
            dl_topics = _select_topics(s, n_questions, Category.DL.value, allowed_ids)
        else:
            ml_topics = dl_topics = _select_topics(s, n_questions, allowed_ids=allowed_ids)

    questions: list[Question] = []
    seen_ids: set[str] = set()
    ar_keys: list[str] = []
    for i, qtype in enumerate(plan):
        pool = (ml_topics if i % 2 == 0 else dl_topics) or ml_topics or dl_topics
        if not pool:
            break
        topic = pool[(i // 2) % len(pool)]
        difficulty = rng.choice([4, 5, 5, 6])
        q = generate_question(
            topic.id, qtype, difficulty, use_llm=use_llm,
            exclude_ids=seen_ids, seed=rng.randint(1, 10 ** 6),
            recent_ar_keys=ar_keys[-4:], min_difficulty=4,
        )
        if q.id in seen_ids:
            q = generate_question(topic.id, qtype, difficulty, use_llm=False,
                                  exclude_ids=seen_ids, seed=rng.randint(1, 10 ** 6),
                                  recent_ar_keys=ar_keys[-4:], min_difficulty=4)
        seen_ids.add(q.id)
        if q.question_type == QuestionType.ASSERTION_REASON and q.correct_option:
            ar_keys.append(q.correct_option)
        questions.append(q)

    with session_scope() as s:
        exam = MockExam(
            label=label,
            n_questions=len(questions),
            duration_minutes=duration_minutes,
            questions_json=json.dumps([q.model_dump(mode="json") for q in questions]),
        )
        s.add(exam)
        s.flush()
        exam_id = int(exam.id)

    log.info("built mock exam %s with %d questions", exam_id, len(questions))
    return {
        "exam_id": exam_id,
        "questions": questions,
        "duration_minutes": duration_minutes,
        "started_at": datetime.utcnow(),
    }


def load_exam(exam_id: int) -> dict[str, Any] | None:
    with session_scope() as s:
        exam = s.get(MockExam, exam_id)
        if exam is None:
            return None
        raw = json.loads(exam.questions_json or "[]")
        return {
            "exam_id": exam.id,
            "label": exam.label,
            "questions": [Question(**q) for q in raw],
            "duration_minutes": exam.duration_minutes,
            "started_at": exam.started_at,
            "completed": exam.completed,
            "answers": json.loads(exam.answers_json or "{}"),
            "report": json.loads(exam.report_json or "{}"),
        }


def submit_exam(
    exam_id: int,
    answers: dict[str, Any],
    duration_seconds: int = 0,
    use_llm: bool = True,
) -> MockExamReport:
    """Grade every answer, persist attempts, and build the performance report."""
    data = load_exam(exam_id)
    if data is None:
        raise ValueError(f"unknown exam {exam_id}")
    questions: list[Question] = data["questions"]

    evaluations: list[tuple[Question, Evaluation]] = []
    for q in questions:
        ans = answers.get(q.id, "" if q.question_type != QuestionType.CALCULATION else {})
        ev = evaluate(q, ans, use_llm=use_llm)
        evaluations.append((q, ev))
        record_attempt(
            q, ev,
            student_answer=json.dumps(ans) if isinstance(ans, dict) else str(ans),
            context="mock", exam_id=exam_id,
            seconds=int(duration_seconds / max(1, len(questions))),
        )

    report = _build_report(exam_id, evaluations, duration_seconds)

    with session_scope() as s:
        exam = s.get(MockExam, exam_id)
        if exam is not None:
            exam.completed = True
            exam.finished_at = datetime.utcnow()
            exam.answers_json = json.dumps(answers, default=str)
            exam.report_json = json.dumps(report.model_dump(mode="json"), default=str)
            exam.percentage = report.percentage
    return report


def _build_report(
    exam_id: int,
    evaluations: list[tuple[Question, Evaluation]],
    duration_seconds: int,
) -> MockExamReport:
    if not evaluations:
        return MockExamReport(exam_id=exam_id)

    total = sum(ev.score for _, ev in evaluations)
    maximum = 10.0 * len(evaluations)
    pct = 100.0 * total / maximum if maximum else 0.0

    def _subset(pred) -> float:
        rows = [ev.score for q, ev in evaluations if pred(q)]
        return round(100.0 * sum(rows) / (10.0 * len(rows)), 1) if rows else 0.0

    ml = _subset(lambda q: q.category == Category.ML)
    dl = _subset(lambda q: q.category == Category.DL)

    by_dimension = {
        d: _subset(lambda q, d=d: q.dimension == d)
        for d in DIMENSIONS
        if any(q.dimension == d for q, _ in evaluations)
    }
    by_type = {
        t.value: _subset(lambda q, t=t: q.question_type == t)
        for t in {q.question_type for q, _ in evaluations}
    }

    # per-topic aggregation
    per_topic: dict[str, list[float]] = {}
    for q, ev in evaluations:
        per_topic.setdefault(q.topic, []).append(ev.score)
    topic_means = {k: sum(v) / len(v) for k, v in per_topic.items()}

    with session_scope() as s:
        lookup = {t.id: t for t in all_topics(s)}
        name = lambda tid: lookup[tid].name if tid in lookup else tid  # noqa: E731
        relevance = lambda tid: lookup[tid].exam_relevance if tid in lookup else 0.5  # noqa: E731
        priority = lambda tid: lookup[tid].priority if tid in lookup else "MEDIUM"  # noqa: E731

        ranked_weak = sorted(topic_means.items(), key=lambda kv: kv[1])
        ranked_strong = sorted(topic_means.items(), key=lambda kv: -kv[1])

        weaknesses = [
            f"{name(t)} - {m:.1f}/10" for t, m in ranked_weak[:5] if m < 7
        ]
        strengths = [
            f"{name(t)} - {m:.1f}/10" for t, m in ranked_strong[:5] if m >= 7
        ]
        dangerous = [
            f"{name(t)} ({priority(t)}, exam relevance {relevance(t):.0%}) scored {m:.1f}/10"
            for t, m in ranked_weak
            if m < 5 and priority(t) == "CRITICAL"
        ][:5]

        immediate = [name(t) for t, m in ranked_weak[:5] if m < 6]

        revision: list[str] = []
        worst_dim = min(by_dimension, key=lambda k: by_dimension[k]) if by_dimension else None
        if worst_dim and by_dimension[worst_dim] < 65:
            revision.append(
                f"Your weakest dimension is **{worst_dim}** at {by_dimension[worst_dim]:.0f}%. "
                f"Spend the next session on {worst_dim} questions only."
            )
        has_ml = any(q.category == Category.ML for q, _ in evaluations)
        has_dl = any(q.category == Category.DL for q, _ in evaluations)
        # note: `if ml and dl` would skip the most extreme case of all - one side
        # scoring exactly zero.
        if has_ml and has_dl and abs(ml - dl) >= 12:
            behind, ahead = ("Machine Learning", "Deep Learning") if ml < dl else (
                "Deep Learning", "Machine Learning")
            revision.append(
                f"{behind} ({min(ml, dl):.0f}%) is well behind {ahead} ({max(ml, dl):.0f}%). "
                f"Rebalance study time toward {behind}."
            )
        worst_type = min(by_type, key=lambda k: by_type[k]) if by_type else None
        if worst_type and by_type[worst_type] < 60:
            revision.append(
                f"You lose the most marks on **{worst_type.replace('_', ' ')}** questions "
                f"({by_type[worst_type]:.0f}%). Practise that format specifically."
            )
        for t, m in ranked_weak[:3]:
            if m < 6:
                revision.append(f"Repair {name(t)} ({m:.1f}/10) before the exam - "
                                f"{priority(t)} priority.")
        if pct >= 75:
            revision.append("Overall performance is exam-ready. Shift to timed practice and "
                            "consolidation rather than new material.")

    return MockExamReport(
        exam_id=exam_id,
        total_score=round(total, 1),
        max_score=maximum,
        percentage=round(pct, 1),
        ml_score=ml,
        dl_score=dl,
        by_dimension={k: round(v, 1) for k, v in by_dimension.items()},
        by_question_type={k: round(v, 1) for k, v in by_type.items()},
        top_weaknesses=weaknesses,
        top_strengths=strengths,
        dangerous_gaps=dangerous,
        immediate_revision=immediate,
        revision_plan=revision,
        duration_seconds=duration_seconds,
        finished_at=datetime.utcnow(),
    )


def exam_history(limit: int = 10) -> list[dict[str, Any]]:
    with session_scope() as s:
        rows = s.query(MockExam).order_by(MockExam.started_at.desc()).limit(limit).all()
        return [
            {
                "id": e.id,
                "label": e.label,
                "n_questions": e.n_questions,
                "percentage": e.percentage,
                "completed": e.completed,
                "started": e.started_at.strftime("%d %b %H:%M"),
                "finished": e.finished_at.strftime("%d %b %H:%M") if e.finished_at else None,
                "report": json.loads(e.report_json or "{}"),
            }
            for e in rows
        ]


def latest_report() -> MockExamReport | None:
    with session_scope() as s:
        e = s.query(MockExam).filter(MockExam.completed.is_(True)).order_by(
            MockExam.finished_at.desc()).first()
        if e is None:
            return None
        try:
            return MockExamReport(**json.loads(e.report_json or "{}"))
        except (TypeError, ValueError):
            return None
