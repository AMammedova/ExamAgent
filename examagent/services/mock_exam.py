"""Timed mock exam: build, run, submit and report.

Exam conditions mean: no hints, no per-question feedback, a clock, and mixed ML
and DL with the same question-type distribution as the real paper. Everything is
scored only after submission.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Callable

from ..config import get_logger
from ..models.db import (
    MockExam,
    Topic,
    all_topics,
    kv_get,
    kv_set,
    session_scope,
)
from ..models.schemas import (
    DIMENSIONS,
    Category,
    Evaluation,
    MockExamReport,
    Priority,
    Question,
    QuestionType,
    points_for,
)
from .evaluator import evaluate
from .progress import record_attempt, weakest_topics
from .question_gen import EXAM_TYPES, generate_question

log = get_logger(__name__)

#: The real paper (AI-CORE-101): each 30-question part is 12 True/False at 1
#: mark, 9 single-best multiple choice at 3 marks and 9 multiple response at 4
#: marks - 30 questions, 75 marks per part, 150 marks over the two parts.
EXAM_BLUEPRINT: list[tuple[QuestionType, int]] = [
    (QuestionType.TRUE_FALSE, 12),
    (QuestionType.MCQ, 9),
    (QuestionType.MULTIPLE_RESPONSE, 9),
]

#: Same proportions, scaled down for a quick paper.
SHORT_BLUEPRINT: list[tuple[QuestionType, int]] = [
    (QuestionType.TRUE_FALSE, 4),
    (QuestionType.MCQ, 3),
    (QuestionType.MULTIPLE_RESPONSE, 3),
]

#: Length and time of the real paper, for the "full mock" default.
FULL_EXAM_QUESTIONS = 60
FULL_EXAM_MINUTES = 150
PASS_MARK_FRACTION = 0.60


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

    if allowed_ids is not None:
        # An explicit allowlist is the caller saying which topics the paper is
        # about, so use all of them - weakest first, but nothing dropped.
        # Ranking them down to `n` would silently sit a wide paper on a
        # handful of topics, and drop every MEDIUM/LOW one entirely.
        order = {t.id: i for i, t in enumerate(weak)}
        return sorted(topics, key=lambda t: order.get(t.id, len(order)))

    relevant = sorted(
        [t for t in topics if t.priority in ("CRITICAL", "HIGH")],
        key=lambda t: -(t.exam_relevance * Priority(t.priority).weight),
    )
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


#: The paper's own running order: Part 1 is Machine Learning, Part 2 Deep
#: Learning, and inside each part the sections run A (True/False), B (multiple
#: choice), C (multiple response). Questions are not shuffled across sections -
#: sitting them in the paper's order is part of practising the paper.
SECTION_LETTERS = {
    QuestionType.TRUE_FALSE: "A",
    QuestionType.MCQ: "B",
    QuestionType.MULTIPLE_RESPONSE: "C",
}
SECTION_TITLES = {
    QuestionType.TRUE_FALSE: "True / False",
    QuestionType.MCQ: "Multiple Choice",
    QuestionType.MULTIPLE_RESPONSE: "Multiple Response",
}


def _section_counts(n_in_part: int) -> list[tuple[QuestionType, int]]:
    """Split one part's questions across its three sections, in the paper's
    12 / 9 / 9 proportion, always totalling exactly `n_in_part`."""
    blueprint = EXAM_BLUEPRINT if n_in_part >= 7 else SHORT_BLUEPRINT
    total_bp = sum(c for _, c in blueprint)
    counts = [(qtype, max(1, round(c * n_in_part / total_bp)))
              for qtype, c in blueprint]

    # rounding rarely lands on the target; settle the difference on Section A,
    # which is the largest and the cheapest to add to
    drift = n_in_part - sum(c for _, c in counts)
    while drift:
        step = 1 if drift > 0 else -1
        for idx, (qtype, c) in enumerate(counts):
            if c + step >= 1:
                counts[idx] = (qtype, c + step)
                drift -= step
                break
        else:  # pragma: no cover - every section already at its floor
            break
    return counts


def _paper_plan(n_questions: int, balance_ml_dl: bool,
                has_ml: bool, has_dl: bool) -> list[tuple[Category, QuestionType]]:
    """Every question of the paper, in the order it is sat."""
    parts: list[tuple[Category, int]]
    if balance_ml_dl and has_ml and has_dl:
        ml_share = n_questions // 2 + n_questions % 2
        parts = [(Category.ML, ml_share), (Category.DL, n_questions - ml_share)]
    else:
        only = Category.ML if has_ml or not has_dl else Category.DL
        parts = [(only, n_questions)]

    plan: list[tuple[Category, QuestionType]] = []
    for category, count in parts:
        for qtype, section_count in _section_counts(count):
            plan.extend([(category, qtype)] * section_count)
    return plan[:n_questions]


def build_exam(
    n_questions: int = 18,
    duration_minutes: int = 75,
    label: str = "Mock Exam",
    use_llm: bool = True,
    balance_ml_dl: bool = True,
    seed: int | None = None,
    topic_ids: list[str] | None = None,
    max_workers: int = 8,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Generate a full paper and persist it.

    `topic_ids`, when given, restricts every question to that set of topics -
    e.g. a quick mock scoped to only what the Learning Path has covered so
    far, rather than the full syllabus.

    `on_progress(done, total)` is called as questions land, so the caller can
    show real movement rather than an indefinite spinner.
    """
    import random

    rng = random.Random(seed)
    allowed_ids = set(topic_ids) if topic_ids else None

    with session_scope() as s:
        if balance_ml_dl:
            ml_topics = _select_topics(s, n_questions, Category.ML.value, allowed_ids)
            dl_topics = _select_topics(s, n_questions, Category.DL.value, allowed_ids)
        else:
            ml_topics = dl_topics = _select_topics(s, n_questions, allowed_ids=allowed_ids)

    plan = _paper_plan(n_questions, balance_ml_dl,
                       has_ml=bool(ml_topics), has_dl=bool(dl_topics))

    # One task per question, then generated concurrently: each item is an
    # independent LLM round trip, and a 60-question paper done one at a time
    # keeps the student staring at a spinner for minutes.
    tasks: list[tuple[int, QuestionType, Any, int, int, int]] = []
    seats: dict[Category, int] = {}
    for i, (category, qtype) in enumerate(plan):
        pool = (ml_topics if category == Category.ML else dl_topics) \
            or ml_topics or dl_topics
        if not pool:
            break
        # Position within this part, so consecutive questions walk the pool one
        # topic at a time. Indexing by the paper position instead advanced every
        # other question and left a 60-question paper sitting on ~23 topics.
        seat = seats.get(category, 0)
        seats[category] = seat + 1
        tasks.append((i, qtype, pool, rng.choice([4, 5, 5, 6]),
                      rng.randint(1, 10 ** 6), seat))

    def _build_one(
        task: tuple[int, QuestionType, Any, int, int, int],
    ) -> tuple[int, Question]:
        i, qtype, pool, difficulty, qseed, seat = task
        topic = pool[seat % len(pool)]
        q = generate_question(topic.id, qtype, difficulty, use_llm=use_llm,
                              seed=qseed, min_difficulty=4)
        # A paper is defined by its blueprint - 150 marks only add up if the
        # planned format actually lands. `generate_question` already tries
        # every closed format for one topic, so if the planned one still did
        # not come out, try neighbouring topics: a couple through the LLM
        # (each is a round trip, so this stays capped), then the whole pool
        # through the bank, which costs nothing but a dictionary lookup.
        if q.question_type != qtype:
            if use_llm:
                for offset in (1, 2):
                    other = pool[(seat + offset) % len(pool)]
                    candidate = generate_question(other.id, qtype, difficulty,
                                                  use_llm=True, seed=qseed + offset,
                                                  min_difficulty=4)
                    if candidate.question_type == qtype:
                        return i, candidate
            for offset in range(len(pool)):
                other = pool[(seat + offset) % len(pool)]
                candidate = generate_question(other.id, qtype, difficulty,
                                              use_llm=False, seed=qseed + offset,
                                              min_difficulty=4)
                if candidate.question_type == qtype:
                    return i, candidate
        return i, q

    results: dict[int, Question] = {}
    if tasks:
        workers = max(1, min(max_workers, len(tasks)))
        with ThreadPoolExecutor(max_workers=workers) as pool_exec:
            futures = [pool_exec.submit(_build_one, t) for t in tasks]
            for done, future in enumerate(as_completed(futures), 1):
                try:
                    i, q = future.result()
                    results[i] = q
                except Exception as exc:  # one bad item must not sink the paper
                    log.warning("question generation failed: %s", exc)
                if on_progress is not None:
                    on_progress(done, len(futures))

    # Walk the paper in order and settle two things the parallel pass cannot:
    # duplicates (workers cannot share a growing exclude set) and slots whose
    # planned section format did not come out. Both are repaired against the
    # bank, which is instant and exact - a section is only as real as the
    # questions actually sitting in it.
    questions: list[Question] = []
    seen_ids: set[str] = set()
    for i, qtype, pool, difficulty, qseed, seat in tasks:
        q = results.get(i)
        if q is None or q.id in seen_ids or q.question_type != qtype:
            for offset in range(len(pool)):
                other = pool[(seat + offset) % len(pool)]
                candidate = generate_question(other.id, qtype, difficulty,
                                              use_llm=False, exclude_ids=seen_ids,
                                              seed=qseed + offset + 977,
                                              min_difficulty=4)
                if candidate.question_type == qtype and candidate.id not in seen_ids:
                    q = candidate
                    break
        if q is None or q.id in seen_ids:
            continue
        seen_ids.add(q.id)
        questions.append(q)

    return _persist(questions, label, duration_minutes)


def _persist(questions: list[Question], label: str,
             duration_minutes: int) -> dict[str, Any]:
    """Store a paper and hand back the dict every exam caller works with."""
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

    log.info("built exam %s with %d questions", exam_id, len(questions))
    return {
        "exam_id": exam_id,
        "questions": questions,
        "duration_minutes": duration_minutes,
        "started_at": datetime.utcnow(),
    }


def build_topic_sweep(
    topic_ids: list[str] | None = None,
    per_topic: int = 1,
    use_llm: bool = True,
    label: str = "Practice paper",
    max_workers: int = 8,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """One question per topic, in the paper's formats - a practice sweep.

    `build_exam` builds *the paper*: fixed sections, fixed marks, so it will
    move to a neighbouring topic when the planned format does not come out.
    A sweep inverts that trade - the topic is the point, and whichever closed
    format that topic can carry is fine. Nothing is dropped, so every topic
    asked for appears.
    """
    import random

    with session_scope() as s:
        pool = [t.id for t in all_topics(s)]
    if topic_ids:
        wanted = set(topic_ids)
        pool = [t for t in pool if t in wanted]

    # cycle the formats in the paper's own 12 / 9 / 9 proportion
    cycle = ([QuestionType.TRUE_FALSE] * 4 + [QuestionType.MCQ] * 3
             + [QuestionType.MULTIPLE_RESPONSE] * 3)
    rng = random.Random(len(pool))
    plan = [(topic_id, cycle[i % len(cycle)])
            for i, topic_id in enumerate(t for t in pool for _ in range(per_topic))]

    def _one(job: tuple[int, str, QuestionType]) -> tuple[int, Question]:
        i, topic_id, qtype = job
        return i, generate_question(topic_id, qtype, difficulty=rng.choice([4, 5, 5]),
                                    use_llm=use_llm, seed=rng.randint(1, 10 ** 6),
                                    min_difficulty=4)

    jobs = [(i, topic_id, qtype) for i, (topic_id, qtype) in enumerate(plan)]
    results: dict[int, Question] = {}
    if jobs:
        with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(jobs)))) as ex:
            futures = [ex.submit(_one, job) for job in jobs]
            for done, future in enumerate(as_completed(futures), 1):
                try:
                    i, q = future.result()
                    results[i] = q
                except Exception as exc:  # one bad topic must not sink the sweep
                    log.warning("sweep question failed: %s", exc)
                if on_progress is not None:
                    on_progress(done, len(futures))

    questions: list[Question] = []
    seen: set[str] = set()
    for i in range(len(jobs)):
        q = results.get(i)
        if q is None or q.id in seen:
            continue
        if q.question_type not in EXAM_TYPES:
            # A sweep is marked question by question, so it can only carry items
            # that mark exactly. Offline, a topic the bank has no facts for has
            # nothing closed-form to offer and is left out rather than shipped
            # as a written question nobody can tick or cross.
            continue
        seen.add(q.id)
        questions.append(q)

    # marked as you go, so the sweep carries no clock of its own
    return _persist(questions, label, duration_minutes=0)


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


# ------------------------------------------------------- work in progress
#: Where an unfinished practice paper lives. Only the exam id and the answers
#: are kept: every format on such a paper marks exactly, so the ticks and
#: crosses are recomputed on resume rather than stored and kept in sync.
PROGRESS_KV = "exam_in_progress"


def save_progress(exam_id: int, answers: dict[str, Any]) -> None:
    """Remember an unfinished paper, so a refresh or a restart does not throw
    away work that took minutes to generate and longer to answer."""
    with session_scope() as s:
        kv_set(s, PROGRESS_KV, {"exam_id": int(exam_id), "answers": answers})


def load_progress() -> dict[str, Any] | None:
    """The unfinished paper, rehydrated - or None if there is none, it was
    already submitted, or its questions have since been deleted."""
    with session_scope() as s:
        saved = kv_get(s, PROGRESS_KV, None)
    if not isinstance(saved, dict) or not saved.get("exam_id"):
        return None

    exam = load_exam(int(saved["exam_id"]))
    if exam is None or exam.get("completed"):
        clear_progress()
        return None
    return {"exam": exam, "answers": dict(saved.get("answers") or {})}


def clear_progress() -> None:
    with session_scope() as s:
        kv_set(s, PROGRESS_KV, None)


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

    # marks, weighted the way the paper weights them: 1 for True/False, 3 for
    # single-best multiple choice, 4 for multiple response
    total = sum(points_for(q.question_type) * ev.score / 10.0 for q, ev in evaluations)
    maximum = float(sum(points_for(q.question_type) for q, _ in evaluations))
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
        pass_mark=round(maximum * PASS_MARK_FRACTION, 1),
        passed=pct >= PASS_MARK_FRACTION * 100,
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
