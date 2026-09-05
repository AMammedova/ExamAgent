"""SQLite persistence layer (SQLAlchemy 2.0 ORM).

Everything the student does is stored here so progress survives restarts:
topic profiles, attempts, mistakes, mock exams, documents, sessions.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Iterator

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    sessionmaker,
)

from ..config import get_logger, get_settings

log = get_logger(__name__)


class Base(DeclarativeBase):
    pass


def _now() -> datetime:
    return datetime.utcnow()


# --------------------------------------------------------------- tables
class Topic(Base):
    """A node of the topic knowledge graph + the student's profile on it."""

    __tablename__ = "topics"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    name: Mapped[str] = mapped_column(String(160), index=True)
    category: Mapped[str] = mapped_column(String(40))  # Machine Learning | Deep Learning
    subtopic: Mapped[str] = mapped_column(String(160), default="")
    source: Mapped[str] = mapped_column(String(200), default="")
    priority: Mapped[str] = mapped_column(String(16), default="MEDIUM", index=True)
    difficulty: Mapped[int] = mapped_column(Integer, default=3)
    exam_relevance: Mapped[float] = mapped_column(Float, default=0.5)
    prerequisites: Mapped[str] = mapped_column(Text, default="[]")  # JSON list of topic ids
    keywords: Mapped[str] = mapped_column(Text, default="[]")  # JSON list

    concept_score: Mapped[float] = mapped_column(Float, default=0.0)
    calculation_score: Mapped[float] = mapped_column(Float, default=0.0)
    reasoning_score: Mapped[float] = mapped_column(Float, default=0.0)
    comparison_score: Mapped[float] = mapped_column(Float, default=0.0)
    application_score: Mapped[float] = mapped_column(Float, default=0.0)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)

    last_reviewed: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    next_review: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    mistake_count: Mapped[int] = mapped_column(Integer, default=0)
    study_seconds: Mapped[int] = mapped_column(Integer, default=0)
    taught_count: Mapped[int] = mapped_column(Integer, default=0)
    has_material: Mapped[bool] = mapped_column(Boolean, default=False)
    notes: Mapped[str] = mapped_column(Text, default="")

    # ---- helpers ----
    @property
    def prereq_ids(self) -> list[str]:
        try:
            return json.loads(self.prerequisites or "[]")
        except json.JSONDecodeError:
            return []

    @property
    def keyword_list(self) -> list[str]:
        try:
            return json.loads(self.keywords or "[]")
        except json.JSONDecodeError:
            return []

    #: relative weight of each dimension in the overall topic score
    DIM_WEIGHTS = {
        "concept": 0.28,
        "calculation": 0.24,
        "reasoning": 0.26,
        "comparison": 0.10,
        "application": 0.12,
    }

    def dimension_scores(self) -> dict[str, float]:
        return {
            "concept": self.concept_score,
            "calculation": self.calculation_score,
            "reasoning": self.reasoning_score,
            "comparison": self.comparison_score,
            "application": self.application_score,
        }

    def overall(self) -> float:
        """Weighted mean across dimensions that have actually been probed.

        Untested dimensions are ignored rather than counted as zero, so a topic
        is not punished for questions the student has not seen yet.
        """
        dims = self.dimension_scores()
        active = {k: v for k, v in dims.items() if v > 0}
        if not active:
            return 0.0
        tw = sum(self.DIM_WEIGHTS[k] for k in active)
        return sum(v * self.DIM_WEIGHTS[k] for k, v in active.items()) / tw

    def tested_dimensions(self) -> int:
        return sum(1 for v in self.dimension_scores().values() if v > 0)

    #: tie-break order when several dimensions score the same
    DIM_ORDER = ("concept", "calculation", "reasoning", "application", "comparison")

    def weakest_dimension(self, only: tuple[str, ...] = ()) -> str:
        """The weakest dimension the student has actually been tested on.

        Untested dimensions are excluded: reporting 'concept' as a weakness when
        no concept question was ever asked produces misleading advice. Use
        `untested_dimensions()` for the coverage gap instead.
        """
        dims = self.dimension_scores()
        if only:
            dims = {k: v for k, v in dims.items() if k in only}
        pool = {k: v for k, v in dims.items() if v > 0} or dims
        return min(pool, key=lambda k: (pool[k], self.DIM_ORDER.index(k)))

    def untested_dimensions(self) -> list[str]:
        return [k for k, v in self.dimension_scores().items() if v <= 0]

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "subtopic": self.subtopic,
            "priority": self.priority,
            "difficulty": self.difficulty,
            "exam_relevance": self.exam_relevance,
            "concept_score": round(self.concept_score, 3),
            "calculation_score": round(self.calculation_score, 3),
            "reasoning_score": round(self.reasoning_score, 3),
            "comparison_score": round(self.comparison_score, 3),
            "application_score": round(self.application_score, 3),
            "confidence": round(self.confidence, 3),
            "overall": round(self.overall(), 3),
            "attempt_count": self.attempt_count,
            "mistake_count": self.mistake_count,
        }


class Attempt(Base):
    """One answered question."""

    __tablename__ = "attempts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    topic_id: Mapped[str] = mapped_column(String(80), ForeignKey("topics.id"), index=True)
    question_id: Mapped[str] = mapped_column(String(80), default="")
    question_type: Mapped[str] = mapped_column(String(40), default="")
    dimension: Mapped[str] = mapped_column(String(20), default="concept")
    difficulty: Mapped[int] = mapped_column(Integer, default=3)
    prompt: Mapped[str] = mapped_column(Text, default="")
    student_answer: Mapped[str] = mapped_column(Text, default="")
    score: Mapped[float] = mapped_column(Float, default=0.0)  # 0..10
    correct: Mapped[bool] = mapped_column(Boolean, default=False)
    evaluation_json: Mapped[str] = mapped_column(Text, default="{}")
    context: Mapped[str] = mapped_column(String(30), default="quiz")  # quiz|study|mock|chat
    exam_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    seconds_spent: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now, index=True)

    def evaluation(self) -> dict[str, Any]:
        try:
            return json.loads(self.evaluation_json or "{}")
        except json.JSONDecodeError:
            return {}


class Mistake(Base):
    """Error log entry; drives Weakness Repair and spaced repetition."""

    __tablename__ = "mistakes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    topic_id: Mapped[str] = mapped_column(String(80), ForeignKey("topics.id"), index=True)
    question_id: Mapped[str] = mapped_column(String(80), default="")
    question: Mapped[str] = mapped_column(Text, default="")
    question_type: Mapped[str] = mapped_column(String(40), default="")
    mistake_type: Mapped[str] = mapped_column(String(30), default="Conceptual")
    severity: Mapped[str] = mapped_column(String(10), default="Medium")
    student_answer: Mapped[str] = mapped_column(Text, default="")
    correct_concept: Mapped[str] = mapped_column(Text, default="")
    retry_required: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    resolved: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    next_retry: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now, index=True)


class Document(Base):
    """An ingested source document."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String(400))
    source_type: Mapped[str] = mapped_column(String(40), index=True)
    source_name: Mapped[str] = mapped_column(String(300), default="")
    lecture: Mapped[str] = mapped_column(String(200), default="")
    content_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    n_chunks: Mapped[int] = mapped_column(Integer, default=0)
    n_chars: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(30), default="pending")
    error: Mapped[str] = mapped_column(Text, default="")
    detected_topics: Mapped[str] = mapped_column(Text, default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class MockExam(Base):
    __tablename__ = "mock_exams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    label: Mapped[str] = mapped_column(String(120), default="Mock Exam")
    n_questions: Mapped[int] = mapped_column(Integer, default=0)
    duration_minutes: Mapped[int] = mapped_column(Integer, default=60)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    questions_json: Mapped[str] = mapped_column(Text, default="[]")
    answers_json: Mapped[str] = mapped_column(Text, default="{}")
    report_json: Mapped[str] = mapped_column(Text, default="{}")
    percentage: Mapped[float] = mapped_column(Float, default=0.0)
    completed: Mapped[bool] = mapped_column(Boolean, default=False, index=True)


class StudySession(Base):
    __tablename__ = "study_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    mode: Mapped[str] = mapped_column(String(40), default="Quick Study")
    topic_id: Mapped[str] = mapped_column(String(80), default="")
    started_at: Mapped[datetime] = mapped_column(DateTime, default=_now, index=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    seconds: Mapped[int] = mapped_column(Integer, default=0)
    questions_answered: Mapped[int] = mapped_column(Integer, default=0)
    mean_score: Mapped[float] = mapped_column(Float, default=0.0)
    state_json: Mapped[str] = mapped_column(Text, default="{}")
    completed: Mapped[bool] = mapped_column(Boolean, default=False)


class KeyValue(Base):
    """Small app-level state (first-run flag, plan snapshot, chat history)."""

    __tablename__ = "kv"

    key: Mapped[str] = mapped_column(String(120), primary_key=True)
    value: Mapped[str] = mapped_column(Text, default="")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)


class QuestionRecord(Base):
    """Persisted question bank (seed + generated), reusable across sessions."""

    __tablename__ = "questions"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    topic_id: Mapped[str] = mapped_column(String(80), index=True)
    question_type: Mapped[str] = mapped_column(String(40), index=True)
    difficulty: Mapped[int] = mapped_column(Integer, default=4)
    payload_json: Mapped[str] = mapped_column(Text, default="{}")
    times_used: Mapped[int] = mapped_column(Integer, default=0)
    last_used: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    generated: Mapped[bool] = mapped_column(Boolean, default=False)


# --------------------------------------------------------------- engine
_engine = None
_SessionFactory = None


def get_engine():
    global _engine, _SessionFactory
    if _engine is None:
        settings = get_settings()
        _engine = create_engine(
            settings.db_url,
            echo=False,
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(_engine)
        _SessionFactory = sessionmaker(bind=_engine, expire_on_commit=False)
        log.info("database ready at %s", settings.db_url)
    return _engine


def get_session_factory():
    get_engine()
    return _SessionFactory


@contextmanager
def session_scope() -> Iterator[Session]:
    factory = get_session_factory()
    s = factory()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def reset_engine() -> None:
    """Drop cached engine (used by tests and after changing DATA_DIR)."""
    global _engine, _SessionFactory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionFactory = None


# --------------------------------------------------------------- kv helpers
def kv_get(session: Session, key: str, default: Any = None) -> Any:
    row = session.get(KeyValue, key)
    if row is None:
        return default
    try:
        return json.loads(row.value)
    except json.JSONDecodeError:
        return row.value


def kv_set(session: Session, key: str, value: Any) -> None:
    payload = json.dumps(value, default=str)
    row = session.get(KeyValue, key)
    if row is None:
        session.add(KeyValue(key=key, value=payload))
    else:
        row.value = payload
        row.updated_at = _now()


# --------------------------------------------------------------- queries
def all_topics(session: Session) -> list[Topic]:
    return list(session.scalars(select(Topic)).all())


def get_topic(session: Session, topic_id: str) -> Topic | None:
    return session.get(Topic, topic_id)


def find_topic_by_name(session: Session, name: str, exact_only: bool = False) -> Topic | None:
    """Look up a topic by name or id.

    The fuzzy pass picks the *closest* match by name-length difference rather
    than the first hit, so "random forest" resolves to "Random Forests" and not
    to "Random Forest Regression".
    """
    name_l = (name or "").strip().lower()
    if not name_l:
        return None
    rows = all_topics(session)
    for t in rows:
        if t.name.lower() == name_l or t.id == name_l:
            return t
    if exact_only:
        return None

    candidates = [
        t for t in rows
        if name_l in t.name.lower() or t.name.lower() in name_l
    ]
    if candidates:
        return min(candidates, key=lambda t: abs(len(t.name) - len(name_l)))
    for t in rows:
        if any(name_l == k.lower() for k in t.keyword_list):
            return t
    return None


def open_mistakes(session: Session, topic_id: str | None = None, due_only: bool = False):
    stmt = select(Mistake).where(Mistake.resolved.is_(False))
    if topic_id:
        stmt = stmt.where(Mistake.topic_id == topic_id)
    if due_only:
        stmt = stmt.where(
            (Mistake.next_retry.is_(None)) | (Mistake.next_retry <= _now())
        )
    return list(session.scalars(stmt.order_by(Mistake.created_at.desc())).all())


def due_topics(session: Session, when: datetime | None = None) -> list[Topic]:
    when = when or _now()
    stmt = select(Topic).where(
        (Topic.next_review.is_(None)) | (Topic.next_review <= when)
    )
    return list(session.scalars(stmt).all())


def recent_attempts(session: Session, limit: int = 50, context: str | None = None):
    stmt = select(Attempt).order_by(Attempt.created_at.desc()).limit(limit)
    if context:
        stmt = select(Attempt).where(Attempt.context == context).order_by(
            Attempt.created_at.desc()
        ).limit(limit)
    return list(session.scalars(stmt).all())


def study_time_today(session: Session) -> int:
    start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    rows = session.scalars(
        select(StudySession).where(StudySession.started_at >= start)
    ).all()
    return sum(r.seconds for r in rows)


def attempts_today(session: Session) -> list[Attempt]:
    start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    return list(
        session.scalars(select(Attempt).where(Attempt.created_at >= start)).all()
    )


def last_n_days_activity(session: Session, days: int = 7) -> dict[str, int]:
    start = datetime.utcnow() - timedelta(days=days)
    rows = session.scalars(select(Attempt).where(Attempt.created_at >= start)).all()
    out: dict[str, int] = {}
    for a in rows:
        key = a.created_at.strftime("%Y-%m-%d")
        out[key] = out.get(key, 0) + 1
    return out
