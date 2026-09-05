"""Adaptive 7-day study planner and session builder.

The plan starts from the day themes in `data/topics.py` but is re-derived from
live data every time it is requested. Two rules dominate:

* Time follows exam damage, not syllabus size. A topic's budget is
  (1 - score) x priority x exam_relevance, so a mastered CRITICAL topic gets
  almost nothing and an untested CRITICAL topic gets a lot.
* Topics are not repeated once mastered, and the weakest *dimension* of a topic
  determines what kind of practice is scheduled.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from sqlalchemy.orm import Session

from ..config import get_logger, get_settings
from ..data.topics import DAY_THEMES, dependents_of
from ..models.db import Mistake, Topic, all_topics, session_scope
from ..models.schemas import (
    DayPlan,
    PlanBlock,
    Priority,
    QuestionType,
    SessionMode,
)
from .calc_engine import topic_has_calculation
from .progress import due_for_review, effective_score, weakest_topics

log = get_logger(__name__)

#: minutes of practice a topic is worth at maximum damage
MAX_BLOCK = 35
MIN_BLOCK = 10


def _damage(topic: Topic, lookup: dict[str, Topic]) -> float:
    """How much this topic can still cost on the exam (0..1+).

    An untested topic is treated as risky, because an unknown is not the same as
    a pass. For a tested topic the *worst measured dimension* carries weight
    alongside the average: a topic at concept 85% / calculation 10% will still
    lose every calculation mark, and a plain average would hide that.
    """
    if topic.attempt_count == 0:
        gap = 1.0 - (0.15 if topic.priority == "CRITICAL" else 0.25)
    else:
        score = effective_score(topic, lookup)
        tested = [v for v in topic.dimension_scores().values() if v > 0]
        worst = min(tested) if tested else score
        gap = 1.0 - (0.65 * score + 0.35 * worst)

    centrality = 1.0 + 0.12 * min(4, len(dependents_of(topic.id)))
    return gap * Priority(topic.priority).weight * (0.35 + 0.65 * topic.exam_relevance) * centrality


def _focus_for(topic: Topic) -> tuple[str, str]:
    """(focus label, human reason) - what kind of practice this topic needs now."""
    if topic.attempt_count == 0:
        return "mixed", "not yet tested - establish a baseline"

    scores = topic.dimension_scores()
    tested = {k: v for k, v in scores.items() if v > 0}
    best = max(tested.values()) if tested else 0.0

    # An untested calculation dimension on a topic that IS examinable numerically
    # is the single most dangerous gap on this paper - surface it first.
    if topic_has_calculation(topic.id) and scores["calculation"] <= 0 and best >= 0.45:
        return "calculation", (
            f"understood at {best:.0%} but never tested on the numbers - "
            "calculation marks are unprotected")

    dim = topic.weakest_dimension()
    if dim == "calculation" and topic_has_calculation(topic.id):
        reference = max(scores["concept"], scores["reasoning"])
        if reference - scores["calculation"] > 0.25:
            return "calculation", (
                f"theory {reference:.0%} vs calculation {scores['calculation']:.0%} - "
                "drill the numbers, skip the theory")
        return "calculation", "calculation is the weakest dimension"
    if dim == "reasoning":
        return "reasoning", "can state facts but not the causal chain"
    if dim == "concept":
        return "concept", "core mechanism is not secure yet"
    if dim == "comparison":
        return "comparison", "needs practice contrasting it with alternatives"
    return "application", "needs applied/scenario practice"


def build_plan(session: Session | None = None, minutes_per_day: int = 240) -> list[DayPlan]:
    """Build (or rebuild) the adaptive plan across the remaining days."""
    def _build(s: Session) -> list[DayPlan]:
        settings = get_settings()
        topics = all_topics(s)
        lookup = {t.id: t for t in topics}
        days_left = max(1, settings.days_remaining())
        total_days = min(len(DAY_THEMES), max(1, settings.study_days))
        first_day = total_days - days_left + 1

        plans: list[DayPlan] = []
        #: topic id -> day offset it was last scheduled on, so a topic is not
        #: repeated on consecutive days unless it is a genuine emergency
        last_seen: dict[str, int] = {}
        weakest_overall = weakest_topics(s, limit=12)

        for offset in range(days_left):
            day_number = min(total_days, max(1, first_day + offset))
            theme = DAY_THEMES[day_number - 1]
            day_date = (date.today() + timedelta(days=offset)).isoformat()
            is_last_day = offset == days_left - 1

            pool_ids = list(theme.get("topics") or [])
            pool = [lookup[t] for t in pool_ids if t in lookup]

            # The final day (and any themeless day) is revision: use live weaknesses
            if not pool or is_last_day:
                pool = list(weakest_overall)
            else:
                # carry forward only genuinely severe topics from other days, and
                # rotate them so the same topic does not appear every single day
                carry = [
                    t for t in weakest_overall
                    if t.id not in {x.id for x in pool}
                    and _damage(t, lookup) > 0.45
                    and offset - last_seen.get(t.id, -99) >= 2
                ][:2]
                pool = pool + carry

            def day_damage(t: Topic) -> float:
                d = _damage(t, lookup)
                gap = offset - last_seen.get(t.id, -99)
                if gap <= 1:
                    d *= 0.35  # seen yesterday - heavily de-prioritise
                elif gap == 2:
                    d *= 0.75
                return d

            ranked = sorted(pool, key=lambda t: -day_damage(t))
            blocks: list[PlanBlock] = []
            used = 0
            budget = minutes_per_day - (60 if is_last_day else 0)  # reserve mock time

            for topic in ranked:
                if used >= budget:
                    break
                dmg = _damage(topic, lookup)
                if dmg < 0.06 and topic.attempt_count > 0:
                    continue  # mastered - do not spend the day here
                minutes = int(max(MIN_BLOCK, min(MAX_BLOCK, round(MAX_BLOCK * dmg / 0.6))))
                minutes = min(minutes, budget - used)
                if minutes < MIN_BLOCK:
                    break
                focus, reason = _focus_for(topic)
                blocks.append(PlanBlock(
                    topic=topic.name,
                    minutes=minutes,
                    focus=focus,
                    reason=reason,
                    priority=Priority(topic.priority),
                ))
                last_seen[topic.id] = offset
                used += minutes

            plans.append(DayPlan(
                day_number=day_number,
                date=day_date,
                theme="Full revision, weakness repair and mock exam" if is_last_day
                else theme["theme"],
                blocks=blocks,
                mock_exam=is_last_day or day_number >= total_days - 1,
            ))
        return plans

    if session is not None:
        return _build(session)
    with session_scope() as s:
        return _build(s)


def today_plan(minutes_per_day: int = 240) -> DayPlan:
    plans = build_plan(minutes_per_day=minutes_per_day)
    return plans[0] if plans else DayPlan(day_number=1, date=date.today().isoformat(),
                                          theme="Revision", blocks=[])


# --------------------------------------------------------------- next action
def next_topic(session: Session | None = None) -> dict[str, Any] | None:
    """The single highest-value thing to study right now, with the reason."""
    def _pick(s: Session) -> dict[str, Any] | None:
        topics = all_topics(s)
        if not topics:
            return None
        lookup = {t.id: t for t in topics}

        # 1. unresolved high-severity mistakes come first
        crit_mistake = s.query(Mistake).filter(
            Mistake.resolved.is_(False),
            Mistake.severity == "High",
            Mistake.retry_required.is_(True),
        ).order_by(Mistake.created_at.desc()).first()
        if crit_mistake and crit_mistake.topic_id in lookup:
            t = lookup[crit_mistake.topic_id]
            focus, _ = _focus_for(t)
            return {
                "topic_id": t.id, "topic": t.name, "focus": focus,
                "mode": SessionMode.REPAIR.value,
                "reason": (f"You have an unresolved high-severity {crit_mistake.mistake_type.lower()} "
                           f"mistake on {t.name}. Repairing a known error is worth more than new material."),
                "priority": t.priority,
            }

        # 2. a due, weak, high-value topic
        due = due_for_review(s, limit=20)
        candidates = due or topics
        ranked = sorted(
            [t for t in candidates if t.priority in ("CRITICAL", "HIGH")],
            key=lambda t: -_damage(t, lookup),
        )
        if not ranked:
            ranked = sorted(topics, key=lambda t: -_damage(t, lookup))
        if not ranked:
            return None
        t = ranked[0]
        focus, reason = _focus_for(t)
        prereq_warning = _weak_prereq(t, lookup)
        if prereq_warning:
            p = prereq_warning
            focus_p, _ = _focus_for(p)
            return {
                "topic_id": p.id, "topic": p.name, "focus": focus_p,
                "mode": SessionMode.QUICK.value,
                "reason": (f"{t.name} is your biggest gap, but it depends on {p.name}, which is at "
                           f"{p.overall():.0%}. Fix the prerequisite first or the practice will not stick."),
                "priority": p.priority,
            }
        return {
            "topic_id": t.id, "topic": t.name, "focus": focus,
            "mode": SessionMode.THIRTY.value if t.attempt_count == 0 else SessionMode.REPAIR.value,
            "reason": (f"{t.priority} priority, exam relevance {t.exam_relevance:.0%}, "
                       f"current level {(effective_score(t, lookup) if t.attempt_count else 0):.0%} - {reason}."),
            "priority": t.priority,
        }

    if session is not None:
        return _pick(session)
    with session_scope() as s:
        return _pick(s)


def _weak_prereq(topic: Topic, lookup: dict[str, Topic]) -> Topic | None:
    for pid in topic.prereq_ids:
        p = lookup.get(pid)
        if p is None:
            continue
        if p.attempt_count > 0 and p.overall() < 0.45:
            return p
    return None


# --------------------------------------------------------------- sessions
#: question-type mixes per session mode
MODE_MIX: dict[str, dict[QuestionType, float]] = {
    SessionMode.QUICK.value: {
        QuestionType.CALCULATION: 0.3, QuestionType.ASSERTION_REASON: 0.3,
        QuestionType.CONCEPTUAL: 0.4,
    },
    SessionMode.THIRTY.value: {
        QuestionType.CONCEPTUAL: 0.3, QuestionType.CALCULATION: 0.25,
        QuestionType.ASSERTION_REASON: 0.25, QuestionType.WHAT_IF: 0.2,
    },
    SessionMode.SIXTY.value: {
        QuestionType.CONCEPTUAL: 0.22, QuestionType.CALCULATION: 0.28,
        QuestionType.ASSERTION_REASON: 0.2, QuestionType.WHAT_IF: 0.12,
        QuestionType.COMPARISON: 0.1, QuestionType.SCENARIO: 0.08,
    },
    SessionMode.DEEP.value: {
        QuestionType.CALCULATION: 0.3, QuestionType.CONCEPTUAL: 0.2,
        QuestionType.ASSERTION_REASON: 0.15, QuestionType.WHAT_IF: 0.12,
        QuestionType.COMPARISON: 0.11, QuestionType.SCENARIO: 0.07,
        QuestionType.DIAGRAM: 0.05,
    },
    SessionMode.RAPID.value: {
        QuestionType.ASSERTION_REASON: 0.5, QuestionType.SHORT_ANSWER: 0.3,
        QuestionType.CONCEPTUAL: 0.2,
    },
    SessionMode.REPAIR.value: {
        QuestionType.CALCULATION: 0.35, QuestionType.CONCEPTUAL: 0.3,
        QuestionType.ASSERTION_REASON: 0.2, QuestionType.WHAT_IF: 0.15,
    },
    SessionMode.EXAM_SIM.value: {
        QuestionType.CALCULATION: 0.3, QuestionType.ASSERTION_REASON: 0.25,
        QuestionType.CONCEPTUAL: 0.2, QuestionType.WHAT_IF: 0.1,
        QuestionType.COMPARISON: 0.08, QuestionType.SCENARIO: 0.07,
    },
}

#: roughly how many questions fit into a mode
MODE_QUESTIONS: dict[str, int] = {
    SessionMode.QUICK.value: 4,
    SessionMode.THIRTY.value: 7,
    SessionMode.SIXTY.value: 12,
    SessionMode.DEEP.value: 16,
    SessionMode.RAPID.value: 10,
    SessionMode.REPAIR.value: 8,
    SessionMode.EXAM_SIM.value: 14,
}


def plan_session(
    mode: str,
    topic_ids: list[str] | None = None,
    session: Session | None = None,
) -> dict[str, Any]:
    """Choose topics, question count, mix and difficulty for a study session."""
    def _plan(s: Session) -> dict[str, Any]:
        topics = all_topics(s)
        lookup = {t.id: t for t in topics}
        n = MODE_QUESTIONS.get(mode, 8)
        mix = MODE_MIX.get(mode, MODE_MIX[SessionMode.THIRTY.value])

        if topic_ids:
            chosen = [lookup[t] for t in topic_ids if t in lookup]
        elif mode == SessionMode.REPAIR.value:
            open_ids = [m.topic_id for m in s.query(Mistake).filter(
                Mistake.resolved.is_(False)).order_by(Mistake.created_at.desc()).limit(30).all()]
            seen: list[str] = []
            for tid in open_ids:
                if tid not in seen and tid in lookup:
                    seen.append(tid)
            chosen = [lookup[t] for t in seen[:4]] or weakest_topics(s, limit=4)
        elif mode == SessionMode.RAPID.value:
            chosen = due_for_review(s, limit=6) or weakest_topics(s, limit=6)
        elif mode == SessionMode.EXAM_SIM.value:
            chosen = weakest_topics(s, limit=8, only_priorities=("CRITICAL", "HIGH"))
        elif mode == SessionMode.QUICK.value:
            chosen = weakest_topics(s, limit=2, only_priorities=("CRITICAL",))
        else:
            chosen = weakest_topics(s, limit=max(2, n // 3))

        if not chosen:
            chosen = sorted(topics, key=lambda t: -_damage(t, lookup))[:3]

        # difficulty follows the student's level: do not throw level-6 at a topic
        # that is at 20%, and do not waste time on level-2 for a strong topic.
        mean_score = sum(
            effective_score(t, lookup) if t.attempt_count else 0.0 for t in chosen
        ) / max(1, len(chosen))
        if mean_score < 0.35:
            drange = (2, 4)
        elif mean_score < 0.6:
            drange = (3, 5)
        else:
            drange = (4, 6)
        if mode == SessionMode.EXAM_SIM.value:
            drange = (4, 6)

        focus_dims = {t.id: _focus_for(t)[0] for t in chosen}
        return {
            "mode": mode,
            "topic_ids": [t.id for t in chosen],
            "topic_names": [t.name for t in chosen],
            "n_questions": n,
            "mix": mix,
            "difficulty_range": drange,
            "minutes": SessionMode(mode).minutes if mode in [m.value for m in SessionMode] else 30,
            "focus": focus_dims,
            "reasons": {t.name: _focus_for(t)[1] for t in chosen},
        }

    if session is not None:
        return _plan(s=session)
    with session_scope() as s:
        return _plan(s)


def plan_summary() -> dict[str, Any]:
    """Aggregate view used by the Progress / Plan screens."""
    plans = build_plan()
    settings = get_settings()
    total_minutes = sum(p.total_minutes for p in plans)
    by_topic: dict[str, int] = {}
    for p in plans:
        for b in p.blocks:
            by_topic[b.topic] = by_topic.get(b.topic, 0) + b.minutes
    return {
        "days": plans,
        "days_remaining": settings.days_remaining(),
        "total_minutes": total_minutes,
        "top_time": sorted(by_topic.items(), key=lambda kv: -kv[1])[:12],
    }
