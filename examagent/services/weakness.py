"""Weakness analysis and error-log review.

Answers the questions the dashboard must answer at a glance:
what am I worst at, what is dangerous, what did I get wrong, and what do I do
about it now.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from ..config import get_logger
from ..models.db import Mistake, all_topics, open_mistakes, session_scope
from ..models.schemas import Mastery
from .calc_engine import topic_has_calculation
from .progress import effective_score, recommend_action, strongest_topics, weakest_topics

log = get_logger(__name__)


def weakness_report(limit: int = 10) -> dict[str, Any]:
    with session_scope() as s:
        topics = all_topics(s)
        lookup = {t.id: t for t in topics}
        weakest = weakest_topics(s, limit=limit)
        strongest = strongest_topics(s, limit=5)

        rows = []
        for t in weakest:
            dims = t.dimension_scores()
            rows.append({
                "id": t.id,
                "name": t.name,
                "category": t.category,
                "priority": t.priority,
                "overall": round(t.overall(), 3),
                "effective": round(effective_score(t, lookup), 3),
                "mastery": Mastery.from_score(t.overall()).value if t.attempt_count
                else "Not tested",
                "weak_dimension": t.weakest_dimension(),
                "dimensions": {k: round(v, 3) for k, v in dims.items()},
                "attempts": t.attempt_count,
                "mistakes": t.mistake_count,
                "exam_relevance": t.exam_relevance,
                "action": recommend_action(t),
            })

        return {
            "weakest": rows,
            "strongest": [
                {"id": t.id, "name": t.name, "overall": round(t.overall(), 3),
                 "mastery": Mastery.from_score(t.overall()).value}
                for t in strongest
            ],
            "dimension_gaps": dimension_gaps(s),
            "mistake_types": mistake_type_counts(s),
            "open_mistakes": len(open_mistakes(s)),
        }


def dimension_gaps(session) -> list[dict[str, Any]]:
    """Topics where one dimension lags far behind another - the actionable case.

    E.g. concept 85% but calculation 31% means: stop explaining, start drilling.
    """
    out: list[dict[str, Any]] = []
    for t in all_topics(session):
        if t.attempt_count < 1:
            continue
        dims = {k: v for k, v in t.dimension_scores().items() if v > 0}
        if len(dims) < 2:
            # a topic tested only conceptually but with a calculation engine
            # available is itself a gap worth reporting
            if "concept" in dims and dims["concept"] >= 0.6 and topic_has_calculation(t.id) \
                    and t.calculation_score == 0:
                out.append({
                    "id": t.id, "name": t.name,
                    "strong": "concept", "strong_score": round(dims["concept"], 3),
                    "weak": "calculation", "weak_score": 0.0,
                    "gap": round(dims["concept"], 3),
                    "advice": (f"You understand {t.name} but have never been tested on the "
                               "numbers. Exam calculation marks are at risk."),
                })
            continue
        best = max(dims, key=lambda k: dims[k])
        worst = min(dims, key=lambda k: dims[k])
        gap = dims[best] - dims[worst]
        if gap >= 0.25:
            out.append({
                "id": t.id, "name": t.name,
                "strong": best, "strong_score": round(dims[best], 3),
                "weak": worst, "weak_score": round(dims[worst], 3),
                "gap": round(gap, 3),
                "advice": _gap_advice(t.name, best, worst),
            })
    return sorted(out, key=lambda d: -d["gap"])[:10]


def _gap_advice(name: str, strong: str, weak: str) -> str:
    if weak == "calculation":
        return (f"Do not re-read the theory of {name}. Your concept score is fine - drill "
                "calculation problems until you can do one cleanly under time pressure.")
    if weak == "reasoning":
        return (f"You know the facts of {name} but not the causal chain. Do assertion-reason "
                "and 'what happens if' questions, not more explanation.")
    if weak == "concept":
        return (f"You can manipulate {name} mechanically but the underlying idea is shaky - "
                "that fails on 'why' questions. Re-learn the mechanism, then self-test.")
    if weak == "comparison":
        return f"Practise contrasting {name} with its nearest alternative along explicit axes."
    return f"Apply {name} to scenario questions - you know it but cannot deploy it."


def mistake_type_counts(session) -> dict[str, int]:
    out: dict[str, int] = {}
    for m in session.query(Mistake).all():
        out[m.mistake_type] = out.get(m.mistake_type, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def error_log(limit: int = 50, unresolved_only: bool = True,
              topic_id: str | None = None) -> list[dict[str, Any]]:
    with session_scope() as s:
        q = s.query(Mistake)
        if unresolved_only:
            q = q.filter(Mistake.resolved.is_(False))
        if topic_id:
            q = q.filter(Mistake.topic_id == topic_id)
        rows = q.order_by(Mistake.created_at.desc()).limit(limit).all()
        lookup = {t.id: t for t in all_topics(s)}
        return [
            {
                "id": m.id,
                "topic_id": m.topic_id,
                "topic": lookup[m.topic_id].name if m.topic_id in lookup else m.topic_id,
                "question": m.question,
                "question_type": m.question_type,
                "mistake_type": m.mistake_type,
                "severity": m.severity,
                "student_answer": m.student_answer,
                "correct_concept": m.correct_concept,
                "retry_required": m.retry_required,
                "retry_count": m.retry_count,
                "resolved": m.resolved,
                "date": m.created_at.strftime("%d %b %H:%M"),
                "due": (m.next_retry is None or m.next_retry <= datetime.utcnow()),
            }
            for m in rows
        ]


def due_retries(limit: int = 10) -> list[dict[str, Any]]:
    """Mistakes whose retry is due - the spaced-repetition queue for errors."""
    return [e for e in error_log(limit=60) if e["due"] and e["retry_required"]][:limit]


def dangerous_gaps(limit: int = 5) -> list[dict[str, Any]]:
    """Weak topics that other topics depend on, ranked by blast radius."""
    from .progress import prerequisite_risks

    with session_scope() as s:
        risks = prerequisite_risks(s)
    return risks[:limit]


def category_breakdown() -> dict[str, dict[str, float]]:
    with session_scope() as s:
        topics = all_topics(s)
        lookup = {t.id: t for t in topics}
        out: dict[str, dict[str, float]] = {}
        for cat in ("Machine Learning", "Deep Learning"):
            group = [t for t in topics if t.category == cat]
            tested = [t for t in group if t.attempt_count > 0]
            out[cat] = {
                "topics": len(group),
                "tested": len(tested),
                "coverage": round(len(tested) / len(group), 3) if group else 0.0,
                "mean_score": round(
                    sum(effective_score(t, lookup) for t in tested) / len(tested), 3
                ) if tested else 0.0,
                "critical_weak": sum(
                    1 for t in group
                    if t.priority == "CRITICAL" and (t.attempt_count == 0 or t.overall() < 0.5)
                ),
            }
        return out


def knowledge_map() -> dict[str, list[dict[str, Any]]]:
    """Topic map grouped by category and subtopic, coloured by mastery."""
    with session_scope() as s:
        topics = all_topics(s)
        lookup = {t.id: t for t in topics}
        out: dict[str, list[dict[str, Any]]] = {}
        for t in topics:
            score = effective_score(t, lookup) if t.attempt_count else 0.0
            mastery = Mastery.from_score(score) if t.attempt_count else None
            out.setdefault(t.category, []).append({
                "id": t.id,
                "name": t.name,
                "subtopic": t.subtopic,
                "priority": t.priority,
                "score": round(score, 3),
                "mastery": mastery.value if mastery else "Not tested",
                "color": mastery.color if mastery else "#8b949e",
                "attempts": t.attempt_count,
                "exam_relevance": t.exam_relevance,
                "weak_dimension": t.weakest_dimension() if t.attempt_count else "-",
                "dimensions": {k: round(v, 3) for k, v in t.dimension_scores().items()},
                "has_material": t.has_material,
            })
        for cat in out:
            out[cat].sort(key=lambda d: (d["subtopic"], -d["exam_relevance"]))
        return out


def revision_priorities(limit: int = 8) -> list[str]:
    """Plain-language ordered instructions for what to fix next."""
    report = weakness_report(limit=limit)
    out: list[str] = []
    for gap in report["dimension_gaps"][:3]:
        out.append(gap["advice"])
    for row in report["weakest"][:limit]:
        if row["attempts"] == 0:
            out.append(f"{row['name']} ({row['priority']}) has never been tested - "
                       "a blind spot on a high-relevance topic.")
        else:
            out.append(f"{row['name']}: {row['action']}")
    seen: set[str] = set()
    unique = []
    for line in out:
        if line not in seen:
            unique.append(line)
            seen.add(line)
    return unique[:limit]
