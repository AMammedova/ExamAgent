"""A single ordered route through the topic graph: learn one topic, quiz on
it immediately, move to the next. No mode to pick, no topic to search for -
one queue, one "Continue" button.

The order is fixed once (prerequisite-respecting, CRITICAL first, ML before
DL, clustered by subtopic so related topics stay adjacent) rather than
recomputed from performance like the adaptive planner - the point of this
page is a stable, predictable march through the syllabus for a first fast
pass. `services.planner` and its damage-based reordering remain what decides
what to *revisit*; this decides what to see *first*.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..data.topics import TOPIC_SEEDS
from ..models.db import Topic, all_topics, kv_get, kv_set, session_scope

KV_KEY = "learning_path_state"

#: quiz questions answered per topic before the path advances - kept small on
#: purpose: the goal here is finishing topics quickly, not the full adaptive
#: depth of a Study session.
QUESTIONS_PER_TOPIC = 3

_ML_SUBTOPIC_ORDER = [
    "Foundations", "Data", "Regression", "Model Selection", "Classification",
    "Evaluation", "Ensembles", "Dimensionality Reduction", "Clustering",
    "Unsupervised", "Association Rules", "Reinforcement Learning", "NLP",
]
_DL_SUBTOPIC_ORDER = ["Foundations", "Training", "CNN", "Sequences", "Transformers", "Modern"]
_PRIORITY_RANK = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
_CATEGORY_RANK = {"Machine Learning": 0, "Deep Learning": 1}


def _subtopic_rank(category: str, subtopic: str) -> int:
    order = _ML_SUBTOPIC_ORDER if category == "Machine Learning" else _DL_SUBTOPIC_ORDER
    return order.index(subtopic) if subtopic in order else len(order)


def _sort_key(seed: dict[str, Any]) -> tuple:
    return (
        _subtopic_rank(seed["category"], seed.get("subtopic", "")),
        _PRIORITY_RANK.get(seed["priority"], 9),
        -seed.get("exam_relevance", 0.0),
        seed["name"],
    )


def curriculum_order() -> list[str]:
    """All 98 topic ids, ML then DL, each prerequisite before what depends on
    it. Computed once from the static registry - deterministic across runs."""
    by_id = {t["id"]: t for t in TOPIC_SEEDS}
    done: set[str] = set()
    order: list[str] = []
    remaining = dict(by_id)

    def ready(seed: dict[str, Any]) -> bool:
        return all(p in done for p in seed.get("prereqs", []))

    while remaining:
        candidates = [s for s in remaining.values() if ready(s)]
        if not candidates:  # a broken/missing prereq id - fall back, don't hang
            candidates = list(remaining.values())
        candidates.sort(key=lambda s: (_CATEGORY_RANK.get(s["category"], 2), _sort_key(s)))
        pick = candidates[0]
        order.append(pick["id"])
        done.add(pick["id"])
        del remaining[pick["id"]]
    return order


#: computed once at import time - the registry it is built from is static
CURRICULUM: list[str] = curriculum_order()


# --------------------------------------------------------------- state
def _load() -> dict[str, Any]:
    with session_scope() as s:
        raw = kv_get(s, KV_KEY, None)
    state = raw if isinstance(raw, dict) else {}
    state.setdefault("completed", [])
    state.setdefault("skipped", [])
    return state


def _save(state: dict[str, Any]) -> None:
    with session_scope() as s:
        kv_set(s, KV_KEY, state)


def mark_complete(topic_id: str) -> None:
    state = _load()
    if topic_id not in state["completed"]:
        state["completed"].append(topic_id)
    if topic_id in state["skipped"]:
        state["skipped"].remove(topic_id)
    _save(state)


def mark_skip(topic_id: str) -> None:
    state = _load()
    if topic_id not in state["skipped"] and topic_id not in state["completed"]:
        state["skipped"].append(topic_id)
    _save(state)


def reopen(topic_id: str) -> None:
    """Send a topic back to not-done - for 'I want to redo this one'."""
    state = _load()
    for bucket in ("completed", "skipped"):
        if topic_id in state[bucket]:
            state[bucket].remove(topic_id)
    _save(state)


def reset() -> None:
    _save({"completed": [], "skipped": []})


# --------------------------------------------------------------- rows
@dataclass
class Row:
    topic: Topic
    position: int
    status: str  # "done" | "skipped" | "current" | "upcoming"


def rows() -> list[Row]:
    """Every topic, in curriculum order, with its path status attached."""
    with session_scope() as s:
        by_id = {t.id: t for t in all_topics(s)}
    state = _load()
    done_set, skip_set = set(state["completed"]), set(state["skipped"])

    out: list[Row] = []
    current_assigned = False
    for i, tid in enumerate(CURRICULUM):
        topic = by_id.get(tid)
        if topic is None:
            continue
        if tid in done_set:
            status = "done"
        elif tid in skip_set:
            status = "skipped"
        elif not current_assigned:
            status = "current"
            current_assigned = True
        else:
            status = "upcoming"
        out.append(Row(topic=topic, position=i, status=status))
    return out


def neighbors(topic_id: str) -> tuple[str | None, str | None]:
    """The topic immediately before and after this one in the curriculum -
    for free browsing, independent of completion state. Either side is None
    at the ends of the path."""
    try:
        i = CURRICULUM.index(topic_id)
    except ValueError:
        return None, None
    prev_id = CURRICULUM[i - 1] if i > 0 else None
    next_id = CURRICULUM[i + 1] if i + 1 < len(CURRICULUM) else None
    return prev_id, next_id


def current_topic_id() -> str | None:
    """The first topic that is neither completed nor skipped - where
    'Continue' resumes."""
    state = _load()
    seen = set(state["completed"]) | set(state["skipped"])
    for tid in CURRICULUM:
        if tid not in seen:
            return tid
    return None


def summary() -> dict[str, Any]:
    state = _load()
    total = len(CURRICULUM)
    done = len(state["completed"])
    return {
        "total": total,
        "done": done,
        "skipped": len(state["skipped"]),
        "remaining": total - done - len(state["skipped"]),
        "fraction": (done / total) if total else 0.0,
        "finished": (done + len(state["skipped"])) >= total,
    }
