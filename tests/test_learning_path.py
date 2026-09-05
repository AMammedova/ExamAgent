"""Learning Path: curriculum order, prerequisite correctness, and state."""
from __future__ import annotations

import pytest

from examagent.data.topics import TOPIC_SEEDS
from examagent.services import learning_path as lp


# ---------------------------------------------------------------- order
def test_curriculum_covers_every_topic_exactly_once() -> None:
    order = lp.curriculum_order()
    assert len(order) == len(TOPIC_SEEDS)
    assert len(set(order)) == len(order)
    assert set(order) == {t["id"] for t in TOPIC_SEEDS}


def test_curriculum_never_places_a_topic_before_its_prerequisite() -> None:
    order = lp.curriculum_order()
    position = {tid: i for i, tid in enumerate(order)}
    by_id = {t["id"]: t for t in TOPIC_SEEDS}
    for tid, seed in by_id.items():
        for prereq in seed.get("prereqs", []):
            assert position[prereq] < position[tid], (
                f"{prereq} must come before {tid}, its dependent"
            )


def test_machine_learning_precedes_deep_learning() -> None:
    order = lp.curriculum_order()
    by_id = {t["id"]: t for t in TOPIC_SEEDS}
    categories = [by_id[tid]["category"] for tid in order]
    last_ml = max(i for i, c in enumerate(categories) if c == "Machine Learning")
    first_dl = min(i for i, c in enumerate(categories) if c == "Deep Learning")
    assert last_ml < first_dl


def test_curriculum_is_deterministic_across_calls() -> None:
    assert lp.curriculum_order() == lp.curriculum_order()


# ---------------------------------------------------------------- state
def test_a_fresh_path_starts_at_the_first_curriculum_topic(clean_db) -> None:
    assert lp.current_topic_id() == lp.CURRICULUM[0]


def test_completing_a_topic_advances_the_pointer(clean_db) -> None:
    first = lp.CURRICULUM[0]
    lp.mark_complete(first)
    assert lp.current_topic_id() == lp.CURRICULUM[1]


def test_skipping_a_topic_also_advances_the_pointer(clean_db) -> None:
    first = lp.CURRICULUM[0]
    lp.mark_skip(first)
    assert lp.current_topic_id() == lp.CURRICULUM[1]


def test_reopening_a_completed_topic_moves_the_pointer_back(clean_db) -> None:
    first = lp.CURRICULUM[0]
    lp.mark_complete(first)
    lp.reopen(first)
    assert lp.current_topic_id() == first


def test_summary_reflects_progress(clean_db) -> None:
    lp.mark_complete(lp.CURRICULUM[0])
    lp.mark_skip(lp.CURRICULUM[1])
    s = lp.summary()
    assert s["done"] == 1
    assert s["skipped"] == 1
    assert s["remaining"] == s["total"] - 2
    assert not s["finished"]


def test_finishing_every_topic_is_reported(clean_db) -> None:
    for tid in lp.CURRICULUM:
        lp.mark_complete(tid)
    assert lp.summary()["finished"] is True
    assert lp.current_topic_id() is None


def test_reset_clears_all_progress(clean_db) -> None:
    lp.mark_complete(lp.CURRICULUM[0])
    lp.mark_skip(lp.CURRICULUM[1])
    lp.reset()
    s = lp.summary()
    assert s["done"] == 0 and s["skipped"] == 0
    assert lp.current_topic_id() == lp.CURRICULUM[0]


# ---------------------------------------------------------------- rows
def test_rows_mark_exactly_one_topic_as_current(clean_db) -> None:
    lp.mark_complete(lp.CURRICULUM[0])
    lp.mark_skip(lp.CURRICULUM[1])
    all_rows = lp.rows()
    current = [r for r in all_rows if r.status == "current"]
    assert len(current) == 1
    assert current[0].topic.id == lp.CURRICULUM[2]


def test_rows_are_in_curriculum_order(clean_db) -> None:
    ids = [r.topic.id for r in lp.rows()]
    assert ids == lp.CURRICULUM
