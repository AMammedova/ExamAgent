"""The revision booklet: coverage, language and shape of its claims."""
from __future__ import annotations

import pytest

from examagent.data import memory_book_az as book
from examagent.data.topics import TOPIC_SEEDS

AZ_MARKERS = ("ə", "ı", "ğ", "ş", "ç", "ö", "ü")


def _all_points() -> list[tuple[str, str]]:
    return [(section.title, point)
            for _category, sections in book.all_sections()
            for section in sections
            for point in section.points]


# ---------------------------------------------------------------- shape
def test_both_parts_of_the_syllabus_are_covered() -> None:
    categories = [category for category, _ in book.all_sections()]
    assert categories == ["Machine Learning", "Deep Learning"]
    for _category, sections in book.all_sections():
        assert len(sections) >= 8, "each part needs more than a couple of sections"


def test_the_booklet_is_substantial() -> None:
    counts = book.stats()
    assert counts["points"] >= 150, f"only {counts['points']} points"
    assert counts["formulas"] >= 8


def test_every_section_has_a_title_and_points() -> None:
    for _category, sections in book.all_sections():
        for section in sections:
            assert section.title.strip()
            assert len(section.points) >= 4, f"{section.title} is too thin"


# ---------------------------------------------------------------- language
def test_every_point_is_written_in_azerbaijani() -> None:
    """A point that slipped through in English would read as a foreign body in
    a booklet meant to be revised from."""
    for title, point in _all_points():
        assert any(marker in point.lower() for marker in AZ_MARKERS), (
            f"English point under {title!r}: {point[:70]}"
        )


def test_points_are_single_claims_not_paragraphs() -> None:
    """The format asked for: numbered sentences you can hold in your head."""
    for title, point in _all_points():
        assert len(point) <= 260, f"too long under {title!r}: {point[:60]}…"
        assert point.strip().endswith((".", "؟", "?")), (
            f"unfinished point under {title!r}: {point[-40:]}"
        )


def test_no_duplicate_points() -> None:
    points = [p for _title, p in _all_points()]
    duplicates = {p for p in points if points.count(p) > 1}
    assert not duplicates, f"repeated: {list(duplicates)[:2]}"


# ---------------------------------------------------------------- content
@pytest.mark.parametrize("term", [
    "overfitting", "cross-validation", "precision", "recall", "ROC",
    "PCA", "K-Means", "DBSCAN", "SVM", "boosting",
    "backpropagation", "ReLU", "softmax", "dropout", "batch normalization",
    "stride", "padding", "pooling", "LSTM", "GRU", "attention", "transformer",
    "BERT", "GPT", "learning rate", "gradient",
])
def test_the_examinable_terms_all_appear(term: str) -> None:
    """If the paper can ask about it, the booklet should say something about it."""
    text = " ".join(p for _title, p in _all_points()).lower()
    assert term.lower() in text, f"the booklet never mentions {term}"


def test_the_formulas_that_get_examined_are_there() -> None:
    formulas = " ".join(f for _category, sections in book.all_sections()
                        for s in sections for f in s.formulas)
    assert "Precision" in formulas and "Recall" in formulas
    assert "W - K + 2P" in formulas, "the conv output-size formula must be listed"
    assert "softmax" in formulas.lower(), "scaled dot-product attention must be listed"


def test_it_speaks_about_topics_the_app_actually_tracks() -> None:
    """The booklet and the topic graph should be about the same course."""
    text = " ".join(p for _title, p in _all_points()).lower()
    named = sum(1 for seed in TOPIC_SEEDS if seed["name"].lower() in text)
    assert named >= 25, f"only {named} registry topics are named in the booklet"
