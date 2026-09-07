"""The three formats the real paper uses: generation, keying and marking."""
from __future__ import annotations

import random

import pytest

from examagent.models.schemas import QuestionType, points_for
from examagent.services import exam_formats as ef
from examagent.services.assertion_engine import AR_BANK
from examagent.services.evaluator import evaluate
from examagent.services.question_gen import generate_question

TOPICS_WITH_FACTS = sorted({i.topic_id for i in AR_BANK})


# ---------------------------------------------------------------- marks
def test_marks_match_the_paper() -> None:
    assert points_for(QuestionType.TRUE_FALSE) == 1
    assert points_for(QuestionType.MCQ) == 3
    assert points_for(QuestionType.MULTIPLE_RESPONSE) == 4


# ---------------------------------------------------------------- facts
def test_every_bank_item_yields_two_verified_statements() -> None:
    facts = ef.all_facts()
    assert len(facts) == 2 * len(AR_BANK)
    assert all(f.text.strip() for f in facts)


def test_a_topics_facts_are_only_its_own() -> None:
    facts = ef.facts_for("pca")
    assert facts
    assert {f.topic_id for f in facts} == {"pca"}


def test_distractor_search_widens_only_as_far_as_it_must() -> None:
    """No topic owns three false statements, so distractors must come from
    further afield - but a false statement is false regardless of topic, so
    this cannot create a second defensible answer."""
    rng = random.Random(1)
    picked = ef.false_facts_near("pca", 3, rng)
    assert len(picked) == 3
    assert all(not f.true for f in picked), "a distractor must be an untrue statement"
    assert len({f.text for f in picked}) == 3, "distractors must not repeat"


# ---------------------------------------------------------------- true/false
@pytest.mark.parametrize("topic_id", ["pca", "backpropagation", "dropout"])
def test_true_false_is_keyed_from_the_recorded_truth(topic_id: str) -> None:
    rng = random.Random(3)
    q = ef.build_true_false(topic_id, rng)
    assert q is not None
    assert q.question_type == QuestionType.TRUE_FALSE
    assert q.correct_option in ("True", "False")

    # the key must match the bank's own flag for that exact statement
    truth = {f.text: f.true for f in ef.facts_for(topic_id)}
    assert q.prompt in truth
    assert q.correct_option == ("True" if truth[q.prompt] else "False")


def test_true_false_is_marked_exactly(clean_db) -> None:
    rng = random.Random(4)
    q = ef.build_true_false("pca", rng)
    wrong = "False" if q.correct_option == "True" else "True"
    assert evaluate(q, q.correct_option, use_llm=False).score == 10.0
    assert evaluate(q, wrong, use_llm=False).score == 0.0
    assert evaluate(q, "", use_llm=False).score == 0.0


# ---------------------------------------------------------------- multiple choice
def test_multiple_choice_has_four_options_and_one_true_answer() -> None:
    rng = random.Random(5)
    q = ef.build_multiple_choice("pca", rng)
    assert q is not None
    assert len(q.options) == 4
    assert q.correct_option in {o.key for o in q.options}

    truth = {f.text: f.true for f in ef.all_facts()}
    for option in q.options:
        assert option.text in truth, "every option must be a verified statement"
        expected = option.key == q.correct_option
        assert truth[option.text] is expected, (
            "exactly the keyed option may be true - otherwise two answers defend"
        )


def test_multiple_choice_is_marked_exactly(clean_db) -> None:
    rng = random.Random(6)
    q = ef.build_multiple_choice("backpropagation", rng)
    wrong = next(o.key for o in q.options if o.key != q.correct_option)
    assert evaluate(q, q.correct_option, use_llm=False).score == 10.0
    assert evaluate(q, wrong, use_llm=False).score == 0.0


# ---------------------------------------------------------------- multiple response
def test_multiple_response_lists_four_statements_and_keys_the_true_set() -> None:
    rng = random.Random(7)
    q = ef.build_multiple_response("pca", rng)
    assert q is not None
    assert len(q.statements) == ef.STATEMENTS_PER_ITEM
    assert len(q.options) == 4

    truth = {f.text: f.true for f in ef.all_facts()}
    true_numbers = [i for i, s in enumerate(q.statements, 1) if truth[s]]
    assert true_numbers, "an item where nothing is true is unanswerable"
    assert len(true_numbers) < len(q.statements), "and so is one where everything is"

    keyed = next(o.text for o in q.options if o.key == q.correct_option)
    assert keyed == ef._render_numbers(true_numbers), (
        f"the key must list exactly the true statements: {keyed} vs {true_numbers}"
    )


def test_multiple_response_options_are_distinct() -> None:
    rng = random.Random(8)
    q = ef.build_multiple_response("backpropagation", rng)
    texts = [o.text for o in q.options]
    assert len(set(texts)) == len(texts), "a repeated option gives the answer away"


def test_multiple_response_is_all_or_nothing(clean_db) -> None:
    rng = random.Random(9)
    q = ef.build_multiple_response("pca", rng)
    wrong = next(o.key for o in q.options if o.key != q.correct_option)
    assert evaluate(q, q.correct_option, use_llm=False).score == 10.0
    ev = evaluate(q, wrong, use_llm=False)
    assert ev.score == 0.0, "there is no partial credit in Section C"
    assert q.correct_option in " ".join(ev.missed)


def test_number_rendering_matches_the_papers_phrasing() -> None:
    assert ef._render_numbers([2]) == "2"
    assert ef._render_numbers([1, 4]) == "1 and 4"
    assert ef._render_numbers([1, 3, 4]) == "1, 3, and 4"


# ---------------------------------------------------------------- integration
@pytest.mark.parametrize("qtype", list(ef.BUILDERS))
def test_generate_question_serves_the_paper_formats_offline(clean_db, qtype) -> None:
    q = generate_question("pca", qtype, difficulty=4, use_llm=False, seed=2)
    assert q.question_type in (qtype, *ef.BUILDERS), (
        "a closed-form request must not fall back to a written question here"
    )
    assert q.options, "a closed-form item must offer options"
    assert q.correct_option


def test_offline_coverage_spans_the_bank() -> None:
    cov = ef.coverage()
    assert cov["true_false"] == cov["topics_with_facts"], (
        "every topic with facts can produce a True/False item"
    )
    assert cov["multiple_choice"] >= 0.9 * cov["topics_with_facts"]
    assert cov["multiple_response"] >= 0.5 * cov["topics_with_facts"]


# ---------------------------------------------------- LLM key verification
def _fake_llm(first: dict, verdict: dict):
    """An LLM that returns `first` for generation and `verdict` for the re-check."""
    calls = {"n": 0}

    class _Fake:
        available = True

        def complete_json(self, prompt, **kwargs):
            calls["n"] += 1
            return (first, None) if calls["n"] == 1 else (verdict, None)

    return _Fake(), calls


def test_multiple_response_is_rejected_when_the_recheck_disagrees(clean_db, monkeypatch):
    """A 4-mark all-or-nothing item keyed off mislabelled statements is worse
    than no item at all - it teaches the wrong answer. Observed live: the model
    keyed 'KNN is classification-only' as true. The re-check must catch it."""
    from examagent.services import question_gen as qg

    generated = {
        "prompt": "Which of the following are true of KNN?",
        "statements": ["KNN is classification-only.", "K is a hyperparameter.",
                       "KNN degrades in high dimensions.", "ANN is slower than KNN."],
        "correct_statements": [1, 3],          # statement 1 is false
        "explanation": "…",
    }
    fake, calls = _fake_llm(generated, {"true_statements": [2, 3]})
    monkeypatch.setattr(qg, "get_llm", lambda: fake)

    q = qg._llm_closed_question("knn", QuestionType.MULTIPLE_RESPONSE, 4, None)
    assert q is None, "a disputed key must not ship"
    assert calls["n"] == 2, "the item must be re-checked, not taken on trust"


def test_multiple_response_is_accepted_when_both_passes_agree(clean_db, monkeypatch):
    from examagent.services import question_gen as qg

    generated = {
        "prompt": "Which of the following are true of KNN?",
        "statements": ["K is a hyperparameter.", "KNN degrades in high dimensions.",
                       "KNN needs no training data.", "KNN is classification-only."],
        "correct_statements": [1, 2],
        "explanation": "…",
    }
    fake, _ = _fake_llm(generated, {"true_statements": [1, 2]})
    monkeypatch.setattr(qg, "get_llm", lambda: fake)

    q = qg._llm_closed_question("knn", QuestionType.MULTIPLE_RESPONSE, 4, None)
    assert q is not None
    assert q.question_type == QuestionType.MULTIPLE_RESPONSE
    keyed = next(o.text for o in q.options if o.key == q.correct_option)
    assert keyed == "1 and 2", "the key must follow the agreed truth flags"


def test_true_false_key_follows_the_models_flag_not_its_prose(clean_db, monkeypatch):
    from examagent.services import question_gen as qg

    fake, _ = _fake_llm({"prompt": "KNN stores the training set.", "answer": True,
                         "explanation": "It is a lazy learner."}, {})
    monkeypatch.setattr(qg, "get_llm", lambda: fake)

    q = qg._llm_closed_question("knn", QuestionType.TRUE_FALSE, 4, None)
    assert q is not None
    assert q.correct_option == "True"
    assert {o.key for o in q.options} == {"True", "False"}


def test_multiple_choice_needs_four_options_and_a_valid_index(clean_db, monkeypatch):
    from examagent.services import question_gen as qg

    fake, _ = _fake_llm({"prompt": "Which is true?", "options": ["a", "b"],
                         "correct_index": 0, "explanation": ""}, {})
    monkeypatch.setattr(qg, "get_llm", lambda: fake)

    assert qg._llm_closed_question("knn", QuestionType.MCQ, 4, None) is None, (
        "a malformed multiple-choice item must be rejected, not padded"
    )


# ------------------------------------------- never fall back to a written type
def test_a_requested_closed_format_that_fails_tries_the_others_before_giving_up(
    clean_db, monkeypatch,
):
    """Regression: observed live in Learning Path on a topic the bank has no
    facts for (svr). Multiple-response kept failing its verification pass,
    and the old code fell straight through to an open conceptual question -
    a format the real paper never uses. It must try True/False and multiple
    choice via the LLM first."""
    from examagent.services import question_gen as qg

    def fake_llm_closed(topic_id, qtype, difficulty, retrieval, avoid_prompts=None):
        if qtype == QuestionType.MULTIPLE_RESPONSE:
            return None  # simulates a persistent verification-pass rejection
        return qg.Question(
            id=f"fake:{qtype.value}", topic=topic_id, category=qg.Category.ML,
            question_type=qtype, difficulty=difficulty, priority=qg.Priority.MEDIUM,
            prompt="stub", options=[qg.AnswerOption(key="A", text="a"),
                                    qg.AnswerOption(key="B", text="b")],
            correct_option="A",
        )

    monkeypatch.setattr(qg, "_llm_closed_question", fake_llm_closed)
    monkeypatch.setattr(qg, "get_llm", lambda: type("F", (), {"available": True})())

    q = qg.generate_question("svr", QuestionType.MULTIPLE_RESPONSE, difficulty=4,
                             use_llm=True, use_rag=False)
    assert q.question_type in qg.EXAM_TYPES, (
        f"must stay closed-form, got {q.question_type.value} - "
        "the real paper has no written questions"
    )
    assert q.question_type != QuestionType.MULTIPLE_RESPONSE, (
        "the fake was rigged to fail exactly this one"
    )
