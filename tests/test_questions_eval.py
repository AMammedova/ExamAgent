"""Question generation, assertion-reason validity and answer evaluation."""
from __future__ import annotations

import pytest

from examagent.data.seed_questions import AR_OPTIONS, SEED_QUESTIONS, ar_key
from examagent.data.topics import topic_index
from examagent.models.schemas import Category, MistakeType, QuestionType
from examagent.services.assertion_engine import (
    AR_BANK,
    bank_for_topic,
    bank_stats,
    evaluate_assertion_reason,
    generate_assertion_reason,
)
from examagent.services.evaluator import evaluate, heuristic_evaluate
from examagent.services.question_gen import (
    _seed_to_question,
    available_types,
    generate_batch,
    generate_question,
)

TOPICS = topic_index()


# ---------------------------------------------------------------- seed bank
def test_seed_bank_meets_the_required_coverage() -> None:
    by_type: dict[str, int] = {}
    for q in SEED_QUESTIONS:
        by_type[q["question_type"]] = by_type.get(q["question_type"], 0) + 1
    assert by_type.get("assertion_reason", 0) >= 10
    assert by_type.get("calculation", 0) >= 10
    assert by_type.get("conceptual_reasoning", 0) >= 10
    assert len(SEED_QUESTIONS) >= 45

    ml = sum(1 for q in SEED_QUESTIONS if q["category"] == "Machine Learning")
    dl = sum(1 for q in SEED_QUESTIONS if q["category"] == "Deep Learning")
    assert ml >= 15 and dl >= 15

    for subject in ("cnn", "rnn", "tfm"):
        assert sum(1 for q in SEED_QUESTIONS if q["id"].startswith(subject)) >= 5


def test_every_seed_question_references_a_real_topic() -> None:
    for q in SEED_QUESTIONS:
        assert q["topic"] in TOPICS, f"{q['id']} -> unknown topic {q['topic']}"


def test_seed_open_questions_have_a_marking_rubric() -> None:
    for q in SEED_QUESTIONS:
        if q.get("calc_generator") or q["question_type"] == "assertion_reason":
            continue
        assert q["prompt"].strip(), f"{q['id']} has no prompt"
        assert q["model_answer"].strip(), f"{q['id']} has no model answer"
        assert len(q["expected_concepts"]) >= 3, f"{q['id']} has too few expected concepts"


# ---------------------------------------------------------------- A-R engine
def test_ar_key_derivation_covers_all_five_options() -> None:
    assert ar_key(True, True, True) == "A"
    assert ar_key(True, True, False) == "B"
    assert ar_key(True, False, False) == "C"
    assert ar_key(False, True, False) == "D"
    assert ar_key(False, False, False) == "E"


def test_ar_bank_items_are_internally_consistent() -> None:
    for item in AR_BANK:
        assert item.assertion.strip() and item.reason.strip()
        assert item.explanation.strip()
        assert item.key in {"A", "B", "C", "D", "E"}
        assert item.key == ar_key(item.a_true, item.r_true, item.explains)
        # a false assertion can never be *explained* by the reason
        if not item.a_true:
            assert not item.explains
        assert item.topic_id in TOPICS


def test_ar_bank_is_not_trivially_all_option_a() -> None:
    stats = bank_stats()
    assert stats["total"] >= 40
    by = stats["by_answer"]
    assert len(by) >= 4, "the bank must exercise at least four of the five patterns"
    assert by.get("A", 0) / stats["total"] < 0.75, "too many 'A' items to be examinable"
    # the two examiner traps must be present
    assert by.get("B", 0) >= 3, "need 'both true but unrelated' traps"
    assert by.get("D", 0) >= 3, "need 'false assertion, true reason' traps"


def test_ar_assertions_are_unique() -> None:
    assertions = [i.assertion for i in AR_BANK]
    assert len(assertions) == len(set(assertions))


def test_generated_ar_hides_the_answer_but_can_be_graded(clean_db) -> None:
    q = generate_assertion_reason("dropout", "Dropout", use_llm=False, seed=1)
    assert q is not None
    assert q.question_type == QuestionType.ASSERTION_REASON
    assert len(q.options) == 5
    assert q.correct_option in {"A", "B", "C", "D", "E"}
    # the prompt must not leak the truth values
    lowered = q.prompt.lower()
    for leak in ("assertion is true", "assertion is false", "correct answer",
                 "reason is true", "reason is false"):
        assert leak not in lowered

    right = evaluate_assertion_reason(q, q.correct_option)
    assert right["correct"]
    wrong_key = next(k for k in "ABCDE" if k != q.correct_option)
    assert not evaluate_assertion_reason(q, wrong_key)["correct"]
    assert len(right["breakdown"]) >= 2


def test_ar_option_text_is_the_standard_five() -> None:
    assert [o["key"] for o in AR_OPTIONS] == ["A", "B", "C", "D", "E"]


# ---------------------------------------------------------------- generation
@pytest.mark.parametrize("topic_id", [
    "backpropagation", "pca", "attention", "knn", "cnn_parameter_count",
    "lstm", "kmeans", "logistic_regression", "transformer", "dropout",
])
def test_generate_question_works_offline_for_key_topics(clean_db, topic_id: str) -> None:
    q = generate_question(topic_id, use_llm=False, seed=4)
    assert q.topic == topic_id
    assert q.prompt.strip()
    assert 1 <= q.difficulty <= 6
    assert q.estimated_time > 0


def test_calculation_questions_carry_a_gradeable_spec(clean_db) -> None:
    q = generate_question("backpropagation", QuestionType.CALCULATION, use_llm=False, seed=2)
    assert q.question_type == QuestionType.CALCULATION
    assert q.calc_spec and q.calc_spec["parts"]
    # the prompt must not contain the worked solution
    assert "dL/dz = y_hat - y" not in q.prompt


def test_generation_falls_back_when_a_topic_has_no_calc_engine(clean_db) -> None:
    q = generate_question("apriori", QuestionType.CALCULATION, use_llm=False, seed=1)
    assert q.question_type != QuestionType.CALCULATION or q.calc_spec is not None


def test_what_if_prompt_never_hands_the_llm_another_topics_examples(clean_db, monkeypatch) -> None:
    """Regression: the WHAT_IF template used to list concrete DL components
    (stride, padding, cell state...) as its only illustration, so an unrelated
    topic like ml_intro would get handed CNN/RNN vocabulary and the model
    would just ask about those instead of the actual topic. The prompt must
    now (a) tell the model to stay on-topic and (b) not offer a bare example
    list a topic without those components could latch onto."""
    from examagent.services import question_gen as qg

    captured: dict[str, str] = {}

    class _FakeLLM:
        available = True

        def complete_json(self, prompt, **kwargs):
            captured["prompt"] = prompt
            return ({"prompt": "stub", "model_answer": "stub",
                     "expected_concepts": ["stub"], "expected_reasoning": "stub",
                     "estimated_time": 120}, None)

    monkeypatch.setattr(qg, "get_llm", lambda: _FakeLLM())
    qg.generate_question("ml_intro", QuestionType.WHAT_IF, use_llm=True, use_rag=False)

    prompt = captured.get("prompt", "")
    assert prompt, "the fake LLM was never called"
    assert "ml_intro" in prompt or "Machine Learning Introduction" in prompt
    assert "belong to" in prompt or "belongs to the topic" in prompt, (
        "the topic-fidelity guardrail must be present in the prompt"
    )
    stray_examples = ("skip connection", "cell state", "attention, learning rate")
    assert not any(s in prompt for s in stray_examples), (
        "a bare list of another topic's components must not be handed to the model "
        "as if it applied here"
    )


def test_generate_batch_spreads_topics_and_types(clean_db) -> None:
    topics = ["backpropagation", "pca", "attention", "knn"]
    batch = generate_batch(topics, 12, use_llm=False, seed=9)
    assert len(batch) == 12
    assert len({q.topic for q in batch}) == len(topics)
    assert len({q.question_type for q in batch}) >= 3
    assert len({q.id for q in batch}) == 12, "questions must not repeat within a batch"


def test_dimension_focus_restricts_question_types(clean_db) -> None:
    batch = generate_batch(["backpropagation"], 6, use_llm=False, seed=3,
                           dimension_focus="calculation")
    assert all(q.question_type == QuestionType.CALCULATION for q in batch)
    assert all(q.dimension == "calculation" for q in batch)


def test_available_types_reports_offline_capability() -> None:
    types = available_types("backpropagation")
    assert QuestionType.CALCULATION in types
    assert QuestionType.ASSERTION_REASON in types


def test_category_is_inherited_from_the_topic_registry(clean_db) -> None:
    dl = generate_question("attention", use_llm=False, seed=1)
    ml = generate_question("kmeans", use_llm=False, seed=1)
    assert dl.category == Category.DL
    assert ml.category == Category.ML


# ---------------------------------------------------------------- evaluation
def test_blank_answer_scores_zero(clean_db) -> None:
    q = generate_question("overfitting", QuestionType.CONCEPTUAL, use_llm=False, seed=1)
    ev = evaluate(q, "", use_llm=False)
    assert ev.score == 0.0
    assert ev.mistake_type == MistakeType.INCOMPLETE
    assert ev.severity == "High"


def test_vague_answers_are_marked_harshly() -> None:
    """The brief is explicit: no high scores for vague answers."""
    vague = ("Dropout is good because it helps the model be better and more robust, "
             "which improves accuracy and makes it work nicely.")
    scores = []
    for raw in SEED_QUESTIONS:
        if raw.get("calc_generator") or raw["question_type"] == "assertion_reason":
            continue
        q = _seed_to_question(raw)
        if not q.expected_concepts:
            continue
        scores.append(heuristic_evaluate(q, vague).score)
    assert scores
    assert max(scores) < 5.0, f"a vague answer scored {max(scores)}"


def test_model_answers_score_well() -> None:
    scores = []
    for raw in SEED_QUESTIONS:
        if raw.get("calc_generator") or raw["question_type"] == "assertion_reason":
            continue
        q = _seed_to_question(raw)
        if not (q.model_answer and q.expected_concepts):
            continue
        scores.append(heuristic_evaluate(q, q.model_answer).score)
    assert scores
    assert sum(scores) / len(scores) >= 7.0
    assert min(scores) >= 4.0


def test_evaluation_discriminates_between_answer_qualities() -> None:
    raw = next(q for q in SEED_QUESTIONS if q["id"] == "dl_c1")
    q = _seed_to_question(raw)
    good = heuristic_evaluate(q, q.model_answer).score
    partial = heuristic_evaluate(
        q,
        "Without activation functions the network becomes linear, because composing "
        "linear layers gives another linear layer, therefore depth adds nothing.",
    ).score
    poor = heuristic_evaluate(q, "It just becomes worse and cannot learn well.").score
    assert good > partial > poor
    assert poor < 3.5


def test_spelling_variants_do_not_cost_marks() -> None:
    raw = next(q for q in SEED_QUESTIONS if q["id"] == "misc_1")
    q = _seed_to_question(raw)
    british = q.model_answer.replace("ization", "isation").replace("ized", "ised")
    assert heuristic_evaluate(q, british).score == pytest.approx(
        heuristic_evaluate(q, q.model_answer).score, abs=0.6)


def test_calculation_evaluation_gives_partial_credit(clean_db) -> None:
    q = generate_question("model_evaluation", QuestionType.CALCULATION,
                          use_llm=False, seed=5)
    parts = q.calc_spec["parts"]
    perfect = {p["key"]: str(p["answer"]) for p in parts}
    full = evaluate(q, perfect, use_llm=False)
    assert full.score == pytest.approx(10.0, abs=0.05)
    assert full.correct
    assert len(full.sub_scores) == len(parts)

    half = {p["key"]: str(p["answer"]) for p in parts[: len(parts) // 2]}
    partial = evaluate(q, half, use_llm=False)
    assert 0 < partial.score < 10
    assert partial.partial


def test_calculation_free_form_answers_are_parsed(clean_db) -> None:
    q = generate_question("model_evaluation", QuestionType.CALCULATION,
                          use_llm=False, seed=7)
    parts = q.calc_spec["parts"]
    text = "\n".join(f"{p['key']} = {p['answer']}" for p in parts)
    ev = evaluate(q, text, use_llm=False)
    assert ev.score == pytest.approx(10.0, abs=0.05)


def test_ar_evaluation_is_all_or_nothing(clean_db) -> None:
    q = generate_question("dropout", QuestionType.ASSERTION_REASON, use_llm=False, seed=2)
    right = evaluate(q, q.correct_option, use_llm=False)
    assert right.score == 10.0 and right.correct
    wrong_key = next(k for k in "ABCDE" if k != q.correct_option)
    wrong = evaluate(q, wrong_key, use_llm=False)
    assert wrong.score == 0.0
    assert wrong.mistake_type == MistakeType.REASONING
    assert len(wrong.sub_scores) == 3  # assertion / reason / explains


def test_evaluation_always_returns_actionable_feedback(clean_db) -> None:
    q = generate_question("pca", QuestionType.CONCEPTUAL, use_llm=False, seed=3)
    ev = evaluate(q, "PCA reduces dimensions somehow.", use_llm=False)
    assert ev.improvement.strip()
    assert ev.examiner_expects.strip()
    assert ev.score < 6
