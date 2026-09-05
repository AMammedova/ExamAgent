"""The calculation engine must be exactly correct — it is the grading authority."""
from __future__ import annotations

import math

import pytest

from examagent.services.calc_engine import (
    GENERATORS,
    TOPIC_GENERATORS,
    fmt,
    generate_problem,
    grade,
    parse_number,
)
from examagent.models.schemas import MistakeType


@pytest.mark.parametrize("name", sorted(GENERATORS))
def test_generator_is_self_consistent(name: str) -> None:
    """A perfect answer sheet must score 10/10; a blank one must score 0."""
    for seed in range(12):
        problem = generate_problem(generator=name, seed=seed)
        spec = problem.spec()
        assert spec["parts"], f"{name} produced no gradeable parts"

        perfect = {p["key"]: fmt(p["answer"]) for p in spec["parts"]}
        result = grade(spec, perfect)
        assert result["score"] == pytest.approx(10.0, abs=0.01), (
            f"{name} seed={seed} scored {result['score']} for a perfect answer: "
            f"{[r for r in result['sub_scores'] if not r['correct']]}"
        )
        assert result["mistake_type"] == MistakeType.NONE.value

        blank = grade(spec, {})
        assert blank["score"] == 0.0
        assert blank["mistake_type"] == MistakeType.INCOMPLETE.value


@pytest.mark.parametrize("name", sorted(GENERATORS))
def test_generator_has_solution_and_metadata(name: str) -> None:
    problem = generate_problem(generator=name, seed=3)
    assert problem.statement.strip()
    assert problem.solution.strip()
    assert problem.concepts
    assert 1 <= problem.difficulty <= 6
    assert problem.category in ("Machine Learning", "Deep Learning")
    for part in problem.parts:
        assert part.label.strip()
        assert part.answer is not None


def test_mlp_backprop_matches_hand_computation() -> None:
    """Reproduce the exam sample: x=(1,2), W1=[[1,1],[1,-1]], W2=[1,2], b2=-3."""
    import numpy as np

    x = np.array([1.0, 2.0])
    W1 = np.array([[1.0, 1.0], [1.0, -1.0]])
    W2 = np.array([1.0, 2.0])
    b2 = -3.0
    y = 1.0
    eta = 0.1

    z1 = W1 @ x                      # [3, -1]
    a1 = np.maximum(z1, 0)           # [3, 0]
    z2 = float(W2 @ a1 + b2)         # 1*3 + 2*0 - 3 = 0
    yhat = 1 / (1 + math.exp(-z2))   # 0.5
    loss = -(y * math.log(yhat) + (1 - y) * math.log(1 - yhat))
    dz2 = yhat - y                   # -0.5
    dW2 = dz2 * a1                   # [-1.5, 0]
    dz1 = dz2 * W2 * (z1 > 0)        # [-0.5, 0]

    assert z1.tolist() == [3.0, -1.0]
    assert a1.tolist() == [3.0, 0.0]
    assert z2 == pytest.approx(0.0)
    assert yhat == pytest.approx(0.5)
    assert loss == pytest.approx(math.log(2), abs=1e-6)
    assert dW2.tolist() == pytest.approx([-1.5, 0.0])
    assert dz1[1] == 0.0, "the dead ReLU unit must receive zero gradient"
    assert (W2 - eta * dW2).tolist() == pytest.approx([1.15, 2.0])


def test_mlp_backprop_generator_always_has_one_dead_unit() -> None:
    """The 'which unit receives zero gradient' part must always be answerable."""
    for seed in range(30):
        problem = generate_problem(generator="mlp_backprop", seed=seed)
        parts = {p.key: p for p in problem.parts}
        assert parts["a1_2"].answer == 0.0, "hidden unit 2 must be ReLU-inactive"
        assert parts["dz1_2"].answer == 0.0
        assert parts["dead_unit"].answer == "unit 2"


def test_cnn_output_shape_formula() -> None:
    problem = generate_problem(generator="cnn_shape", seed=1)
    parts = {p.key: p for p in problem.parts}
    # the statement must contain everything needed to derive the answer
    assert "stride" in problem.statement.lower()
    assert "padding" in problem.statement.lower()
    assert isinstance(parts["out_hw"].answer, int)
    assert parts["out_hw"].answer > 0
    assert parts["params"].answer > 0


def test_receptive_field_recursion() -> None:
    """RF grows by (k-1)*jump and jump multiplies by stride."""
    for seed in range(15):
        problem = generate_problem(generator="receptive_field", seed=seed)
        parts = {p.key: p for p in problem.parts}
        assert parts["rf"].answer >= 3
        assert parts["jump"].answer >= 1
        # a stacked network always sees more than a single kernel
        assert parts["rf"].answer > 1


def test_metrics_precision_recall_are_not_swapped() -> None:
    problem = generate_problem(generator="metrics", seed=2)
    parts = {p.key: p for p in problem.parts}
    p_val, r_val = parts["precision"].answer, parts["recall"].answer
    f1 = parts["f1"].answer
    assert f1 == pytest.approx(2 * p_val * r_val / (p_val + r_val), abs=1e-6)
    assert parts["accuracy"].answer + parts["error_rate"].answer == pytest.approx(1.0)


def test_partial_credit_is_awarded() -> None:
    problem = generate_problem(generator="metrics", seed=4)
    spec = problem.spec()
    parts = spec["parts"]
    half = {p["key"]: fmt(p["answer"]) for p in parts[: len(parts) // 2]}
    result = grade(spec, half)
    assert 0 < result["score"] < 10
    assert result["n_correct"] == len(half)


def test_known_wrong_path_is_diagnosed_as_formula_error() -> None:
    """Answering recall where precision was asked must be caught specifically."""
    problem = generate_problem(generator="metrics", seed=6)
    spec = problem.spec()
    parts = {p["key"]: p for p in spec["parts"]}
    answers = {"precision": fmt(parts["recall"]["answer"])}
    result = grade(spec, answers)
    note = next(r["note"] for r in result["sub_scores"] if r["key"] == "precision")
    assert "recall" in note.lower()
    assert result["mistake_type"] in (MistakeType.FORMULA.value, MistakeType.CONCEPTUAL.value)


def test_near_miss_is_diagnosed_as_arithmetic() -> None:
    problem = generate_problem(generator="linreg", seed=8)
    spec = problem.spec()
    parts = {p["key"]: p for p in spec["parts"]}
    target = float(parts["b1"]["answer"])
    answers = {"b1": fmt(target * 1.06)}  # 6% off: right method, bad arithmetic
    result = grade(spec, answers)
    assert result["mistake_type"] == MistakeType.ARITHMETIC.value
    assert result["score"] > 0


def test_bias_omission_is_caught_in_parameter_counting() -> None:
    problem = generate_problem(generator="mlp_params", seed=5)
    spec = problem.spec()
    parts = {p["key"]: p for p in spec["parts"]}
    no_bias = next(e for e in parts["total"]["common_errors"])
    result = grade(spec, {"total": fmt(no_bias[0])})
    note = next(r["note"] for r in result["sub_scores"] if r["key"] == "total")
    assert "bias" in note.lower()


def test_tolerance_accepts_rounded_answers() -> None:
    problem = generate_problem(generator="mlp_backprop", seed=11)
    spec = problem.spec()
    answers = {}
    for p in spec["parts"]:
        if isinstance(p["answer"], float):
            answers[p["key"]] = f"{p['answer']:.3f}"
        else:
            answers[p["key"]] = fmt(p["answer"])
    result = grade(spec, answers)
    assert result["score"] >= 9.9, "3-decimal rounding must not be penalised"


def test_choice_parts_accept_natural_phrasing() -> None:
    problem = generate_problem(generator="mlp_backprop", seed=13)
    spec = problem.spec()
    answers = {"dead_unit": "2"}  # student writes just the number
    result = grade(spec, answers)
    dead = next(r for r in result["sub_scores"] if r["key"] == "dead_unit")
    assert dead["correct"]


@pytest.mark.parametrize(
    "text,expected",
    [("3.5", 3.5), ("  -2 ", -2.0), ("1e-3", 1e-3), ("3/4", 0.75),
     ("about 0.25", 0.25), ("", None), ("abc", None), ("0,5", 0.5)],
)
def test_parse_number(text: str, expected) -> None:
    got = parse_number(text)
    if expected is None:
        assert got is None
    else:
        assert got == pytest.approx(expected)


def test_every_registered_topic_generator_exists() -> None:
    for topic_id, names in TOPIC_GENERATORS.items():
        assert names, f"{topic_id} maps to no generator"
        for name in names:
            assert name in GENERATORS, f"{topic_id} -> unknown generator {name}"


def test_generate_problem_by_topic_keeps_topic_id() -> None:
    problem = generate_problem(topic_id="receptive_field", seed=1)
    assert problem.topic_id == "receptive_field"
