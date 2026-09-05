"""Deterministic calculation engine.

Every problem here is generated *and solved* in Python, so grading is exact and
does not depend on an LLM. Each problem is broken into parts; each part is
graded independently which gives real partial credit and lets the evaluator
distinguish arithmetic errors from conceptual/formula/dimension errors.

Add a generator by writing a function `(rng) -> CalcProblem` and registering it
in GENERATORS.
"""
from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..models.schemas import MistakeType


# --------------------------------------------------------------- primitives
@dataclass
class CalcPart:
    """One gradeable sub-answer."""

    key: str
    label: str
    answer: Any
    kind: str = "number"  # number | int | choice | text
    tolerance: float = 1e-2  # absolute tolerance for floats
    rel_tolerance: float = 0.02
    unit: str = ""
    hint_formula: str = ""
    step: str = ""  # worked solution line
    weight: float = 1.0
    #: value -> (mistake type, explanation) for known wrong paths
    common_errors: list[tuple[Any, MistakeType, str]] = field(default_factory=list)
    choices: list[str] = field(default_factory=list)

    def format_answer(self) -> str:
        return fmt(self.answer)


@dataclass
class CalcProblem:
    problem_id: str
    topic_id: str
    title: str
    statement: str
    parts: list[CalcPart]
    solution: str
    difficulty: int = 4
    category: str = "Deep Learning"
    estimated_time: int = 300
    concepts: list[str] = field(default_factory=list)

    def spec(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "topic_id": self.topic_id,
            "generator": self.problem_id.split(":")[0],
            "parts": [
                {
                    "key": p.key,
                    "label": p.label,
                    "answer": p.answer,
                    "kind": p.kind,
                    "tolerance": p.tolerance,
                    "rel_tolerance": p.rel_tolerance,
                    "unit": p.unit,
                    "step": p.step,
                    "weight": p.weight,
                    "choices": p.choices,
                    "common_errors": [
                        [e[0], e[1].value if isinstance(e[1], MistakeType) else str(e[1]), e[2]]
                        for e in p.common_errors
                    ],
                }
                for p in self.parts
            ],
            "solution": self.solution,
            "concepts": self.concepts,
        }


def fmt(v: Any, nd: int = 4) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, float):
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return f"{v:.{nd}g}"
    if isinstance(v, (list, tuple, np.ndarray)):
        return "[" + ", ".join(fmt(x, nd) for x in np.asarray(v).tolist()) + "]"
    return str(v)


def mat(a: np.ndarray, nd: int = 4) -> str:
    """Render a matrix/vector as a markdown-safe bracketed literal."""
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        return "[" + ", ".join(fmt(x, nd) for x in a) + "]"
    rows = ["[" + ", ".join(fmt(x, nd) for x in row) + "]" for row in a]
    return "[" + ", ".join(rows) + "]"


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def relu(z: float) -> float:
    return max(0.0, z)


# --------------------------------------------------------------- generators
def gen_mlp_backprop(rng: random.Random) -> CalcProblem:
    """2-2-1 MLP: forward pass, BCE loss, backprop, one GD update.

    Mirrors the university exam sample. Values are randomised but the structure
    guarantees exactly one hidden unit is switched off by ReLU, so the
    'which unit receives zero gradient and why' part is always meaningful.
    """
    x = np.array([rng.choice([1, 2, 3]), rng.choice([1, 2, 3])], dtype=float)
    y = float(rng.choice([0, 1]))
    eta = rng.choice([0.1, 0.1, 0.05, 0.2])

    # build W1 so that unit 0 is active and unit 1 is dead under ReLU
    w1_row0 = np.array([rng.choice([1, 1, 2]), rng.choice([1, 1, 2])], dtype=float)
    w1_row1 = np.array([rng.choice([1, 2]), -rng.choice([1, 2, 3])], dtype=float)
    W1 = np.vstack([w1_row0, w1_row1])
    b1 = np.zeros(2)
    z1 = W1 @ x + b1
    tries = 0
    while (z1[0] <= 0 or z1[1] >= 0) and tries < 40:
        W1[1, 1] -= 1.0
        z1 = W1 @ x + b1
        tries += 1

    W2 = np.array([rng.choice([1, 2]), rng.choice([1, 2, 3])], dtype=float)
    b2 = float(rng.choice([-3, -2, -1]))

    a1 = np.maximum(z1, 0.0)
    z2 = float(W2 @ a1 + b2)
    yhat = sigmoid(z2)
    eps = 1e-12
    loss = -(y * math.log(yhat + eps) + (1 - y) * math.log(1 - yhat + eps))

    dz2 = yhat - y                      # dL/dz2 for sigmoid + BCE
    dW2 = dz2 * a1                      # (2,)
    db2 = dz2
    da1 = dz2 * W2
    dz1 = da1 * (z1 > 0).astype(float)  # ReLU derivative
    dW1 = np.outer(dz1, x)              # (2,2)

    W2_new = W2 - eta * dW2
    b2_new = b2 - eta * db2

    dead = int(np.argmin(z1)) if z1[1] <= 0 else 1

    statement = f"""A 2-2-1 multilayer perceptron is trained with **binary cross-entropy**.

- Input: **x = [{fmt(x[0])}, {fmt(x[1])}]^T**, target **y = {fmt(y)}**, learning rate **eta = {fmt(eta)}**
- Layer 1 (ReLU): **W1 = [[{fmt(W1[0,0])}, {fmt(W1[0,1])}], [{fmt(W1[1,0])}, {fmt(W1[1,1])}]]**, **b1 = [0, 0]^T**
- Layer 2 (sigmoid): **W2 = [{fmt(W2[0])}, {fmt(W2[1])}]**, **b2 = {fmt(b2)}**

Compute the full forward pass, the loss, the backward pass, and one gradient-descent update."""

    parts = [
        CalcPart("z1_1", "z1 (pre-activation of hidden unit 1)", float(z1[0]),
                 step=f"z1_1 = {fmt(W1[0,0])}*{fmt(x[0])} + {fmt(W1[0,1])}*{fmt(x[1])} + 0 = {fmt(z1[0])}",
                 hint_formula="z = W1 x + b1"),
        CalcPart("z1_2", "z2 (pre-activation of hidden unit 2)", float(z1[1]),
                 step=f"z1_2 = {fmt(W1[1,0])}*{fmt(x[0])} + {fmt(W1[1,1])}*{fmt(x[1])} + 0 = {fmt(z1[1])}"),
        CalcPart("a1_1", "a1 (ReLU output of hidden unit 1)", float(a1[0]),
                 step=f"a1_1 = max(0, {fmt(z1[0])}) = {fmt(a1[0])}"),
        CalcPart("a1_2", "a2 (ReLU output of hidden unit 2)", float(a1[1]),
                 step=f"a1_2 = max(0, {fmt(z1[1])}) = {fmt(a1[1])}",
                 common_errors=[(float(z1[1]), MistakeType.CONCEPTUAL,
                                 "You passed the negative pre-activation through unchanged - ReLU clamps it to 0.")]),
        CalcPart("z2", "Output pre-activation z (layer 2)", z2,
                 step=f"z = W2 . a1 + b2 = {fmt(W2[0])}*{fmt(a1[0])} + {fmt(W2[1])}*{fmt(a1[1])} + ({fmt(b2)}) = {fmt(z2)}"),
        CalcPart("yhat", "Predicted probability y_hat = sigmoid(z)", yhat, tolerance=5e-3,
                 step=f"y_hat = 1/(1+e^(-{fmt(z2)})) = {fmt(yhat)}"),
        CalcPart("loss", "Binary cross-entropy loss L", loss, tolerance=1e-2,
                 step=f"L = -[y log y_hat + (1-y) log(1-y_hat)] = {fmt(loss)}",
                 hint_formula="L = -(y log p + (1-y) log(1-p))"),
        CalcPart("dz2", "dL/dz (gradient at the output pre-activation)", dz2, tolerance=5e-3,
                 step=f"dL/dz = y_hat - y = {fmt(yhat)} - {fmt(y)} = {fmt(dz2)}",
                 hint_formula="sigmoid + BCE collapses to (y_hat - y)",
                 common_errors=[(float(y - yhat), MistakeType.FORMULA,
                                 "Sign flipped: for BCE with sigmoid the gradient is (y_hat - y), not (y - y_hat).")]),
        CalcPart("dW2_1", "dL/dW2[1]", float(dW2[0]), tolerance=5e-3,
                 step=f"dL/dW2_1 = dL/dz * a1_1 = {fmt(dz2)} * {fmt(a1[0])} = {fmt(dW2[0])}"),
        CalcPart("dW2_2", "dL/dW2[2]", float(dW2[1]), tolerance=5e-3,
                 step=f"dL/dW2_2 = dL/dz * a1_2 = {fmt(dz2)} * {fmt(a1[1])} = {fmt(dW2[1])}"),
        CalcPart("db2", "dL/db2", float(db2), tolerance=5e-3,
                 step=f"dL/db2 = dL/dz = {fmt(db2)}"),
        CalcPart("dz1_1", "dL/dz1 for hidden unit 1", float(dz1[0]), tolerance=5e-3,
                 step=f"dL/dz1_1 = dL/dz * W2_1 * 1[z1_1>0] = {fmt(dz2)} * {fmt(W2[0])} * 1 = {fmt(dz1[0])}"),
        CalcPart("dz1_2", "dL/dz1 for hidden unit 2", float(dz1[1]), tolerance=5e-3,
                 step="dL/dz1_2 = dL/dz * W2_2 * 1[z1_2>0] = ... * 0 = 0 (ReLU is inactive)"),
        CalcPart("dW1_11", "dL/dW1[1,1]", float(dW1[0, 0]), tolerance=5e-3,
                 step=f"dL/dW1_11 = dL/dz1_1 * x1 = {fmt(dz1[0])} * {fmt(x[0])} = {fmt(dW1[0,0])}"),
        CalcPart("W2_1_new", "Updated W2[1] after one GD step", float(W2_new[0]), tolerance=5e-3,
                 step=f"W2_1 <- {fmt(W2[0])} - {fmt(eta)}*{fmt(dW2[0])} = {fmt(W2_new[0])}",
                 hint_formula="w <- w - eta * dL/dw"),
        CalcPart("b2_new", "Updated b2 after one GD step", float(b2_new), tolerance=5e-3,
                 step=f"b2 <- {fmt(b2)} - {fmt(eta)}*{fmt(db2)} = {fmt(b2_new)}"),
        CalcPart("dead_unit", "Which hidden unit receives zero gradient?", f"unit {dead + 1}",
                 kind="choice", choices=["unit 1", "unit 2", "neither", "both"],
                 step=f"Hidden unit {dead+1} has z = {fmt(z1[dead])} <= 0, so ReLU'(z) = 0 and the "
                      f"gradient is blocked at that unit."),
    ]

    solution = "\n".join(
        ["**Forward pass**"] + [f"- {p.step}" for p in parts[:7]]
        + ["", "**Backward pass**"] + [f"- {p.step}" for p in parts[7:13]]
        + ["", "**Update**"] + [f"- {p.step}" for p in parts[13:]]
        + ["", f"Because ReLU is inactive for hidden unit {dead+1} (pre-activation {fmt(z1[dead])} <= 0), "
             "its local derivative is 0, so no gradient flows into its incoming weights - they are "
             "unchanged by this update."]
    )

    return CalcProblem(
        problem_id=f"mlp_backprop:{rng.randint(1000, 9999)}",
        topic_id="backpropagation",
        title="MLP forward pass, BCE loss and one backpropagation step",
        statement=statement,
        parts=parts,
        solution=solution,
        difficulty=5,
        category="Deep Learning",
        estimated_time=600,
        concepts=["chain rule", "ReLU derivative", "sigmoid+BCE gradient", "gradient descent update"],
    )


def gen_cnn_shape(rng: random.Random) -> CalcProblem:
    """Convolution output shape + parameter count (+ pooling)."""
    H = rng.choice([28, 32, 64, 128, 224])
    C_in = rng.choice([1, 3, 16, 32])
    k = rng.choice([3, 3, 5, 7])
    s = rng.choice([1, 1, 2])
    p = rng.choice([0, 1, k // 2])
    C_out = rng.choice([8, 16, 32, 64])
    pool = rng.choice([2, 2, 0])

    out = (H + 2 * p - k) // s + 1
    params = (k * k * C_in + 1) * C_out
    params_nobias = k * k * C_in * C_out
    after_pool = out // pool if pool else out
    fc_units = rng.choice([10, 64, 128])
    flat = after_pool * after_pool * C_out
    fc_params = flat * fc_units + fc_units

    statement = f"""A convolutional layer receives an input feature map of shape **{H} x {H} x {C_in}**
(height x width x channels).

The layer applies **{C_out} filters** of size **{k} x {k}**, stride **{s}**, padding **{p}**, each with a bias term.
""" + (f"\nA **{pool} x {pool} max-pooling** layer with stride {pool} follows the convolution.\n" if pool else "") + f"""
The result is then flattened and fed into a fully connected layer with **{fc_units} units** (with biases)."""

    parts = [
        CalcPart("out_hw", "Spatial output size of the conv layer (one side)", int(out), kind="int",
                 tolerance=0,
                 hint_formula="out = floor((H + 2P - K)/S) + 1",
                 step=f"out = floor(({H} + 2*{p} - {k})/{s}) + 1 = {out}",
                 common_errors=[
                     ((H - k) // s + 1, MistakeType.FORMULA, "You ignored the padding term 2P."),
                     ((H + 2 * p - k) // max(s, 1), MistakeType.FORMULA, "You forgot the +1 in the output-size formula."),
                 ]),
        CalcPart("out_channels", "Number of output channels", int(C_out), kind="int", tolerance=0,
                 step=f"Output depth equals the number of filters = {C_out}",
                 common_errors=[(C_in, MistakeType.CONCEPTUAL,
                                 "Output depth is set by the number of filters, not the input channels.")]),
        CalcPart("params", "Trainable parameters in the conv layer (incl. biases)", int(params), kind="int",
                 tolerance=0,
                 hint_formula="params = (K*K*C_in + 1) * C_out",
                 step=f"params = ({k}*{k}*{C_in} + 1) * {C_out} = {params}",
                 common_errors=[
                     (params_nobias, MistakeType.FORMULA, "You forgot the bias term (+1 per filter)."),
                     (k * k * C_out, MistakeType.DIMENSION, "You dropped the input-channel depth C_in from the kernel volume."),
                 ]),
    ]
    if pool:
        parts.append(CalcPart("after_pool", f"Spatial size after {pool}x{pool} max pooling", int(after_pool),
                              kind="int", tolerance=0,
                              step=f"{out} / {pool} = {after_pool}"))
        parts.append(CalcPart("pool_params", "Trainable parameters in the pooling layer", 0, kind="int",
                              tolerance=0,
                              step="Max pooling has no learnable parameters.",
                              common_errors=[(pool * pool, MistakeType.CONCEPTUAL,
                                              "Pooling is a fixed operation - it has zero trainable parameters.")]))
    parts.append(CalcPart("flat", "Length of the flattened vector entering the FC layer", int(flat), kind="int",
                          tolerance=0,
                          step=f"{after_pool} * {after_pool} * {C_out} = {flat}"))
    parts.append(CalcPart("fc_params", "Trainable parameters in the fully connected layer", int(fc_params),
                          kind="int", tolerance=0,
                          hint_formula="params = in*out + out",
                          step=f"{flat} * {fc_units} + {fc_units} = {fc_params}"))

    return CalcProblem(
        problem_id=f"cnn_shape:{rng.randint(1000, 9999)}",
        topic_id="cnn_parameter_count",
        title="CNN output shape and parameter count",
        statement=statement,
        parts=parts,
        solution="\n".join(f"- **{p.label}**: {p.step}" for p in parts)
        + "\n\nNote how the convolution's parameter count is independent of the input resolution "
          "(weight sharing), while the fully connected layer's count explodes with it.",
        difficulty=4,
        category="Deep Learning",
        estimated_time=300,
        concepts=["output size formula", "weight sharing", "parameter counting"],
    )


def gen_receptive_field(rng: random.Random) -> CalcProblem:
    """Receptive field of a stack of conv layers."""
    n = rng.choice([3, 3, 4])
    layers = []
    for _ in range(n):
        layers.append((rng.choice([3, 3, 5]), rng.choice([1, 1, 2])))

    rf, jump = 1, 1
    steps = []
    for i, (k, s) in enumerate(layers, 1):
        rf = rf + (k - 1) * jump
        jump = jump * s
        steps.append(f"Layer {i} (k={k}, s={s}): RF = RF + (k-1)*jump = {rf}, jump = {jump}")

    naive_sum = sum(k for k, _ in layers) - (n - 1)

    desc = ", ".join(f"conv {k}x{k} stride {s}" for k, s in layers)
    statement = f"""A CNN stacks {n} convolutional layers in sequence (no pooling):

**{desc}**

Every layer uses padding that preserves the formula's structure; you only need the receptive field growth.
Compute the receptive field of one unit in the final feature map, measured in input pixels."""

    parts = [
        CalcPart("rf", "Receptive field (one side, in input pixels)", int(rf), kind="int", tolerance=0,
                 hint_formula="RF_l = RF_{l-1} + (k_l - 1) * jump_{l-1};  jump_l = jump_{l-1} * s_l",
                 step="; ".join(steps),
                 common_errors=[(naive_sum, MistakeType.FORMULA,
                                 "You added kernel sizes without multiplying by the accumulated stride (jump).")]),
        CalcPart("jump", "Final jump (effective stride of the last feature map)", int(jump), kind="int",
                 tolerance=0,
                 step=f"jump = product of strides = {jump}"),
    ]

    return CalcProblem(
        problem_id=f"receptive_field:{rng.randint(1000, 9999)}",
        topic_id="receptive_field",
        title="Receptive field of a convolutional stack",
        statement=statement,
        parts=parts,
        solution="\n".join(f"- {s}" for s in steps)
        + "\n\nStride multiplies the *jump*, so later layers grow the receptive field much faster. "
          "This is why strided/pooled deep stacks see far more context than the sum of their kernel sizes.",
        difficulty=5,
        category="Deep Learning",
        estimated_time=240,
        concepts=["receptive field recursion", "jump/effective stride"],
    )


def gen_classification_metrics(rng: random.Random) -> CalcProblem:
    tp = rng.randint(15, 90)
    fp = rng.randint(3, 40)
    fn = rng.randint(3, 40)
    tn = rng.randint(30, 160)
    n = tp + fp + fn + tn

    acc = (tp + tn) / n
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2 * prec * rec / (prec + rec)
    spec = tn / (tn + fp)
    err = 1 - acc

    statement = f"""A binary classifier produces the following confusion matrix on a test set of {n} samples:

|                | Predicted Positive | Predicted Negative |
|----------------|--------------------|--------------------|
| **Actual Positive** | TP = {tp} | FN = {fn} |
| **Actual Negative** | FP = {fp} | TN = {tn} |

Compute the standard evaluation metrics (4 decimal places)."""

    parts = [
        CalcPart("accuracy", "Accuracy", acc, tolerance=5e-3,
                 hint_formula="(TP+TN)/(TP+TN+FP+FN)",
                 step=f"({tp}+{tn})/{n} = {fmt(acc)}"),
        CalcPart("precision", "Precision", prec, tolerance=5e-3,
                 hint_formula="TP/(TP+FP)",
                 step=f"{tp}/({tp}+{fp}) = {fmt(prec)}",
                 common_errors=[(tp / (tp + fn), MistakeType.FORMULA,
                                 "That is recall. Precision divides by predicted positives (TP+FP).")]),
        CalcPart("recall", "Recall (sensitivity)", rec, tolerance=5e-3,
                 hint_formula="TP/(TP+FN)",
                 step=f"{tp}/({tp}+{fn}) = {fmt(rec)}",
                 common_errors=[(tp / (tp + fp), MistakeType.FORMULA,
                                 "That is precision. Recall divides by actual positives (TP+FN).")]),
        CalcPart("f1", "F1 score", f1, tolerance=5e-3,
                 hint_formula="2PR/(P+R)",
                 step=f"2*{fmt(prec)}*{fmt(rec)}/({fmt(prec)}+{fmt(rec)}) = {fmt(f1)}",
                 common_errors=[((prec + rec) / 2, MistakeType.FORMULA,
                                 "F1 is the harmonic mean, not the arithmetic mean, of precision and recall.")]),
        CalcPart("specificity", "Specificity (true negative rate)", spec, tolerance=5e-3,
                 hint_formula="TN/(TN+FP)",
                 step=f"{tn}/({tn}+{fp}) = {fmt(spec)}"),
        CalcPart("error_rate", "Error rate", err, tolerance=5e-3,
                 step=f"1 - {fmt(acc)} = {fmt(err)}"),
    ]

    return CalcProblem(
        problem_id=f"metrics:{rng.randint(1000, 9999)}",
        topic_id="model_evaluation",
        title="Classification metrics from a confusion matrix",
        statement=statement,
        parts=parts,
        solution="\n".join(f"- **{p.label}** = {p.hint_formula} = {p.step}" for p in parts)
        + "\n\nExaminer note: state *which* denominator you used. Precision and recall differ only in "
          "whether you condition on the prediction or on the truth.",
        difficulty=3,
        category="Machine Learning",
        estimated_time=240,
        concepts=["precision vs recall", "harmonic mean", "confusion matrix"],
    )


def gen_knn(rng: random.Random) -> CalcProblem:
    k = rng.choice([1, 3, 3])
    pts = []
    labels = []
    for i in range(6):
        pts.append((rng.randint(0, 9), rng.randint(0, 9)))
        labels.append(rng.choice(["A", "B"]))
    # guarantee both classes present
    labels[0], labels[1] = "A", "B"
    q = (rng.randint(1, 8), rng.randint(1, 8))

    P = np.array(pts, dtype=float)
    qv = np.array(q, dtype=float)
    d_euc = np.sqrt(((P - qv) ** 2).sum(axis=1))
    d_man = np.abs(P - qv).sum(axis=1)
    order = np.argsort(d_euc)
    nearest = order[:k]
    votes = [labels[i] for i in nearest]
    pred = max(set(votes), key=votes.count)

    rows = "\n".join(
        f"| P{i+1} | {pts[i][0]} | {pts[i][1]} | {labels[i]} |" for i in range(len(pts))
    )
    statement = f"""A KNN classifier with **k = {k}** and **Euclidean distance** is given this training set:

| Point | x1 | x2 | Class |
|-------|----|----|-------|
{rows}

Classify the query point **q = ({q[0]}, {q[1]})**."""

    parts = [
        CalcPart("d1", "Euclidean distance from q to P1", float(d_euc[0]), tolerance=5e-3,
                 hint_formula="sqrt((x1-q1)^2 + (x2-q2)^2)",
                 step=f"sqrt(({pts[0][0]}-{q[0]})^2 + ({pts[0][1]}-{q[1]})^2) = {fmt(d_euc[0])}",
                 common_errors=[(float(d_man[0]), MistakeType.FORMULA,
                                 "That is the Manhattan (L1) distance; the question asked for Euclidean (L2).")]),
        CalcPart("d2", "Euclidean distance from q to P2", float(d_euc[1]), tolerance=5e-3,
                 step=f"= {fmt(d_euc[1])}"),
        CalcPart("nearest", f"Indices of the {k} nearest neighbour(s), e.g. P1,P3",
                 ",".join(f"P{i+1}" for i in sorted(nearest)), kind="text",
                 step="Sorted distances: " + ", ".join(
                     f"P{i+1}={fmt(d_euc[i])}" for i in order)),
        CalcPart("prediction", "Predicted class of q", pred, kind="choice", choices=["A", "B"],
                 step=f"Votes among the {k} nearest: {votes} -> {pred}"),
    ]

    return CalcProblem(
        problem_id=f"knn:{rng.randint(1000, 9999)}",
        topic_id="knn",
        title="KNN distance computation and prediction",
        statement=statement,
        parts=parts,
        solution="\n".join(f"- {p.step}" for p in parts)
        + "\n\nRemember: KNN is distance-based, so unscaled features with larger ranges dominate the "
          "distance. Feature scaling is not optional for KNN.",
        difficulty=3,
        category="Machine Learning",
        estimated_time=240,
        concepts=["Euclidean distance", "majority vote", "scale sensitivity"],
    )


def gen_bayes(rng: random.Random) -> CalcProblem:
    prior = rng.choice([0.01, 0.02, 0.005, 0.1])
    sens = rng.choice([0.95, 0.99, 0.9])
    fpr = rng.choice([0.05, 0.02, 0.1])

    p_pos = sens * prior + fpr * (1 - prior)
    post = sens * prior / p_pos

    statement = f"""A disease affects **{prior*100:.3g}%** of a population.
A diagnostic test has:

- Sensitivity P(test positive | disease) = **{sens}**
- False positive rate P(test positive | no disease) = **{fpr}**

A randomly selected person tests positive."""

    parts = [
        CalcPart("p_pos", "P(test positive) - the marginal / evidence", p_pos, tolerance=2e-3,
                 hint_formula="P(+) = P(+|D)P(D) + P(+|~D)P(~D)",
                 step=f"{sens}*{prior} + {fpr}*{1-prior:.4g} = {fmt(p_pos)}"),
        CalcPart("posterior", "P(disease | test positive)", post, tolerance=2e-3,
                 hint_formula="P(D|+) = P(+|D)P(D)/P(+)",
                 step=f"{sens}*{prior}/{fmt(p_pos)} = {fmt(post)}",
                 common_errors=[(sens, MistakeType.CONCEPTUAL,
                                 "You reported the sensitivity P(+|D). The question asks for the reversed "
                                 "conditional P(D|+) - this is the base-rate fallacy.")]),
    ]

    return CalcProblem(
        problem_id=f"bayes:{rng.randint(1000, 9999)}",
        topic_id="naive_bayes",
        title="Bayes theorem / posterior probability",
        statement=statement,
        parts=parts,
        solution="\n".join(f"- {p.step}" for p in parts)
        + f"\n\nThe posterior ({fmt(post)}) is far below the sensitivity ({sens}) because the prior is small - "
          "the classic base-rate effect that Naive Bayes questions test.",
        difficulty=4,
        category="Machine Learning",
        estimated_time=180,
        concepts=["Bayes rule", "prior", "likelihood", "evidence", "base rate"],
    )


def gen_naive_bayes_counts(rng: random.Random) -> CalcProblem:
    """Naive Bayes with categorical counts and Laplace smoothing."""
    n_spam, n_ham = rng.choice([40, 60]), rng.choice([60, 90])
    n = n_spam + n_ham
    c_free_spam = rng.randint(20, n_spam - 5)
    c_free_ham = rng.randint(2, 15)
    c_win_spam = rng.randint(10, n_spam - 10)
    c_win_ham = rng.randint(1, 10)

    p_spam = n_spam / n
    p_ham = n_ham / n
    p_free_spam = c_free_spam / n_spam
    p_win_spam = c_win_spam / n_spam
    p_free_ham = c_free_ham / n_ham
    p_win_ham = c_win_ham / n_ham

    score_spam = p_spam * p_free_spam * p_win_spam
    score_ham = p_ham * p_free_ham * p_win_ham
    post_spam = score_spam / (score_spam + score_ham)
    pred = "spam" if post_spam > 0.5 else "ham"

    statement = f"""A Naive Bayes spam filter is trained on {n} emails: **{n_spam} spam**, **{n_ham} ham**.

Word counts (number of emails of that class containing the word):

| Word | in spam | in ham |
|------|---------|--------|
| "free" | {c_free_spam} | {c_free_ham} |
| "win"  | {c_win_spam} | {c_win_ham} |

A new email contains both "free" and "win". Classify it (no smoothing needed)."""

    parts = [
        CalcPart("prior_spam", "Prior P(spam)", p_spam, tolerance=2e-3,
                 step=f"{n_spam}/{n} = {fmt(p_spam)}"),
        CalcPart("lik_free_spam", 'P("free" | spam)', p_free_spam, tolerance=2e-3,
                 step=f"{c_free_spam}/{n_spam} = {fmt(p_free_spam)}"),
        CalcPart("score_spam", 'Unnormalised score P(spam) * P("free"|spam) * P("win"|spam)',
                 score_spam, tolerance=1e-3, rel_tolerance=0.05,
                 hint_formula="conditional independence lets you multiply the likelihoods",
                 step=f"{fmt(p_spam)} * {fmt(p_free_spam)} * {fmt(p_win_spam)} = {fmt(score_spam)}"),
        CalcPart("score_ham", "Unnormalised score for ham", score_ham, tolerance=1e-3, rel_tolerance=0.05,
                 step=f"{fmt(p_ham)} * {fmt(p_free_ham)} * {fmt(p_win_ham)} = {fmt(score_ham)}"),
        CalcPart("posterior_spam", "Normalised P(spam | both words)", post_spam, tolerance=5e-3,
                 step=f"{fmt(score_spam)} / ({fmt(score_spam)} + {fmt(score_ham)}) = {fmt(post_spam)}"),
        CalcPart("prediction", "Predicted class", pred, kind="choice", choices=["spam", "ham"],
                 step=f"P(spam|x) = {fmt(post_spam)} -> {pred}"),
    ]

    return CalcProblem(
        problem_id=f"nb_counts:{rng.randint(1000, 9999)}",
        topic_id="naive_bayes",
        title="Naive Bayes classification from counts",
        statement=statement,
        parts=parts,
        solution="\n".join(f"- {p.step}" for p in parts)
        + "\n\nThe 'naive' assumption is conditional independence of features given the class - that is "
          "the only reason you may multiply the per-word likelihoods.",
        difficulty=4,
        category="Machine Learning",
        estimated_time=300,
        concepts=["conditional independence", "prior", "likelihood", "normalisation"],
    )


def gen_linear_regression(rng: random.Random) -> CalcProblem:
    n = 5
    xs = sorted(rng.sample(range(1, 12), n))
    slope_true = rng.choice([2, 3, -2, 1.5])
    ys = [round(slope_true * x + rng.choice([-2, -1, 0, 1, 2]) + 3, 2) for x in xs]

    X = np.array(xs, dtype=float)
    Y = np.array(ys, dtype=float)
    xbar, ybar = X.mean(), Y.mean()
    sxy = float(((X - xbar) * (Y - ybar)).sum())
    sxx = float(((X - xbar) ** 2).sum())
    b1 = sxy / sxx
    b0 = ybar - b1 * xbar
    pred_x = rng.choice(xs) + 1
    pred = b0 + b1 * pred_x
    resid = Y - (b0 + b1 * X)
    mse = float((resid ** 2).mean())
    sst = float(((Y - ybar) ** 2).sum())
    sse = float((resid ** 2).sum())
    r2 = 1 - sse / sst

    rows = " | ".join(str(v) for v in xs)
    rows_y = " | ".join(str(v) for v in ys)
    statement = f"""Fit a simple linear regression **y = b0 + b1*x** by ordinary least squares:

| x | {rows} |
|---|{'---|' * n}
| y | {rows_y} |

Compute the fitted line and evaluate it."""

    parts = [
        CalcPart("xbar", "Mean of x", float(xbar), tolerance=5e-3, step=f"= {fmt(xbar)}"),
        CalcPart("ybar", "Mean of y", float(ybar), tolerance=5e-3, step=f"= {fmt(ybar)}"),
        CalcPart("b1", "Slope b1", b1, tolerance=5e-3,
                 hint_formula="b1 = sum((x-xbar)(y-ybar)) / sum((x-xbar)^2)",
                 step=f"{fmt(sxy)} / {fmt(sxx)} = {fmt(b1)}"),
        CalcPart("b0", "Intercept b0", b0, tolerance=1e-2,
                 hint_formula="b0 = ybar - b1*xbar",
                 step=f"{fmt(ybar)} - {fmt(b1)}*{fmt(xbar)} = {fmt(b0)}"),
        CalcPart("pred", f"Prediction at x = {pred_x}", pred, tolerance=2e-2,
                 step=f"{fmt(b0)} + {fmt(b1)}*{pred_x} = {fmt(pred)}"),
        CalcPart("mse", "Mean squared error on the training data", mse, tolerance=2e-2, rel_tolerance=0.05,
                 hint_formula="MSE = (1/n) sum(y - yhat)^2",
                 step=f"= {fmt(mse)}",
                 common_errors=[(sse, MistakeType.FORMULA, "That is the sum of squared errors; MSE divides by n.")]),
        CalcPart("r2", "R^2", r2, tolerance=1e-2,
                 hint_formula="R^2 = 1 - SSE/SST",
                 step=f"1 - {fmt(sse)}/{fmt(sst)} = {fmt(r2)}"),
    ]

    return CalcProblem(
        problem_id=f"linreg:{rng.randint(1000, 9999)}",
        topic_id="linear_regression",
        title="Ordinary least squares by hand",
        statement=statement,
        parts=parts,
        solution="\n".join(f"- **{p.label}**: {p.step}" for p in parts),
        difficulty=3,
        category="Machine Learning",
        estimated_time=360,
        concepts=["least squares", "slope formula", "R^2", "MSE"],
    )


def gen_logistic_step(rng: random.Random) -> CalcProblem:
    w = np.array([rng.choice([0.5, -0.5, 1.0, 0.2]), rng.choice([0.5, -1.0, 0.3])], dtype=float)
    b = rng.choice([0.0, -0.5, 0.5])
    x = np.array([rng.choice([1, 2, -1]), rng.choice([1, 2, 3])], dtype=float)
    y = float(rng.choice([0, 1]))
    eta = rng.choice([0.1, 0.5])

    z = float(w @ x + b)
    p = sigmoid(z)
    loss = -(y * math.log(p) + (1 - y) * math.log(1 - p))
    dz = p - y
    gw = dz * x
    gb = dz
    w_new = w - eta * gw
    b_new = b - eta * gb
    label = 1 if p >= 0.5 else 0

    statement = f"""A logistic regression model has weights **w = [{fmt(w[0])}, {fmt(w[1])}]**, bias **b = {fmt(b)}**.

For the training sample **x = [{fmt(x[0])}, {fmt(x[1])}]** with true label **y = {fmt(y)}**
and learning rate **eta = {eta}**, perform one gradient descent step on the binary cross-entropy loss."""

    parts = [
        CalcPart("z", "Linear score z = w.x + b", z, tolerance=5e-3,
                 step=f"{fmt(w[0])}*{fmt(x[0])} + {fmt(w[1])}*{fmt(x[1])} + {fmt(b)} = {fmt(z)}"),
        CalcPart("p", "Predicted probability sigma(z)", p, tolerance=5e-3,
                 step=f"1/(1+e^-{fmt(z)}) = {fmt(p)}"),
        CalcPart("label", "Predicted class at threshold 0.5", label, kind="int", tolerance=0,
                 step=f"p = {fmt(p)} -> class {label}"),
        CalcPart("loss", "Binary cross-entropy loss", loss, tolerance=1e-2,
                 step=f"= {fmt(loss)}"),
        CalcPart("gw1", "dL/dw1", float(gw[0]), tolerance=5e-3,
                 hint_formula="dL/dw = (sigma(z) - y) * x",
                 step=f"({fmt(p)} - {fmt(y)}) * {fmt(x[0])} = {fmt(gw[0])}"),
        CalcPart("gb", "dL/db", float(gb), tolerance=5e-3,
                 step=f"{fmt(p)} - {fmt(y)} = {fmt(gb)}"),
        CalcPart("w1_new", "Updated w1", float(w_new[0]), tolerance=5e-3,
                 step=f"{fmt(w[0])} - {eta}*{fmt(gw[0])} = {fmt(w_new[0])}"),
        CalcPart("b_new", "Updated b", float(b_new), tolerance=5e-3,
                 step=f"{fmt(b)} - {eta}*{fmt(gb)} = {fmt(b_new)}"),
    ]

    return CalcProblem(
        problem_id=f"logistic:{rng.randint(1000, 9999)}",
        topic_id="logistic_regression",
        title="Logistic regression: one gradient descent step",
        statement=statement,
        parts=parts,
        solution="\n".join(f"- {p.step}" for p in parts)
        + "\n\nThe elegant fact worth quoting in the exam: with a sigmoid output and cross-entropy loss "
          "the gradient simplifies to (sigma(z) - y)x - the sigmoid derivative cancels.",
        difficulty=4,
        category="Machine Learning",
        estimated_time=300,
        concepts=["sigmoid", "cross entropy gradient", "gradient descent update"],
    )


def gen_attention(rng: random.Random) -> CalcProblem:
    d_k = rng.choice([2, 4])
    n = 2
    Q = np.array([[rng.choice([1, 0, 2]) for _ in range(d_k)] for _ in range(n)], dtype=float)
    K = np.array([[rng.choice([1, 0, 2]) for _ in range(d_k)] for _ in range(n)], dtype=float)
    V = np.array([[rng.choice([1, 2, 3, 4]) for _ in range(2)] for _ in range(n)], dtype=float)
    if np.allclose(Q, 0):
        Q[0, 0] = 1.0
    if np.allclose(K, 0):
        K[0, 0] = 1.0

    scores = Q @ K.T
    scaled = scores / math.sqrt(d_k)
    ex = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    attn = ex / ex.sum(axis=1, keepdims=True)
    out = attn @ V

    statement = f"""Scaled dot-product attention with **d_k = {d_k}** and {n} tokens.

- Q = {mat(Q)}
- K = {mat(K)}
- V = {mat(V)}

Compute Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V for the **first query row**."""

    parts = [
        CalcPart("score_11", "Raw score q1.k1 (before scaling)", float(scores[0, 0]), tolerance=1e-3,
                 step=f"q1 . k1 = {fmt(scores[0,0])}"),
        CalcPart("score_12", "Raw score q1.k2", float(scores[0, 1]), tolerance=1e-3,
                 step=f"q1 . k2 = {fmt(scores[0,1])}"),
        CalcPart("scaled_11", "Scaled score (q1.k1)/sqrt(d_k)", float(scaled[0, 0]), tolerance=5e-3,
                 hint_formula="divide by sqrt(d_k)",
                 step=f"{fmt(scores[0,0])}/sqrt({d_k}) = {fmt(scaled[0,0])}",
                 common_errors=[(float(scores[0, 0] / d_k), MistakeType.FORMULA,
                                 "You divided by d_k instead of sqrt(d_k).")]),
        CalcPart("alpha_11", "Attention weight alpha_11 after softmax", float(attn[0, 0]), tolerance=5e-3,
                 step=f"softmax over row 1 -> {fmt(attn[0,0])}"),
        CalcPart("alpha_12", "Attention weight alpha_12", float(attn[0, 1]), tolerance=5e-3,
                 step=f"-> {fmt(attn[0,1])}"),
        CalcPart("alpha_sum", "Sum of the attention weights in row 1", 1.0, tolerance=1e-3,
                 step="Softmax rows always sum to 1."),
        CalcPart("out_1", "First component of the output vector for query 1", float(out[0, 0]),
                 tolerance=1e-2,
                 step=f"{fmt(attn[0,0])}*{fmt(V[0,0])} + {fmt(attn[0,1])}*{fmt(V[1,0])} = {fmt(out[0,0])}"),
    ]

    return CalcProblem(
        problem_id=f"attention:{rng.randint(1000, 9999)}",
        topic_id="scaled_dot_product",
        title="Scaled dot-product attention by hand",
        statement=statement,
        parts=parts,
        solution="\n".join(f"- {p.step}" for p in parts)
        + f"\n\nWhy divide by sqrt(d_k)? With d_k = {d_k}, the dot product of two random vectors has "
          "variance proportional to d_k. Without scaling, large-magnitude logits push softmax into a "
          "saturated, near one-hot regime where gradients vanish. Dividing by sqrt(d_k) keeps the "
          "logit variance ~1 and the gradients usable.",
        difficulty=5,
        category="Deep Learning",
        estimated_time=360,
        concepts=["dot product similarity", "scaling by sqrt(d_k)", "softmax", "weighted sum of values"],
    )


def gen_mlp_params(rng: random.Random) -> CalcProblem:
    sizes = [rng.choice([4, 8, 10, 784]), rng.choice([16, 32, 64]), rng.choice([16, 32]),
             rng.choice([1, 3, 10])]
    layer_params = []
    for i in range(len(sizes) - 1):
        layer_params.append(sizes[i] * sizes[i + 1] + sizes[i + 1])
    total = sum(layer_params)
    no_bias = sum(sizes[i] * sizes[i + 1] for i in range(len(sizes) - 1))

    arch = " -> ".join(str(s) for s in sizes)
    statement = f"""A fully connected network has the architecture:

**{arch}**  (input -> hidden -> hidden -> output)

Every layer has a bias vector. Count the trainable parameters."""

    parts = [
        CalcPart("l1", f"Parameters in layer 1 ({sizes[0]} -> {sizes[1]})", layer_params[0], kind="int",
                 tolerance=0, hint_formula="in*out + out",
                 step=f"{sizes[0]}*{sizes[1]} + {sizes[1]} = {layer_params[0]}"),
        CalcPart("l2", f"Parameters in layer 2 ({sizes[1]} -> {sizes[2]})", layer_params[1], kind="int",
                 tolerance=0, step=f"{sizes[1]}*{sizes[2]} + {sizes[2]} = {layer_params[1]}"),
        CalcPart("l3", f"Parameters in layer 3 ({sizes[2]} -> {sizes[3]})", layer_params[2], kind="int",
                 tolerance=0, step=f"{sizes[2]}*{sizes[3]} + {sizes[3]} = {layer_params[2]}"),
        CalcPart("total", "Total trainable parameters", total, kind="int", tolerance=0,
                 step=f"{' + '.join(str(p) for p in layer_params)} = {total}",
                 common_errors=[(no_bias, MistakeType.FORMULA, "You omitted the bias terms.")]),
    ]

    return CalcProblem(
        problem_id=f"mlp_params:{rng.randint(1000, 9999)}",
        topic_id="mlp",
        title="MLP parameter counting",
        statement=statement,
        parts=parts,
        solution="\n".join(f"- {p.step}" for p in parts),
        difficulty=3,
        category="Deep Learning",
        estimated_time=180,
        concepts=["dense layer parameter formula", "biases"],
    )


def gen_lstm_params(rng: random.Random) -> CalcProblem:
    in_dim = rng.choice([50, 100, 300])
    hid = rng.choice([64, 128, 256])
    lstm = 4 * ((in_dim + hid) * hid + hid)
    gru = 3 * ((in_dim + hid) * hid + hid)
    rnn = (in_dim + hid) * hid + hid

    statement = f"""Compare recurrent cells with input dimension **{in_dim}** and hidden size **{hid}**
(single layer, biases included).

Recall that a vanilla RNN has one weight block, a GRU has three gates' worth,
and an LSTM has four."""

    parts = [
        CalcPart("rnn", "Parameters in a vanilla RNN cell", rnn, kind="int", tolerance=0,
                 hint_formula="(input + hidden)*hidden + hidden",
                 step=f"({in_dim}+{hid})*{hid} + {hid} = {rnn}"),
        CalcPart("lstm", "Parameters in an LSTM cell", lstm, kind="int", tolerance=0,
                 hint_formula="4 * [(input + hidden)*hidden + hidden]",
                 step=f"4 * [({in_dim}+{hid})*{hid} + {hid}] = {lstm}",
                 common_errors=[(3 * rnn, MistakeType.CONCEPTUAL,
                                 "3 blocks is a GRU. LSTM has four: forget, input, candidate and output.")]),
        CalcPart("gru", "Parameters in a GRU cell", gru, kind="int", tolerance=0,
                 step=f"3 * [({in_dim}+{hid})*{hid} + {hid}] = {gru}"),
        CalcPart("ratio", "LSTM parameters divided by GRU parameters", lstm / gru, tolerance=1e-2,
                 step=f"{lstm}/{gru} = {fmt(lstm/gru)} (i.e. 4/3)"),
    ]

    return CalcProblem(
        problem_id=f"lstm_params:{rng.randint(1000, 9999)}",
        topic_id="lstm",
        title="RNN / LSTM / GRU parameter counts",
        statement=statement,
        parts=parts,
        solution="\n".join(f"- {p.step}" for p in parts)
        + "\n\nGRU merges the forget and input gates into a single update gate and drops the separate "
          "cell state, giving ~3/4 of the LSTM's parameters with usually comparable accuracy.",
        difficulty=4,
        category="Deep Learning",
        estimated_time=240,
        concepts=["gate count", "recurrent parameter formula"],
    )


def gen_transformer_params(rng: random.Random) -> CalcProblem:
    d_model = rng.choice([256, 512, 768])
    heads = rng.choice([4, 8, 12])
    d_ff = d_model * 4
    d_head = d_model // heads

    attn = 4 * (d_model * d_model + d_model)   # W_Q, W_K, W_V, W_O with biases
    ffn = d_model * d_ff + d_ff + d_ff * d_model + d_model
    ln = 2 * (2 * d_model)
    block = attn + ffn + ln

    statement = f"""A Transformer encoder block uses **d_model = {d_model}**, **{heads} attention heads**,
and a feed-forward inner dimension **d_ff = {d_ff}**.

Assume W_Q, W_K, W_V and W_O are each ({d_model} x {d_model}) with a bias vector,
the FFN is two linear layers with biases, and there are 2 LayerNorms (gain + bias each)."""

    parts = [
        CalcPart("d_head", "Dimension per attention head", d_head, kind="int", tolerance=0,
                 hint_formula="d_model / h",
                 step=f"{d_model}/{heads} = {d_head}"),
        CalcPart("attn_params", "Parameters in the multi-head attention sublayer", attn, kind="int",
                 tolerance=0,
                 step=f"4 * ({d_model}*{d_model} + {d_model}) = {attn}",
                 common_errors=[(3 * (d_model * d_model + d_model), MistakeType.CONCEPTUAL,
                                 "You counted only Q, K, V and forgot the output projection W_O.")]),
        CalcPart("ffn_params", "Parameters in the feed-forward sublayer", ffn, kind="int", tolerance=0,
                 step=f"{d_model}*{d_ff}+{d_ff} + {d_ff}*{d_model}+{d_model} = {ffn}"),
        CalcPart("block_params", "Total parameters in the block (incl. LayerNorms)", block, kind="int",
                 tolerance=0,
                 step=f"{attn} + {ffn} + {ln} = {block}"),
        CalcPart("scale_factor", "By what factor does sqrt(d_k) scale attention logits here?",
                 math.sqrt(d_head), tolerance=1e-2,
                 step=f"sqrt(d_head) = sqrt({d_head}) = {fmt(math.sqrt(d_head))}"),
    ]

    return CalcProblem(
        problem_id=f"tfm_params:{rng.randint(1000, 9999)}",
        topic_id="transformer",
        title="Transformer block dimensions and parameter count",
        statement=statement,
        parts=parts,
        solution="\n".join(f"- {p.step}" for p in parts)
        + "\n\nMulti-head attention does not increase parameters relative to single-head attention of "
          "the same d_model - it *splits* the same projection into h subspaces, each of size d_model/h.",
        difficulty=5,
        category="Deep Learning",
        estimated_time=300,
        concepts=["multi-head splitting", "projection matrices", "FFN expansion ratio"],
    )


def gen_kmeans_iteration(rng: random.Random) -> CalcProblem:
    pts = [(rng.randint(0, 10), rng.randint(0, 10)) for _ in range(6)]
    c1 = (float(pts[0][0]), float(pts[0][1]))
    c2 = (float(pts[-1][0]), float(pts[-1][1]))
    P = np.array(pts, dtype=float)
    C = np.array([c1, c2])

    d = np.sqrt(((P[:, None, :] - C[None, :, :]) ** 2).sum(axis=2))
    assign = d.argmin(axis=1)
    new_c = []
    for j in range(2):
        m = P[assign == j]
        new_c.append(m.mean(axis=0) if len(m) else C[j])
    new_c = np.array(new_c)
    wcss = float(sum(((P[assign == j] - new_c[j]) ** 2).sum() for j in range(2)))

    rows = "\n".join(f"| P{i+1} | {p[0]} | {p[1]} |" for i, p in enumerate(pts))
    statement = f"""Run **one iteration of K-Means** (k = 2, Euclidean distance) on:

| Point | x | y |
|-------|---|---|
{rows}

Initial centroids: **C1 = ({fmt(c1[0])}, {fmt(c1[1])})**, **C2 = ({fmt(c2[0])}, {fmt(c2[1])})**."""

    parts = [
        CalcPart("assign", "Cluster assignment of P2 (1 or 2)", int(assign[1]) + 1, kind="int", tolerance=0,
                 step=f"d(P2,C1)={fmt(d[1,0])}, d(P2,C2)={fmt(d[1,1])} -> cluster {assign[1]+1}"),
        CalcPart("c1_new_x", "x-coordinate of the updated centroid C1", float(new_c[0, 0]), tolerance=1e-2,
                 hint_formula="mean of the points assigned to the cluster",
                 step=f"mean x of cluster 1 = {fmt(new_c[0,0])}"),
        CalcPart("c1_new_y", "y-coordinate of the updated centroid C1", float(new_c[0, 1]), tolerance=1e-2,
                 step=f"mean y of cluster 1 = {fmt(new_c[0,1])}"),
        CalcPart("c2_new_x", "x-coordinate of the updated centroid C2", float(new_c[1, 0]), tolerance=1e-2,
                 step=f"mean x of cluster 2 = {fmt(new_c[1,0])}"),
        CalcPart("wcss", "WCSS (inertia) after the update", wcss, tolerance=5e-2, rel_tolerance=0.03,
                 hint_formula="sum of squared distances of points to their own centroid",
                 step=f"= {fmt(wcss)}"),
    ]

    return CalcProblem(
        problem_id=f"kmeans:{rng.randint(1000, 9999)}",
        topic_id="kmeans",
        title="One K-Means iteration",
        statement=statement,
        parts=parts,
        solution="\n".join(f"- {p.step}" for p in parts)
        + "\n\nK-Means alternates the **assignment step** (each point to the nearest centroid) with the "
          "**update step** (each centroid to the mean of its members). Both steps can only decrease "
          "WCSS, which is why the algorithm converges - but only to a local optimum.",
        difficulty=4,
        category="Machine Learning",
        estimated_time=360,
        concepts=["assignment step", "update step", "WCSS", "local optimum"],
    )


def gen_pca_2d(rng: random.Random) -> CalcProblem:
    n = 4
    xs = [rng.randint(1, 9) for _ in range(n)]
    ys = [x + rng.choice([-1, 0, 1]) for x in xs]
    X = np.array([xs, ys], dtype=float).T
    mean = X.mean(axis=0)
    Xc = X - mean
    cov = (Xc.T @ Xc) / (n - 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    total = float(evals.sum())
    ratio = float(evals[0] / total) if total else 0.0

    rows = "\n".join(f"| {xs[i]} | {ys[i]} |" for i in range(n))
    statement = f"""Perform PCA on this 2-D dataset:

| x1 | x2 |
|----|----|
{rows}

Use the **sample covariance** (divide by n-1)."""

    parts = [
        CalcPart("mean_x1", "Mean of x1", float(mean[0]), tolerance=1e-2, step=f"= {fmt(mean[0])}"),
        CalcPart("cov_11", "Covariance matrix entry S[1,1] (variance of x1)", float(cov[0, 0]),
                 tolerance=2e-2, rel_tolerance=0.03,
                 step=f"= {fmt(cov[0,0])}",
                 common_errors=[(float((Xc[:, 0] ** 2).sum() / n), MistakeType.FORMULA,
                                 "You divided by n; the sample covariance divides by n-1.")]),
        CalcPart("cov_12", "Covariance matrix entry S[1,2]", float(cov[0, 1]), tolerance=2e-2,
                 rel_tolerance=0.03, step=f"= {fmt(cov[0,1])}"),
        CalcPart("eval_1", "Largest eigenvalue", float(evals[0]), tolerance=3e-2, rel_tolerance=0.03,
                 step=f"= {fmt(evals[0])}"),
        CalcPart("eval_2", "Smallest eigenvalue", float(evals[1]), tolerance=3e-2, rel_tolerance=0.05,
                 step=f"= {fmt(evals[1])}"),
        CalcPart("var_explained", "Fraction of variance explained by PC1", ratio, tolerance=1e-2,
                 hint_formula="lambda_1 / sum(lambda)",
                 step=f"{fmt(evals[0])}/{fmt(total)} = {fmt(ratio)}"),
    ]

    return CalcProblem(
        problem_id=f"pca:{rng.randint(1000, 9999)}",
        topic_id="pca",
        title="PCA: covariance, eigenvalues, variance explained",
        statement=statement,
        parts=parts,
        solution="\n".join(f"- {p.step}" for p in parts)
        + "\n\nPCA finds the orthogonal directions of maximal variance: eigenvectors of the covariance "
          "matrix, ordered by eigenvalue. Because it is variance-based it is scale sensitive - always "
          "standardise features first when their units differ.",
        difficulty=5,
        category="Machine Learning",
        estimated_time=420,
        concepts=["centering", "covariance matrix", "eigen-decomposition", "explained variance"],
    )


def gen_entropy_gain(rng: random.Random) -> CalcProblem:
    n_pos, n_neg = rng.randint(4, 12), rng.randint(4, 12)
    n = n_pos + n_neg
    l_pos = rng.randint(1, n_pos - 1)
    l_neg = rng.randint(0, n_neg - 1)
    r_pos, r_neg = n_pos - l_pos, n_neg - l_neg
    nl, nr = l_pos + l_neg, r_pos + r_neg

    def H(a: int, b: int) -> float:
        t = a + b
        if t == 0:
            return 0.0
        out = 0.0
        for c in (a, b):
            if c > 0:
                p = c / t
                out -= p * math.log2(p)
        return out

    def G(a: int, b: int) -> float:
        t = a + b
        if t == 0:
            return 0.0
        return 1 - (a / t) ** 2 - (b / t) ** 2

    h_parent = H(n_pos, n_neg)
    h_l, h_r = H(l_pos, l_neg), H(r_pos, r_neg)
    h_child = nl / n * h_l + nr / n * h_r
    ig = h_parent - h_child
    gini_parent = G(n_pos, n_neg)

    statement = f"""A decision tree node contains **{n_pos} positive** and **{n_neg} negative** samples.

A candidate split divides it into:

- Left child: {l_pos} positive, {l_neg} negative
- Right child: {r_pos} positive, {r_neg} negative

Use log base 2."""

    parts = [
        CalcPart("h_parent", "Entropy of the parent node", h_parent, tolerance=5e-3,
                 hint_formula="H = -sum p log2 p",
                 step=f"H({n_pos}+,{n_neg}-) = {fmt(h_parent)}"),
        CalcPart("h_left", "Entropy of the left child", h_l, tolerance=5e-3,
                 step=f"= {fmt(h_l)}"),
        CalcPart("h_weighted", "Weighted entropy of the children", h_child, tolerance=5e-3,
                 hint_formula="(n_L/n)H_L + (n_R/n)H_R",
                 step=f"({nl}/{n})*{fmt(h_l)} + ({nr}/{n})*{fmt(h_r)} = {fmt(h_child)}",
                 common_errors=[((h_l + h_r) / 2, MistakeType.FORMULA,
                                 "Children must be weighted by their sample counts, not averaged equally.")]),
        CalcPart("info_gain", "Information gain of the split", ig, tolerance=5e-3,
                 step=f"{fmt(h_parent)} - {fmt(h_child)} = {fmt(ig)}"),
        CalcPart("gini_parent", "Gini impurity of the parent", gini_parent, tolerance=5e-3,
                 hint_formula="1 - sum p^2",
                 step=f"= {fmt(gini_parent)}"),
    ]

    return CalcProblem(
        problem_id=f"entropy:{rng.randint(1000, 9999)}",
        topic_id="decision_trees",
        title="Entropy, information gain and Gini impurity",
        statement=statement,
        parts=parts,
        solution="\n".join(f"- {p.step}" for p in parts)
        + "\n\nA tree greedily picks the split with the highest information gain (or lowest weighted "
          "Gini). Information gain is always >= 0 for a valid split.",
        difficulty=4,
        category="Machine Learning",
        estimated_time=360,
        concepts=["entropy", "weighted child impurity", "information gain", "Gini"],
    )


def gen_scaling(rng: random.Random) -> CalcProblem:
    vals = [rng.randint(1, 40) for _ in range(5)]
    a = np.array(vals, dtype=float)
    mu = float(a.mean())
    sd_pop = float(a.std())
    lo, hi = float(a.min()), float(a.max())
    target = vals[rng.randint(0, 4)]
    z = (target - mu) / sd_pop
    mm = (target - lo) / (hi - lo)

    statement = f"""A feature has the values **{vals}**.

Compute standardization and min-max normalization (use the population standard deviation, divide by n)."""

    parts = [
        CalcPart("mean", "Mean", mu, tolerance=1e-2, step=f"= {fmt(mu)}"),
        CalcPart("std", "Standard deviation (population)", sd_pop, tolerance=2e-2, rel_tolerance=0.03,
                 step=f"= {fmt(sd_pop)}"),
        CalcPart("zscore", f"Standardized value (z-score) of {target}", z, tolerance=2e-2,
                 hint_formula="z = (x - mu)/sigma",
                 step=f"({target} - {fmt(mu)})/{fmt(sd_pop)} = {fmt(z)}"),
        CalcPart("minmax", f"Min-max normalized value of {target}", mm, tolerance=1e-2,
                 hint_formula="(x - min)/(max - min)",
                 step=f"({target} - {fmt(lo)})/({fmt(hi)} - {fmt(lo)}) = {fmt(mm)}"),
    ]

    return CalcProblem(
        problem_id=f"scaling:{rng.randint(1000, 9999)}",
        topic_id="feature_scaling",
        title="Standardization vs min-max normalization",
        statement=statement,
        parts=parts,
        solution="\n".join(f"- {p.step}" for p in parts)
        + "\n\nCritical exam point: fit the scaler on the **training set only** and apply the same "
          "statistics to validation/test - otherwise you leak information.",
        difficulty=2,
        category="Machine Learning",
        estimated_time=180,
        concepts=["z-score", "min-max", "fit on train only"],
    )


def gen_gd_step(rng: random.Random) -> CalcProblem:
    a = rng.choice([1, 2, 3])
    b = rng.choice([-4, -6, 2, 4])
    w0 = rng.choice([0.0, 1.0, 2.0, -1.0])
    eta = rng.choice([0.1, 0.2, 0.5])
    lam = rng.choice([0.0, 0.1, 0.5])

    # f(w) = a w^2 + b w  (+ L2 penalty lam w^2)
    grad = 2 * a * w0 + b + 2 * lam * w0
    w1 = w0 - eta * grad
    grad2 = 2 * a * w1 + b + 2 * lam * w1
    w2 = w1 - eta * grad2
    wstar = -b / (2 * (a + lam))

    pen = f" + {lam}*w^2 (L2 penalty)" if lam else ""
    statement = f"""Minimise the objective **J(w) = {a}w^2 + ({b})w{pen}** by gradient descent.

Start at **w0 = {fmt(w0)}** with learning rate **eta = {eta}**."""

    parts = [
        CalcPart("grad0", "dJ/dw at w0", grad, tolerance=5e-3,
                 hint_formula=f"dJ/dw = {2*a}w + ({b})" + (f" + {2*lam}w" if lam else ""),
                 step=f"= {fmt(grad)}"),
        CalcPart("w1", "w after one update", w1, tolerance=5e-3,
                 step=f"{fmt(w0)} - {eta}*{fmt(grad)} = {fmt(w1)}",
                 common_errors=[(w0 + eta * grad, MistakeType.FORMULA,
                                 "Sign error: gradient descent subtracts the gradient (moves downhill).")]),
        CalcPart("w2", "w after two updates", w2, tolerance=1e-2,
                 step=f"{fmt(w1)} - {eta}*{fmt(grad2)} = {fmt(w2)}"),
        CalcPart("w_star", "Analytic minimiser w*", wstar, tolerance=1e-2,
                 hint_formula="set the gradient to zero",
                 step=f"w* = -b/(2(a+lambda)) = {fmt(wstar)}"),
    ]
    if lam:
        parts.append(CalcPart(
            "shrink", "Does the L2 penalty move w* toward zero? (yes/no)", "yes",
            kind="choice", choices=["yes", "no"],
            step="Yes - the penalty adds curvature, so the minimiser shrinks toward the origin."))

    return CalcProblem(
        problem_id=f"gd:{rng.randint(1000, 9999)}",
        topic_id="gradient_descent",
        title="Gradient descent steps on a quadratic objective",
        statement=statement,
        parts=parts,
        solution="\n".join(f"- {p.step}" for p in parts)
        + f"\n\nWith eta = {eta} and curvature {2*(a+lam)}, the update is stable iff eta < 2/curvature = "
          f"{fmt(2/(2*(a+lam)))}. A learning rate above that makes the iterates oscillate and diverge.",
        difficulty=3,
        category="Deep Learning",
        estimated_time=240,
        concepts=["derivative", "update rule", "stability condition", "L2 shrinkage"],
    )


def gen_softmax_ce(rng: random.Random) -> CalcProblem:
    logits = np.array([rng.choice([0.0, 1.0, 2.0, -1.0, 3.0]) for _ in range(3)])
    if len(set(logits.tolist())) == 1:
        logits[0] += 1.0
    true_c = rng.randint(0, 2)
    ex = np.exp(logits - logits.max())
    p = ex / ex.sum()
    loss = -math.log(p[true_c])
    grad = p.copy()
    grad[true_c] -= 1.0

    statement = f"""A 3-class classifier outputs the logits **z = [{fmt(logits[0])}, {fmt(logits[1])}, {fmt(logits[2])}]**.

The true class is **class {true_c + 1}** (one-hot target).

Compute the softmax probabilities, the cross-entropy loss, and the gradient dL/dz."""

    parts = [
        CalcPart("p1", "softmax probability of class 1", float(p[0]), tolerance=5e-3,
                 hint_formula="exp(z_i)/sum(exp(z_j))",
                 step=f"= {fmt(p[0])}"),
        CalcPart("p2", "softmax probability of class 2", float(p[1]), tolerance=5e-3,
                 step=f"= {fmt(p[1])}"),
        CalcPart("p_sum", "Sum of the three probabilities", 1.0, tolerance=1e-3,
                 step="Softmax outputs form a probability distribution: they sum to 1."),
        CalcPart("loss", "Cross-entropy loss", loss, tolerance=1e-2,
                 hint_formula="L = -log p_true",
                 step=f"-log({fmt(p[true_c])}) = {fmt(loss)}"),
        CalcPart("grad_true", f"dL/dz for the true class (class {true_c+1})", float(grad[true_c]),
                 tolerance=5e-3,
                 hint_formula="dL/dz = p - y",
                 step=f"{fmt(p[true_c])} - 1 = {fmt(grad[true_c])}"),
    ]

    return CalcProblem(
        problem_id=f"softmax:{rng.randint(1000, 9999)}",
        topic_id="loss_functions",
        title="Softmax and cross-entropy",
        statement=statement,
        parts=parts,
        solution="\n".join(f"- {p.step}" for p in parts)
        + "\n\nThe gradient of softmax + cross-entropy w.r.t. the logits is simply **p - y**. "
          "Softmax is shift-invariant, which is why implementations subtract max(z) for numerical stability.",
        difficulty=4,
        category="Deep Learning",
        estimated_time=240,
        concepts=["softmax", "cross entropy", "p - y gradient", "shift invariance"],
    )


def gen_conv_output_volume(rng: random.Random) -> CalcProblem:
    """Encoder/decoder resolution reasoning as seen in the CNN exam patterns."""
    H = rng.choice([224, 256, 128])
    n_down = rng.choice([3, 4, 5])
    obj = rng.choice([8, 16, 32])

    sizes = [H]
    for _ in range(n_down):
        sizes.append(sizes[-1] // 2)
    final = sizes[-1]
    stride_total = 2 ** n_down
    obj_px = obj / stride_total

    statement = f"""An encoder halves the spatial resolution **{n_down} times** (stride-2 stages),
starting from a **{H} x {H}** input.

An object in the input image occupies about **{obj} x {obj}** pixels."""

    parts = [
        CalcPart("final_res", "Spatial resolution of the final feature map (one side)", final, kind="int",
                 tolerance=0,
                 step=" -> ".join(str(s) for s in sizes)),
        CalcPart("total_stride", "Total downsampling factor", stride_total, kind="int", tolerance=0,
                 step=f"2^{n_down} = {stride_total}"),
        CalcPart("obj_size", "Size of that object on the final feature map (in cells)", obj_px,
                 tolerance=1e-2,
                 step=f"{obj}/{stride_total} = {fmt(obj_px)}",
                 common_errors=[(obj * stride_total, MistakeType.DIMENSION,
                                 "Downsampling divides object size by the stride; it does not multiply it.")]),
        CalcPart("detectable", "Is that object still well localised at this resolution? (yes/no)",
                 "yes" if obj_px >= 1 else "no", kind="choice", choices=["yes", "no"],
                 step=f"{fmt(obj_px)} cells: " + (
                     "at least one cell, so it survives - though detail is coarse."
                     if obj_px >= 1 else
                     "less than one cell, so small objects are lost. This is why detectors use "
                     "multi-scale features / skip connections from higher-resolution layers.")),
    ]

    return CalcProblem(
        problem_id=f"conv_volume:{rng.randint(1000, 9999)}",
        topic_id="object_detection",
        title="Feature map resolution and small object detection",
        statement=statement,
        parts=parts,
        solution="\n".join(f"- {p.step}" for p in parts)
        + "\n\nThis is the standard trade-off: deeper layers give large receptive fields and semantics "
          "but coarse resolution. Skip connections / FPNs recombine fine spatial detail with deep "
          "semantics, which is why they matter for small objects.",
        difficulty=4,
        category="Deep Learning",
        estimated_time=240,
        concepts=["downsampling factor", "resolution vs semantics", "small object problem"],
    )


def gen_dropout_expectation(rng: random.Random) -> CalcProblem:
    p = rng.choice([0.2, 0.5, 0.3])
    n = rng.choice([100, 200, 512])
    acts = [rng.choice([1.0, 2.0, 4.0]) for _ in range(4)]
    s = float(sum(acts))
    keep = 1 - p
    expected = s * keep
    inverted = s  # inverted dropout keeps the expectation at train time

    statement = f"""A hidden layer with **{n} units** uses dropout with drop probability **p = {p}**.

Consider four of its activations: **{acts}** (sum = {fmt(s)})."""

    parts = [
        CalcPart("keep_p", "Keep probability", keep, tolerance=1e-3, step=f"1 - {p} = {fmt(keep)}"),
        CalcPart("expected_units", "Expected number of ACTIVE units in the layer", n * keep,
                 tolerance=1e-2, rel_tolerance=0.01,
                 step=f"{n} * {fmt(keep)} = {fmt(n*keep)}"),
        CalcPart("expected_sum", "Expected sum of those 4 activations at training time (vanilla dropout)",
                 expected, tolerance=1e-2,
                 step=f"{fmt(s)} * {fmt(keep)} = {fmt(expected)}"),
        CalcPart("inverted_sum",
                 "Expected sum with INVERTED dropout (scaling by 1/keep during training)", inverted,
                 tolerance=1e-2,
                 step=f"{fmt(expected)} / {fmt(keep)} = {fmt(inverted)} - the expectation is preserved, "
                      "so no rescaling is needed at test time.",
                 common_errors=[(expected, MistakeType.CONCEPTUAL,
                                 "Inverted dropout divides by the keep probability during training "
                                 "precisely so the expectation matches test time.")]),
        CalcPart("test_time", "Is dropout applied at test time? (yes/no)", "no", kind="choice",
                 choices=["yes", "no"],
                 step="No. Dropout is a training-time regulariser; at inference the full network is used."),
    ]

    return CalcProblem(
        problem_id=f"dropout:{rng.randint(1000, 9999)}",
        topic_id="dropout",
        title="Dropout expectation and train/test behaviour",
        statement=statement,
        parts=parts,
        solution="\n".join(f"- {p.step}" for p in parts),
        difficulty=3,
        category="Deep Learning",
        estimated_time=200,
        concepts=["keep probability", "expectation preservation", "train vs test behaviour"],
    )


GENERATORS: dict[str, Callable[[random.Random], CalcProblem]] = {
    "mlp_backprop": gen_mlp_backprop,
    "cnn_shape": gen_cnn_shape,
    "receptive_field": gen_receptive_field,
    "metrics": gen_classification_metrics,
    "knn": gen_knn,
    "bayes": gen_bayes,
    "nb_counts": gen_naive_bayes_counts,
    "linreg": gen_linear_regression,
    "logistic": gen_logistic_step,
    "attention": gen_attention,
    "mlp_params": gen_mlp_params,
    "lstm_params": gen_lstm_params,
    "tfm_params": gen_transformer_params,
    "kmeans": gen_kmeans_iteration,
    "pca": gen_pca_2d,
    "entropy": gen_entropy_gain,
    "scaling": gen_scaling,
    "gd": gen_gd_step,
    "softmax": gen_softmax_ce,
    "conv_volume": gen_conv_output_volume,
    "dropout": gen_dropout_expectation,
}

#: topic id -> generators that can examine it
TOPIC_GENERATORS: dict[str, list[str]] = {
    "backpropagation": ["mlp_backprop"],
    "chain_rule": ["mlp_backprop"],
    "forward_propagation": ["mlp_backprop", "mlp_params"],
    "loss_functions": ["softmax", "mlp_backprop"],
    "activation_functions": ["mlp_backprop", "softmax"],
    "mlp": ["mlp_params", "mlp_backprop"],
    "neural_networks": ["mlp_params"],
    "gradient_descent": ["gd", "mlp_backprop"],
    "learning_rate": ["gd"],
    "optimizers": ["gd"],
    "weight_decay": ["gd"],
    "regularization_ml": ["gd"],
    "regularization_dl": ["dropout", "gd"],
    "dropout": ["dropout"],
    "cnn_basics": ["cnn_shape"],
    "convolution": ["cnn_shape"],
    "stride_padding": ["cnn_shape"],
    "pooling": ["cnn_shape"],
    "cnn_parameter_count": ["cnn_shape"],
    "receptive_field": ["receptive_field"],
    "cnn_architectures": ["cnn_shape", "receptive_field"],
    "object_detection": ["conv_volume"],
    "residual_connections": ["conv_volume"],
    "rnn": ["lstm_params"],
    "lstm": ["lstm_params"],
    "gru": ["lstm_params"],
    "vanishing_gradients": ["lstm_params"],
    "attention": ["attention"],
    "scaled_dot_product": ["attention"],
    "transformer": ["tfm_params", "attention"],
    "self_vs_cross_attention": ["attention"],
    "bert": ["tfm_params"],
    "gpt": ["tfm_params"],
    "model_evaluation": ["metrics"],
    "confusion_matrix": ["metrics"],
    "knn": ["knn"],
    "naive_bayes": ["bayes", "nb_counts"],
    "linear_regression": ["linreg"],
    "multiple_linear_regression": ["linreg"],
    "polynomial_regression": ["linreg"],
    "logistic_regression": ["logistic"],
    "kmeans": ["kmeans"],
    "hierarchical_clustering": ["kmeans"],
    "dbscan": ["kmeans"],
    "gmm": ["kmeans", "bayes"],
    "expectation_maximization": ["bayes"],
    "pca": ["pca"],
    "lda": ["pca"],
    "kernel_pca": ["pca"],
    "decision_trees": ["entropy"],
    "random_forests": ["entropy"],
    "boosting": ["entropy"],
    "decision_tree_regression": ["entropy"],
    "feature_scaling": ["scaling"],
    "data_preprocessing": ["scaling"],
    "density_estimation": ["bayes"],
    "kde": ["bayes"],
    "svm": ["knn"],
    "batch_normalization": ["scaling"],
    "cross_validation": ["metrics"],
    "overfitting": ["metrics"],
}


def generate_problem(
    topic_id: str | None = None,
    generator: str | None = None,
    seed: int | None = None,
) -> CalcProblem:
    """Generate a calculation problem for a topic (or a named generator)."""
    rng = random.Random(seed)
    if generator and generator in GENERATORS:
        name = generator
    elif topic_id and topic_id in TOPIC_GENERATORS:
        name = rng.choice(TOPIC_GENERATORS[topic_id])
    else:
        name = rng.choice(list(GENERATORS))
    problem = GENERATORS[name](rng)
    if topic_id:
        problem.topic_id = topic_id
    return problem


def topic_has_calculation(topic_id: str) -> bool:
    return topic_id in TOPIC_GENERATORS


# --------------------------------------------------------------- grading
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_number(text: str) -> float | None:
    """Extract a single number from free-form student input."""
    if text is None:
        return None
    t = str(text).strip().lower()
    if not t:
        return None
    t = t.replace(",", ".") if t.count(",") == 1 and "." not in t else t.replace(",", "")
    # simple fractions like 3/4
    frac = re.fullmatch(r"\s*([-+]?\d*\.?\d+)\s*/\s*([-+]?\d*\.?\d+)\s*", t)
    if frac:
        try:
            den = float(frac.group(2))
            return float(frac.group(1)) / den if den else None
        except ValueError:
            return None
    m = _NUM_RE.search(t)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _matches(part: dict[str, Any], student: str) -> tuple[bool, float | None]:
    kind = part.get("kind", "number")
    ans = part.get("answer")
    s = (student or "").strip()
    if not s:
        return False, None

    if kind in ("choice", "text"):
        norm_s = re.sub(r"[^a-z0-9]+", "", s.lower())
        norm_a = re.sub(r"[^a-z0-9]+", "", str(ans).lower())
        if norm_s == norm_a:
            return True, None
        # accept "2" for "unit 2", or a subset match for text lists
        digits_s = re.findall(r"\d+", s.lower())
        digits_a = re.findall(r"\d+", str(ans).lower())
        if digits_a and digits_s == digits_a and len(norm_s) <= len(norm_a) + 4:
            return True, None
        if norm_a and norm_a in norm_s:
            return True, None
        return False, None

    val = parse_number(s)
    if val is None:
        return False, None
    try:
        target = float(ans)
    except (TypeError, ValueError):
        return False, val
    tol = float(part.get("tolerance", 1e-2))
    rel = float(part.get("rel_tolerance", 0.02))
    ok = abs(val - target) <= max(tol, abs(target) * rel)
    if kind == "int":
        ok = abs(val - target) <= max(float(part.get("tolerance", 0)), 1e-9)
    return ok, val


def grade(spec: dict[str, Any], answers: dict[str, str]) -> dict[str, Any]:
    """Grade a calculation problem, part by part, with error classification.

    Returns a dict consumable by the evaluator: score (0-10), per-part results,
    dominant mistake type and targeted feedback.
    """
    parts = spec.get("parts", [])
    if not parts:
        return {"score": 0.0, "sub_scores": [], "mistake_type": MistakeType.NONE.value}

    results: list[dict[str, Any]] = []
    total_w = 0.0
    earned = 0.0
    error_kinds: list[str] = []
    notes: list[str] = []

    for part in parts:
        w = float(part.get("weight", 1.0))
        total_w += w
        student = answers.get(part["key"], "")
        ok, val = _matches(part, student)
        note = ""
        if ok:
            earned += w
        elif val is not None or part.get("kind") in ("choice", "text"):
            # look for a known wrong path
            for ce in part.get("common_errors", []):
                try:
                    ce_val = float(ce[0])
                except (TypeError, ValueError):
                    continue
                if val is not None and abs(val - ce_val) <= max(1e-6, abs(ce_val) * 0.02):
                    error_kinds.append(ce[1])
                    note = ce[2]
                    notes.append(f"{part['label']}: {ce[2]}")
                    earned += 0.35 * w  # method credit for a recognised near-miss
                    break
            if not note and val is not None:
                try:
                    target = float(part.get("answer"))
                    if target != 0 and abs(val - target) / max(abs(target), 1e-9) < 0.12:
                        error_kinds.append(MistakeType.ARITHMETIC.value)
                        note = "Right method, arithmetic slip - your value is close but not exact."
                        earned += 0.5 * w
                    else:
                        error_kinds.append(MistakeType.CONCEPTUAL.value)
                except (TypeError, ValueError):
                    error_kinds.append(MistakeType.CONCEPTUAL.value)
        else:
            error_kinds.append(MistakeType.INCOMPLETE.value)

        results.append({
            "key": part["key"],
            "label": part["label"],
            "student": student,
            "expected": fmt(part.get("answer")),
            "correct": ok,
            "note": note,
            "step": part.get("step", ""),
        })

    ratio = earned / total_w if total_w else 0.0
    score = round(ratio * 10, 2)

    #: diagnostic value order used to break ties between equally frequent errors
    _SEVERITY_ORDER = [
        MistakeType.FORMULA.value, MistakeType.CONCEPTUAL.value,
        MistakeType.DIMENSION.value, MistakeType.ARITHMETIC.value,
        MistakeType.INCOMPLETE.value,
    ]

    if not error_kinds:
        mistake = MistakeType.NONE.value
    else:
        # A substantive error outranks blank parts even when blanks are more
        # numerous: "you used the wrong formula" is actionable, whereas
        # incompleteness is already visible in the score and the per-part table.
        substantive = [k for k in error_kinds if k != MistakeType.INCOMPLETE.value]
        pool = substantive or error_kinds
        mistake = sorted(
            set(pool),
            key=lambda k: (
                -pool.count(k),
                _SEVERITY_ORDER.index(k) if k in _SEVERITY_ORDER else 9,
            ),
        )[0]

    n_correct = sum(1 for r in results if r["correct"])
    return {
        "score": score,
        "ratio": ratio,
        "n_correct": n_correct,
        "n_parts": len(results),
        "sub_scores": results,
        "mistake_type": mistake,
        "notes": notes,
        "solution": spec.get("solution", ""),
    }
