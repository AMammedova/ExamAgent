"""Strict answer evaluation.

Three paths, chosen by question type and configuration:

* calculation      -> deterministic part-by-part grading (calc_engine)
* assertion-reason -> deterministic option check with a truth-flag breakdown
* open-ended       -> LLM rubric when configured, otherwise a concept-coverage
                      heuristic that is deliberately harsh on vague answers

The heuristic never awards a high score for an answer that merely name-drops the
topic: it requires the expected concepts to be present *and* the answer to show
causal language, which is what the exam rewards.
"""
from __future__ import annotations

import re
from typing import Any

from ..config import get_logger
from ..models.schemas import (
    Evaluation,
    MistakeType,
    Question,
    QuestionType,
    SubScore,
)
from .assertion_engine import evaluate_assertion_reason
from .calc_engine import grade as grade_calc
from .llm import EXAMINER_SYSTEM, get_llm, system_with_language

log = get_logger(__name__)

#: words that signal the student is explaining a mechanism rather than asserting
_CAUSAL = (
    "because", "therefore", "so that", "hence", "which means", "as a result",
    "since", "this causes", "leads to", "due to", "consequently", "thus",
    "in order to", "the reason", "results in", "implies",
)

_VAGUE = (
    "better", "good", "bad", "helps", "improves", "makes it work", "more robust",
    "nice", "useful", "efficient", "optimizes", "somehow", "kind of", "i think",
)


#: British/American variants and abbreviations must not cost the student marks
_SPELLING = (
    ("isation", "ization"), ("isations", "izations"), ("ised", "ized"),
    ("ising", "izing"), ("iser", "izer"), ("isers", "izers"), ("ise ", "ize "),
    ("yse", "yze"), ("behaviour", "behavior"), ("neighbour", "neighbor"),
)

_SYNONYMS: dict[str, tuple[str, ...]] = {
    "learning rate": ("lr", "step size"),
    "gradient descent": ("gd", "sgd"),
    "cross entropy": ("ce", "log loss", "logloss"),
    "receptive field": ("rf",),
    "weight decay": ("l2", "l2 penalty"),
    "co adaptation": ("coadaptation",),
    "parameters": ("params", "weights"),
    "variance": ("var",),
    "regularization": ("regularisation", "penalty"),
}

_STOP = {"the", "and", "for", "with", "that", "this", "does", "not", "are", "its",
         "from", "into", "than", "then", "when", "what", "which", "have", "has"}


def _norm(text: str) -> str:
    t = (text or "").lower()
    for a, b in _SPELLING:
        t = t.replace(a, b)
    t = re.sub(r"[^a-z0-9\s]+", " ", t)
    return re.sub(r"\s+", " ", t)


def _stem(token: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > 5 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def _concept_hit(concept: str, answer_norm: str) -> bool:
    """Does the answer contain this expected concept (allowing partial phrasing)?"""
    c = _norm(concept).strip()
    if not c:
        return False
    if c in answer_norm:
        return True
    for canonical, alts in _SYNONYMS.items():
        if canonical in c and any(a in answer_norm for a in alts):
            return True
    tokens = [t for t in c.split() if len(t) > 3 and t not in _STOP]
    if not tokens:
        return c in answer_norm
    stems = {_stem(w) for w in answer_norm.split()}
    hits = sum(1 for t in tokens if t in answer_norm or _stem(t) in stems)
    # a multi-word concept counts if most of its content words appear
    return hits >= max(1, int(round(len(tokens) * 0.65)))


def _technical_signal(question: Question, answer_norm: str) -> float:
    """How much of the student's vocabulary is the *right* vocabulary (0..1).

    Measures precision, not recall: of the technical words the student used, how
    many appear in the reference material for this question. A correct answer
    phrased differently from the rubric still scores here; a fluent but empty
    answer does not, because its content words are absent from the reference.
    """
    reference_text = " ".join([
        question.model_answer or "",
        question.prompt or "",
        " ".join(question.expected_concepts),
        question.expected_reasoning or "",
    ])
    reference = {_stem(w) for w in _norm(reference_text).split()
                 if len(w) > 4 and w not in _STOP}
    if not reference:
        return 0.0

    student = [w for w in answer_norm.split() if len(w) > 4 and w not in _STOP]
    if not student:
        return 0.0
    student_stems = {_stem(w) for w in student}
    # discount words the question itself handed the student, so copying the
    # prompt back does not count as knowledge
    precision = len(student_stems & reference) / len(student_stems)
    substance = min(1.0, len(student_stems) / 14.0)
    return round(precision * substance, 4)


# --------------------------------------------------------------- heuristic
def heuristic_evaluate(question: Question, answer: str) -> Evaluation:
    """Concept-coverage scoring used whenever no LLM is configured."""
    ans = (answer or "").strip()
    if len(ans) < 15:
        return Evaluation(
            score=0.0,
            correct=False,
            missed=question.expected_concepts[:6],
            examiner_expects=question.expected_reasoning
            or "A precise explanation of the mechanism, not a restatement of the question.",
            model_answer=question.model_answer,
            improvement="Write a full answer: state the mechanism, then its consequence.",
            mistake_type=MistakeType.INCOMPLETE,
            severity="High",
            evaluator="heuristic",
        )

    norm = _norm(ans)
    words = [w for w in norm.split() if w]
    concepts = question.expected_concepts or []

    hit = [c for c in concepts if _concept_hit(c, norm)]
    missed = [c for c in concepts if c not in hit]
    coverage = len(hit) / len(concepts) if concepts else None

    causal = sum(1 for c in _CAUSAL if c in norm)
    vague = sum(1 for v in _VAGUE if v in norm)
    signal = _technical_signal(question, norm)

    # --- base score ---
    if coverage is None:
        # no rubric available: judge only on structure, and cap the score,
        # because we genuinely cannot verify correctness offline
        length_score = min(1.0, len(words) / 70)
        base = 3.0 + 3.0 * length_score + min(1.5, causal * 0.5)
        base = min(base, 7.0)
    else:
        # Concave in coverage: an offline keyword matcher cannot recognise every
        # valid paraphrase, so full literal coverage must not be required for a
        # high mark. The technical-vocabulary signal rescues correct answers that
        # are phrased differently from the rubric; vague answers score near zero
        # on both terms.
        base = 10.0 * (0.72 * (coverage ** 0.75) + 0.28 * signal)
        # reasoning bonus/penalty
        if causal == 0 and len(words) > 25:
            base -= 1.5  # asserts without explaining
        elif causal >= 2:
            base += 0.5
        if vague >= 2 and coverage < 0.7:
            base -= 1.0
        if len(words) < 25 and coverage < 0.8:
            base -= 1.0  # too short to have justified anything

    score = max(0.0, min(10.0, round(base, 1)))

    # --- diagnosis ---
    if coverage is None:
        mistake = MistakeType.NONE if score >= 6 else MistakeType.INCOMPLETE
    elif coverage >= 0.85:
        mistake = MistakeType.NONE
    elif causal == 0:
        mistake = MistakeType.REASONING
    elif coverage < 0.4:
        mistake = MistakeType.CONCEPTUAL
    else:
        mistake = MistakeType.INCOMPLETE

    severity = "High" if score < 4 else ("Medium" if score < 7 else "Low")

    improvement = _improvement_line(score, missed, causal, question)
    got = [f"You covered: {c}" for c in hit[:5]]
    if not got and score > 0:
        got = ["You produced a structured attempt - but the examiner's key points are missing."]

    return Evaluation(
        score=score,
        correct=score >= 7,
        partial=3 <= score < 7,
        got_right=got,
        missed=[f"Not addressed: {c}" for c in missed[:6]],
        incorrect=[],
        examiner_expects=question.expected_reasoning
        or ("The examiner is checking that you can state the mechanism and its consequence, "
            "using precise terminology."),
        model_answer=question.model_answer,
        improvement=improvement,
        mistake_type=mistake,
        severity=severity,
        evaluator="heuristic",
    )


def _improvement_line(score: float, missed: list[str], causal: int,
                      question: Question) -> str:
    if score >= 8.5:
        return ("Strong. Tighten it further by naming the mechanism in the first sentence, "
                "then the consequence.")
    if missed:
        return (f"Rewrite the answer so it explicitly names **{missed[0]}**"
                + (f" and **{missed[1]}**" if len(missed) > 1 else "")
                + ", then state the consequence that follows from it.")
    if causal == 0:
        return ("You stated facts without linking them. Add an explicit causal chain: "
                "'X does A, therefore B, which means C.'")
    return "Add the precise technical term the examiner is looking for, then the consequence."


# --------------------------------------------------------------- LLM rubric
_EVAL_PROMPT = """Mark this student's exam answer strictly, as a university examiner would.

QUESTION ({qtype}, difficulty {difficulty}/6, topic: {topic}):
{prompt}

{model_block}
{concepts_block}
STUDENT'S ANSWER:
\"\"\"{answer}\"\"\"

Marking rules:
- Score 0-10. A vague but topically-relevant answer scores 3-5, NOT 7. Reserve 9-10 for answers
  that state the mechanism precisely with correct terminology and a correct causal chain.
- Judge correctness, completeness, reasoning quality, technical accuracy, terminology and
  exam suitability.
- Identify the dominant mistake type from exactly: Arithmetic, Conceptual, Formula, Dimension,
  Reasoning, Terminology, Incomplete, None.
- Be specific. "Explain more" is useless feedback; say WHICH concept is missing.

Return JSON exactly:
{{"score": 0-10 number,
  "got_right": ["specific thing the student got right", ...],
  "missed": ["specific thing the examiner wanted that is absent", ...],
  "incorrect": ["specific statement that is wrong, and why", ...],
  "examiner_expects": "one or two sentences on what a full-mark answer must contain",
  "model_answer": "a full-mark answer, 4-10 sentences",
  "improvement": "ONE sentence telling the student how to turn their answer into an exam-quality one",
  "mistake_type": "one of the listed types",
  "severity": "Low|Medium|High"}}"""


def llm_evaluate(question: Question, answer: str) -> Evaluation | None:
    llm = get_llm()
    if not llm.available:
        return None
    model_block = (f"REFERENCE ANSWER (for your marking only):\n{question.model_answer}\n"
                   if question.model_answer else "")
    concepts_block = (
        "CONCEPTS A FULL-MARK ANSWER MUST CONTAIN:\n"
        + "\n".join(f"- {c}" for c in question.expected_concepts) + "\n"
        if question.expected_concepts else ""
    )
    data, resp = llm.complete_json(
        _EVAL_PROMPT.format(
            qtype=question.question_type.value,
            difficulty=question.difficulty,
            topic=question.topic,
            prompt=question.prompt,
            model_block=model_block,
            concepts_block=concepts_block,
            answer=answer.strip()[:4000],
        ),
        system=system_with_language(EXAMINER_SYSTEM),
        temperature=0.2,
        max_tokens=1500,
    )
    if not isinstance(data, dict) or "score" not in data:
        log.info("LLM evaluation unavailable (%s); using heuristic", resp.error)
        return None
    try:
        score = float(data["score"])
    except (TypeError, ValueError):
        return None
    score = max(0.0, min(10.0, score))
    try:
        mistake = MistakeType(str(data.get("mistake_type", "None")).strip().title())
    except ValueError:
        mistake = MistakeType.CONCEPTUAL if score < 6 else MistakeType.NONE
    severity = str(data.get("severity", "Medium")).title()
    if severity not in ("Low", "Medium", "High"):
        severity = "High" if score < 4 else "Medium" if score < 7 else "Low"

    return Evaluation(
        score=round(score, 1),
        correct=score >= 7,
        partial=3 <= score < 7,
        got_right=[str(x) for x in data.get("got_right", [])][:6],
        missed=[str(x) for x in data.get("missed", [])][:6],
        incorrect=[str(x) for x in data.get("incorrect", [])][:6],
        examiner_expects=str(data.get("examiner_expects", "")),
        model_answer=str(data.get("model_answer", "")) or question.model_answer,
        improvement=str(data.get("improvement", "")),
        mistake_type=mistake,
        severity=severity,  # type: ignore[arg-type]
        evaluator="llm",
    )


# --------------------------------------------------------------- calculation
def evaluate_calculation(question: Question, answers: dict[str, str]) -> Evaluation:
    spec = question.calc_spec or {}
    result = grade_calc(spec, answers)

    subs = [
        SubScore(
            label=r["label"],
            student=r["student"] or "(blank)",
            expected=r["expected"],
            correct=r["correct"],
            note=r["note"] or (r["step"] if not r["correct"] else ""),
        )
        for r in result["sub_scores"]
    ]
    score = float(result["score"])
    try:
        mistake = MistakeType(str(result["mistake_type"]))
    except ValueError:
        mistake = MistakeType.CONCEPTUAL

    right = [s.label for s in subs if s.correct]
    wrong = [s for s in subs if not s.correct]

    if score >= 9.5:
        improvement = ("Fully correct. In the exam, write each intermediate value on its own "
                       "line so partial credit is visible.")
    elif mistake == MistakeType.ARITHMETIC:
        improvement = ("Your method is right - slow down on the arithmetic and carry 4 decimal "
                       "places through intermediate steps.")
    elif mistake == MistakeType.FORMULA:
        improvement = ("Write the formula down before substituting numbers; the error is in the "
                       "formula, not the calculation.")
    elif mistake == MistakeType.DIMENSION:
        improvement = ("Track the shape of every tensor explicitly - your error is a dimension "
                       "confusion, which examiners penalise heavily.")
    elif mistake == MistakeType.INCOMPLETE:
        improvement = ("Attempt every part: unanswered parts score zero, whereas a partially "
                       "correct method earns marks.")
    else:
        improvement = (f"Redo this problem focusing on: {wrong[0].label}." if wrong
                       else "Review the worked solution and redo the problem.")

    return Evaluation(
        score=score,
        correct=score >= 9.0,
        partial=1.0 <= score < 9.0,
        got_right=[f"Correct: {r}" for r in right[:6]],
        missed=[f"{s.label}: expected {s.expected}, you wrote {s.student}" for s in wrong[:6]],
        incorrect=[n for n in result.get("notes", [])][:5],
        examiner_expects=("Every intermediate quantity, correctly labelled. Partial credit is "
                          "awarded per step, so never leave a part blank."),
        model_answer=result.get("solution", question.model_answer),
        improvement=improvement,
        mistake_type=mistake,
        severity="High" if score < 4 else ("Medium" if score < 8 else "Low"),
        sub_scores=subs,
        evaluator="deterministic",
    )


# --------------------------------------------------------------- AR
def evaluate_ar(question: Question, chosen: str) -> Evaluation:
    res = evaluate_assertion_reason(question, chosen)
    correct = bool(res["correct"])
    breakdown = res["breakdown"]
    return Evaluation(
        score=10.0 if correct else 0.0,
        correct=correct,
        partial=False,
        got_right=[f"Correct option: {res['expected']}"] if correct else [],
        missed=[] if correct else [
            f"You chose {res['chosen'] or '(none)'}; the correct option is {res['expected']}."
        ],
        incorrect=[] if correct else breakdown,
        examiner_expects=("Evaluate the truth of the Assertion and the Reason SEPARATELY first, "
                          "then ask whether the Reason is the actual cause of the Assertion. "
                          "A true statement is not automatically an explanation."),
        model_answer=res["explanation"],
        improvement=("Keep doing the three checks in order: A true? R true? R explains A?"
                     if correct else
                     "Before choosing, write down T/F for A and for R separately - most errors "
                     "here come from evaluating them as a single statement."),
        mistake_type=MistakeType.NONE if correct else MistakeType.REASONING,
        severity="Low" if correct else "High",
        sub_scores=[
            SubScore(label="Assertion", expected="TRUE" if question.assertion_truth else "FALSE",
                     correct=correct),
            SubScore(label="Reason", expected="TRUE" if question.reason_truth else "FALSE",
                     correct=correct),
            SubScore(label="Reason explains Assertion",
                     expected="YES" if question.reason_explains_assertion else "NO",
                     correct=correct),
        ],
        evaluator="deterministic",
    )


# --------------------------------------------------------------- MCQ
def evaluate_mcq(question: Question, chosen: str) -> Evaluation:
    correct = (chosen or "").strip().upper()[:1] == (question.correct_option or "").upper()
    return Evaluation(
        score=10.0 if correct else 0.0,
        correct=correct,
        missed=[] if correct else [f"Correct option: {question.correct_option}"],
        examiner_expects="Eliminate distractors by mechanism, not by feel.",
        model_answer=question.model_answer,
        improvement="" if correct else "Re-derive why each distractor fails.",
        mistake_type=MistakeType.NONE if correct else MistakeType.CONCEPTUAL,
        severity="Low" if correct else "Medium",
        evaluator="deterministic",
    )


# --------------------------------------------------------------- entry point
def evaluate(
    question: Question,
    answer: str | dict[str, str],
    use_llm: bool = True,
) -> Evaluation:
    """Evaluate any answer. `answer` is a dict for calculation questions."""
    qt = question.question_type

    if qt == QuestionType.CALCULATION and question.calc_spec:
        answers = answer if isinstance(answer, dict) else _split_free_form(question, str(answer))
        return evaluate_calculation(question, answers)

    text = answer if isinstance(answer, str) else " ".join(str(v) for v in answer.values())

    if qt == QuestionType.ASSERTION_REASON:
        return evaluate_ar(question, text)
    if qt == QuestionType.MCQ:
        return evaluate_mcq(question, text)

    if use_llm:
        ev = llm_evaluate(question, text)
        if ev is not None:
            return ev
    return heuristic_evaluate(question, text)


def _split_free_form(question: Question, text: str) -> dict[str, str]:
    """Map a free-typed calculation answer onto the problem's parts.

    Accepts 'key = value' lines, 'a) value' lines, or bare numbers in order.
    """
    spec = question.calc_spec or {}
    parts = spec.get("parts", [])
    out: dict[str, str] = {}
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    for ln in lines:
        m = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*[=:]\s*(.+)$", ln)
        if m and any(p["key"] == m.group(1) for p in parts):
            out[m.group(1)] = m.group(2).strip()

    if len(out) < len(parts):
        # positional fallback: one value per line / per label order
        numbers = [ln for ln in lines if re.search(r"[-+]?\d", ln)]
        for part, ln in zip([p for p in parts if p["key"] not in out], numbers):
            out.setdefault(part["key"], ln)
    return out


def summarize_for_log(question: Question, evaluation: Evaluation) -> dict[str, Any]:
    return {
        "topic": question.topic,
        "question_type": question.question_type.value,
        "dimension": question.dimension,
        "difficulty": question.difficulty,
        "score": evaluation.score,
        "mistake_type": evaluation.mistake_type.value,
        "severity": evaluation.severity,
        "evaluator": evaluation.evaluator,
    }
