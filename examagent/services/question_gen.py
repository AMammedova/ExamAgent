"""Question generator.

Resolution order for every request (first one that succeeds wins):

1. Deterministic engine  - calculation problems and assertion-reason items.
2. LLM + RAG             - open-ended exam questions grounded in course material.
3. Seed bank             - hand-written exam-style items for the topic.
4. Template fallback     - keyword-driven prompt so the app never dead-ends.

This ordering is what makes the app fully usable with no API key.
"""
from __future__ import annotations

import random
from typing import Any, Sequence

from ..config import get_logger
from ..data.seed_questions import SEED_QUESTIONS
from ..data.topics import topic_index
from ..models.schemas import (
    Category,
    Citation,
    Priority,
    Question,
    QuestionType,
    RetrievalResult,
)
from . import rag
from .assertion_engine import generate_assertion_reason
from .calc_engine import generate_problem, topic_has_calculation
from .llm import EXAMINER_SYSTEM, get_llm, system_with_language

log = get_logger(__name__)

TOPIC_INDEX = topic_index()

OPEN_TYPES = (
    QuestionType.CONCEPTUAL,
    QuestionType.COMPARISON,
    QuestionType.SCENARIO,
    QuestionType.WHAT_IF,
    QuestionType.DIAGRAM,
    QuestionType.GRAPH,
    QuestionType.SHORT_ANSWER,
)

#: The exam is reasoning-first, so the default mix is weighted accordingly.
DEFAULT_MIX: dict[QuestionType, float] = {
    QuestionType.ASSERTION_REASON: 0.24,
    QuestionType.CALCULATION: 0.26,
    QuestionType.CONCEPTUAL: 0.20,
    QuestionType.WHAT_IF: 0.10,
    QuestionType.COMPARISON: 0.10,
    QuestionType.SCENARIO: 0.06,
    QuestionType.DIAGRAM: 0.04,
}

#: Mix used when the goal is to repair one specific weak dimension.
DIMENSION_TYPES: dict[str, list[QuestionType]] = {
    "concept": [QuestionType.CONCEPTUAL, QuestionType.SHORT_ANSWER],
    "calculation": [QuestionType.CALCULATION],
    "reasoning": [QuestionType.ASSERTION_REASON, QuestionType.WHAT_IF,
                  QuestionType.CONCEPTUAL],
    "comparison": [QuestionType.COMPARISON],
    "application": [QuestionType.SCENARIO, QuestionType.DIAGRAM, QuestionType.GRAPH],
}


def _category_of(topic_id: str) -> Category:
    seed = TOPIC_INDEX.get(topic_id)
    if seed and seed.get("category") == "Deep Learning":
        return Category.DL
    return Category.ML


def _priority_of(topic_id: str) -> Priority:
    seed = TOPIC_INDEX.get(topic_id)
    try:
        return Priority(seed["priority"]) if seed else Priority.MEDIUM
    except (KeyError, ValueError):
        return Priority.MEDIUM


def _topic_name(topic_id: str) -> str:
    seed = TOPIC_INDEX.get(topic_id)
    return seed["name"] if seed else topic_id.replace("_", " ").title()


# --------------------------------------------------------------- calculation
def build_calculation_question(topic_id: str, seed: int | None = None,
                               generator: str | None = None) -> Question | None:
    if not (generator or topic_has_calculation(topic_id)):
        return None
    problem = generate_problem(topic_id=topic_id, generator=generator, seed=seed)
    return Question(
        id=f"calc:{problem.problem_id}",
        topic=topic_id,
        subtopic=problem.title,
        category=_category_of(topic_id),
        question_type=QuestionType.CALCULATION,
        difficulty=problem.difficulty,
        priority=_priority_of(topic_id),
        prompt=f"### {problem.title}\n\n{problem.statement}",
        model_answer=problem.solution,
        expected_concepts=problem.concepts,
        estimated_time=problem.estimated_time,
        source_basis="calculation engine",
        calc_spec=problem.spec(),
    )


# --------------------------------------------------------------- seed bank
def _seed_pool(topic_id: str, qtype: QuestionType | None = None,
               min_difficulty: int = 1) -> list[dict[str, Any]]:
    """Seed questions for a topic.

    `min_difficulty` is a hard floor: returning an easier item would put a
    level-2 recall question on a paper that asked for exam level, and the
    reported difficulty would then misrepresent the result. Callers fall through
    to the template generator instead, which honours the requested level.
    """
    pool = [q for q in SEED_QUESTIONS if q["topic"] == topic_id and not q.get("calc_generator")]
    if qtype:
        pool = [q for q in pool if q["question_type"] == qtype.value]
    if min_difficulty > 1:
        pool = [q for q in pool if int(q.get("difficulty", 4)) >= min_difficulty]
    return pool


def _seed_to_question(raw: dict[str, Any]) -> Question:
    data = dict(raw)
    data.pop("calc_generator", None)
    opts = data.pop("options", None)
    q = Question(
        id=data["id"],
        topic=data["topic"],
        subtopic=data.get("subtopic") or None,
        category=Category.DL if data.get("category") == "Deep Learning" else Category.ML,
        question_type=QuestionType(data["question_type"]),
        difficulty=int(data.get("difficulty", 4)),
        priority=Priority(data.get("priority", "HIGH")),
        prompt=data.get("prompt", ""),
        model_answer=data.get("model_answer", ""),
        expected_concepts=list(data.get("expected_concepts", [])),
        expected_reasoning=data.get("expected_reasoning", ""),
        estimated_time=int(data.get("estimated_time", 240)),
        source_basis="seed bank",
        correct_option=data.get("correct_option"),
        assertion=data.get("assertion"),
        reason=data.get("reason"),
        assertion_truth=data.get("assertion_truth"),
        reason_truth=data.get("reason_truth"),
        reason_explains_assertion=data.get("reason_explains_assertion"),
    )
    if opts:
        from ..models.schemas import AnswerOption

        q.options = [AnswerOption(key=o["key"], text=o["text"]) for o in opts]
    if q.question_type == QuestionType.ASSERTION_REASON and q.assertion:
        q.prompt = (f"**Assertion (A):** {q.assertion}\n\n"
                    f"**Reason (R):** {q.reason}\n\nSelect the correct option.")
    return q


# --------------------------------------------------------------- LLM
_GEN_PROMPT = """Write ONE exam question on the topic **{topic}** ({category}).

Question type: **{qtype_desc}**
Target difficulty: **{difficulty}/6** ({difficulty_desc})

{context_block}

This is for a university final exam that tests REASONING, not definitions. The question must:
- require the student to explain a mechanism, make a comparison, predict a consequence, or
  diagnose a scenario - never "what is X";
- be answerable in {minutes} minutes of writing;
- have a precise, defensible model answer that an examiner could mark against.

Return JSON exactly:
{{"prompt": "the question as the student sees it",
  "model_answer": "a full-mark examiner answer, 4-10 sentences, technically precise",
  "expected_concepts": ["concept the answer must contain", "..." (4-8 items)],
  "expected_reasoning": "one sentence on the reasoning step the examiner is really testing",
  "estimated_time": seconds_as_integer}}"""

_QTYPE_DESC: dict[QuestionType, str] = {
    QuestionType.CONCEPTUAL: (
        "Conceptual reasoning - ask WHY a mechanism produces an effect. Force the student to "
        "explain the causal chain."),
    QuestionType.COMPARISON: (
        "Comparison - contrast two methods/architectures along several explicit axes "
        "(mechanism, cost, assumptions, failure modes) and say when each is preferred."),
    QuestionType.SCENARIO: (
        "Scenario diagnosis - describe a concrete failing model or experiment; ask the student "
        "to diagnose the cause, justify it, propose an intervention, and name an intervention "
        "that would NOT help and why."),
    QuestionType.WHAT_IF: (
        "'What happens if...' - remove or change one component (activation, skip connection, "
        "padding, stride, cell state, attention, learning rate) and ask for the consequence and "
        "the mechanism behind it."),
    QuestionType.DIAGRAM: (
        "Architecture interpretation - describe an architecture in text (layers, shapes, "
        "connections) and ask what each component does, what flows through it, what dimensions "
        "are involved, and what breaks if a component is removed."),
    QuestionType.GRAPH: (
        "Graph interpretation - describe a training curve or performance plot in words and ask "
        "the student to identify the phenomenon, the correct intervention, and what the curve "
        "implies about what the model is learning."),
    QuestionType.SHORT_ANSWER: (
        "Short answer - a precise technical question answerable in 3-5 sentences, requiring "
        "correct terminology."),
}

_DIFF_DESC = {
    1: "basic recognition", 2: "understanding", 3: "application",
    4: "reasoning", 5: "exam level", 6: "hard exam level",
}


def _llm_question(topic_id: str, qtype: QuestionType, difficulty: int,
                  retrieval: RetrievalResult | None) -> Question | None:
    llm = get_llm()
    if not llm.available:
        return None
    topic_name = _topic_name(topic_id)
    category = _category_of(topic_id)

    context_block = "Use standard university-level knowledge of this topic.\n"
    citations: list[Citation] = []
    if retrieval and retrieval.grounded:
        context_block = (
            "Base the question on this course material where possible:\n\n"
            + retrieval.context_block(4000)
            + "\n\nDo not invent course-specific facts that the material does not support.\n"
        )
        citations = retrieval.citations()[:3]

    data, resp = llm.complete_json(
        _GEN_PROMPT.format(
            topic=topic_name,
            category=category.value,
            qtype_desc=_QTYPE_DESC.get(qtype, qtype.value),
            difficulty=difficulty,
            difficulty_desc=_DIFF_DESC.get(difficulty, "exam level"),
            context_block=context_block,
            minutes=max(2, difficulty),
        ),
        system=system_with_language(EXAMINER_SYSTEM),
        temperature=0.75,
        max_tokens=1400,
    )
    if not isinstance(data, dict) or not str(data.get("prompt", "")).strip():
        log.info("LLM question generation unavailable (%s)", resp.error)
        return None
    return Question(
        id=f"gen:{topic_id}:{qtype.value}:{abs(hash(str(data['prompt']))) & 0xffffff}",
        topic=topic_id,
        category=category,
        question_type=qtype,
        difficulty=difficulty,
        priority=_priority_of(topic_id),
        prompt=str(data["prompt"]).strip(),
        model_answer=str(data.get("model_answer", "")).strip(),
        expected_concepts=[str(c) for c in data.get("expected_concepts", [])][:10],
        expected_reasoning=str(data.get("expected_reasoning", "")),
        estimated_time=int(data.get("estimated_time", 240) or 240),
        source_basis="llm+rag" if citations else "llm",
        citations=citations,
    )


# --------------------------------------------------------------- templates
_TEMPLATES: dict[QuestionType, list[str]] = {
    QuestionType.CONCEPTUAL: [
        "Explain the mechanism behind {topic}. Do not define it - explain *why* it works and "
        "what would go wrong without it. Reference {kw1} and {kw2} explicitly.",
        "A fellow student says they understand {topic} because they can state its definition. "
        "Give the question you would ask to prove they do not, and give the full-mark answer to "
        "your own question.",
    ],
    QuestionType.WHAT_IF: [
        "What happens if {kw1} is removed or set to an extreme value in {topic}? Describe the "
        "immediate effect, the mechanism that causes it, and whether the model can still be "
        "trained.",
        "In {topic}, what changes if {kw2} is doubled? Address the effect on the computation, on "
        "the result, and on the cost.",
    ],
    QuestionType.COMPARISON: [
        "Compare {topic} with the most closely related alternative method you know. Contrast "
        "them on mechanism, assumptions, computational cost and failure modes, and state when "
        "each is preferred.",
    ],
    QuestionType.SCENARIO: [
        "A model using {topic} performs well in development but fails on new data. Diagnose the "
        "most likely cause given how {topic} works, justify the diagnosis, propose an "
        "intervention, and name one intervention that would NOT address it.",
    ],
    QuestionType.DIAGRAM: [
        "Describe the structure of {topic} component by component. For each component state what "
        "it computes, what flows through it, what dimensions are involved, and what breaks if it "
        "is removed.",
    ],
    QuestionType.GRAPH: [
        "Sketch in words how a training and validation curve would look for a model using "
        "{topic} that is (a) working correctly and (b) failing. Explain how you would tell the "
        "two apart and what you would do in case (b).",
    ],
    QuestionType.SHORT_ANSWER: [
        "State precisely what {topic} does and why it is used, in at most four sentences, using "
        "correct technical terminology. Include the role of {kw1}.",
    ],
}


def _template_question(topic_id: str, qtype: QuestionType, difficulty: int) -> Question:
    seed_data = TOPIC_INDEX.get(topic_id, {})
    kws = list(seed_data.get("keywords", [])) or ["its main parameter", "its main component"]
    name = _topic_name(topic_id)
    templates = _TEMPLATES.get(qtype) or _TEMPLATES[QuestionType.CONCEPTUAL]
    text = random.choice(templates).format(
        topic=name, kw1=kws[0], kw2=kws[1] if len(kws) > 1 else kws[0]
    )
    return Question(
        id=f"tpl:{topic_id}:{qtype.value}:{random.randint(1000, 9999)}",
        topic=topic_id,
        category=_category_of(topic_id),
        question_type=qtype,
        difficulty=difficulty,
        priority=_priority_of(topic_id),
        prompt=text,
        model_answer="",
        expected_concepts=kws[:6],
        expected_reasoning=f"Tests whether the student can reason about {name} rather than "
                           "recite its definition.",
        estimated_time=240,
        source_basis="template",
    )


# --------------------------------------------------------------- public API
def generate_question(
    topic_id: str,
    question_type: QuestionType | None = None,
    difficulty: int = 4,
    use_llm: bool = True,
    use_rag: bool = True,
    exclude_ids: set[str] | None = None,
    seed: int | None = None,
    recent_ar_keys: list[str] | None = None,
    min_difficulty: int = 1,
) -> Question:
    """Produce one exam question for a topic, degrading gracefully.

    `min_difficulty` is a floor applied to bank-sourced items so that an exam
    asking for level 4-6 never receives a level-2 recall question.
    """
    exclude_ids = exclude_ids or set()
    rng = random.Random(seed)
    qtype = question_type or _weighted_type(rng)
    difficulty = max(difficulty, min_difficulty)

    if qtype == QuestionType.CALCULATION:
        q = build_calculation_question(topic_id, seed=seed)
        if q is not None:
            return q
        qtype = QuestionType.CONCEPTUAL  # topic has no numeric engine

    if qtype == QuestionType.ASSERTION_REASON:
        q = generate_assertion_reason(
            topic_id,
            _topic_name(topic_id),
            _category_of(topic_id).value,
            context=_context_text(topic_id) if use_rag else "",
            exclude=exclude_ids,
            use_llm=use_llm,
            seed=seed,
            recent_keys=recent_ar_keys,
            min_difficulty=min_difficulty,
        )
        if q is not None:
            return q
        qtype = QuestionType.CONCEPTUAL

    retrieval = _retrieval(topic_id) if use_rag else None
    if use_llm:
        q = _llm_question(topic_id, qtype, difficulty, retrieval)
        if q is not None:
            return q

    pool = [q for q in _seed_pool(topic_id, qtype, min_difficulty)
            if q["id"] not in exclude_ids]
    if not pool:
        pool = [q for q in _seed_pool(topic_id, None, min_difficulty)
                if q["id"] not in exclude_ids]
    if pool:
        return _seed_to_question(rng.choice(pool))

    return _template_question(topic_id, qtype, difficulty)


def _weighted_type(rng: random.Random) -> QuestionType:
    types = list(DEFAULT_MIX)
    weights = [DEFAULT_MIX[t] for t in types]
    return rng.choices(types, weights=weights, k=1)[0]


def _retrieval(topic_id: str) -> RetrievalResult | None:
    try:
        return rag.retrieve_for_topic(
            _topic_name(topic_id), topic_id, category=_category_of(topic_id).value
        )
    except Exception as exc:  # retrieval must never break question generation
        log.warning("retrieval failed for %s: %s", topic_id, exc)
        return None


def _context_text(topic_id: str, max_chars: int = 2500) -> str:
    res = _retrieval(topic_id)
    return res.context_block(max_chars) if res and res.grounded else ""


def generate_batch(
    topic_ids: Sequence[str],
    n: int,
    type_mix: dict[QuestionType, float] | None = None,
    difficulty_range: tuple[int, int] = (3, 6),
    use_llm: bool = True,
    seed: int | None = None,
    dimension_focus: str | None = None,
) -> list[Question]:
    """Build a set of questions spread over topics and question types."""
    if not topic_ids:
        return []
    rng = random.Random(seed)
    mix = type_mix or DEFAULT_MIX
    if dimension_focus and dimension_focus in DIMENSION_TYPES:
        allowed = DIMENSION_TYPES[dimension_focus]
        mix = {t: w for t, w in mix.items() if t in allowed} or {allowed[0]: 1.0}

    types = list(mix)
    weights = [mix[t] for t in types]
    out: list[Question] = []
    seen: set[str] = set()
    ar_keys: list[str] = []

    for i in range(n):
        topic = topic_ids[i % len(topic_ids)]
        qtype = rng.choices(types, weights=weights, k=1)[0]
        diff = rng.randint(*difficulty_range)
        q = generate_question(
            topic, qtype, diff, use_llm=use_llm, exclude_ids=seen,
            seed=rng.randint(1, 10 ** 6), recent_ar_keys=ar_keys[-4:],
        )
        if q.id in seen:
            q = generate_question(topic, qtype, diff, use_llm=False, exclude_ids=seen,
                                  seed=rng.randint(1, 10 ** 6), recent_ar_keys=ar_keys[-4:])
        seen.add(q.id)
        if q.question_type == QuestionType.ASSERTION_REASON and q.correct_option:
            ar_keys.append(q.correct_option)
        out.append(q)
    return out


def available_types(topic_id: str) -> list[QuestionType]:
    """Which question types this topic can actually produce offline."""
    types: list[QuestionType] = [QuestionType.CONCEPTUAL]
    if topic_has_calculation(topic_id):
        types.append(QuestionType.CALCULATION)
    from .assertion_engine import bank_for_topic

    if bank_for_topic(topic_id):
        types.append(QuestionType.ASSERTION_REASON)
    for q in _seed_pool(topic_id):
        t = QuestionType(q["question_type"])
        if t not in types:
            types.append(t)
    for t in (QuestionType.WHAT_IF, QuestionType.COMPARISON, QuestionType.SCENARIO):
        if t not in types:
            types.append(t)
    return types
