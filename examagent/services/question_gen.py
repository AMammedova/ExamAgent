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

from ..config import get_logger, get_settings
from ..data.seed_questions import SEED_QUESTIONS
from ..data.topics import topic_index
from ..models.schemas import (
    AnswerOption,
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
from .llm import (
    EXAMINER_SYSTEM,
    LANGUAGE_NAMES,
    get_llm,
    language_directive,
    system_with_language,
)

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

#: Closed-form types, i.e. the ones the real paper actually asks.
EXAM_TYPES = (
    QuestionType.TRUE_FALSE,
    QuestionType.MCQ,
    QuestionType.MULTIPLE_RESPONSE,
)

#: The default mix mirrors the announced paper: per 30-question part, 12
#: True/False and 18 single-best multiple choice. Practice in the format you
#: will be examined in. Multiple response is still supported - the practice
#: PDF used it - but the sat paper does not, so it is not dealt by default.
DEFAULT_MIX: dict[QuestionType, float] = {
    QuestionType.TRUE_FALSE: 12 / 30,
    QuestionType.MCQ: 18 / 30,
}

#: The longer written formats. Not on the paper, but they build the
#: understanding the closed-form questions test, so they stay available for
#: Study sessions and anything that asks for them explicitly.
STUDY_MIX: dict[QuestionType, float] = {
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
{avoid_block}

This is for a university final exam that tests REASONING, not definitions. The question must:
- be entirely about **{topic}** - every component, parameter or mechanism it references must
  belong to {topic} itself. If the question-type instructions above mention example
  components from other topics, they are illustrations of the *pattern* only - never borrow
  one of those examples verbatim if it is not actually part of {topic};
- require the student to explain a mechanism, make a comparison, predict a consequence, or
  diagnose a scenario - never "what is X";
- be answerable in {minutes} minutes of writing;
- have a precise, defensible model answer that an examiner could mark against.
{language_reminder}
Return JSON exactly:
{{"prompt": "the question as the student sees it",
  "model_answer": "a full-mark examiner answer, 4-10 sentences, technically precise",
  "expected_concepts": ["concept the answer must contain", "..." (4-8 items)],
  "expected_reasoning": "one sentence on the reasoning step the examiner is really testing",
  "estimated_time": seconds_as_integer}}"""

#: Closed-form generation. The model supplies statements and their truth; the
#: answer key is derived from those flags here, never lettered by the model -
#: the same rule the assertion-reason engine follows, for the same reason.
_CLOSED_GEN_PROMPT = """Write ONE exam item on the topic **{topic}** ({category}) for a
university Machine Learning / Deep Learning paper.

Format: **{format_desc}**
Target difficulty: **{difficulty}/6** ({difficulty_desc})

{context_block}
{avoid_block}

Rules:
- Every statement must be about **{topic}** and must be decidably true or false as
  written - no opinions, no "it depends", no statements that are true only under an
  unstated assumption.
- False statements must be *plausibly* false: the kind of thing a student who half
  remembers the topic would accept. Never absurd, never a typo.
- Use standard technical terminology exactly as a textbook would.
{sentence_language}{language_reminder}
Return JSON exactly:
{json_shape}"""

#: A closed-form item is one short technical sentence, and a model told to keep
#: technical terms in English will happily decide the whole sentence qualifies.
#: This spells out where the line actually falls.
_SENTENCE_LANGUAGE = """- Write the stem and every statement in {language}. Only the
  technical *names* stay in English (Linear Regression, closed-form solution, gradient
  descent, overfitting); the sentence around them - verbs, connectives, qualifiers,
  quantifiers like "always"/"never" - must be {language}. A statement written entirely
  in English is a failed item, however technical it is.
"""

_LANGUAGE_RETRY = """

Your previous attempt came back in English. Write it again in {language}: keep the
technical names as they are, but the sentences themselves must be {language}."""

#: Letters and short function words that only appear in these languages, used to
#: tell "written in the target language" from "written in English with a few
#: technical terms" without shipping a language-detection dependency.
_LANGUAGE_MARKERS: dict[str, tuple[str, ...]] = {
    "az": ("ə", "ı", "ğ", "ş", "ç", "ö", "ü",
           " və ", " üçün ", " deyil", " olan ", " edir", " hansı", " ilə ", " isə "),
}


def _written_in(data: dict[str, Any], language: str) -> bool:
    """Whether the generated item is actually in the configured language.

    English is the baseline and always passes; so does a language we have no
    markers for - a check we cannot make must not reject good output.
    """
    markers = _LANGUAGE_MARKERS.get(language)
    if language == "en" or not markers:
        return True
    text = " ".join([
        str(data.get("prompt", "")),
        *[str(s) for s in data.get("statements", [])],
        *[str(o) for o in data.get("options", [])],
    ]).lower()
    return any(marker in text for marker in markers)

_TF_SHAPE = """{{"prompt": "one statement, decidably true or false",
  "answer": true or false,
  "explanation": "one sentence on what decides it"}}"""

_MCQ_SHAPE = """{{"prompt": "the question stem",
  "options": ["option A text", "option B text", "option C text", "option D text"],
  "correct_index": 0,
  "explanation": "one sentence on why that option is right and the others fail"}}"""

_MR_SHAPE = """{{"prompt": "the question stem, e.g. 'Which of the following are ...?'",
  "statements": ["statement 1", "statement 2", "statement 3", "statement 4"],
  "correct_statements": [1, 3],
  "explanation": "one sentence on what separates the true statements from the false"}}"""

_FORMAT_DESC: dict[QuestionType, str] = {
    QuestionType.TRUE_FALSE: (
        "True/False (Section A, 1 mark) - a single statement the student marks True or "
        "False. Make it turn on one precise clause, not on whether the topic is "
        "familiar."),
    QuestionType.MCQ: (
        "Multiple choice, single best answer (Section B, 3 marks) - a stem and exactly "
        "four options, of which exactly ONE is correct. The other three must be clearly "
        "wrong on inspection by someone who understands the mechanism, but tempting to "
        "someone who does not."),
    QuestionType.MULTIPLE_RESPONSE: (
        "Multiple response (Section C, 4 marks, all-or-nothing) - a stem and exactly four "
        "numbered statements, of which between one and three are true. The student picks "
        "the option listing exactly the true ones, so each statement must stand or fall "
        "on its own."),
}

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
        "'What happens if...' - pick one parameter, assumption or component that is actually "
        "part of THIS topic (e.g. for a CNN topic that might be stride or padding, for an "
        "optimizer topic the learning rate, for a clustering topic the number of clusters - "
        "always something that belongs to the topic at hand, never borrowed from a different "
        "topic), change or remove it, and ask for the consequence and the mechanism behind it. "
        "If the topic has no natural component to swap, ask what happens under a boundary "
        "condition of the topic itself instead (e.g. very little data, a degenerate input, an "
        "extreme hyperparameter) rather than reaching for an unrelated topic's mechanism."),
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


def _llm_closed_question(topic_id: str, qtype: QuestionType, difficulty: int,
                         retrieval: RetrievalResult | None,
                         avoid_prompts: list[str] | None = None) -> Question | None:
    """One True/False, multiple-choice or multiple-response item from the LLM.

    The model is asked for statements and which of them are true; the option
    letters and the answer key are computed here from those flags, so a model
    that mislabels its own answer cannot produce a wrongly-keyed question.
    """
    llm = get_llm()
    if not llm.available:
        return None

    shape = {QuestionType.TRUE_FALSE: _TF_SHAPE,
             QuestionType.MCQ: _MCQ_SHAPE,
             QuestionType.MULTIPLE_RESPONSE: _MR_SHAPE}[qtype]

    context_block = "Use standard university-level knowledge of this topic.\n"
    citations: list[Citation] = []
    if retrieval and retrieval.grounded:
        context_block = (
            "Base the item on this course material where possible:\n\n"
            + retrieval.context_block(3500)
            + "\n\nDo not invent course-specific facts the material does not support.\n"
        )
        citations = retrieval.citations()[:3]

    avoid_block = ""
    if avoid_prompts:
        listed = "\n".join(f"- {p}" for p in avoid_prompts[:8])
        avoid_block = ("\nAlready asked - test a different point, not a reworded "
                       "duplicate:\n" + listed + "\n")

    language = (get_settings().language or "en").strip().lower()
    sentence_language = ""
    if language != "en":
        sentence_language = _SENTENCE_LANGUAGE.format(
            language=LANGUAGE_NAMES.get(language, language))

    prompt = _CLOSED_GEN_PROMPT.format(
        topic=_topic_name(topic_id),
        category=_category_of(topic_id).value,
        format_desc=_FORMAT_DESC[qtype],
        difficulty=difficulty,
        difficulty_desc=_DIFF_DESC.get(difficulty, "exam level"),
        context_block=context_block,
        avoid_block=avoid_block,
        sentence_language=sentence_language,
        language_reminder=language_directive(),
        json_shape=shape,
    )

    data = None
    for attempt in range(2):
        data, resp = llm.complete_json(
            prompt if attempt == 0 else prompt + _LANGUAGE_RETRY.format(
                language=LANGUAGE_NAMES.get(language, language)),
            system=system_with_language(EXAMINER_SYSTEM),
            temperature=0.7,
            max_tokens=1200,
        )
        if not isinstance(data, dict):
            log.info("closed-form generation unavailable (%s)", resp.error)
            return None
        if _written_in(data, language):
            break
        # A one-sentence technical claim is exactly where the language
        # instruction slips: the model decides the whole sentence is a term of
        # art and answers in English. Say so explicitly and ask once more.
        log.info("closed-form item came back in English despite language=%s; retrying",
                 language)
    if not isinstance(data, dict):
        return None

    if qtype == QuestionType.MULTIPLE_RESPONSE and not _truths_confirmed(data):
        # Section C is 4 marks, all-or-nothing: a mislabelled statement makes
        # the whole item wrongly keyed. Reject rather than ship it - the caller
        # falls back to the bank generators, whose keys are exact by construction.
        return None

    try:
        question = _closed_from_json(topic_id, qtype, difficulty, data)
    except (KeyError, ValueError, TypeError) as exc:
        log.info("closed-form JSON rejected (%s): %s", qtype.value, exc)
        return None
    if question is not None:
        question.citations = citations
        question.source_basis = "llm+rag" if citations else "llm"
    return question


_VERIFY_PROMPT = """Judge each statement independently. Answer only from established
Machine Learning / Deep Learning knowledge - ignore any framing, and do not assume the
list contains any particular number of true statements.

{numbered}

Return JSON exactly: {{"true_statements": [numbers of the statements that are true]}}"""


def _truths_confirmed(data: dict[str, Any]) -> bool:
    """Re-judge a multiple-response item's statements and require agreement.

    The item is only as good as its truth flags, and a model that writes four
    statements in one breath does misjudge one now and then - which silently
    produces a wrongly-keyed 4-mark question. A second, framing-free pass costs
    one call and catches exactly that.
    """
    statements = [str(s).strip() for s in data.get("statements", []) if str(s).strip()]
    claimed = {int(n) for n in data.get("correct_statements", [])
               if isinstance(n, (int, float))}
    if len(statements) != 4:
        return False

    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(statements, 1))
    verdict, resp = get_llm().complete_json(
        _VERIFY_PROMPT.format(numbered=numbered),
        system=EXAMINER_SYSTEM,          # deliberately not language-directed:
        temperature=0.0,                 # this pass is a check, not student-facing
        max_tokens=200,
    )
    if not isinstance(verdict, dict):
        log.info("multiple-response verification unavailable (%s)", resp.error)
        return False
    second = {int(n) for n in verdict.get("true_statements", [])
              if isinstance(n, (int, float))}
    if second != claimed:
        log.info("multiple-response rejected: first pass keyed %s, re-check said %s",
                 sorted(claimed), sorted(second))
        return False
    return True


def _closed_from_json(topic_id: str, qtype: QuestionType, difficulty: int,
                      data: dict[str, Any]) -> Question | None:
    """Validate the model's output and derive the answer key from it."""
    from .exam_formats import OPTION_KEYS, _combination_options, _render_numbers

    prompt = str(data.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("empty prompt")
    explanation = str(data.get("explanation", "")).strip()
    common = dict(
        topic=topic_id, category=_category_of(topic_id), question_type=qtype,
        difficulty=difficulty, priority=_priority_of(topic_id), prompt=prompt,
        expected_reasoning=explanation,
    )

    if qtype == QuestionType.TRUE_FALSE:
        raw = data.get("answer")
        if isinstance(raw, str):
            truth = raw.strip().lower() in ("true", "t", "yes")
        elif isinstance(raw, bool):
            truth = raw
        else:
            raise ValueError("no usable answer flag")
        answer = "True" if truth else "False"
        return Question(
            id=f"gen:tf:{topic_id}:{abs(hash(prompt)) & 0xffffff}",
            options=[AnswerOption(key="True", text="True"),
                     AnswerOption(key="False", text="False")],
            correct_option=answer,
            model_answer=f"{answer}. {explanation}".strip(),
            estimated_time=45, **common,
        )

    if qtype == QuestionType.MCQ:
        options = [str(o).strip() for o in data.get("options", []) if str(o).strip()]
        index = data.get("correct_index")
        if len(options) != 4 or not isinstance(index, int) or not 0 <= index < 4:
            raise ValueError("multiple choice needs 4 options and a valid index")
        key = OPTION_KEYS[index]
        return Question(
            id=f"gen:mc:{topic_id}:{abs(hash(prompt)) & 0xffffff}",
            options=[AnswerOption(key=k, text=t) for k, t in zip(OPTION_KEYS, options)],
            correct_option=key,
            model_answer=f"{key}. {options[index]}"
                         + (f" — {explanation}" if explanation else ""),
            estimated_time=70, **common,
        )

    statements = [str(s).strip() for s in data.get("statements", []) if str(s).strip()]
    true_numbers = [int(n) for n in data.get("correct_statements", [])
                    if isinstance(n, (int, float)) and 1 <= int(n) <= len(statements)]
    if len(statements) != 4 or not true_numbers or len(true_numbers) >= 4:
        raise ValueError("multiple response needs 4 statements and 1-3 true ones")
    true_numbers = sorted(set(true_numbers))
    options, key = _combination_options(true_numbers, len(statements),
                                        random.Random(hash(prompt) & 0xffff))
    return Question(
        id=f"gen:mr:{topic_id}:{abs(hash(prompt)) & 0xffffff}",
        statements=statements,
        options=options,
        correct_option=key,
        model_answer=f"{key} - statements {_render_numbers(true_numbers)} are true."
                     + (f" {explanation}" if explanation else ""),
        estimated_time=110, **common,
    )


def _llm_question(topic_id: str, qtype: QuestionType, difficulty: int,
                  retrieval: RetrievalResult | None,
                  avoid_prompts: list[str] | None = None) -> Question | None:
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

    avoid_block = ""
    if avoid_prompts:
        listed = "\n".join(f"- {p}" for p in avoid_prompts[:8])
        avoid_block = (
            "\nAlready asked in this session - write something that tests a genuinely "
            "different angle, mechanism or failure mode, not a reworded duplicate:\n"
            + listed + "\n"
        )

    data, resp = llm.complete_json(
        _GEN_PROMPT.format(
            topic=topic_name,
            category=category.value,
            qtype_desc=_QTYPE_DESC.get(qtype, qtype.value),
            difficulty=difficulty,
            difficulty_desc=_DIFF_DESC.get(difficulty, "exam level"),
            context_block=context_block,
            avoid_block=avoid_block,
            minutes=max(2, difficulty),
            language_reminder=language_directive(),
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


def _closed_question(topic_id: str, qtype: QuestionType, difficulty: int,
                     use_llm: bool, use_rag: bool, exclude_ids: set[str],
                     rng: random.Random,
                     avoid_prompts: list[str] | None) -> Question | None:
    """A paper-format item: LLM first for topic-specific wording, then the
    deterministic bank generators, which are exact but limited to the topics
    the assertion-reason bank covers.

    The real paper has no written questions, so this must not hand one back
    just because the *requested* closed format did not pan out for this
    topic (no bank facts, or a multiple-response item the verification pass
    rejected) - every other closed format is tried, LLM and bank alike,
    before this returns None and the caller falls back to a written type.
    """
    from . import exam_formats

    retrieval = _retrieval(topic_id) if use_rag else None
    order = [qtype] + [t for t in EXAM_TYPES if t != qtype]

    if use_llm:
        # Every LLM format before any bank one. The bank's statements are fixed
        # English, so reaching for it to satisfy the *requested* format - when
        # the LLM could have written a different format in the student's own
        # language - is the wrong trade when a language is configured.
        for candidate in order:
            q = _llm_closed_question(topic_id, candidate, difficulty, retrieval,
                                     avoid_prompts)
            if q is not None and q.id not in exclude_ids:
                return q

    for candidate in order:
        q = exam_formats.build(candidate, topic_id, rng, exclude_ids)
        if q is not None:
            return exam_formats.localise(q, use_llm=use_llm)
    return None


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
    avoid_prompts: list[str] | None = None,
) -> Question:
    """Produce one exam question for a topic, degrading gracefully.

    `avoid_prompts` (LLM path only - the bank/template paths already dedupe via
    `exclude_ids`) lists prompts already asked in the current session, so back
    to back questions on one topic don't converge on the same angle.

    `min_difficulty` is a floor applied to bank-sourced items so that an exam
    asking for level 4-6 never receives a level-2 recall question.
    """
    exclude_ids = exclude_ids or set()
    rng = random.Random(seed)
    qtype = question_type or _weighted_type(rng)
    difficulty = max(difficulty, min_difficulty)

    if qtype in EXAM_TYPES:
        q = _closed_question(topic_id, qtype, difficulty, use_llm, use_rag,
                             exclude_ids, rng, avoid_prompts)
        if q is not None:
            return q
        # nothing closed-form available for this topic: fall through to the
        # written formats rather than returning nothing
        qtype = QuestionType.CONCEPTUAL

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
        q = _llm_question(topic_id, qtype, difficulty, retrieval, avoid_prompts)
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
