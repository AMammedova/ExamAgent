"""The three question formats the real paper uses, generated offline.

AI-CORE-101 is entirely closed-form: Section A True/False (1 pt), Section B
single-best multiple choice (3 pts), Section C multiple response - "choose the
option listing *all* correct statements", all-or-nothing (4 pts).

All three need one thing the app already has in bulk: statements whose truth is
known for certain. The assertion-reason bank stores `assertion`/`a_true` and
`reason`/`r_true` for 88 curated items, which is ~176 verified statements to
build on - so these generators are exact, need no LLM, and cannot hallucinate a
wrong answer key.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from ..models.schemas import AnswerOption, Category, Priority, Question, QuestionType
from .assertion_engine import AR_BANK, ARItem

#: how many numbered statements a Section C item lists
STATEMENTS_PER_ITEM = 4

#: the option letters a closed-form item offers
OPTION_KEYS = ("A", "B", "C", "D")


@dataclass(frozen=True)
class Fact:
    """One statement whose truth value is known."""

    text: str
    true: bool
    topic_id: str
    category: str
    difficulty: int


def facts_for(topic_id: str) -> list[Fact]:
    """Every verified statement the bank holds about one topic.

    Both halves of an assertion-reason item are usable: each is a standalone
    claim with a recorded truth value.
    """
    out: list[Fact] = []
    for item in AR_BANK:
        if item.topic_id != topic_id:
            continue
        out.extend(_facts_from(item))
    return out


def all_facts() -> list[Fact]:
    out: list[Fact] = []
    for item in AR_BANK:
        out.extend(_facts_from(item))
    return out


def false_facts_near(topic_id: str, wanted: int, rng: random.Random) -> list[Fact]:
    """False statements to use as distractors, nearest-first.

    A false statement is false whatever topic it was written for, so widening
    the pool past the topic cannot create a second defensible answer - and it
    has to widen, because no single topic carries three false statements of
    its own. Same topic first, then same category, then the whole bank, so
    distractors stay as close to the subject as the bank allows.
    """
    everything = all_facts()
    category = next((f.category for f in everything if f.topic_id == topic_id), "")
    tiers = [
        [f for f in everything if not f.true and f.topic_id == topic_id],
        [f for f in everything if not f.true and f.topic_id != topic_id
         and f.category == category],
        [f for f in everything if not f.true and f.category != category],
    ]
    picked: list[Fact] = []
    seen: set[str] = set()
    for tier in tiers:
        rng.shuffle(tier)
        for fact in tier:
            if len(picked) >= wanted:
                return picked
            if fact.text not in seen:
                seen.add(fact.text)
                picked.append(fact)
    return picked


def _facts_from(item: ARItem) -> list[Fact]:
    return [
        Fact(item.assertion, item.a_true, item.topic_id, item.category, item.difficulty),
        Fact(item.reason, item.r_true, item.topic_id, item.category, item.difficulty),
    ]


def _category(cat: str) -> Category:
    return Category.DL if cat == "Deep Learning" else Category.ML


def _priority(topic_id: str) -> Priority:
    from .question_gen import _priority_of

    return _priority_of(topic_id)


# --------------------------------------------------------------- true / false
def build_true_false(topic_id: str, rng: random.Random,
                     exclude: set[str] | None = None) -> Question | None:
    """A Section A item: one statement, answered True or False.

    Truth comes straight from the bank's recorded flag, so the answer key is
    exact rather than a model's opinion.
    """
    exclude = exclude or set()
    pool = [f for f in facts_for(topic_id) if _tf_id(topic_id, f.text) not in exclude]
    if not pool:
        return None
    fact = rng.choice(pool)
    answer = "True" if fact.true else "False"
    return Question(
        id=_tf_id(topic_id, fact.text),
        topic=topic_id,
        category=_category(fact.category),
        question_type=QuestionType.TRUE_FALSE,
        difficulty=fact.difficulty,
        priority=_priority(topic_id),
        prompt=fact.text,
        options=[AnswerOption(key="True", text="True"),
                 AnswerOption(key="False", text="False")],
        correct_option=answer,
        model_answer=f"{answer}. " + (
            "The statement holds as written." if fact.true else
            "The statement is false as written - check the clause that overstates it."
        ),
        expected_concepts=[],
        expected_reasoning="Section A tests whether a claim is precisely true, not "
                           "roughly familiar.",
        estimated_time=45,
        source_basis="bank",
    )


def _tf_id(topic_id: str, text: str) -> str:
    return f"tf:{topic_id}:{abs(hash(text)) & 0xffffff}"


# --------------------------------------------------- multiple response (Sec C)
def build_multiple_response(topic_id: str, rng: random.Random,
                            exclude: set[str] | None = None) -> Question | None:
    """A Section C item: four numbered statements, pick the option listing
    exactly the true ones.

    Needs a mix of true and false statements to be answerable at all, so it
    returns None rather than emitting a degenerate item where every statement
    is true (which would always be "All of them").
    """
    exclude = exclude or set()
    true_facts = [f for f in facts_for(topic_id) if f.true]
    if len(true_facts) < 2:
        return None

    n_true = min(len(true_facts), rng.choice([2, 3]))
    n_false = STATEMENTS_PER_ITEM - n_true
    false_facts = false_facts_near(topic_id, n_false, rng)
    if len(false_facts) < n_false:
        return None
    chosen = rng.sample(true_facts, n_true) + false_facts
    if len(chosen) < STATEMENTS_PER_ITEM:
        return None
    rng.shuffle(chosen)

    statements = [f.text for f in chosen]
    correct_numbers = [i for i, f in enumerate(chosen, 1) if f.true]
    qid = f"mr:{topic_id}:{abs(hash(tuple(statements))) & 0xffffff}"
    if qid in exclude:
        return None

    options, correct_key = _combination_options(correct_numbers, len(chosen), rng)
    return Question(
        id=qid,
        topic=topic_id,
        category=_category(chosen[0].category),
        question_type=QuestionType.MULTIPLE_RESPONSE,
        difficulty=max(f.difficulty for f in chosen),
        priority=_priority(topic_id),
        prompt="Which of the following statements are correct?",
        statements=statements,
        options=options,
        correct_option=correct_key,
        model_answer=f"{correct_key} - statements {_render_numbers(correct_numbers)} "
                     "are the true ones; the rest are false as written.",
        expected_reasoning="Section C is all-or-nothing: one wrong statement in the "
                           "combination loses the whole item.",
        estimated_time=110,
        source_basis="bank",
    )


def _render_numbers(numbers: list[int]) -> str:
    """The paper's own phrasing: '1', '1 and 4', '1, 3, and 4'."""
    if len(numbers) == 1:
        return str(numbers[0])
    if len(numbers) == 2:
        return f"{numbers[0]} and {numbers[1]}"
    return ", ".join(str(n) for n in numbers[:-1]) + f", and {numbers[-1]}"


def _combination_options(correct: list[int], total: int,
                         rng: random.Random) -> tuple[list[AnswerOption], str]:
    """Four combination options in the paper's style, one of them exactly right.

    Distractors are near misses - a statement added, one dropped, or the
    complement - because a distractor that shares no structure with the answer
    is guessable by shape alone.
    """
    all_numbers = list(range(1, total + 1))
    correct_set = set(correct)
    candidates: list[list[int]] = []

    def add(nums: list[int]) -> None:
        nums = sorted(set(nums))
        if nums and nums != correct and nums not in candidates:
            candidates.append(nums)

    missing = [n for n in all_numbers if n not in correct_set]
    if missing:
        add(correct + [missing[0]])          # one false statement swept in
    if len(correct) > 1:
        add(correct[:-1])                    # one true statement dropped
    add(missing)                             # the exact complement
    for _ in range(4):
        add(rng.sample(all_numbers, rng.choice([2, 3])))

    distractors = candidates[:3]
    while len(distractors) < 3:              # tiny pools: pad with the full set
        fallback = all_numbers if all_numbers != correct else all_numbers[:2]
        if fallback not in distractors and fallback != correct:
            distractors.append(fallback)
        else:
            break

    pool = [correct] + distractors
    rng.shuffle(pool)
    options: list[AnswerOption] = []
    correct_key = "A"
    for key, nums in zip(OPTION_KEYS, pool):
        text = "All of them" if len(nums) == total else _render_numbers(nums)
        options.append(AnswerOption(key=key, text=text))
        if nums == correct:
            correct_key = key
    return options, correct_key


# ------------------------------------------------ multiple choice (Section B)
def build_multiple_choice(topic_id: str, rng: random.Random,
                          exclude: set[str] | None = None) -> Question | None:
    """A Section B item: one true statement as the answer, three false ones as
    distractors, all about the same topic.

    Built from the same verified facts, so the key is exact - the true fact is
    the only defensible choice and every distractor is a recorded falsehood.
    """
    exclude = exclude or set()
    true_facts = [f for f in facts_for(topic_id) if f.true]
    if not true_facts:
        return None

    answer = rng.choice(true_facts)
    qid = f"mc:{topic_id}:{abs(hash(answer.text)) & 0xffffff}"
    if qid in exclude:
        return None
    distractors = false_facts_near(topic_id, 3, rng)
    if len(distractors) < 3:
        return None

    entries = [answer, *distractors]
    rng.shuffle(entries)
    options = [AnswerOption(key=k, text=f.text) for k, f in zip(OPTION_KEYS, entries)]
    correct_key = next(k for k, f in zip(OPTION_KEYS, entries) if f is answer)

    from .question_gen import _topic_name

    return Question(
        id=qid,
        topic=topic_id,
        category=_category(answer.category),
        question_type=QuestionType.MCQ,
        difficulty=answer.difficulty,
        priority=_priority(topic_id),
        prompt=f"Which statement about {_topic_name(topic_id)} is correct?",
        options=options,
        correct_option=correct_key,
        model_answer=f"{correct_key}. {answer.text}",
        expected_reasoning="Section B rewards eliminating distractors by mechanism.",
        estimated_time=70,
        source_basis="bank",
    )


#: dispatch used by question_gen's offline path
BUILDERS = {
    QuestionType.TRUE_FALSE: build_true_false,
    QuestionType.MCQ: build_multiple_choice,
    QuestionType.MULTIPLE_RESPONSE: build_multiple_response,
}


def build(question_type: QuestionType, topic_id: str, rng: random.Random,
          exclude: set[str] | None = None) -> Question | None:
    builder = BUILDERS.get(question_type)
    return builder(topic_id, rng, exclude) if builder else None


def coverage() -> dict[str, int]:
    """How many topics each closed-form generator can actually serve offline."""
    topics = {i.topic_id for i in AR_BANK}
    rng = random.Random(0)
    return {
        "topics_with_facts": len(topics),
        "true_false": sum(1 for t in topics if build_true_false(t, rng)),
        "multiple_choice": sum(1 for t in topics if build_multiple_choice(t, rng)),
        "multiple_response": sum(1 for t in topics if build_multiple_response(t, rng)),
    }
