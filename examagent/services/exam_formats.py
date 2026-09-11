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

from ..config import get_logger
from ..models.schemas import AnswerOption, Category, Priority, Question, QuestionType
from .assertion_engine import AR_BANK, ARItem

log = get_logger(__name__)

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


# --------------------------------------------------------------- translation
#: Bank statements are fixed English - that is what makes them exact, and it is
#: also why a student studying in another language kept meeting English items
#: among otherwise translated ones. Translating them keeps the exactness (the
#: key comes from the truth flags, which translation never touches) and is
#: cached per sentence, so the bank's ~174 statements are paid for once.
_TRANSLATION_KV = "exam_formats_translation"

_TRANSLATE_PROMPT = """Translate each numbered exam statement into {language}.

Rules, in order of importance:
- Preserve the meaning exactly. These are statements a student must mark true or
  false, so a dropped negation, a softened "always"/"never", or a changed
  quantifier turns a true statement false. Translate, never rephrase.
- Keep technical names in English (Linear Regression, gradient descent, ReLU,
  overfitting). Everything around them - verbs, connectives, qualifiers - is
  {language}.
- Same number of items, same order.

{numbered}

Return JSON exactly: {{"translations": ["...", "..."]}}"""


def _translation_cache(language: str) -> dict[str, str]:
    from ..models.db import kv_get, session_scope

    with session_scope() as s:
        stored = kv_get(s, f"{_TRANSLATION_KV}:{language}", {}) or {}
    return stored if isinstance(stored, dict) else {}


def _store_translations(language: str, pairs: dict[str, str]) -> None:
    from ..models.db import kv_get, kv_set, session_scope

    with session_scope() as s:
        key = f"{_TRANSLATION_KV}:{language}"
        stored = kv_get(s, key, {}) or {}
        if not isinstance(stored, dict):
            stored = {}
        stored.update(pairs)
        kv_set(s, key, stored)


def translate_all(texts: list[str], language: str) -> dict[str, str]:
    """English -> `language` for a batch of statements.

    Checked-in translations first (see `data.translations_az`): they cost
    nothing, work with no API credit at all, and are the reason an offline
    paper is not in English. The LLM only covers whatever they miss, and its
    output is cached so each sentence is paid for once.
    """
    from .llm import get_llm

    # checked-in wording wins over anything a previous run cached: it has been
    # read by a person, the cache has not
    cache = {**_translation_cache(language), **_checked_in(language)}

    missing = [t for t in dict.fromkeys(texts) if t and t not in cache]
    if not missing:
        return cache

    llm = get_llm()
    if not llm.available:
        return cache

    numbered = "\n".join(f"{i}. {t}" for i, t in enumerate(missing, 1))
    from .llm import LANGUAGE_NAMES

    data, resp = llm.complete_json(
        _TRANSLATE_PROMPT.format(
            language=LANGUAGE_NAMES.get(language, language), numbered=numbered),
        temperature=0.0,
        max_tokens=min(3000, 120 * len(missing) + 200),
    )
    translations = (data or {}).get("translations") if isinstance(data, dict) else None
    if not isinstance(translations, list) or len(translations) != len(missing):
        # a partial or misaligned translation would attach the wrong sentence to
        # the wrong key - keep the English rather than risk that
        log.info("bank translation unusable (%s); keeping English",
                 resp.error if resp else "shape mismatch")
        return cache

    fresh = {src: str(out).strip() for src, out in zip(missing, translations)
             if str(out).strip()}
    if fresh:
        _store_translations(language, fresh)
        cache.update(fresh)
    return cache


def localise(question: Question, use_llm: bool = True,
             language: str | None = None) -> Question:
    """Return the item with its text in the configured language.

    `use_llm` only governs whether the *LLM* may be asked for anything the
    checked-in translations miss - the translations themselves always apply,
    which is what lets an offline paper, or one built with no API credit left,
    still come out in the student's language.

    Only the wording moves: the correct option, the truth flags behind it and
    the option lettering are untouched, so a translation cannot mis-key an item
    - at worst it reads awkwardly.
    """
    from ..config import get_settings

    language = (language or get_settings().language or "en").strip().lower()
    if language == "en":
        return question

    # 'True'/'False' are the answer values themselves, not prose: translating
    # them would stop the marker recognising the answer
    fixed = {"True", "False"}
    texts = [question.prompt, *question.statements, question.model_answer,
             *[o.text for o in question.options if o.text not in fixed]]
    texts = [t for t in texts if t and t.strip()]
    if not texts:
        return question

    try:
        table = translate_all(texts, language) if use_llm else _static_table(language)
    except Exception as exc:  # translation is a nicety, never a blocker
        log.warning("bank translation failed: %s", exc)
        return question

    def _t(text: str) -> str:
        return table.get(text, text)

    translated = question.model_copy(deep=True)
    translated.prompt = _localise_prompt(question.prompt, table, language)
    translated.statements = [_t(s) for s in question.statements]
    for option in translated.options:
        if option.text not in fixed:
            option.text = _localise_option(option.text, table, language)
    translated.model_answer = _localise_model_answer(translated, table, language)
    return translated


def _localise_prompt(prompt: str, table: dict[str, str], language: str) -> str:
    """Stems are either fixed (and in the table) or built around a topic name."""
    if prompt in table:
        return table[prompt]
    if language == "az" and prompt.startswith("Which statement about "):
        topic = prompt[len("Which statement about "):].removesuffix(" is correct?")
        return f"{topic} haqqında hansı ifadə doğrudur?"
    return prompt


def _localise_option(text: str, table: dict[str, str], language: str) -> str:
    """Combination options are built from statement numbers, so they are joined
    here rather than translated - '1, 3, and 4' is not a sentence to look up."""
    if text in table:
        return table[text]
    if language == "az" and text and text[0].isdigit():
        return text.replace(", and ", ", ").replace(" and ", " və ")
    return text


def _static_table(language: str) -> dict[str, str]:
    """Translations that need no API call at all."""
    return {**_translation_cache(language), **_checked_in(language)}


def _checked_in(language: str) -> dict[str, str]:
    """Translations shipped with the app - no API call, no API budget."""
    if language != "az":
        return {}
    from ..data import translations_az

    return {**translations_az.STATEMENTS, **translations_az.PHRASES}


def _localise_model_answer(question: Question, table: dict[str, str],
                           language: str) -> str:
    """The generators' own wording around the answer, in the student's language.

    Rebuilt rather than translated word for word: these strings are assembled
    from the key and the statement numbers, so the parts are known here and a
    round trip through a translator would only risk mangling them.
    """
    if language != "az":
        return question.model_answer

    key = question.correct_option or ""
    if question.question_type == QuestionType.TRUE_FALSE:
        verdict = table.get(
            "The statement holds as written." if key == "True" else
            "The statement is false as written - check the clause that overstates it.",
            "",
        )
        return f"{key}. {verdict}".strip()

    if question.question_type == QuestionType.MULTIPLE_RESPONSE:
        listed = next((o.text for o in question.options if o.key == key), "")
        return (f"{key} — {listed} nömrəli ifadələr doğrudur; "
                "qalanları yazıldığı kimi yanlışdır.")

    if question.question_type == QuestionType.MCQ:
        answer = next((o.text for o in question.options if o.key == key), "")
        return f"{key}. {table.get(answer, answer)}"

    return question.model_answer
