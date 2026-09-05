"""Tutor: teaching flow, chat and command routing.

The teaching flow is deliberately front-loaded with a short explanation and then
switches to interrogation - explanation is capped so the student spends most of
the session retrieving rather than reading. Offline it uses the knowledge base
plus a structured fallback built from the topic registry and the seed material.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..config import get_logger
from ..data.topics import topic_index
from ..models.db import find_topic_by_name, session_scope
from ..models.schemas import (
    Citation,
    Question,
    QuestionType,
    RetrievalResult,
    SessionMode,
)
from . import rag
from .llm import TUTOR_SYSTEM, get_llm, language_directive, system_with_language
from .progress import topic_report
from .question_gen import generate_question

log = get_logger(__name__)
TOPIC_INDEX = topic_index()


# --------------------------------------------------------------- teaching
@dataclass
class Lesson:
    topic_id: str
    topic_name: str
    explanation: str = ""
    intuition: str = ""
    mathematics: str = ""
    exam_points: list[str] = field(default_factory=list)
    example: str = ""
    citations: list[Citation] = field(default_factory=list)
    grounded: bool = False
    source: str = "offline"

    def as_markdown(self) -> str:
        parts = []
        if self.explanation:
            parts.append(f"**What it is**\n\n{self.explanation}")
        if self.intuition:
            parts.append(f"**Intuition**\n\n{self.intuition}")
        if self.mathematics:
            parts.append(f"**The mathematics that matters**\n\n{self.mathematics}")
        if self.exam_points:
            bullets = "\n".join(f"- {p}" for p in self.exam_points)
            parts.append(f"**What to remember for the exam**\n\n{bullets}")
        if self.example:
            parts.append(f"**Example**\n\n{self.example}")
        return "\n\n".join(parts)


_LESSON_PROMPT = """Teach the topic **{topic}** to a student whose exam is in {days} days.

Their current profile on this topic:
{profile}

{context_block}

Produce a TIGHT lesson. The student has minutes, not hours. Rules:
- No filler, no "in this lesson we will". Start with the substance.
- Explanation: at most 6 sentences.
- Intuition: at most 3 sentences, concrete.
- Mathematics: the one or two formulas that get examined, with each symbol named.
- Exam points: 4-6 bullets of exactly what an examiner rewards, phrased as the student
  should write them.
- Example: one small worked example or scenario, at most 5 sentences.
- If the source material is provided, ground your claims in it and cite [S1], [S2].
  If it does not cover something, rely on standard knowledge but do not invent
  course-specific details (lecture numbers, notation the course uses, etc.).
{language_reminder}
Return JSON exactly:
{{"explanation": "...", "intuition": "...", "mathematics": "...",
  "exam_points": ["...", "..."], "example": "..."}}"""


def build_lesson(topic_id: str, use_llm: bool = True, use_rag: bool = True) -> Lesson:
    from ..config import get_settings

    seed = TOPIC_INDEX.get(topic_id, {})
    name = seed.get("name", topic_id.replace("_", " ").title())
    report = topic_report(topic_id)

    retrieval: RetrievalResult | None = None
    if use_rag:
        try:
            retrieval = rag.retrieve_for_topic(name, topic_id, k=5,
                                               category=seed.get("category"))
        except Exception as exc:
            log.warning("lesson retrieval failed: %s", exc)

    grounded = bool(retrieval and retrieval.grounded)
    citations = retrieval.citations()[:4] if grounded else []

    llm = get_llm()
    if use_llm and llm.available:
        profile = _profile_line(report)
        context_block = (
            "SOURCE MATERIAL FROM THE STUDENT'S COURSE:\n\n"
            + retrieval.context_block(5000) + "\n"
            if grounded else
            "No course material has been uploaded for this topic - use standard "
            "university-level knowledge and do not fabricate course specifics.\n"
        )
        data, resp = llm.complete_json(
            _LESSON_PROMPT.format(
                topic=name,
                days=get_settings().days_remaining(),
                profile=profile,
                context_block=context_block,
                # repeated here, not just in the system prompt: a long English
                # context_block right before this can otherwise pull the model
                # back into English by the time it writes the JSON fields
                language_reminder=language_directive(),
            ),
            system=system_with_language(TUTOR_SYSTEM),
            temperature=0.4,
            max_tokens=1600,
        )
        if isinstance(data, dict) and data.get("explanation"):
            return Lesson(
                topic_id=topic_id,
                topic_name=name,
                explanation=str(data.get("explanation", "")),
                intuition=str(data.get("intuition", "")),
                mathematics=str(data.get("mathematics", "")),
                exam_points=[str(x) for x in data.get("exam_points", [])][:6],
                example=str(data.get("example", "")),
                citations=citations,
                grounded=grounded,
                source="llm+rag" if grounded else "llm",
            )
        log.info("lesson LLM path unavailable (%s); using offline lesson", resp.error)

    return _offline_lesson(topic_id, name, seed, retrieval, grounded, citations)


def _profile_line(report: dict[str, Any]) -> str:
    if not report:
        return "No data yet - this is their first exposure."
    t = report.get("topic", {})
    dims = ", ".join(
        f"{d} {t.get(f'{d}_score', 0):.0%}"
        for d in ("concept", "calculation", "reasoning", "comparison", "application")
        if t.get(f"{d}_score", 0) > 0
    ) or "no dimension tested yet"
    return (f"Overall {t.get('overall', 0):.0%} ({report.get('mastery', 'unknown')}); {dims}. "
            f"Weakest dimension: {report.get('weak_dimension', 'unknown')}. "
            f"{t.get('attempt_count', 0)} attempts, {t.get('mistake_count', 0)} mistakes.")


def _offline_lesson(topic_id: str, name: str, seed: dict[str, Any],
                    retrieval: RetrievalResult | None, grounded: bool,
                    citations: list[Citation]) -> Lesson:
    """Structured lesson from the knowledge base + topic registry, no LLM."""
    keywords = list(seed.get("keywords", []))
    kw_line = ", ".join(keywords[:8]) if keywords else "its core mechanism"

    if grounded and retrieval:
        excerpt = "\n\n".join(
            f"> {c.text.strip()[:700]}\n>\n> *Source: {c.citation.label()}*"
            for c in retrieval.chunks[:3]
        )
        explanation = (
            f"From your uploaded course material on **{name}**:\n\n{excerpt}"
        )
    else:
        explanation = (
            f"**{name}** is a {seed.get('priority', 'HIGH').lower()}-priority "
            f"{seed.get('category', 'Machine Learning')} topic in this course "
            f"(exam relevance {float(seed.get('exam_relevance', 0.5)):.0%}).\n\n"
            f"No uploaded material covers it yet, so this app will not state "
            f"course-specific claims about it. What the exam will test is your ability to "
            f"reason about: **{kw_line}**.\n\n"
            f"Upload the relevant lecture slides on the Materials page to get a grounded "
            f"explanation, or configure an LLM provider in Settings for a full explanation."
        )

    intuition = (
        f"Ask yourself the three questions the examiner will ask about {name}: "
        f"what mechanism does it implement, what breaks without it, and what does it cost?"
    )
    maths = ""
    from .calc_engine import TOPIC_GENERATORS, generate_problem

    if topic_id in TOPIC_GENERATORS:
        prob = generate_problem(topic_id=topic_id, seed=1)
        formulas = [p.hint_formula for p in prob.parts if p.hint_formula]
        if formulas:
            maths = ("The formulas this topic is examined on:\n\n"
                     + "\n".join(f"- `{f}`" for f in dict.fromkeys(formulas)))
        else:
            maths = "This topic is examined numerically - see the calculation practice below."

    exam_points = [f"Be able to explain **{k}** in one precise sentence." for k in keywords[:4]]
    prereqs = seed.get("prereqs", [])
    if prereqs:
        names = [TOPIC_INDEX[p]["name"] for p in prereqs if p in TOPIC_INDEX]
        if names:
            exam_points.append(f"Depends on: {', '.join(names)} - a gap there shows up here.")
    if topic_id in TOPIC_GENERATORS:
        exam_points.append("This topic carries calculation marks - practise the numbers, "
                           "not just the theory.")

    return Lesson(
        topic_id=topic_id,
        topic_name=name,
        explanation=explanation,
        intuition=intuition,
        mathematics=maths,
        exam_points=exam_points,
        example="",
        citations=citations,
        grounded=grounded,
        source="rag" if grounded else "offline",
    )


# --------------------------------------------------------------- explanations
def explain(query: str, topic_id: str | None = None, beginner: bool = False,
            analogy: bool = False, use_llm: bool = True) -> dict[str, Any]:
    """Answer a question, grounded in course material where possible."""
    retrieval = None
    try:
        retrieval = rag.retrieve(query, k=6)
    except Exception as exc:
        log.warning("retrieval failed: %s", exc)
    grounded = bool(retrieval and retrieval.grounded)

    llm = get_llm()
    if use_llm and llm.available:
        style = ""
        if beginner:
            style = "Explain as if to a beginner: plain language first, then the technical term.\n"
        if analogy:
            style += "Use one concrete analogy, then immediately map every part of the analogy " \
                     "back to the real mechanism.\n"
        context_block = (
            "SOURCE MATERIAL FROM THE COURSE:\n\n" + retrieval.context_block(5000)
            + "\n\nGround your answer in this material and cite [S1], [S2]. If the material "
              "does not establish part of the answer, say so explicitly.\n"
            if grounded else
            "No course material was retrieved for this question. Answer from standard "
            "knowledge, and state clearly that the course material does not cover it.\n"
        )
        resp = llm.complete(
            f"{context_block}\nSTUDENT'S QUESTION: {query}\n\n{style}"
            "Answer concisely (at most 8 sentences) and end with one question that tests "
            "whether they actually understood."
            f"{language_directive()}",
            system=system_with_language(TUTOR_SYSTEM),
            max_tokens=1200,
        )
        if resp.ok:
            return {
                "answer": resp.text,
                "citations": retrieval.citations()[:4] if grounded else [],
                "grounded": grounded,
                "source": "llm+rag" if grounded else "llm",
            }

    # offline: return the material itself rather than inventing an answer
    if grounded and retrieval:
        body = "\n\n".join(
            f"**{c.citation.label()}**\n\n{c.text.strip()[:900]}"
            for c in retrieval.chunks[:3]
        )
        return {
            "answer": ("No LLM is configured, so here is the relevant course material "
                       f"verbatim rather than a generated answer:\n\n{body}"),
            "citations": retrieval.citations()[:4],
            "grounded": True,
            "source": "rag",
        }
    return {
        "answer": ("The uploaded source material does not establish an answer to this, and no "
                   "LLM provider is configured. Upload the relevant lecture material on the "
                   "**Materials** page, or add an API key in **Settings**, and ask again.\n\n"
                   "In the meantime the Quiz and Mock Exam pages work fully offline."),
        "citations": [],
        "grounded": False,
        "source": "none",
    }


def compare_sources_answer(query: str) -> dict[str, Any]:
    """Show university vs Udemy treatment of a topic and flag conflicts."""
    res = rag.compare_sources(query, k=3)
    uni, udemy = res["university"], res["udemy"]
    out = {
        "university": [{"text": c.text[:800], "citation": c.citation.label()} for c in uni.chunks],
        "udemy": [{"text": c.text[:800], "citation": c.citation.label()} for c in udemy.chunks],
        "conflict_note": "",
    }
    if uni.chunks and udemy.chunks:
        out["conflict_note"] = (
            "Both sources cover this. **For the exam, follow the university material** - "
            "the Udemy course is for intuition and practical implementation. If the two "
            "disagree on notation or emphasis, answer with the university's version."
        )
    elif uni.chunks:
        out["conflict_note"] = "Only university material covers this."
    elif udemy.chunks:
        out["conflict_note"] = (
            "Only Udemy material covers this. Treat it as supporting intuition; the exam "
            "follows the university syllabus, so verify against your lecture notes."
        )
    else:
        out["conflict_note"] = "Neither source covers this topic yet."
    return out


# --------------------------------------------------------------- commands
COMMANDS = {
    "/study": "Start a study session on the highest-value topic (or /study <topic>)",
    "/quiz": "Quiz me (optionally /quiz <topic>)",
    "/exam": "Ask one exam-level question",
    "/mock": "Build a full mock exam",
    "/review": "Review my recent mistakes",
    "/weakness": "Show my weakest topics",
    "/progress": "Show my progress and readiness",
    "/plan": "Show the adaptive study plan",
    "/explain": "/explain <topic or question>",
    "/calculate": "/calculate <topic> - give me a calculation problem",
    "/assertion": "/assertion <topic> - give me an assertion-reason question",
    "/compare": "/compare <a> vs <b>",
    "/rapid_review": "Rapid-fire active recall",
    "/help": "List commands",
}


@dataclass
class CommandResult:
    kind: str                      # text | question | navigate | lesson | report
    text: str = ""
    question: Question | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    navigate: str | None = None


#: chat words that must never be treated as topic evidence
_CHAT_STOPWORDS = {
    "teach", "explain", "study", "learn", "quiz", "test", "give", "show", "tell",
    "about", "with", "what", "which", "when", "why", "how", "the", "and", "for",
    "please", "session", "question", "questions", "minute", "minutes", "some",
    "more", "hard", "easy", "like", "beginner", "analogy", "simple", "want",
}

#: short but genuine topic terms that must still resolve
_ACRONYMS = {
    "pca": "pca", "knn": "knn", "svm": "svm", "cnn": "cnn_basics", "rnn": "rnn",
    "lstm": "lstm", "gru": "gru", "gmm": "gmm", "kde": "kde", "em": "expectation_maximization",
    "mlp": "mlp", "bert": "bert", "gpt": "gpt", "lda": "lda", "ucb": "ucb",
    "vit": "vision_transformers", "clip": "clip", "rag": "rag", "svr": "svr",
    "dbscan": "dbscan", "xgboost": "xgboost", "relu": "activation_functions",
    "backprop": "backpropagation", "sgd": "optimizers", "adam": "optimizers",
    "dropout": "dropout", "batchnorm": "batch_normalization", "attention": "attention",
    "transformer": "transformer", "transformers": "transformer", "softmax": "loss_functions",
}


#: multi-word phrases that name a topic but do not share its registry wording
_ALIASES: dict[str, str] = {
    "convolutional neural network": "cnn_basics",
    "convolutional network": "cnn_basics",
    "recurrent neural network": "rnn",
    "vanishing gradient": "vanishing_gradients",
    "exploding gradient": "vanishing_gradients",
    "gradient clipping": "vanishing_gradients",
    "skip connection": "residual_connections",
    "residual connection": "residual_connections",
    "scaled dot product": "scaled_dot_product",
    "dot product attention": "scaled_dot_product",
    "self attention": "transformer",
    "cross attention": "self_vs_cross_attention",
    "positional encoding": "transformer",
    "confusion matrix": "confusion_matrix",
    "bias variance": "bias_variance",
    "weight decay": "weight_decay",
    "batch normalization": "batch_normalization",
    "batch normalisation": "batch_normalization",
    "learning rate": "learning_rate",
    "activation function": "activation_functions",
    "loss function": "loss_functions",
    "feature scaling": "feature_scaling",
    "cross validation": "cross_validation",
    "k fold": "cross_validation",
    "decision tree": "decision_trees",
    "random forest": "random_forests",
    "naive bayes": "naive_bayes",
    "logistic regression": "logistic_regression",
    "linear regression": "linear_regression",
    "transfer learning": "transfer_learning",
    "object detection": "object_detection",
    "receptive field": "receptive_field",
    "parameter count": "cnn_parameter_count",
    "word embedding": "embeddings",
    "language model": "gpt",
}


def _stem(word: str) -> str:
    for suffix in ("ing", "es", "s"):
        if len(word) > 4 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _word_set(text: str) -> set[str]:
    return {_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w}


def resolve_topic(text: str) -> tuple[str, str] | None:
    """Map free text to a topic id + name.

    Scores whole-word overlap so chat filler ("teach me ...") cannot match, and
    ignores one/two-character keywords which would otherwise match everything.
    """
    if not text.strip():
        return None
    # exact name / id match is authoritative
    with session_scope() as s:
        t = find_topic_by_name(s, text.strip(), exact_only=True)
        if t is not None:
            return t.id, t.name

    low = " " + text.lower().strip() + " "
    words = {w for w in _word_set(text) if w not in _CHAT_STOPWORDS}

    # explicit multi-word aliases are the most reliable signal
    for phrase, tid in sorted(_ALIASES.items(), key=lambda kv: -len(kv[0])):
        if phrase in low and tid in TOPIC_INDEX:
            return tid, TOPIC_INDEX[tid]["name"]

    # then unambiguous acronyms
    for token in words:
        if token in _ACRONYMS:
            tid = _ACRONYMS[token]
            if tid in TOPIC_INDEX:
                return tid, TOPIC_INDEX[tid]["name"]

    best: tuple[float, str, str] | None = None
    for tid, seed in TOPIC_INDEX.items():
        score = 0.0
        # strip a parenthesised acronym from the name for matching
        bare = re.sub(r"\s*\(.*?\)\s*", " ", seed["name"].lower()).strip()
        if f" {bare} " in low:
            score += 6 + len(bare) / 20
        else:
            name_words = {w for w in _word_set(bare) if w not in _CHAT_STOPWORDS}
            if name_words:
                overlap = len(name_words & words) / len(name_words)
                if name_words <= words:
                    # exact name coverage; longer names are more specific
                    score += 5 + 0.4 * len(name_words)
                elif overlap >= 0.5:
                    score += 3.5 * overlap

        for kw in seed.get("keywords", []):
            kw_low = kw.lower()
            if len(kw_low) < 4:
                continue  # 'a', 'z', 'k', 'c' would match anything
            kw_words = _word_set(kw_low)
            if f" {kw_low} " in low or (kw_words and kw_words <= words):
                score += 2

        if score and (best is None or score > best[0]):
            best = (score, tid, seed["name"])
    if best and best[0] >= 2:
        return best[1], best[2]
    # last resort: the database's fuzzy match
    with session_scope() as s:
        t = find_topic_by_name(s, text.strip())
        return (t.id, t.name) if t is not None else None


def route_command(text: str, use_llm: bool = True) -> CommandResult:
    """Interpret a chat message; slash commands take precedence."""
    raw = text.strip()
    if not raw:
        return CommandResult("text", "Say something, or type /help.")

    if not raw.startswith("/"):
        return _natural_language(raw, use_llm=use_llm)

    parts = raw.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd == "/help":
        lines = "\n".join(f"- `{k}` - {v}" for k, v in COMMANDS.items())
        return CommandResult("text", f"**Commands**\n\n{lines}")

    if cmd == "/study":
        target = resolve_topic(arg) if arg else None
        if target is None:
            from .planner import next_topic

            nxt = next_topic()
            if nxt is None:
                return CommandResult("text", "No topics available yet.")
            return CommandResult(
                "navigate",
                f"Start with **{nxt['topic']}** ({nxt['focus']} focus).\n\n{nxt['reason']}",
                navigate="Study",
                payload={"topic_id": nxt["topic_id"], "mode": nxt["mode"]},
            )
        return CommandResult(
            "navigate", f"Opening a study session on **{target[1]}**.",
            navigate="Study", payload={"topic_id": target[0]},
        )

    if cmd in ("/quiz", "/exam"):
        target = resolve_topic(arg) if arg else None
        difficulty = 6 if cmd == "/exam" else 4
        if target is None:
            from .planner import next_topic

            nxt = next_topic()
            if nxt is None:
                return CommandResult("text", "No topics available yet.")
            target = (nxt["topic_id"], nxt["topic"])
        q = generate_question(target[0], None, difficulty, use_llm=use_llm)
        return CommandResult("question", f"**{target[1]}** - answer first, I will mark it after.",
                             question=q)

    if cmd == "/mock":
        return CommandResult("navigate", "Opening the mock exam builder.", navigate="Mock Exam")

    if cmd == "/review":
        from .weakness import error_log

        errors = error_log(limit=8)
        if not errors:
            return CommandResult("text", "No unresolved mistakes. Take a mock exam to find some.")
        lines = [
            f"**{e['topic']}** - {e['mistake_type']} ({e['severity']}), {e['date']}\n"
            f"  - {e['question'][:180]}"
            for e in errors
        ]
        return CommandResult("text", "**Unresolved mistakes**\n\n" + "\n\n".join(lines),
                             payload={"errors": errors})

    if cmd == "/weakness":
        from .weakness import weakness_report

        rep = weakness_report(limit=8)
        lines = [
            f"{i}. **{r['name']}** - {r['mastery']} ({r['effective']:.0%}), "
            f"weakest: {r['weak_dimension']}\n   {r['action']}"
            for i, r in enumerate(rep["weakest"], 1)
        ]
        return CommandResult("text", "**Weakest topics**\n\n" + "\n".join(lines),
                             navigate=None)

    if cmd == "/progress":
        from .progress import compute_readiness, dimension_profile

        r = compute_readiness()
        dims = dimension_profile()
        body = (
            f"**Exam readiness: {r.overall:.0%}**\n\n"
            f"- Critical topic mastery: {r.critical_mastery:.0%}\n"
            f"- Calculation: {r.calculation:.0%}\n"
            f"- Reasoning: {r.reasoning:.0%}\n"
            f"- Exam performance: {r.exam_performance:.0%}\n"
            f"- Coverage: {r.coverage:.0%}\n"
            f"- Confidence: {r.confidence:.0%}\n\n"
            f"ML {r.ml_score:.0%} | DL {r.dl_score:.0%}\n\n"
            f"Dimensions: " + ", ".join(f"{k} {v:.0%}" for k, v in dims.items())
        )
        return CommandResult("text", body)

    if cmd == "/plan":
        return CommandResult("navigate", "Opening the study plan.", navigate="Dashboard")

    if cmd == "/explain":
        if not arg:
            return CommandResult("text", "Usage: `/explain backpropagation`")
        target = resolve_topic(arg)
        res = explain(arg, target[0] if target else None, use_llm=use_llm)
        return CommandResult("text", res["answer"], payload=res)

    if cmd == "/calculate":
        target = resolve_topic(arg) if arg else None
        from .calc_engine import TOPIC_GENERATORS

        if target is None or target[0] not in TOPIC_GENERATORS:
            candidates = ", ".join(sorted(TOPIC_INDEX[t]["name"] for t in TOPIC_GENERATORS
                                          if t in TOPIC_INDEX)[:12])
            if target is None:
                return CommandResult("text", f"Which topic? Try one of: {candidates}")
        tid = target[0] if target else "backpropagation"
        q = generate_question(tid, QuestionType.CALCULATION, 5, use_llm=False)
        return CommandResult("question", "Solve it fully before I mark it.", question=q)

    if cmd == "/assertion":
        target = resolve_topic(arg) if arg else None
        tid = target[0] if target else "backpropagation"
        q = generate_question(tid, QuestionType.ASSERTION_REASON, 5, use_llm=use_llm)
        return CommandResult("question", "Choose one option.", question=q)

    if cmd == "/compare":
        if not arg:
            return CommandResult("text", "Usage: `/compare LSTM vs GRU`")
        target = resolve_topic(arg.split(" vs ")[0] if " vs " in arg else arg)
        tid = target[0] if target else "lstm"
        q = generate_question(tid, QuestionType.COMPARISON, 5, use_llm=use_llm)
        if arg and use_llm and get_llm().available:
            q.prompt = f"Compare: {arg}.\n\n{q.prompt}"
        return CommandResult("question", "Write the comparison as you would in the exam.",
                             question=q)

    if cmd == "/rapid_review":
        return CommandResult("navigate", "Starting rapid revision.", navigate="Study",
                             payload={"mode": SessionMode.RAPID.value})

    return CommandResult("text", f"Unknown command `{cmd}`. Type `/help`.")


#: ordered - the first match wins, so specific intents must precede general ones
_NL_PATTERNS: list[tuple[str, str]] = [
    (r"what (should|do) i (study|do|learn)|what'?s next|what next\b|where do i (start|begin)",
     "next"),
    (r"\bmock\b|\bexam simulation\b|\bfull exam\b|\bpractice exam\b", "mock"),
    (r"\bassertion\b|\bassertion[- ]reason\b", "assertion"),
    (r"\bcalculat|\bcompute\b|\bnumerical\b|\bwork(ed)? example\b|\bsolve\b", "calculate"),
    (r"\bquiz\b|\bask me\b|\btest me\b|\bexam question\b", "quiz"),
    (r"\breview (my )?(mistakes?|errors?)\b|\bmistake|\bwhat did i get wrong\b", "review"),
    (r"\bweak(est)?\b|\bbad at\b|\bstruggl", "weakness"),
    (r"\bprogress\b|\breadiness\b|\bhow am i doing\b|\bhow ready\b", "progress"),
    (r"\bcompare\b|\bdifference between\b|\s+vs\.?\s+|\bversus\b", "compare"),
    (r"\b(\d+)\s*minute", "timed"),
    (r"\b(study )?plan\b|\bschedule\b", "plan"),
    (r"\b(teach|study|learn)\b", "teach"),
    (r"\bexplain\b|\bwhy\b|\bhow does\b|\bwhat is\b|\bwhat are\b", "explain"),
]


def _natural_language(text: str, use_llm: bool = True) -> CommandResult:
    low = text.lower()
    intent = None
    for pattern, name in _NL_PATTERNS:
        if re.search(pattern, low):
            intent = name
            break

    topic = resolve_topic(text)

    if intent == "teach":
        if topic:
            return CommandResult("navigate", f"Teaching **{topic[1]}**.", navigate="Study",
                                 payload={"topic_id": topic[0]})
        return route_command("/study", use_llm)
    if intent == "quiz":
        return route_command(f"/quiz {topic[1] if topic else ''}".strip(), use_llm)
    if intent == "mock":
        return route_command("/mock", use_llm)
    if intent == "assertion":
        return route_command(f"/assertion {topic[1] if topic else ''}".strip(), use_llm)
    if intent == "calculate":
        return route_command(f"/calculate {topic[1] if topic else ''}".strip(), use_llm)
    if intent == "next":
        return route_command("/study", use_llm)
    if intent == "weakness":
        return route_command("/weakness", use_llm)
    if intent == "review":
        return route_command("/review", use_llm)
    if intent == "progress":
        return route_command("/progress", use_llm)
    if intent == "plan":
        return route_command("/plan", use_llm)
    if intent == "compare":
        return route_command(f"/compare {text}", use_llm)
    if intent == "timed":
        m = re.search(r"(\d+)\s*minute", low)
        minutes = int(m.group(1)) if m else 30
        mode = (SessionMode.QUICK if minutes <= 20 else
                SessionMode.THIRTY if minutes <= 40 else
                SessionMode.SIXTY if minutes <= 70 else SessionMode.DEEP)
        return CommandResult("navigate", f"Starting a {mode.value.lower()}.",
                             navigate="Study", payload={"mode": mode.value})

    beginner = "beginner" in low or "simple" in low or "like i am" in low
    analogy = "analogy" in low or "analogies" in low
    res = explain(text, topic[0] if topic else None, beginner=beginner,
                  analogy=analogy, use_llm=use_llm)
    return CommandResult("text", res["answer"], payload=res)


# --------------------------------------------------------------- follow-ups
def followup_question(topic_id: str, previous: Question, score: float,
                      use_llm: bool = True) -> Question:
    """Pick the right next question after an answer, per the adaptive rules."""
    if score >= 8:
        # move up in difficulty, or switch to the next untested dimension
        harder = min(6, previous.difficulty + 1)
        if previous.question_type == QuestionType.CALCULATION:
            return generate_question(topic_id, QuestionType.WHAT_IF, harder, use_llm=use_llm)
        from .calc_engine import topic_has_calculation

        if topic_has_calculation(topic_id):
            return generate_question(topic_id, QuestionType.CALCULATION, harder, use_llm=use_llm)
        return generate_question(topic_id, QuestionType.ASSERTION_REASON, harder, use_llm=use_llm)

    if score >= 5:
        # same difficulty, same dimension - consolidate
        return generate_question(topic_id, previous.question_type, previous.difficulty,
                                 use_llm=use_llm, exclude_ids={previous.id})

    # struggling: same dimension, one level easier
    easier = max(2, previous.difficulty - 1)
    return generate_question(topic_id, previous.question_type, easier, use_llm=use_llm,
                             exclude_ids={previous.id})


def teaching_steps() -> list[str]:
    return [
        "Explanation",
        "Intuition",
        "The mathematics",
        "Exam points",
        "Example",
        "Explain it back",
        "Evaluation",
        "Harder reasoning question",
        "Calculation",
        "Profile updated",
    ]
