"""Guided walkthrough of a single uploaded document.

Uploaded material otherwise only ever surfaces as background context for
retrieval - useful, but passive. This module turns one document into an ordered
study route: read a section, get a brief grounded in that section alone, answer
a question written from it, get marked, move on. Per-document progress lives in
the key-value store, so the walk survives a restart.

Everything degrades. Without an LLM the brief is *extracted* from the section
rather than written about it - the sentences that carry a definition, a cause,
a contrast or a formula, in the section's own words - and the question becomes a
recall prompt graded against the technical terms the section actually uses.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..config import get_logger
from ..data.topics import TOPIC_SEEDS
from ..models.db import kv_get, kv_set, session_scope
from ..models.schemas import (
    Citation,
    Question,
    QuestionType,
    RetrievalResult,
    RetrievedChunk,
)
from .llm import TUTOR_SYSTEM, get_llm, system_with_language
from .question_gen import (
    _category_of,
    _priority_of,
    _topic_name,
    question_from_material,
)
from .vectorstore import get_vector_store

log = get_logger(__name__)

KV_PREFIX = "walkthrough:"

#: minutes budgeted per section (read + answer + read the marking)
MINUTES_PER_SECTION = 3

#: A slide deck chunks one slide at a time, and a single bullet is not worth a
#: question. Consecutive sections are merged until a step carries at least this
#: much text, and never grown past the ceiling.
MIN_STEP_CHARS = 900
MAX_STEP_CHARS = 2600


@dataclass
class Section:
    """One chunk of a document, presented as a study step."""

    chunk_id: str
    text: str
    label: str
    page: int | None = None
    page_end: int | None = None
    topics: list[str] = field(default_factory=list)
    citation: Citation | None = None
    index: int = 0

    @property
    def topic_id(self) -> str:
        """The registry topic this section is about, or '' when it matches none."""
        return self.topics[0] if self.topics else ""

    @property
    def heading(self) -> str:
        if self.page is None:
            return self.label
        if self.page_end is not None and self.page_end != self.page:
            return f"{self.label} · p.{self.page}-{self.page_end}"
        return f"{self.label} · p.{self.page}"


# --------------------------------------------------------------- sections
def sections(filename: str) -> list[Section]:
    """The document's chunks as an ordered walk.

    Order is the document's own: the store returns chunks in ingestion order,
    and a stable sort by page keeps that order inside each page while fixing
    backends that do not preserve it.
    """
    chunks = get_vector_store().list_by(filename=filename)
    out: list[Section] = []
    for c in chunks:
        meta = c.metadata or {}
        page = meta.get("page")
        label = meta.get("section") or (f"Page {page}" if page is not None else "Section")
        topics = [t for t in str(meta.get("topics") or "").split(",") if t]
        out.append(Section(
            chunk_id=c.chunk_id,
            text=c.text,
            label=str(label),
            page=page if isinstance(page, int) else None,
            topics=topics,
            citation=Citation(
                source_type=str(meta.get("source_type", "")),
                source_name=str(meta.get("source_name", "")),
                lecture=meta.get("lecture") or None,
                page=page if isinstance(page, int) else None,
                section=str(label),
                score=1.0,
            ),
        ))
    out.sort(key=lambda s: s.page if s.page is not None else -1)
    grouped = _group(out)
    for i, s in enumerate(grouped):
        s.index = i
    return grouped


_SLIDE_LABEL = re.compile(r"^Slide \d+$")


def _merge(parts: list[Section]) -> Section:
    """Fold consecutive chunks into one study step, keeping the page range so
    the citation still points at exactly where the text came from."""
    if len(parts) == 1:
        return parts[0]
    head, tail = parts[0], parts[-1]
    if all(_SLIDE_LABEL.match(p.label) for p in parts):
        label = f"Slides {head.label.split()[-1]}-{tail.label.split()[-1]}"
    else:
        label = head.label if head.label == tail.label else f"{head.label} → {tail.label}"

    topics: list[str] = []
    for p in parts:
        for t in p.topics:
            if t not in topics:
                topics.append(t)

    citation = head.citation
    if citation is not None:
        citation = citation.model_copy(update={"section": label})
    return Section(
        chunk_id=head.chunk_id,
        text="\n\n".join(p.text for p in parts),
        label=label,
        page=head.page,
        page_end=tail.page,
        topics=topics[:4],
        citation=citation,
    )


def _group(secs: list[Section]) -> list[Section]:
    """Batch short consecutive sections into steps worth answering a question on."""
    steps: list[Section] = []
    buf: list[Section] = []
    size = 0
    for s in secs:
        if buf and size + len(s.text) > MAX_STEP_CHARS:
            steps.append(_merge(buf))
            buf, size = [], 0
        buf.append(s)
        size += len(s.text)
        if size >= MIN_STEP_CHARS:
            steps.append(_merge(buf))
            buf, size = [], 0
    if buf:
        # a short tail joins the previous step rather than standing alone
        if steps and size < MIN_STEP_CHARS // 2:
            steps[-1] = _merge([steps[-1], *buf])
        else:
            steps.append(_merge(buf))
    return steps


# --------------------------------------------------------------- progress
def _key(filename: str) -> str:
    return f"{KV_PREFIX}{filename}"


def load_state(filename: str) -> dict[str, Any]:
    with session_scope() as s:
        raw = kv_get(s, _key(filename), None)
    state = raw if isinstance(raw, dict) else {}
    state.setdefault("scores", {})     # chunk_id -> last score out of 10
    state.setdefault("flagged", [])    # chunk_ids the student marked as hard
    return state


def _save_state(filename: str, state: dict[str, Any]) -> None:
    with session_scope() as s:
        kv_set(s, _key(filename), state)


def record(filename: str, chunk_id: str, score: float) -> None:
    """Store the mark for one section. Re-answering overwrites, so a section
    the student came back and fixed counts as fixed."""
    state = load_state(filename)
    state["scores"][chunk_id] = round(float(score), 1)
    if score >= 7 and chunk_id in state["flagged"]:
        state["flagged"].remove(chunk_id)
    _save_state(filename, state)


def toggle_flag(filename: str, chunk_id: str) -> bool:
    """Mark/unmark a section as one to come back to. Returns the new state."""
    state = load_state(filename)
    if chunk_id in state["flagged"]:
        state["flagged"].remove(chunk_id)
        flagged = False
    else:
        state["flagged"].append(chunk_id)
        flagged = True
    _save_state(filename, state)
    return flagged


def reset(filename: str) -> None:
    _save_state(filename, {"scores": {}, "flagged": []})


def overview(filename: str, secs: list[Section] | None = None) -> dict[str, Any]:
    """Where the student stands on this document."""
    secs = secs if secs is not None else sections(filename)
    state = load_state(filename)
    scores = state["scores"]
    done = [s for s in secs if s.chunk_id in scores]
    weak = [s for s in done if scores[s.chunk_id] < 6]
    marks = [scores[s.chunk_id] for s in done]
    return {
        "total": len(secs),
        "done": len(done),
        "remaining": len(secs) - len(done),
        "fraction": (len(done) / len(secs)) if secs else 0.0,
        "mean_score": (sum(marks) / len(marks)) if marks else 0.0,
        "weak_sections": weak,
        "flagged": [s for s in secs if s.chunk_id in state["flagged"]],
        "minutes_left": (len(secs) - len(done)) * MINUTES_PER_SECTION,
    }


def next_index(filename: str, secs: list[Section] | None = None) -> int:
    """The first section not yet answered - where 'Continue' should land."""
    secs = secs if secs is not None else sections(filename)
    scores = load_state(filename)["scores"]
    for s in secs:
        if s.chunk_id not in scores:
            return s.index
    return 0


# --------------------------------------------------------------- briefs
_DEFINITION = ("is a ", "is an ", "is the ", "are the ", "is defined", "refers to",
               "means that", "consists of", "denotes")
_CAUSAL = ("because", "therefore", "so that", "hence", "thus", "as a result",
           "which means", "leads to", "causes")
_CONTRAST = ("unlike", "whereas", "however", "in contrast", "rather than",
             "instead of", "but not")
_RULE = ("must", "cannot", "never", "always", "only if", "requires", "note that")


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if len(p.strip()) > 25]


def _looks_like_formula(s: str) -> bool:
    return bool(re.search(r"[=∑√×·≤≥∂]|\b\d+\s*[x×*/+-]\s*\d+", s))


def _sentence_score(s: str) -> int:
    low = s.lower()
    score = 0
    score += 3 * sum(1 for c in _DEFINITION if c in low)
    score += 3 * sum(1 for c in _CAUSAL if c in low)
    score += 2 * sum(1 for c in _CONTRAST if c in low)
    score += 2 * sum(1 for c in _RULE if c in low)
    if _looks_like_formula(s):
        score += 4
    return score


def key_sentences(text: str, limit: int = 5) -> list[str]:
    """The load-bearing sentences of a passage, in their original order.

    Used for the offline brief: it is the section's own words, never a
    paraphrase, so nothing can be invented on the way through.
    """
    sents = _sentences(text)
    if not sents:
        return []
    ranked = sorted(range(len(sents)), key=lambda i: (-_sentence_score(sents[i]), i))
    keep = sorted(ranked[:limit])
    picked = [sents[i] for i in keep if _sentence_score(sents[i]) > 0]
    return picked or sents[:min(limit, 2)]


def concepts_in(text: str, limit: int = 8) -> list[str]:
    """Curated technical terms the passage actually uses.

    Drawn from the topic registry rather than raw word frequency, so the
    offline evaluator grades against real terminology instead of whatever
    happened to be common in the slide.
    """
    low = text.lower()
    found: list[tuple[int, str]] = []
    seen: set[str] = set()
    for t in TOPIC_SEEDS:
        for kw in [str(t["name"])] + [str(k) for k in t.get("keywords", [])]:
            k = kw.lower()
            if len(k) <= 3 or k in seen:
                continue
            pos = low.find(k)
            if pos >= 0:
                seen.add(k)
                found.append((pos, kw))
    found.sort()
    return [kw for _, kw in found[:limit]]


_BRIEF_PROMPT = """One section of the student's own course material is below.

SECTION ({label}):
\"\"\"
{text}
\"\"\"

Write a brief that helps them hold this section in memory. Rules:
- State only what THIS section establishes. Add nothing it does not say.
- 3 to 5 points, each one sentence, mechanism before consequence.
- Then one line naming how this could be examined.
- Write in the language your system instruction requires, keeping technical
  terms in English. The section itself may be in a different language - that
  does not change the language you answer in.

Return JSON: {{"points": ["...", "..."], "exam_angle": "..."}}"""


def brief(section: Section, use_llm: bool = True) -> dict[str, Any]:
    """A short, honest account of what this section says.

    With an LLM: written from the section, in the configured language. Without
    one: the section's own key sentences, extracted verbatim.
    """
    llm = get_llm()
    if use_llm and llm.available:
        data, resp = llm.complete_json(
            _BRIEF_PROMPT.format(label=section.heading, text=section.text[:4000]),
            system=system_with_language(TUTOR_SYSTEM),
            max_tokens=700,
        )
        if isinstance(data, dict) and data.get("points"):
            points = [str(p).strip() for p in data["points"] if str(p).strip()]
            if points:
                return {
                    "points": points[:5],
                    "exam_angle": str(data.get("exam_angle", "")).strip(),
                    "source": "llm",
                    "verbatim": False,
                }
        log.info("brief fell back to extraction: %s", resp.error or "unusable JSON")

    return {
        "points": key_sentences(section.text),
        "exam_angle": "",
        "source": "extract",
        "verbatim": True,
    }


# --------------------------------------------------------------- questions
def _retrieval_for(section: Section) -> RetrievalResult:
    """The section itself, packaged as a retrieval result so the existing
    grounded-question machinery can be reused unchanged."""
    return RetrievalResult(
        query=section.heading,
        chunks=[RetrievedChunk(
            chunk_id=section.chunk_id,
            text=section.text,
            citation=section.citation or Citation(source_type="", source_name=""),
            score=1.0,
        )],
    )


def _recall_question(section: Section) -> Question:
    """Deterministic fallback: recall of this section, graded against the
    terminology it uses. Honest about what it can check - it verifies coverage
    and phrasing, not correctness."""
    concepts = concepts_in(section.text, limit=6)
    topic_id = section.topic_id
    subject = _topic_name(topic_id) if topic_id else section.label
    prompt = (
        f"Close the material. In your own words, state what this section "
        f"(\"{section.heading}\") establishes about {subject}: the mechanism "
        f"first, then why it matters."
    )
    return Question(
        id=f"walk:{section.chunk_id}",
        topic=topic_id,
        subtopic=section.label,
        category=_category_of(topic_id),
        question_type=QuestionType.CONCEPTUAL,
        difficulty=4,
        priority=_priority_of(topic_id),
        prompt=prompt,
        expected_concepts=concepts,
        expected_reasoning="The section's own claim, reproduced without reading it back.",
        citations=[section.citation] if section.citation else [],
        estimated_time=150,
        source_basis="walkthrough",
    )


def question_for(section: Section, use_llm: bool = True, difficulty: int = 4) -> Question:
    """One question that can only be answered by someone who read this section."""
    if use_llm and section.topic_id:
        try:
            q = question_from_material(
                section.topic_id, _retrieval_for(section), difficulty=difficulty
            )
        except Exception as exc:  # a generation failure must not stop the walk
            log.warning("walkthrough question generation failed: %s", exc)
            q = None
        if q is not None:
            q.subtopic = section.label
            return q
    return _recall_question(section)


# --------------------------------------------------------------- library view
def documents_with_progress(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach walkthrough progress to the document rows from `materials`."""
    out = []
    for d in docs:
        if d.get("status") != "indexed":
            continue
        secs = sections(d["filename"])
        if not secs:
            continue
        row = dict(d)
        row["progress"] = overview(d["filename"], secs)
        out.append(row)
    return out
