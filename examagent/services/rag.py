"""Retrieval layer.

Two rules the rest of the app depends on:

1. **University material outranks Udemy material** when the question is about
   what the course expects. Udemy is used for intuition; exam samples are used
   for question style.
2. If retrieval is not grounded, callers must say so rather than invent a
   course-specific fact. ``RetrievalResult.grounded`` is that signal.
"""
from __future__ import annotations

from typing import Any, Iterable

from ..config import get_logger, get_settings
from ..models.schemas import Citation, RetrievalResult, RetrievedChunk, SourceType
from .vectorstore import get_vector_store

log = get_logger(__name__)

#: multiplicative trust applied to a chunk's similarity score
SOURCE_WEIGHTS: dict[str, float] = {
    SourceType.UNIVERSITY_ML.value: 1.30,
    SourceType.UNIVERSITY_DL.value: 1.30,
    SourceType.EXAM_SAMPLES.value: 1.25,
    SourceType.STUDENT_NOTES.value: 1.05,
    SourceType.UDEMY_ML.value: 0.95,
    SourceType.UDEMY_DL.value: 0.95,
}

UNIVERSITY_SOURCES = [SourceType.UNIVERSITY_ML.value, SourceType.UNIVERSITY_DL.value]
UDEMY_SOURCES = [SourceType.UDEMY_ML.value, SourceType.UDEMY_DL.value]
ML_SOURCES = [SourceType.UNIVERSITY_ML.value, SourceType.UDEMY_ML.value]
DL_SOURCES = [SourceType.UNIVERSITY_DL.value, SourceType.UDEMY_DL.value]


def _to_citation(meta: dict[str, Any], score: float) -> Citation:
    page = meta.get("page")
    try:
        page = int(page) if page not in (None, "", "None") else None
    except (TypeError, ValueError):
        page = None
    return Citation(
        source_type=str(meta.get("source_type", "UNKNOWN")),
        source_name=str(meta.get("source_name") or meta.get("filename") or "unknown"),
        lecture=str(meta.get("lecture") or "") or None,
        topic=str(meta.get("topics") or "") or None,
        page=page,
        section=str(meta.get("section") or "") or None,
        score=round(score, 4),
    )


def retrieve(
    query: str,
    k: int | None = None,
    source_types: Iterable[str] | None = None,
    topics: Iterable[str] | None = None,
    prefer_university: bool = True,
    fallback_to_all: bool = False,
) -> RetrievalResult:
    """Semantic search over the ingested course material.

    A source filter is a hard constraint by default: returning material from a
    source the caller excluded would make citations misleading. Callers that
    genuinely want a broader second pass must ask for it with `fallback_to_all`.
    """
    settings = get_settings()
    k = k or settings.retrieval_top_k
    store = get_vector_store()
    if store.count() == 0:
        return RetrievalResult(chunks=[], query=query)

    filters: dict[str, Any] = {}
    if source_types:
        filters["source_type"] = list(source_types)
    if topics:
        filters["topics"] = list(topics)

    # over-fetch, then re-rank by source trust
    hits = store.query(query, k=k * 3, filters=filters or None)
    if not hits and filters and fallback_to_all:
        hits = store.query(query, k=k * 3, filters=None)

    scored: list[tuple[float, RetrievedChunk]] = []
    for chunk, score in hits:
        st = str(chunk.metadata.get("source_type", ""))
        weight = SOURCE_WEIGHTS.get(st, 1.0) if prefer_university else 1.0
        final = score * weight
        scored.append((
            final,
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                citation=_to_citation(chunk.metadata, final),
                score=round(final, 4),
            ),
        ))
    scored.sort(key=lambda t: -t[0])
    return RetrievalResult(chunks=[c for _, c in scored[:k]], query=query)


def retrieve_for_topic(topic_name: str, topic_id: str = "", k: int | None = None,
                       category: str | None = None) -> RetrievalResult:
    """Retrieve material for a topic, biased toward the matching course half."""
    sources = None
    if category == "Machine Learning":
        sources = ML_SOURCES + [SourceType.STUDENT_NOTES.value, SourceType.EXAM_SAMPLES.value]
    elif category == "Deep Learning":
        sources = DL_SOURCES + [SourceType.STUDENT_NOTES.value, SourceType.EXAM_SAMPLES.value]
    res = retrieve(topic_name, k=k, source_types=sources)
    if not res.chunks and sources:
        res = retrieve(topic_name, k=k)
    return res


def retrieve_exam_style(topic_name: str = "", k: int = 4) -> RetrievalResult:
    """Pull previous-exam material to imitate question style and difficulty."""
    q = f"{topic_name} exam question".strip()
    return retrieve(q, k=k, source_types=[SourceType.EXAM_SAMPLES.value])


def compare_sources(query: str, k: int = 3) -> dict[str, RetrievalResult]:
    """University vs Udemy views of the same query, for conflict detection."""
    return {
        "university": retrieve(query, k=k, source_types=UNIVERSITY_SOURCES, prefer_university=False),
        "udemy": retrieve(query, k=k, source_types=UDEMY_SOURCES, prefer_university=False),
    }


def coverage_by_topic() -> dict[str, int]:
    """How many stored chunks mention each topic id (drives 'missing material')."""
    store = get_vector_store()
    stats = store.stats()
    out: dict[str, int] = {}
    docs = getattr(store, "_docs", None)
    if docs is None:  # chroma
        try:
            got = store.collection.get(include=["metadatas"])  # type: ignore[attr-defined]
            metas = got.get("metadatas", []) or []
        except Exception:
            metas = []
    else:
        metas = [d["metadata"] for d in docs]
    for meta in metas:
        for t in str((meta or {}).get("topics", "")).split(","):
            t = t.strip()
            if t:
                out[t] = out.get(t, 0) + 1
    log.debug("coverage computed over %s chunks", stats.get("chunks"))
    return out


def knowledge_base_summary() -> dict[str, Any]:
    store = get_vector_store()
    stats = store.stats()
    cov = coverage_by_topic()
    stats["topics_with_material"] = len(cov)
    stats["coverage"] = cov
    return stats
