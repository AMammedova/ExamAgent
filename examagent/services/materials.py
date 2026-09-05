"""Material ingestion pipeline: file -> chunks -> vector store -> topic coverage."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, BinaryIO

from ..config import get_logger, get_settings
from ..models.db import Document, all_topics, session_scope
from ..models.schemas import SourceType
from .ingest import SUPPORTED, file_hash, prepare_document
from .rag import coverage_by_topic
from .vectorstore import get_vector_store

log = get_logger(__name__)


def save_upload(data: BinaryIO | bytes, filename: str) -> Path:
    settings = get_settings()
    dest = settings.upload_path / filename
    counter = 1
    while dest.exists():
        dest = settings.upload_path / f"{Path(filename).stem}_{counter}{Path(filename).suffix}"
        counter += 1
    if isinstance(data, bytes):
        dest.write_bytes(data)
    else:
        with dest.open("wb") as fh:
            shutil.copyfileobj(data, fh)
    return dest


def ingest_file(
    path: Path,
    source_type: str,
    source_name: str | None = None,
    lecture: str = "",
) -> dict[str, Any]:
    """Ingest one file. Duplicate content (by hash) is skipped, not re-indexed."""
    path = Path(path)
    result: dict[str, Any] = {
        "filename": path.name,
        "source_type": source_type,
        "status": "pending",
        "chunks": 0,
        "error": "",
    }
    if path.suffix.lower() not in SUPPORTED:
        result.update(status="unsupported",
                      error=f"unsupported file type {path.suffix}")
        return result

    try:
        digest = file_hash(path)
    except OSError as exc:
        result.update(status="failed", error=str(exc))
        return result

    with session_scope() as s:
        existing = s.query(Document).filter(Document.content_hash == digest).first()
        if existing is not None:
            result.update(status="duplicate", chunks=existing.n_chunks,
                          error=f"identical content already ingested as '{existing.filename}'")
            return result

    try:
        chunks, summary = prepare_document(path, source_type, source_name, lecture)
    except Exception as exc:
        log.exception("ingestion failed for %s", path.name)
        with session_scope() as s:
            s.add(Document(filename=path.name, source_type=source_type,
                           source_name=source_name or path.stem, lecture=lecture,
                           content_hash=digest, status="failed", error=str(exc)[:500]))
        result.update(status="failed", error=str(exc))
        return result

    if not chunks:
        note = ("no extractable text - if this is a scanned PDF or an image, install an OCR "
                "engine (pytesseract + Pillow) or supply a text version")
        with session_scope() as s:
            s.add(Document(filename=path.name, source_type=source_type,
                           source_name=source_name or path.stem, lecture=lecture,
                           content_hash=digest, status="empty", error=note,
                           n_chars=summary.get("n_chars", 0)))
        result.update(status="empty", error=note)
        return result

    store = get_vector_store()
    added = store.add(chunks)

    with session_scope() as s:
        s.add(Document(
            filename=path.name,
            source_type=source_type,
            source_name=source_name or path.stem,
            lecture=lecture,
            content_hash=digest,
            n_chunks=len(chunks),
            n_chars=summary["n_chars"],
            status="indexed",
            detected_topics=json.dumps(summary["topics"]),
        ))
        _mark_topics_with_material(s)

    result.update(status="indexed", chunks=added, topics=summary["topics"],
                  n_chars=summary["n_chars"])
    log.info("ingested %s (%s): %d chunks", path.name, source_type, added)
    return result


def ingest_uploads(
    uploads: list[tuple[str, bytes]],
    source_type: str,
    lecture: str = "",
) -> list[dict[str, Any]]:
    out = []
    for name, data in uploads:
        path = save_upload(data, name)
        out.append(ingest_file(path, source_type, source_name=Path(name).stem, lecture=lecture))
    return out


def _mark_topics_with_material(session) -> None:
    coverage = coverage_by_topic()
    for t in all_topics(session):
        t.has_material = coverage.get(t.id, 0) > 0


def documents() -> list[dict[str, Any]]:
    with session_scope() as s:
        rows = s.query(Document).order_by(Document.created_at.desc()).all()
        return [
            {
                "id": d.id,
                "filename": d.filename,
                "source_type": d.source_type,
                "source_name": d.source_name,
                "lecture": d.lecture,
                "chunks": d.n_chunks,
                "chars": d.n_chars,
                "status": d.status,
                "error": d.error,
                "topics": json.loads(d.detected_topics or "[]"),
                "date": d.created_at.strftime("%d %b %H:%M"),
            }
            for d in rows
        ]


def delete_document(doc_id: int) -> bool:
    with session_scope() as s:
        d = s.get(Document, doc_id)
        if d is None:
            return False
        get_vector_store().delete_by(filename=d.filename)
        s.delete(d)
        _mark_topics_with_material(s)
    return True


def library_status() -> dict[str, Any]:
    """What the knowledge base contains and, more usefully, what it is missing."""
    from .rag import knowledge_base_summary

    kb = knowledge_base_summary()
    coverage = kb.get("coverage", {})
    with session_scope() as s:
        topics = all_topics(s)
        missing_critical = [
            {"id": t.id, "name": t.name, "category": t.category}
            for t in topics
            if t.priority == "CRITICAL" and coverage.get(t.id, 0) == 0
        ]
        covered = sum(1 for t in topics if coverage.get(t.id, 0) > 0)
        by_cat: dict[str, dict[str, int]] = {}
        for t in topics:
            c = by_cat.setdefault(t.category, {"total": 0, "covered": 0})
            c["total"] += 1
            if coverage.get(t.id, 0) > 0:
                c["covered"] += 1

    present = set(kb.get("by_source_type", {}))
    all_sources = [st.value for st in SourceType]
    return {
        "backend": kb.get("backend"),
        "chunks": kb.get("chunks", 0),
        "by_source_type": kb.get("by_source_type", {}),
        "by_file": kb.get("by_file", {}),
        "topics_with_material": covered,
        "total_topics": len(coverage) and len(topics) or len(topics),
        "missing_critical": missing_critical,
        "missing_sources": [s for s in all_sources if s not in present],
        "by_category": by_cat,
    }


def reset_knowledge_base() -> None:
    get_vector_store().reset()
    with session_scope() as s:
        s.query(Document).delete()
        for t in all_topics(s):
            t.has_material = False
