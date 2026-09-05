"""Vector store with two interchangeable backends.

* ``local``  - TF-IDF (char + word n-grams) cosine similarity, persisted as JSONL.
               Zero network, zero API cost, works offline. Default.
* ``chroma`` - ChromaDB, used when VECTOR_BACKEND=chroma and chromadb installs.

Both expose the same tiny interface, so switching is a config change.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Iterable, Protocol

import numpy as np

from ..config import get_logger, get_settings
from .ingest import Chunk

log = get_logger(__name__)


def _matches_filters(meta: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    for key, want in filters.items():
        have = meta.get(key)
        if want is None:
            continue
        if key == "topics":
            have_list = [t for t in str(have or "").split(",") if t]
            want_list = want if isinstance(want, (list, tuple, set)) else [want]
            if not set(have_list) & set(want_list):
                return False
        elif isinstance(want, (list, tuple, set)):
            if have not in want:
                return False
        else:
            if have != want:
                return False
    return True


class VectorStore(Protocol):
    def add(self, chunks: Iterable[Chunk]) -> int: ...
    def query(self, text: str, k: int = 6,
              filters: dict[str, Any] | None = None) -> list[tuple[Chunk, float]]: ...
    def count(self) -> int: ...
    def delete_by(self, **meta: Any) -> int: ...
    def reset(self) -> None: ...
    def stats(self) -> dict[str, Any]: ...


# --------------------------------------------------------------- local
class LocalVectorStore:
    """TF-IDF backed store. Rebuilds its index lazily after writes."""

    def __init__(self, path: Path | None = None) -> None:
        settings = get_settings()
        self.path = path or (settings.data_path / "vectorstore" / "chunks.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._docs: list[dict[str, Any]] = []
        self._matrix = None
        self._vec = None
        self._dirty = True
        self._load()

    # ---- persistence ----
    def _load(self) -> None:
        self._docs = []
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self._docs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        self._dirty = True
        log.info("local vector store loaded: %d chunks", len(self._docs))

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for d in self._docs:
                fh.write(json.dumps(d, ensure_ascii=False) + "\n")
        tmp.replace(self.path)

    # ---- index ----
    def _build(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        if not self._docs:
            self._vec, self._matrix = None, None
            self._dirty = False
            return
        texts = [d["text"] for d in self._docs]
        # On a small corpus a max_df below 1.0 deletes every term the documents
        # share - which is exactly the vocabulary a course-material search needs.
        max_df = 1.0 if len(texts) < 25 else 0.92
        self._vec = TfidfVectorizer(
            lowercase=True,
            sublinear_tf=True,
            ngram_range=(1, 2),
            min_df=1,
            max_df=max_df,
            strip_accents="unicode",
            stop_words="english",
        )
        try:
            self._matrix = self._vec.fit_transform(texts)
        except ValueError:
            # degenerate corpora (e.g. one tiny doc) - fall back to raw counts
            self._vec = TfidfVectorizer(lowercase=True, min_df=1)
            self._matrix = self._vec.fit_transform(texts)
        self._dirty = False
        log.info("tf-idf index built: %s", self._matrix.shape)

    # ---- api ----
    def add(self, chunks: Iterable[Chunk]) -> int:
        chunks = list(chunks)
        if not chunks:
            return 0
        with self._lock:
            known = {d["chunk_id"] for d in self._docs}
            added = 0
            for c in chunks:
                if c.chunk_id in known:
                    continue
                self._docs.append(
                    {"chunk_id": c.chunk_id, "text": c.text, "metadata": c.metadata}
                )
                known.add(c.chunk_id)
                added += 1
            if added:
                self._save()
                self._dirty = True
            return added

    def query(self, text: str, k: int = 6,
              filters: dict[str, Any] | None = None) -> list[tuple[Chunk, float]]:
        if not text.strip() or not self._docs:
            return []
        with self._lock:
            if self._dirty:
                self._build()
            if self._matrix is None or self._vec is None:
                return []
            try:
                qv = self._vec.transform([text])
            except Exception:
                return []
            sims = (self._matrix @ qv.T).toarray().ravel()

        idx = np.argsort(-sims)
        out: list[tuple[Chunk, float]] = []
        for i in idx:
            score = float(sims[i])
            if score <= 0:
                break
            d = self._docs[i]
            if not _matches_filters(d["metadata"], filters):
                continue
            out.append((Chunk(d["chunk_id"], d["text"], d["metadata"]), score))
            if len(out) >= k:
                break
        return out

    def count(self) -> int:
        return len(self._docs)

    def delete_by(self, **meta: Any) -> int:
        with self._lock:
            before = len(self._docs)
            self._docs = [
                d for d in self._docs
                if not all(d["metadata"].get(kk) == vv for kk, vv in meta.items())
            ]
            removed = before - len(self._docs)
            if removed:
                self._save()
                self._dirty = True
            return removed

    def reset(self) -> None:
        with self._lock:
            self._docs = []
            self._matrix = None
            self._vec = None
            self._dirty = True
            if self.path.exists():
                self.path.unlink()

    def stats(self) -> dict[str, Any]:
        by_source: dict[str, int] = {}
        by_file: dict[str, int] = {}
        for d in self._docs:
            st = d["metadata"].get("source_type", "?")
            by_source[st] = by_source.get(st, 0) + 1
            fn = d["metadata"].get("filename", "?")
            by_file[fn] = by_file.get(fn, 0) + 1
        return {
            "backend": "local (tf-idf)",
            "chunks": len(self._docs),
            "by_source_type": by_source,
            "by_file": by_file,
        }


# --------------------------------------------------------------- chroma
class ChromaVectorStore:
    """ChromaDB backend (optional dependency)."""

    def __init__(self, path: Path | None = None) -> None:
        import chromadb  # noqa: F401  (import error handled by the factory)
        from chromadb.config import Settings as ChromaSettings

        settings = get_settings()
        self.dir = path or (settings.data_path / "chroma")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(self.dir), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name="examagent", metadata={"hnsw:space": "cosine"}
        )

    @staticmethod
    def _flatten(meta: dict[str, Any]) -> dict[str, Any]:
        # chroma only accepts scalar metadata values
        return {k: ("" if v is None else v if isinstance(v, (str, int, float, bool)) else str(v))
                for k, v in meta.items()}

    def add(self, chunks: Iterable[Chunk]) -> int:
        chunks = list(chunks)
        if not chunks:
            return 0
        existing = set()
        try:
            got = self.collection.get(ids=[c.chunk_id for c in chunks])
            existing = set(got.get("ids", []))
        except Exception:
            pass
        new = [c for c in chunks if c.chunk_id not in existing]
        if not new:
            return 0
        self.collection.add(
            ids=[c.chunk_id for c in new],
            documents=[c.text for c in new],
            metadatas=[self._flatten(c.metadata) for c in new],
        )
        return len(new)

    def query(self, text: str, k: int = 6,
              filters: dict[str, Any] | None = None) -> list[tuple[Chunk, float]]:
        if not text.strip() or self.count() == 0:
            return []
        where = None
        if filters:
            clauses = []
            for key, want in filters.items():
                if key == "topics" or want is None:
                    continue
                if isinstance(want, (list, tuple, set)):
                    clauses.append({key: {"$in": list(want)}})
                else:
                    clauses.append({key: want})
            if len(clauses) == 1:
                where = clauses[0]
            elif clauses:
                where = {"$and": clauses}
        res = self.collection.query(
            query_texts=[text], n_results=min(k * 3, max(self.count(), 1)), where=where
        )
        out: list[tuple[Chunk, float]] = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            if filters and "topics" in filters and not _matches_filters(meta, {"topics": filters["topics"]}):
                continue
            out.append((Chunk(cid, doc, dict(meta)), max(0.0, 1.0 - float(dist))))
            if len(out) >= k:
                break
        return out

    def count(self) -> int:
        try:
            return int(self.collection.count())
        except Exception:
            return 0

    def delete_by(self, **meta: Any) -> int:
        before = self.count()
        try:
            self.collection.delete(where=self._flatten(meta))
        except Exception as exc:
            log.warning("chroma delete failed: %s", exc)
        return before - self.count()

    def reset(self) -> None:
        try:
            self.client.delete_collection("examagent")
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name="examagent", metadata={"hnsw:space": "cosine"}
        )

    def stats(self) -> dict[str, Any]:
        by_source: dict[str, int] = {}
        by_file: dict[str, int] = {}
        try:
            got = self.collection.get(include=["metadatas"])
            for meta in got.get("metadatas", []) or []:
                st = (meta or {}).get("source_type", "?")
                by_source[st] = by_source.get(st, 0) + 1
                fn = (meta or {}).get("filename", "?")
                by_file[fn] = by_file.get(fn, 0) + 1
        except Exception:
            pass
        return {
            "backend": "chromadb",
            "chunks": self.count(),
            "by_source_type": by_source,
            "by_file": by_file,
        }


# --------------------------------------------------------------- factory
_store: VectorStore | None = None


def get_vector_store(force: bool = False) -> VectorStore:
    global _store
    if _store is not None and not force:
        return _store
    settings = get_settings()
    if settings.vector_backend == "chroma":
        try:
            _store = ChromaVectorStore()
            log.info("using ChromaDB vector backend")
            return _store
        except Exception as exc:
            log.warning("chromadb unavailable (%s); falling back to local TF-IDF", exc)
    _store = LocalVectorStore()
    return _store


def reset_vector_store() -> None:
    global _store
    _store = None
