"""Central configuration. Everything is env-driven; nothing is hardcoded."""
from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM ---
    llm_provider: str = Field(default="anthropic", alias="LLM_PROVIDER")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    model_name: str = Field(default="claude-sonnet-4-5", alias="MODEL_NAME")
    max_tokens: int = Field(default=2000, alias="MAX_TOKENS")
    temperature: float = Field(default=0.3, alias="TEMPERATURE")

    # --- retrieval ---
    embedding_backend: str = Field(default="local", alias="EMBEDDING_BACKEND")
    vector_backend: str = Field(default="local", alias="VECTOR_BACKEND")
    chunk_size: int = Field(default=1100, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=180, alias="CHUNK_OVERLAP")
    retrieval_top_k: int = Field(default=6, alias="RETRIEVAL_TOP_K")

    # --- schedule ---
    exam_date: str = Field(default="", alias="EXAM_DATE")
    study_days: int = Field(default=7, alias="STUDY_DAYS")

    # --- readiness weights ---
    w_critical: float = Field(default=0.30, alias="READINESS_W_CRITICAL")
    w_calculation: float = Field(default=0.20, alias="READINESS_W_CALCULATION")
    w_reasoning: float = Field(default=0.20, alias="READINESS_W_REASONING")
    w_exam: float = Field(default=0.15, alias="READINESS_W_EXAM")
    w_coverage: float = Field(default=0.10, alias="READINESS_W_COVERAGE")
    w_confidence: float = Field(default=0.05, alias="READINESS_W_CONFIDENCE")

    # --- language ---
    #: en | az - language of LLM-generated lessons, explanations and feedback.
    #: Deterministic offline content (calc engine, assertion-reason bank, seed
    #: questions) stays in English regardless, since it is fixed pedagogical text.
    language: str = Field(default="en", alias="LANGUAGE")

    # --- storage ---
    data_dir: str = Field(default="data", alias="DATA_DIR")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @field_validator("llm_provider", "embedding_backend", "vector_backend", "language")
    @classmethod
    def _lower(cls, v: str) -> str:
        return (v or "").strip().lower()

    # ---------- derived ----------
    @property
    def data_path(self) -> Path:
        p = Path(self.data_dir)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def upload_path(self) -> Path:
        p = self.data_path / "uploads"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def db_url(self) -> str:
        return f"sqlite:///{(self.data_path / 'examagent.db').as_posix()}"

    @property
    def api_key(self) -> str:
        if self.llm_provider == "anthropic":
            return self.anthropic_api_key.strip()
        if self.llm_provider == "openai":
            return self.openai_api_key.strip()
        return ""

    @property
    def llm_available(self) -> bool:
        return bool(self.api_key) and self.llm_provider in ("anthropic", "openai")

    @property
    def exam_day(self) -> date:
        """Exam date; defaults to today + STUDY_DAYS when unset."""
        if self.exam_date:
            try:
                return datetime.strptime(self.exam_date.strip(), "%Y-%m-%d").date()
            except ValueError:
                pass
        return date.today() + timedelta(days=self.study_days)

    def days_remaining(self, today: date | None = None) -> int:
        return max(0, (self.exam_day - (today or date.today())).days)

    def readiness_weights(self) -> dict[str, float]:
        w = {
            "critical": self.w_critical,
            "calculation": self.w_calculation,
            "reasoning": self.w_reasoning,
            "exam": self.w_exam,
            "coverage": self.w_coverage,
            "confidence": self.w_confidence,
        }
        total = sum(w.values()) or 1.0
        return {k: v / total for k, v in w.items()}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reload_settings() -> Settings:
    """Re-read .env after the Settings page writes to it."""
    get_settings.cache_clear()
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    return get_settings()


_LOG_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _LOG_CONFIGURED
    if not _LOG_CONFIGURED:
        logging.basicConfig(
            level=os.getenv("LOG_LEVEL", "INFO").upper(),
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        )
        _LOG_CONFIGURED = True
    return logging.getLogger(name)
