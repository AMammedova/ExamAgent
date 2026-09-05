"""Pydantic domain models shared across services and UI."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------- enums
class SourceType(str, Enum):
    UNIVERSITY_ML = "UNIVERSITY_ML"
    UNIVERSITY_DL = "UNIVERSITY_DL"
    UDEMY_ML = "UDEMY_ML"
    UDEMY_DL = "UDEMY_DL"
    EXAM_SAMPLES = "EXAM_SAMPLES"
    STUDENT_NOTES = "STUDENT_NOTES"


class Priority(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

    @property
    def weight(self) -> float:
        return {"CRITICAL": 1.0, "HIGH": 0.65, "MEDIUM": 0.35, "LOW": 0.15}[self.value]


class Category(str, Enum):
    ML = "Machine Learning"
    DL = "Deep Learning"


class QuestionType(str, Enum):
    MCQ = "multiple_choice"
    ASSERTION_REASON = "assertion_reason"
    SHORT_ANSWER = "short_answer"
    CALCULATION = "calculation"
    CONCEPTUAL = "conceptual_reasoning"
    COMPARISON = "comparison"
    SCENARIO = "scenario"
    DIAGRAM = "diagram_interpretation"
    WHAT_IF = "what_happens_if"
    GRAPH = "graph_interpretation"


#: which knowledge dimension a question type primarily measures
DIMENSION_OF_TYPE: dict[QuestionType, str] = {
    QuestionType.MCQ: "concept",
    QuestionType.ASSERTION_REASON: "reasoning",
    QuestionType.SHORT_ANSWER: "concept",
    QuestionType.CALCULATION: "calculation",
    QuestionType.CONCEPTUAL: "reasoning",
    QuestionType.COMPARISON: "comparison",
    QuestionType.SCENARIO: "application",
    QuestionType.DIAGRAM: "application",
    QuestionType.WHAT_IF: "reasoning",
    QuestionType.GRAPH: "application",
}

DIMENSIONS = ("concept", "calculation", "reasoning", "comparison", "application")


class Mastery(str, Enum):
    MASTERED = "Mastered"
    STRONG = "Strong"
    MEDIUM = "Medium"
    WEAK = "Weak"
    CRITICAL_WEAKNESS = "Critical weakness"

    @classmethod
    def from_score(cls, score01: float) -> "Mastery":
        if score01 >= 0.88:
            return cls.MASTERED
        if score01 >= 0.72:
            return cls.STRONG
        if score01 >= 0.55:
            return cls.MEDIUM
        if score01 >= 0.35:
            return cls.WEAK
        return cls.CRITICAL_WEAKNESS

    @property
    def color(self) -> str:
        return {
            "Mastered": "#1a7f37",
            "Strong": "#2da44e",
            "Medium": "#d4a72c",
            "Weak": "#e8804a",
            "Critical weakness": "#cf222e",
        }[self.value]


class MistakeType(str, Enum):
    ARITHMETIC = "Arithmetic"
    CONCEPTUAL = "Conceptual"
    FORMULA = "Formula"
    DIMENSION = "Dimension"
    REASONING = "Reasoning"
    TERMINOLOGY = "Terminology"
    INCOMPLETE = "Incomplete"
    NONE = "None"


class SessionMode(str, Enum):
    QUICK = "Quick Study"
    THIRTY = "30 Minute Study"
    SIXTY = "60 Minute Study"
    DEEP = "Deep Study"
    RAPID = "Rapid Revision"
    REPAIR = "Weakness Repair"
    EXAM_SIM = "Exam Simulation"

    @property
    def minutes(self) -> int:
        return {
            "Quick Study": 15,
            "30 Minute Study": 30,
            "60 Minute Study": 60,
            "Deep Study": 90,
            "Rapid Revision": 20,
            "Weakness Repair": 45,
            "Exam Simulation": 75,
        }[self.value]


# ---------------------------------------------------------------- RAG
class Citation(BaseModel):
    source_type: str
    source_name: str
    lecture: str | None = None
    topic: str | None = None
    page: int | None = None
    section: str | None = None
    score: float = 0.0

    def label(self) -> str:
        bits = [self.source_type.replace("_", " ").title(), self.source_name]
        if self.lecture:
            bits.append(self.lecture)
        if self.section:
            bits.append(self.section)
        if self.page is not None:
            bits.append("p." + str(self.page))
        return " - ".join(b for b in bits if b)


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    citation: Citation
    score: float = 0.0


class RetrievalResult(BaseModel):
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    query: str = ""

    @property
    def grounded(self) -> bool:
        """True when retrieval found enough material to base a claim on."""
        return bool(self.chunks) and max((c.score for c in self.chunks), default=0.0) >= 0.12

    def context_block(self, max_chars: int = 6000) -> str:
        out: list[str] = []
        used = 0
        for i, c in enumerate(self.chunks, 1):
            piece = "[S{}] ({})\n{}\n".format(i, c.citation.label(), c.text.strip())
            if used + len(piece) > max_chars:
                break
            out.append(piece)
            used += len(piece)
        return "\n".join(out)

    def citations(self) -> list[Citation]:
        return [c.citation for c in self.chunks]


# ---------------------------------------------------------------- questions
class AnswerOption(BaseModel):
    key: str
    text: str


class Question(BaseModel):
    id: str
    topic: str
    subtopic: str | None = None
    category: Category = Category.ML
    question_type: QuestionType
    difficulty: int = Field(ge=1, le=6, default=4)
    priority: Priority = Priority.HIGH
    prompt: str
    options: list[AnswerOption] = Field(default_factory=list)
    correct_option: str | None = None
    model_answer: str = ""
    expected_concepts: list[str] = Field(default_factory=list)
    expected_reasoning: str = ""
    source_basis: str = "seed"
    citations: list[Citation] = Field(default_factory=list)
    estimated_time: int = 120  # seconds
    # assertion-reason internals (never shown before evaluation)
    assertion: str | None = None
    reason: str | None = None
    assertion_truth: bool | None = None
    reason_truth: bool | None = None
    reason_explains_assertion: bool | None = None
    # calculation internals
    calc_spec: dict[str, Any] | None = None

    @property
    def dimension(self) -> str:
        return DIMENSION_OF_TYPE.get(self.question_type, "concept")


class SubScore(BaseModel):
    label: str
    student: str = ""
    expected: str = ""
    correct: bool = False
    note: str = ""


class Evaluation(BaseModel):
    score: float = Field(ge=0, le=10, default=0)
    correct: bool = False
    partial: bool = False
    got_right: list[str] = Field(default_factory=list)
    missed: list[str] = Field(default_factory=list)
    incorrect: list[str] = Field(default_factory=list)
    examiner_expects: str = ""
    model_answer: str = ""
    improvement: str = ""
    mistake_type: MistakeType = MistakeType.NONE
    severity: Literal["Low", "Medium", "High"] = "Medium"
    sub_scores: list[SubScore] = Field(default_factory=list)
    evaluator: str = "heuristic"  # heuristic | llm | deterministic

    @property
    def score01(self) -> float:
        return self.score / 10.0


class TopicScores(BaseModel):
    concept: float = 0.0
    calculation: float = 0.0
    reasoning: float = 0.0
    comparison: float = 0.0
    application: float = 0.0
    confidence: float = 0.0


class PlanBlock(BaseModel):
    topic: str
    minutes: int
    focus: str  # concept | calculation | reasoning | mixed | revision
    reason: str = ""
    priority: Priority = Priority.HIGH


class DayPlan(BaseModel):
    day_number: int
    date: str
    theme: str
    blocks: list[PlanBlock] = Field(default_factory=list)
    mock_exam: bool = False

    @property
    def total_minutes(self) -> int:
        return sum(b.minutes for b in self.blocks)


class ReadinessBreakdown(BaseModel):
    overall: float = 0.0
    critical_mastery: float = 0.0
    calculation: float = 0.0
    reasoning: float = 0.0
    exam_performance: float = 0.0
    coverage: float = 0.0
    confidence: float = 0.0
    ml_score: float = 0.0
    dl_score: float = 0.0
    weights: dict[str, float] = Field(default_factory=dict)


class MockExamReport(BaseModel):
    exam_id: int
    total_score: float = 0.0
    max_score: float = 0.0
    percentage: float = 0.0
    ml_score: float = 0.0
    dl_score: float = 0.0
    by_dimension: dict[str, float] = Field(default_factory=dict)
    by_question_type: dict[str, float] = Field(default_factory=dict)
    top_weaknesses: list[str] = Field(default_factory=list)
    top_strengths: list[str] = Field(default_factory=list)
    dangerous_gaps: list[str] = Field(default_factory=list)
    immediate_revision: list[str] = Field(default_factory=list)
    revision_plan: list[str] = Field(default_factory=list)
    duration_seconds: int = 0
    finished_at: datetime | None = None
