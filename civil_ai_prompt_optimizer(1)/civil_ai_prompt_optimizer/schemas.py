"""Pydantic models used by the prompt-optimisation pipeline."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class TaskType(str, Enum):
    """Supported prompt intents for structured analysis."""

    EXPLANATION = "explanation"
    EXAM_ANSWER = "exam_answer"
    VIVA_ANSWER = "viva_answer"
    TECHNICAL_REPORT = "technical_report"
    NUMERICAL_SOLUTION = "numerical_solution"
    COMPARISON = "comparison"
    REVISION_NOTES = "revision_notes"
    RESEARCH_SUMMARY = "research_summary"
    OTHER = "other"


class EvaluationVerdict(str, Enum):
    """Possible evaluator decisions."""

    PASS = "pass"
    REVISE = "revise"
    INSUFFICIENT_CONTEXT = "insufficient_context"


class PromptAnalysis(BaseModel):
    """Structured output returned by the prompt-analysis stage."""

    model_config = ConfigDict(extra="forbid")

    intent: str = Field(min_length=1, description="Likely purpose of the user's question.")
    audience: str = Field(min_length=1, description="Detected or selected audience.")
    task_type: TaskType
    missing_information: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    needs_clarification: bool
    clarification_questions: list[str] = Field(default_factory=list)
    improved_prompt: str = Field(min_length=1)


class AnswerEvaluation(BaseModel):
    """Adversarial technical review of a generated answer."""

    model_config = ConfigDict(extra="forbid")

    accuracy_score: int = Field(ge=1, le=5)
    relevance_score: int = Field(ge=1, le=5)
    completeness_score: int = Field(ge=1, le=5)
    clarity_score: int = Field(ge=1, le=5)
    terminology_score: int = Field(ge=1, le=5)
    grounding_score: int = Field(ge=1, le=5)
    verdict: EvaluationVerdict
    unsupported_claims: list[str] = Field(default_factory=list)
    technical_issues: list[str] = Field(default_factory=list)
    missing_assumptions: list[str] = Field(default_factory=list)
    missing_caveats: list[str] = Field(default_factory=list)
    revision_instructions: list[str] = Field(default_factory=list)


class PipelineResult(BaseModel):
    """Complete result displayed by the Streamlit interface."""

    model_config = ConfigDict(extra="forbid")

    original_prompt: str
    analysis: PromptAnalysis
    draft_answer: str | None = None
    evaluation: AnswerEvaluation | None = None
    revised_answer: str | None = None
    stopped_for_clarification: bool = False
