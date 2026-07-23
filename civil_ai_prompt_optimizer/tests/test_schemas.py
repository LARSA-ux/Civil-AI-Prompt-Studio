"""Tests for strict Pydantic schema validation."""

import pytest
from pydantic import ValidationError

from schemas import AnswerEvaluation, EvaluationVerdict, PromptAnalysis, TaskType


def test_prompt_analysis_accepts_supported_task_type() -> None:
    analysis = PromptAnalysis(
        intent="Explain a structural-engineering concept",
        audience="First-year engineering student",
        task_type=TaskType.EXPLANATION,
        missing_information=[],
        assumptions=["A conceptual explanation is requested."],
        risk_flags=[],
        needs_clarification=False,
        clarification_questions=[],
        improved_prompt="Explain storey drift conceptually for a first-year student.",
    )

    assert analysis.task_type is TaskType.EXPLANATION
    assert analysis.needs_clarification is False


def test_answer_evaluation_rejects_score_above_five() -> None:
    with pytest.raises(ValidationError):
        AnswerEvaluation(
            accuracy_score=6,
            relevance_score=5,
            completeness_score=5,
            clarity_score=5,
            terminology_score=5,
            grounding_score=5,
            verdict=EvaluationVerdict.PASS,
            unsupported_claims=[],
            technical_issues=[],
            missing_assumptions=[],
            missing_caveats=[],
            revision_instructions=[],
        )


def test_answer_evaluation_accepts_insufficient_context_verdict() -> None:
    evaluation = AnswerEvaluation(
        accuracy_score=3,
        relevance_score=4,
        completeness_score=2,
        clarity_score=4,
        terminology_score=4,
        grounding_score=2,
        verdict=EvaluationVerdict.INSUFFICIENT_CONTEXT,
        unsupported_claims=[],
        technical_issues=["Required project data are absent."],
        missing_assumptions=["Governing code"],
        missing_caveats=[],
        revision_instructions=["Request the governing code and project data."],
    )

    assert evaluation.verdict.value == "insufficient_context"
