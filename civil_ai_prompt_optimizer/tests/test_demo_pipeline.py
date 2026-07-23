"""Tests for the free deterministic demonstration pipeline."""

from demo_pipeline import DemoCivilAIPipeline
from schemas import EvaluationVerdict


def test_demo_mode_runs_without_api_key() -> None:
    pipeline = DemoCivilAIPipeline()

    result, was_truncated = pipeline.run(
        original_prompt="Explain storey drift.",
        domain="Civil / Structural Engineering",
        selected_audience="First-year engineering student",
        output_style="Clear explanation",
        strict_mode=False,
        automatic_revision=True,
        trusted_context="",
    )

    assert was_truncated is False
    assert result.stopped_for_clarification is False
    assert "relative horizontal displacement" in (result.draft_answer or "")
    assert result.evaluation is not None
    assert result.evaluation.verdict is EvaluationVerdict.REVISE
    assert "Demo Mode verification limitation" in (result.revised_answer or "")


def test_demo_strict_mode_stops_unsafe_assessment() -> None:
    pipeline = DemoCivilAIPipeline()

    result, _ = pipeline.run(
        original_prompt="Is this building safe under seismic loads?",
        domain="Civil / Structural Engineering",
        selected_audience="Infer automatically",
        output_style="Technical report",
        strict_mode=True,
        automatic_revision=True,
        trusted_context="",
    )

    assert result.stopped_for_clarification is True
    assert result.draft_answer is None
    assert "Which design code and jurisdiction apply?" in result.analysis.clarification_questions


def test_demo_table_analysis_reports_insufficient_context() -> None:
    pipeline = DemoCivilAIPipeline()

    result, _ = pipeline.run(
        original_prompt="Analyse this storey-drift table.",
        domain="Civil / Structural Engineering",
        selected_audience="M.Tech structural-engineering student",
        output_style="Technical report",
        strict_mode=False,
        automatic_revision=True,
        trusted_context="",
    )

    assert result.evaluation is not None
    assert result.evaluation.verdict is EvaluationVerdict.INSUFFICIENT_CONTEXT
    assert result.revised_answer is None
