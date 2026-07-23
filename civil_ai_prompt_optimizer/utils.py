"""Validation, formatting, and deterministic safety helpers."""

from __future__ import annotations

from collections.abc import Iterable

from schemas import PromptAnalysis


HIGH_RISK_TERMS = {
    "safe",
    "safety",
    "capacity",
    "adequate",
    "compliance",
    "design",
    "load",
    "seismic",
    "earthquake",
    "foundation",
    "slope stability",
    "retaining wall",
    "punching shear",
    "bearing capacity",
    "reinforcement",
    "column",
    "beam",
    "slab",
}


def validate_user_prompt(prompt: str) -> str:
    """Return a trimmed prompt or raise a user-friendly validation error."""

    cleaned = prompt.strip()
    if not cleaned:
        raise ValueError("Enter a civil-engineering question before running the app.")
    if len(cleaned) < 3:
        raise ValueError("The question is too short to analyse reliably.")
    return cleaned


def validate_model_name(model: str) -> str:
    """Validate a user-configurable model identifier."""

    cleaned = model.strip()
    if not cleaned:
        raise ValueError("Enter a model name in the sidebar.")
    if any(character.isspace() for character in cleaned):
        raise ValueError("Model names cannot contain spaces.")
    return cleaned


def truncate_context(context: str, maximum_characters: int) -> tuple[str, bool]:
    """Trim trusted context to a predictable local limit."""

    cleaned = context.strip()
    if len(cleaned) <= maximum_characters:
        return cleaned, False
    return cleaned[:maximum_characters], True


def format_bullets(items: Iterable[str], empty_message: str = "None identified.") -> str:
    """Convert a collection to Markdown bullets for Streamlit."""

    values = [item.strip() for item in items if item and item.strip()]
    if not values:
        return empty_message
    return "\n".join(f"- {item}" for item in values)


def contains_high_risk_topic(prompt: str) -> bool:
    """Perform a conservative keyword check for design/safety-sensitive topics."""

    lowered = prompt.casefold()
    return any(term in lowered for term in HIGH_RISK_TERMS)


def ensure_question(questions: list[str], question: str) -> None:
    """Add a clarification question only when an equivalent one is absent."""

    normalized = {item.casefold().rstrip("?") for item in questions}
    candidate = question.casefold().rstrip("?")
    if candidate not in normalized:
        questions.append(question)


def apply_deterministic_analysis_rules(
    analysis: PromptAnalysis,
    *,
    original_prompt: str,
    selected_audience: str,
    output_style: str,
    strict_mode: bool,
    trusted_context_supplied: bool,
) -> PromptAnalysis:
    """Enforce non-negotiable application rules after model analysis."""

    updated = analysis.model_copy(deep=True)

    if selected_audience != "Infer automatically":
        updated.audience = selected_audience

    high_risk = contains_high_risk_topic(original_prompt)
    if high_risk and "Safety- or design-sensitive civil-engineering topic" not in updated.risk_flags:
        updated.risk_flags.append("Safety- or design-sensitive civil-engineering topic")

    if not trusted_context_supplied:
        boundary_note = "No trusted reference material was supplied; code- and project-specific claims cannot be verified."
        if boundary_note not in updated.assumptions:
            updated.assumptions.append(boundary_note)

    if strict_mode:
        if selected_audience == "Infer automatically":
            ensure_question(updated.clarification_questions, "Who is the intended audience?")
            if "Intended audience" not in updated.missing_information:
                updated.missing_information.append("Intended audience")

        if output_style == "Step-by-step numerical solution":
            ensure_question(updated.clarification_questions, "What numerical data and units should be used?")
            if "Numerical inputs and units" not in updated.missing_information:
                updated.missing_information.append("Numerical inputs and units")

        if high_risk:
            ensure_question(updated.clarification_questions, "Which design code and jurisdiction apply?")
            if "Governing design code and jurisdiction" not in updated.missing_information:
                updated.missing_information.append("Governing design code and jurisdiction")

        if updated.clarification_questions:
            updated.needs_clarification = True

    return updated
