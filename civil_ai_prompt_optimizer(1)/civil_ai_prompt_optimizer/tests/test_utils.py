"""Tests for local validation and formatting helpers."""

import pytest

from utils import (
    contains_high_risk_topic,
    format_bullets,
    truncate_context,
    validate_model_name,
    validate_user_prompt,
)


def test_validate_user_prompt_strips_whitespace() -> None:
    assert validate_user_prompt("  Explain storey drift.  ") == "Explain storey drift."


def test_validate_user_prompt_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="Enter a civil-engineering question"):
        validate_user_prompt("   ")


def test_validate_model_name_rejects_spaces() -> None:
    with pytest.raises(ValueError, match="cannot contain spaces"):
        validate_model_name("example model")


def test_truncate_context_reports_truncation() -> None:
    context, was_truncated = truncate_context("abcdefgh", 5)
    assert context == "abcde"
    assert was_truncated is True


def test_high_risk_detection_is_conservative() -> None:
    assert contains_high_risk_topic("Check the foundation bearing capacity") is True
    assert contains_high_risk_topic("Define a contour line") is False


def test_format_bullets_handles_empty_values() -> None:
    assert format_bullets([]) == "None identified."
    assert format_bullets(["One", "Two"]) == "- One\n- Two"
