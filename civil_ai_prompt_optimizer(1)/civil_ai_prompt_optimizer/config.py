"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


DEFAULT_MODEL = "gpt-4.1-mini"


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Runtime configuration for the OpenAI client and UI."""

    api_key: str | None
    default_model: str
    request_timeout_seconds: float = 90.0
    max_context_characters: int = 60_000


def load_config() -> AppConfig:
    """Load configuration without printing or logging secrets."""

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    timeout_raw = os.getenv("OPENAI_TIMEOUT_SECONDS", "90")

    try:
        timeout = max(10.0, float(timeout_raw))
    except ValueError:
        timeout = 90.0

    return AppConfig(
        api_key=api_key.strip() if api_key else None,
        default_model=model,
        request_timeout_seconds=timeout,
    )
