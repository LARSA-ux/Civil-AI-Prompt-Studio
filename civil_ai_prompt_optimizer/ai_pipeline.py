"""OpenAI Responses API orchestration for CIVIL AI PROMPT OPTIMIZER."""

from __future__ import annotations

from typing import TypeVar, cast

from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)
from pydantic import BaseModel, ValidationError

from prompts import (
    ANSWER_EVALUATOR_SYSTEM_PROMPT,
    ANSWER_GENERATOR_SYSTEM_PROMPT,
    ANSWER_REVISION_SYSTEM_PROMPT,
    PROMPT_ANALYSER_SYSTEM_PROMPT,
    build_analysis_input,
    build_evaluation_input,
    build_generation_input,
    build_revision_input,
)
from schemas import (
    AnswerEvaluation,
    EvaluationVerdict,
    PipelineResult,
    PromptAnalysis,
)
from utils import (
    apply_deterministic_analysis_rules,
    truncate_context,
    validate_model_name,
    validate_user_prompt,
)


StructuredModelT = TypeVar("StructuredModelT", bound=BaseModel)


class PipelineError(RuntimeError):
    """A safe, user-facing error raised by the AI pipeline."""

    def __init__(self, message: str, *, category: str = "pipeline") -> None:
        super().__init__(message)
        self.category = category


class CivilAIPipeline:
    """Run prompt analysis, answer generation, evaluation, and revision."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: float = 90.0,
        max_context_characters: int = 60_000,
    ) -> None:
        if not api_key.strip():
            raise PipelineError(
                "OPENAI_API_KEY is missing. Add it to your .env file and restart the app.",
                category="missing_api_key",
            )

        self.model = validate_model_name(model)
        self.max_context_characters = max_context_characters
        self.client = OpenAI(api_key=api_key, timeout=timeout_seconds, max_retries=2)

    def _parse_structured(
        self,
        *,
        instructions: str,
        input_text: str,
        schema: type[StructuredModelT],
        max_output_tokens: int,
    ) -> StructuredModelT:
        """Call Responses structured parsing and validate the result."""

        try:
            response = self.client.responses.parse(
                model=self.model,
                instructions=instructions,
                input=input_text,
                text_format=schema,
                max_output_tokens=max_output_tokens,
            )
            if getattr(response, "status", None) == "incomplete":
                reason = getattr(getattr(response, "incomplete_details", None), "reason", None)
                suffix = f" ({reason})" if reason else ""
                raise PipelineError(
                    f"The model returned an incomplete structured response{suffix}. Try again or choose another model.",
                    category="incomplete_response",
                )

            parsed = getattr(response, "output_parsed", None)
            if parsed is not None:
                return cast(StructuredModelT, parsed)

            output_text = getattr(response, "output_text", "")
            if output_text:
                return schema.model_validate_json(output_text)

            raise PipelineError(
                "The model returned an empty structured response.",
                category="empty_response",
            )
        except PipelineError:
            raise
        except ValidationError as exc:
            raise PipelineError(
                "The model returned malformed structured output. Try again or choose a model that supports structured outputs.",
                category="malformed_structured_output",
            ) from exc
        except Exception as exc:  # OpenAI errors are mapped centrally below.
            raise self._map_openai_error(exc) from exc

    def _generate_text(
        self,
        *,
        instructions: str,
        input_text: str,
        max_output_tokens: int,
    ) -> str:
        """Generate a non-empty text response through the Responses API."""

        try:
            response = self.client.responses.create(
                model=self.model,
                instructions=instructions,
                input=input_text,
                max_output_tokens=max_output_tokens,
            )
            if getattr(response, "status", None) == "incomplete":
                reason = getattr(getattr(response, "incomplete_details", None), "reason", None)
                suffix = f" ({reason})" if reason else ""
                raise PipelineError(
                    f"The model returned an incomplete answer{suffix}. Try again or choose another model.",
                    category="incomplete_response",
                )

            text = getattr(response, "output_text", "").strip()
            if not text:
                raise PipelineError(
                    "The model returned an empty answer.",
                    category="empty_response",
                )
            return text
        except PipelineError:
            raise
        except Exception as exc:
            raise self._map_openai_error(exc) from exc

    def _map_openai_error(self, exc: Exception) -> PipelineError:
        """Convert SDK errors to readable messages without leaking secrets."""

        if isinstance(exc, AuthenticationError):
            return PipelineError(
                "The OpenAI API key was rejected. Check OPENAI_API_KEY in .env.",
                category="invalid_api_key",
            )
        if isinstance(exc, RateLimitError):
            return PipelineError(
                "The OpenAI API rate limit or account quota was reached. Check API billing and try again later.",
                category="rate_limit",
            )
        if isinstance(exc, APIConnectionError):
            return PipelineError(
                "The app could not connect to the OpenAI API. Check your internet connection and firewall settings.",
                category="connection",
            )
        if isinstance(exc, (NotFoundError, PermissionDeniedError)):
            return PipelineError(
                f"The model '{self.model}' is unavailable to this API project. Choose an accessible model in the sidebar.",
                category="model_access",
            )
        if isinstance(exc, BadRequestError):
            return PipelineError(
                f"The API rejected the request. The selected model '{self.model}' may not support this structured-output configuration.",
                category="bad_request",
            )
        if isinstance(exc, APIStatusError):
            return PipelineError(
                f"The OpenAI API returned an unexpected service error (HTTP {exc.status_code}).",
                category="api_status",
            )
        return PipelineError(
            "An unexpected error occurred while calling the OpenAI API.",
            category="unknown",
        )

    def analyse_prompt(
        self,
        *,
        original_prompt: str,
        domain: str,
        selected_audience: str,
        output_style: str,
        strict_mode: bool,
        trusted_context_supplied: bool,
    ) -> PromptAnalysis:
        """Analyse and improve the prompt without answering it."""

        analysis = self._parse_structured(
            instructions=PROMPT_ANALYSER_SYSTEM_PROMPT,
            input_text=build_analysis_input(
                original_prompt=original_prompt,
                domain=domain,
                selected_audience=selected_audience,
                output_style=output_style,
                strict_mode=strict_mode,
                trusted_context_supplied=trusted_context_supplied,
            ),
            schema=PromptAnalysis,
            max_output_tokens=2_000,
        )
        return apply_deterministic_analysis_rules(
            analysis,
            original_prompt=original_prompt,
            selected_audience=selected_audience,
            output_style=output_style,
            strict_mode=strict_mode,
            trusted_context_supplied=trusted_context_supplied,
        )

    def generate_answer(
        self,
        *,
        original_prompt: str,
        analysis: PromptAnalysis,
        domain: str,
        output_style: str,
        trusted_context: str,
    ) -> str:
        """Generate the first technical draft."""

        return self._generate_text(
            instructions=ANSWER_GENERATOR_SYSTEM_PROMPT,
            input_text=build_generation_input(
                original_prompt=original_prompt,
                improved_prompt=analysis.improved_prompt,
                analysis=analysis.model_dump(mode="json"),
                domain=domain,
                audience=analysis.audience,
                output_style=output_style,
                trusted_context=trusted_context,
            ),
            max_output_tokens=3_000,
        )

    def evaluate_answer(
        self,
        *,
        improved_prompt: str,
        draft_answer: str,
        trusted_context: str,
    ) -> AnswerEvaluation:
        """Adversarially evaluate the draft answer."""

        return self._parse_structured(
            instructions=ANSWER_EVALUATOR_SYSTEM_PROMPT,
            input_text=build_evaluation_input(
                improved_prompt=improved_prompt,
                draft_answer=draft_answer,
                trusted_context=trusted_context,
            ),
            schema=AnswerEvaluation,
            max_output_tokens=2_000,
        )

    def revise_answer(
        self,
        *,
        improved_prompt: str,
        draft_answer: str,
        evaluation: AnswerEvaluation,
        trusted_context: str,
    ) -> str:
        """Revise a failed draft using all evaluator feedback."""

        return self._generate_text(
            instructions=ANSWER_REVISION_SYSTEM_PROMPT,
            input_text=build_revision_input(
                improved_prompt=improved_prompt,
                draft_answer=draft_answer,
                evaluator_feedback=evaluation.model_dump(mode="json"),
                trusted_context=trusted_context,
            ),
            max_output_tokens=3_000,
        )

    def run(
        self,
        *,
        original_prompt: str,
        domain: str,
        selected_audience: str,
        output_style: str,
        strict_mode: bool,
        automatic_revision: bool,
        trusted_context: str,
    ) -> tuple[PipelineResult, bool]:
        """Execute the complete pipeline and report whether context was truncated."""

        prompt = validate_user_prompt(original_prompt)
        context, context_was_truncated = truncate_context(
            trusted_context,
            self.max_context_characters,
        )

        analysis = self.analyse_prompt(
            original_prompt=prompt,
            domain=domain,
            selected_audience=selected_audience,
            output_style=output_style,
            strict_mode=strict_mode,
            trusted_context_supplied=bool(context),
        )

        if strict_mode and analysis.needs_clarification:
            return (
                PipelineResult(
                    original_prompt=prompt,
                    analysis=analysis,
                    stopped_for_clarification=True,
                ),
                context_was_truncated,
            )

        draft_answer = self.generate_answer(
            original_prompt=prompt,
            analysis=analysis,
            domain=domain,
            output_style=output_style,
            trusted_context=context,
        )
        evaluation = self.evaluate_answer(
            improved_prompt=analysis.improved_prompt,
            draft_answer=draft_answer,
            trusted_context=context,
        )

        revised_answer: str | None = None
        if automatic_revision and evaluation.verdict == EvaluationVerdict.REVISE:
            revised_answer = self.revise_answer(
                improved_prompt=analysis.improved_prompt,
                draft_answer=draft_answer,
                evaluation=evaluation,
                trusted_context=context,
            )

        return (
            PipelineResult(
                original_prompt=prompt,
                analysis=analysis,
                draft_answer=draft_answer,
                evaluation=evaluation,
                revised_answer=revised_answer,
                stopped_for_clarification=False,
            ),
            context_was_truncated,
        )
