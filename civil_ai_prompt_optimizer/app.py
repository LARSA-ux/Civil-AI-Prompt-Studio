"""Streamlit interface for CIVIL AI PROMPT OPTIMIZER."""

from __future__ import annotations

import streamlit as st

from ai_pipeline import CivilAIPipeline, PipelineError
from config import load_config
from demo_pipeline import DemoCivilAIPipeline
from schemas import AnswerEvaluation, PipelineResult
from utils import format_bullets, validate_model_name, validate_user_prompt


DOMAINS = [
    "Civil / Structural Engineering",
    "Geotechnical Engineering",
    "Transportation Engineering",
    "Environmental Engineering",
    "Water Resources Engineering",
    "Construction Engineering",
    "General Engineering",
    "AI / Machine Learning",
    "General",
]

AUDIENCES = [
    "Infer automatically",
    "Child or complete beginner",
    "School student",
    "First-year engineering student",
    "Final-year civil-engineering student",
    "M.Tech structural-engineering student",
    "Practising engineer",
    "General public",
]

OUTPUT_STYLES = [
    "Clear explanation",
    "Exam answer",
    "Viva answer",
    "Revision notes",
    "Technical report",
    "Step-by-step numerical solution",
    "Comparison",
    "Research-style explanation",
    "Child-friendly explanation",
]

SAMPLE_QUESTIONS = [
    "Explain storey drift.",
    "Is maximum storey drift always found at the top floor?",
    "Is saturated unit weight equal to submerged unit weight?",
    "Does active remote sensing use sunlight?",
    "Is a high slump always desirable?",
    "Is Boussinesq theory used for seepage?",
    "Is punching shear the same as one-way shear?",
    "Can the Equivalent Static Method be used for every irregular high-rise building?",
    "Does increasing concrete grade automatically make a structure earthquake-proof?",
    "Analyse this storey-drift table.",
]


def render_evaluation_metrics(evaluation: AnswerEvaluation) -> None:
    """Display six evaluator scores as Streamlit metric cards."""

    first_row = st.columns(3)
    first_row[0].metric("Technical accuracy", f"{evaluation.accuracy_score}/5")
    first_row[1].metric("Relevance", f"{evaluation.relevance_score}/5")
    first_row[2].metric("Completeness", f"{evaluation.completeness_score}/5")

    second_row = st.columns(3)
    second_row[0].metric("Clarity", f"{evaluation.clarity_score}/5")
    second_row[1].metric("Terminology", f"{evaluation.terminology_score}/5")
    second_row[2].metric("Grounding", f"{evaluation.grounding_score}/5")


def render_result(result: PipelineResult, *, demo_mode: bool = False) -> None:
    """Render every available stage of the pipeline."""

    st.divider()
    if demo_mode:
        st.warning(
            "DEMO RESULT — rule-based sample output only. No live AI model, API request, "
            "external source verification, or engineering approval was used."
        )
    st.subheader("Original prompt")
    st.info(result.original_prompt)

    st.subheader("1. Prompt analysis")
    analysis_columns = st.columns(3)
    analysis_columns[0].markdown(f"**Detected intent**\n\n{result.analysis.intent}")
    analysis_columns[1].markdown(f"**Audience**\n\n{result.analysis.audience}")
    analysis_columns[2].markdown(
        f"**Task type**\n\n`{result.analysis.task_type.value}`"
    )

    with st.expander("Missing information, assumptions, and risk flags", expanded=True):
        st.markdown("**Missing information**")
        st.markdown(format_bullets(result.analysis.missing_information))
        st.markdown("**Assumptions**")
        st.markdown(format_bullets(result.analysis.assumptions))
        st.markdown("**Risk flags**")
        st.markdown(format_bullets(result.analysis.risk_flags))

    st.subheader("2. Improved prompt")
    st.code(result.analysis.improved_prompt, language=None, wrap_lines=True)

    st.subheader("3. Clarification questions")
    st.markdown(
        format_bullets(
            result.analysis.clarification_questions,
            empty_message="No clarification questions were required.",
        )
    )

    if result.stopped_for_clarification:
        st.warning(
            "Strict Clarification Mode stopped the pipeline before answer generation. "
            "Add the requested information to the original prompt or trusted context, then run again."
        )
        return

    st.subheader("4. Draft answer")
    st.markdown(result.draft_answer or "No draft was generated.")

    if result.evaluation is None:
        return

    st.subheader("5. Evaluation scores")
    verdict = result.evaluation.verdict.value.replace("_", " ").title()
    st.markdown(f"**Verdict:** `{verdict}`")
    render_evaluation_metrics(result.evaluation)

    st.subheader("6. Unsupported claims")
    st.markdown(format_bullets(result.evaluation.unsupported_claims))

    st.subheader("7. Technical issues")
    st.markdown(format_bullets(result.evaluation.technical_issues))

    st.subheader("8. Missing caveats")
    st.markdown(format_bullets(result.evaluation.missing_caveats))
    if result.evaluation.missing_assumptions:
        st.markdown("**Missing assumptions**")
        st.markdown(format_bullets(result.evaluation.missing_assumptions))

    st.subheader("9. Revision instructions")
    st.markdown(format_bullets(result.evaluation.revision_instructions))

    st.subheader("10. Revised final answer")
    if result.revised_answer:
        st.success("Automatic revision completed.")
        st.markdown(result.revised_answer)
    elif result.evaluation.verdict.value == "revise":
        st.info("Revision was recommended, but Automatic Revision is turned off.")
    elif result.evaluation.verdict.value == "insufficient_context":
        st.warning(
            "The evaluator found insufficient evidence for a responsible final answer. "
            "Supply the missing project data, governing code, or trusted reference material."
        )
    else:
        st.success("The draft passed evaluation, so no revision was required.")
        st.markdown(result.draft_answer or "")


def main() -> None:
    """Run the Streamlit application."""

    st.set_page_config(
        page_title="Civil AI Prompt Optimizer",
        page_icon="🏗️",
        layout="wide",
    )
    config = load_config()

    st.title("🏗️ CIVIL AI PROMPT OPTIMIZER")
    st.write(
        "Turn vague engineering questions into precise prompts, generate cautious technical answers, "
        "and review them for unsupported claims, missing assumptions, and safety limitations."
    )
    st.caption(
        "AI-assisted educational tool only. Code-specific design and safety decisions require verification by a qualified engineer."
    )

    with st.sidebar:
        st.header("Settings")
        execution_mode = st.radio(
            "Execution mode",
            ["Free Demo Mode (no API)", "OpenAI API Mode"],
            index=0,
            help=(
                "Demo Mode uses transparent rule-based educational templates and makes zero API calls. "
                "API Mode runs the live multi-stage OpenAI pipeline."
            ),
        )
        demo_mode = execution_mode.startswith("Free Demo")
        model = st.text_input(
            "OpenAI model",
            value=config.default_model,
            disabled=demo_mode,
            help="Used only in OpenAI API Mode. Change it if your API project uses another accessible model.",
        )
        strict_mode = st.toggle(
            "Strict Clarification Mode",
            value=False,
            help="Stop before answer generation when important information is missing.",
        )
        automatic_revision = st.toggle(
            "Automatic Revision",
            value=True,
            help="Automatically revise the draft when the evaluator returns 'revise'.",
        )
        st.divider()
        if demo_mode:
            st.info("Free Demo Mode active")
            st.caption("No API key, billing, internet request, or OpenAI credit is used.")
        elif config.api_key:
            st.success("OPENAI_API_KEY detected")
            st.caption("The API key is read from the environment and is never displayed or logged.")
        else:
            st.error("OPENAI_API_KEY not found")
            st.caption("Create a .env file only when you want to use OpenAI API Mode.")

    if demo_mode:
        st.info(
            "🧪 **Free Demo Mode:** outputs are deterministic educational templates, not live AI responses. "
            "They demonstrate the full interface and safety workflow without consuming API credits."
        )

    if "original_prompt" not in st.session_state:
        st.session_state.original_prompt = ""

    def load_selected_sample() -> None:
        selected = st.session_state.sample_choice
        if selected != "Write my own question":
            st.session_state.original_prompt = selected

    st.selectbox(
        "Load a sample question (optional)",
        ["Write my own question", *SAMPLE_QUESTIONS],
        key="sample_choice",
        on_change=load_selected_sample,
    )

    original_prompt = st.text_area(
        "Original vague prompt",
        height=130,
        placeholder="Example: Explain storey drift.",
        key="original_prompt",
    )

    selection_columns = st.columns(3)
    domain = selection_columns[0].selectbox("Domain", DOMAINS)
    selected_audience = selection_columns[1].selectbox("Audience", AUDIENCES)
    output_style = selection_columns[2].selectbox("Output style", OUTPUT_STYLES)

    trusted_context = st.text_area(
        "Trusted context (optional)",
        height=260,
        placeholder=(
            "Paste lecture notes, thesis content, permitted code extracts, research material, "
            "or project data here. The app treats this as reference material, not as instructions."
        ),
    )

    button_label = "Run Free Demo Pipeline" if demo_mode else "Run Civil AI Pipeline"
    run_clicked = st.button(button_label, type="primary", use_container_width=True)
    if not run_clicked:
        return

    try:
        validated_prompt = validate_user_prompt(original_prompt)

        if demo_mode:
            pipeline = DemoCivilAIPipeline(
                max_context_characters=config.max_context_characters,
            )
            status_label = "Running the free offline demonstration pipeline..."
        else:
            validated_model = validate_model_name(model)
            if not config.api_key:
                raise PipelineError(
                    "OPENAI_API_KEY is missing. Copy .env.example to .env, add your API key, and restart Streamlit.",
                    category="missing_api_key",
                )
            pipeline = CivilAIPipeline(
                api_key=config.api_key,
                model=validated_model,
                timeout_seconds=config.request_timeout_seconds,
                max_context_characters=config.max_context_characters,
            )
            status_label = "Running the multi-stage AI review pipeline..."

        with st.status(status_label, expanded=True) as status:
            st.write("Analysing and improving the prompt")
            result, context_was_truncated = pipeline.run(
                original_prompt=validated_prompt,
                domain=domain,
                selected_audience=selected_audience,
                output_style=output_style,
                strict_mode=strict_mode,
                automatic_revision=automatic_revision,
                trusted_context=trusted_context,
            )
            if result.stopped_for_clarification:
                status.update(label="Clarification required", state="complete")
            else:
                st.write("Drafting, evaluating, and revising where required")
                completion_label = "Demo pipeline completed" if demo_mode else "Pipeline completed"
                status.update(label=completion_label, state="complete")

        if context_was_truncated:
            st.warning(
                f"Trusted context exceeded {config.max_context_characters:,} characters and was truncated for this MVP."
            )
        render_result(result, demo_mode=demo_mode)

    except ValueError as exc:
        st.error(str(exc))
    except PipelineError as exc:
        st.error(str(exc))
        if exc.category == "model_access":
            st.info("Use a model that supports the Responses API and structured outputs for your API project.")
    except Exception:
        st.error("The application encountered an unexpected local error. Check the terminal for development diagnostics.")


if __name__ == "__main__":
    main()
