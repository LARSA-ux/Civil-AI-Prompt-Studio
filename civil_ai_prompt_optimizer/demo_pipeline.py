"""Offline rule-based demonstration pipeline for CIVIL AI PROMPT OPTIMIZER.

The demo pipeline deliberately makes no network or OpenAI API calls. It exists so
students can demonstrate the application's workflow without paid API credits.
Its outputs are deterministic educational templates, not model-generated or
independently verified engineering advice.
"""

from __future__ import annotations

from dataclasses import dataclass

from schemas import (
    AnswerEvaluation,
    EvaluationVerdict,
    PipelineResult,
    PromptAnalysis,
    TaskType,
)
from utils import (
    apply_deterministic_analysis_rules,
    contains_high_risk_topic,
    truncate_context,
    validate_user_prompt,
)


TASK_TYPE_BY_STYLE = {
    "Clear explanation": TaskType.EXPLANATION,
    "Exam answer": TaskType.EXAM_ANSWER,
    "Viva answer": TaskType.VIVA_ANSWER,
    "Revision notes": TaskType.REVISION_NOTES,
    "Technical report": TaskType.TECHNICAL_REPORT,
    "Step-by-step numerical solution": TaskType.NUMERICAL_SOLUTION,
    "Comparison": TaskType.COMPARISON,
    "Research-style explanation": TaskType.RESEARCH_SUMMARY,
    "Child-friendly explanation": TaskType.EXPLANATION,
}


@dataclass(frozen=True, slots=True)
class DemoTopic:
    """A recognised rule-based demonstration topic."""

    key: str
    label: str


class DemoCivilAIPipeline:
    """Run a deterministic no-cost version of the multi-stage workflow."""

    def __init__(self, *, max_context_characters: int = 60_000) -> None:
        self.max_context_characters = max_context_characters

    @staticmethod
    def _detect_topic(prompt: str) -> DemoTopic:
        lowered = prompt.casefold()

        if ("storey drift" in lowered or "story drift" in lowered) and (
            "table" in lowered or "analyse" in lowered or "analyze" in lowered
        ):
            return DemoTopic("drift_table", "storey-drift table analysis")
        if ("storey drift" in lowered or "story drift" in lowered) and (
            "maximum" in lowered or "top floor" in lowered or "top storey" in lowered
        ):
            return DemoTopic("maximum_drift", "location of maximum storey drift")
        if "storey drift" in lowered or "story drift" in lowered:
            return DemoTopic("storey_drift", "storey drift")
        if "saturated" in lowered and "submerged" in lowered:
            return DemoTopic("unit_weight", "saturated and submerged unit weight")
        if "active remote sensing" in lowered or (
            "remote sensing" in lowered and "sunlight" in lowered
        ):
            return DemoTopic("remote_sensing", "active and passive remote sensing")
        if "slump" in lowered:
            return DemoTopic("slump", "concrete slump")
        if "boussinesq" in lowered:
            return DemoTopic("boussinesq", "Boussinesq theory")
        if "punching shear" in lowered or (
            "one-way shear" in lowered and "shear" in lowered
        ):
            return DemoTopic("punching_shear", "punching and one-way shear")
        if "equivalent static" in lowered:
            return DemoTopic("equivalent_static", "Equivalent Static Method")
        if "concrete grade" in lowered and (
            "earthquake" in lowered or "seismic" in lowered
        ):
            return DemoTopic("concrete_grade", "concrete grade and seismic performance")
        if any(term in lowered for term in ("is this building safe", "structure safe", "design safe")):
            return DemoTopic("safety_assessment", "project-specific structural safety")
        return DemoTopic("generic", "the requested engineering topic")

    @staticmethod
    def _infer_audience(selected_audience: str, output_style: str) -> tuple[str, str | None]:
        if selected_audience != "Infer automatically":
            return selected_audience, None
        if output_style == "Child-friendly explanation":
            return "Child or complete beginner", "Audience inferred from the child-friendly output style."
        if output_style == "Viva answer":
            return "Final-year civil-engineering student", "Audience inferred from the viva-answer output style."
        return "First-year engineering student", "Audience inferred as a first-year engineering student for demo mode."

    @staticmethod
    def _intent_for(topic: DemoTopic, output_style: str) -> str:
        action = {
            "Exam answer": "Prepare an exam-ready answer about",
            "Viva answer": "Prepare a concise viva response about",
            "Revision notes": "Create revision notes about",
            "Technical report": "Prepare a cautious technical discussion of",
            "Step-by-step numerical solution": "Solve a numerical problem involving",
            "Comparison": "Compare the relevant concepts for",
            "Research-style explanation": "Provide a research-style explanation of",
            "Child-friendly explanation": "Explain in beginner-friendly language",
        }.get(output_style, "Explain")
        return f"{action} {topic.label}."

    @staticmethod
    def _build_improved_prompt(
        *,
        original_prompt: str,
        topic: DemoTopic,
        domain: str,
        audience: str,
        output_style: str,
        trusted_context_supplied: bool,
    ) -> str:
        evidence_instruction = (
            "Use the supplied trusted context as reference material and identify any information that it does not support."
            if trusted_context_supplied
            else "Provide only cautious general background; do not claim code-specific or project-specific verification."
        )
        return (
            f"In the domain of {domain}, {DemoCivilAIPipeline._intent_for(topic, output_style).lower()} "
            f"Write for a {audience} using the output format '{output_style}'. "
            f"Clarify key terminology, distinguish commonly confused concepts, state assumptions and limitations, "
            f"and do not invent numerical values, design-code clauses, project conditions, citations, or safety conclusions. "
            f"{evidence_instruction} Original request: {original_prompt}"
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
        """Create deterministic prompt analysis without answering the question."""

        topic = self._detect_topic(original_prompt)
        audience, audience_assumption = self._infer_audience(selected_audience, output_style)
        task_type = TASK_TYPE_BY_STYLE.get(output_style, TaskType.OTHER)
        missing_information: list[str] = []
        assumptions: list[str] = [
            "Demo Mode uses deterministic educational templates rather than a language model."
        ]
        if audience_assumption:
            assumptions.append(audience_assumption)

        risk_flags: list[str] = []
        clarification_questions: list[str] = []

        high_risk = contains_high_risk_topic(original_prompt)
        if high_risk:
            risk_flags.append("Safety- or design-sensitive civil-engineering topic")
            missing_information.append("Governing design code and jurisdiction")
            clarification_questions.append("Which design code and jurisdiction apply?")

        if output_style == "Step-by-step numerical solution":
            missing_information.append("Numerical inputs and units")
            clarification_questions.append("What numerical data and units should be used?")

        if topic.key == "drift_table":
            missing_information.extend(
                [
                    "Complete storey-drift table and units",
                    "Storey heights and storey numbering convention",
                    "Load case or load combination",
                    "Analysis method and model description",
                ]
            )
            clarification_questions.extend(
                [
                    "Please paste the complete drift table with units.",
                    "What are the storey heights and how are the storeys numbered?",
                    "Which load case or combination and analysis method produced the results?",
                ]
            )

        if topic.key == "safety_assessment":
            missing_information.extend(
                [
                    "Verified geometry and member dimensions",
                    "Material properties and detailing",
                    "Loads and load combinations",
                    "Analysis results, drawings, and site conditions",
                ]
            )
            clarification_questions.extend(
                [
                    "What verified geometry, material, loading, detailing, and analysis data are available?",
                    "Are checked drawings and project-specific investigation results available?",
                ]
            )

        if not trusted_context_supplied:
            assumptions.append(
                "No trusted reference material was supplied; code- and project-specific claims cannot be verified."
            )

        needs_clarification = strict_mode and bool(clarification_questions)
        analysis = PromptAnalysis(
            intent=self._intent_for(topic, output_style),
            audience=audience,
            task_type=task_type,
            missing_information=self._unique(missing_information),
            assumptions=self._unique(assumptions),
            risk_flags=self._unique(risk_flags),
            needs_clarification=needs_clarification,
            clarification_questions=self._unique(clarification_questions),
            improved_prompt=self._build_improved_prompt(
                original_prompt=original_prompt,
                topic=topic,
                domain=domain,
                audience=audience,
                output_style=output_style,
                trusted_context_supplied=trusted_context_supplied,
            ),
        )
        return apply_deterministic_analysis_rules(
            analysis,
            original_prompt=original_prompt,
            selected_audience=selected_audience,
            output_style=output_style,
            strict_mode=strict_mode,
            trusted_context_supplied=trusted_context_supplied,
        )

    @staticmethod
    def _unique(items: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for item in items:
            normalized = item.casefold().strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                result.append(item)
        return result

    @staticmethod
    def _answer_for_topic(topic: DemoTopic) -> str:
        answers = {
            "storey_drift": r"""**Storey drift** is the relative horizontal displacement between two consecutive floors of a building during lateral movement, such as wind or earthquake response.

If the lateral displacements of two adjacent floors are \(\Delta_i\) and \(\Delta_{i-1}\), the storey drift is their difference:

\[
\text{Storey drift} = \Delta_i - \Delta_{i-1}
\]

A related quantity is the **drift ratio**, obtained by dividing the storey drift by the storey height. It indicates how much one floor moves sideways relative to the floor below. Excessive drift can damage partitions, cladding and services, and may indicate undesirable deformation of the structural system.

Storey drift is not the same as total roof displacement. Roof displacement is movement relative to the base; storey drift is the displacement difference between adjacent levels.""",
            "maximum_drift": """No. **Maximum storey drift is not automatically located at the top floor.**

Storey drift is the difference in lateral displacement between two adjacent floors, not the total displacement of a floor from the base. Total displacement often increases with height, but the largest difference between consecutive floors may occur near the bottom, middle or upper part of the building.

Its location depends on the distribution of stiffness, mass and lateral forces; storey heights; structural irregularities; infill walls; setbacks; soft or weak storeys; and the selected load case or dynamic mode. The actual drift profile must therefore be read from the analysis results rather than assumed from building height alone.""",
            "unit_weight": r"""No. **Saturated unit weight and submerged unit weight are different.**

- **Saturated unit weight, \(\gamma_{sat}\):** total weight per unit volume when all soil voids are filled with water.
- **Submerged or buoyant unit weight, \(\gamma'\):** effective unit weight of saturated soil below the water table after accounting for buoyancy.

Their relationship is:

\[
\gamma' = \gamma_{sat} - \gamma_w
\]

where \(\gamma_w\) is the unit weight of water. Saturated unit weight is used for total-weight calculations, while submerged unit weight is commonly used when evaluating effective stress below the water table.""",
            "remote_sensing": """No. **Active remote sensing does not depend on sunlight.**

An active sensor emits its own energy toward a target and measures the returned signal. Radar and LiDAR are common examples. Because the instrument supplies the energy, some active systems can operate at night.

**Passive remote sensing** measures naturally available radiation, such as reflected sunlight or thermal radiation emitted by objects. Therefore, visible-light passive sensing generally depends on sunlight, while thermal sensing can also use naturally emitted energy.""",
            "slump": """No. **A high slump is not always desirable.** Slump is mainly an indicator of the consistency and workability of fresh concrete.

A higher slump may help concrete flow and compact around congested reinforcement, but an unnecessarily high slump can be associated with segregation, bleeding or excess water when the mix is not properly designed. A low slump can also be unsuitable when placement and compaction become difficult.

The desirable slump depends on the member, placement method, reinforcement congestion, compaction method, mix design and project specification. Slump should therefore be judged against the specified range, not by assuming that higher is always better.""",
            "boussinesq": """No. **Boussinesq theory is not primarily a seepage theory.** In soil mechanics, Boussinesq equations are used to estimate stress increases within an idealised elastic, homogeneous, isotropic half-space due to surface loading.

Seepage problems are instead commonly analysed using Darcy's law, continuity, Laplace's equation, flow nets or numerical groundwater-flow methods. Boussinesq stress theory and seepage analysis may both appear in geotechnical engineering, but they address different physical problems.""",
            "punching_shear": """No. **Punching shear and one-way shear are different failure mechanisms.**

- **One-way shear** behaves broadly like beam shear. The critical section extends across the width of the slab or footing, and the potential failure surface is associated with shear in one principal direction.
- **Punching shear**, also called two-way shear, occurs locally around a concentrated support or load, such as a column supporting a flat slab. The potential failure surface forms around the loaded area and can resemble a truncated pyramid or cone.

Both are shear checks, but they use different critical sections and represent different load-transfer behaviour.""",
            "equivalent_static": """No. **The Equivalent Static Method cannot automatically be used for every irregular high-rise building.**

The method represents earthquake action using an equivalent set of lateral static forces. It is most suitable when the structure falls within the applicability limits of the governing seismic code and its response can be represented adequately by the assumptions of the method.

Tall, dynamically sensitive or significantly irregular buildings may require response-spectrum analysis, time-history analysis or another code-permitted dynamic procedure. The permitted method depends on factors such as height, structural system, regularity, seismicity and the governing jurisdiction. No universal height or irregularity limit should be quoted without the applicable code.""",
            "concrete_grade": """No. **Increasing concrete grade does not make a structure earthquake-proof.** Higher concrete strength can increase certain material and member capacities, but earthquake performance also depends on the structural system, ductility, reinforcement detailing, confinement, load path, stiffness distribution, member proportions, foundation behaviour, construction quality and avoidance of brittle failure modes.

A structure may use high-strength concrete and still perform poorly if it lacks suitable seismic detailing or has major irregularities. Seismic safety requires a complete code-based design and verification process, not one material-property change.""",
            "drift_table": """A responsible storey-drift-table assessment cannot be completed from the question alone.

The table should include the storey identifier, drift value, units or drift-ratio definition, load case or combination, direction, storey height and enough model information to interpret the pattern. The governing code is also required before checking a numerical limit.

Once those data are supplied, examine the drift profile for the maximum value, the storey where it occurs, abrupt changes between adjacent storeys, directional differences, possible stiffness irregularities and whether the reported quantity is displacement, drift or drift ratio.""",
            "safety_assessment": """There is not enough verified information to determine whether a building is safe.

A structural-safety conclusion requires, at minimum, the governing code and jurisdiction; checked geometry and member sizes; material properties and reinforcement/detailing; loads and combinations; structural system; analysis assumptions and results; foundation and soil information; construction condition; and review of drawings and site-specific hazards.

An AI-generated response must not replace design checks or approval by a qualified engineer.""",
            "generic": """The offline demo recognised the request but does not contain a reliable technical template for this specific topic.

It can still demonstrate prompt analysis, missing-information detection and the evaluation workflow. For a trustworthy technical answer, supply reviewed reference material in the Trusted Context box or use API Mode when credits are available. Project-specific and code-specific conclusions must be checked independently.""",
        }
        return answers[topic.key]

    @staticmethod
    def _adapt_output_style(answer: str, output_style: str) -> str:
        if output_style == "Viva answer":
            compact = " ".join(line.strip() for line in answer.splitlines() if line.strip())
            return f"**Viva response:** {compact}"
        if output_style == "Revision notes":
            return f"### Revision notes\n\n{answer}"
        if output_style == "Exam answer":
            return f"### Answer\n\n{answer}\n\n**Conclusion:** Use the definition and distinctions above; do not assume code compliance without the governing code."
        if output_style == "Technical report":
            return f"### Technical discussion\n\n{answer}\n\n### Limitation\nThis demo output is not a project-specific design verification."
        if output_style == "Research-style explanation":
            return f"### Conceptual background\n\n{answer}\n\n### Evidence limitation\nNo literature search or independent source verification was performed in offline Demo Mode."
        if output_style == "Child-friendly explanation":
            return f"### Simple explanation\n\n{answer}\n\nThink of this as a learning example rather than a real design check."
        return answer

    def generate_answer(
        self,
        *,
        original_prompt: str,
        output_style: str,
        trusted_context: str,
    ) -> str:
        """Return a deterministic educational draft for recognised topics."""

        topic = self._detect_topic(original_prompt)
        answer = self._adapt_output_style(self._answer_for_topic(topic), output_style)
        if trusted_context:
            answer += (
                "\n\n> **Trusted-context note:** Reference material was supplied, but the offline "
                "demo does not semantically verify arbitrary pasted documents. Review the supplied material manually."
            )
        return answer

    def evaluate_answer(
        self,
        *,
        original_prompt: str,
        draft_answer: str,
        trusted_context: str,
    ) -> AnswerEvaluation:
        """Create a transparent deterministic demonstration evaluation."""

        topic = self._detect_topic(original_prompt)
        high_risk = contains_high_risk_topic(original_prompt)
        requires_project_evidence = topic.key in {"drift_table", "safety_assessment"}

        if requires_project_evidence and not trusted_context:
            return AnswerEvaluation(
                accuracy_score=3,
                relevance_score=5,
                completeness_score=2,
                clarity_score=4,
                terminology_score=4,
                grounding_score=1,
                verdict=EvaluationVerdict.INSUFFICIENT_CONTEXT,
                unsupported_claims=[],
                technical_issues=["The requested project-specific assessment cannot be completed from the supplied information."],
                missing_assumptions=["Governing code, verified project data, units and analysis basis."],
                missing_caveats=[],
                revision_instructions=["Request the missing project data and do not produce a numerical safety or compliance conclusion."],
            )

        if topic.key == "generic":
            return AnswerEvaluation(
                accuracy_score=3,
                relevance_score=3,
                completeness_score=2,
                clarity_score=4,
                terminology_score=3,
                grounding_score=1 if not trusted_context else 2,
                verdict=EvaluationVerdict.INSUFFICIENT_CONTEXT,
                unsupported_claims=[],
                technical_issues=["Demo Mode has no reviewed technical template for this topic."],
                missing_assumptions=["A reviewed technical source or supported AI model response."],
                missing_caveats=[],
                revision_instructions=["Supply trusted reference material or use API Mode when available."],
            )

        if not trusted_context:
            return AnswerEvaluation(
                accuracy_score=4,
                relevance_score=5,
                completeness_score=4,
                clarity_score=5,
                terminology_score=5,
                grounding_score=2,
                verdict=EvaluationVerdict.REVISE,
                unsupported_claims=[],
                technical_issues=[],
                missing_assumptions=["No governing code, jurisdiction or project data were supplied." if high_risk else "No trusted reference was supplied."],
                missing_caveats=["State explicitly that the educational template was not independently source-verified."],
                revision_instructions=["Add a visible Demo Mode verification limitation without changing the correct conceptual explanation."],
            )

        return AnswerEvaluation(
            accuracy_score=4,
            relevance_score=5,
            completeness_score=4,
            clarity_score=5,
            terminology_score=5,
            grounding_score=3,
            verdict=EvaluationVerdict.PASS,
            unsupported_claims=[],
            technical_issues=[],
            missing_assumptions=[],
            missing_caveats=[],
            revision_instructions=[],
        )

    @staticmethod
    def revise_answer(draft_answer: str) -> str:
        """Add the limitation requested by the deterministic evaluator."""

        return (
            f"{draft_answer}\n\n---\n\n"
            "> **Demo Mode verification limitation:** This is a rule-based educational sample. "
            "It was not generated from a live AI model and was not independently checked against a design code, "
            "project file or external technical source. Verify code-specific and project-specific decisions with "
            "the governing documents and a qualified engineer."
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
        """Execute the no-cost demonstration pipeline."""

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
            output_style=output_style,
            trusted_context=context,
        )
        evaluation = self.evaluate_answer(
            original_prompt=prompt,
            draft_answer=draft_answer,
            trusted_context=context,
        )
        revised_answer: str | None = None
        if automatic_revision and evaluation.verdict == EvaluationVerdict.REVISE:
            revised_answer = self.revise_answer(draft_answer)

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
