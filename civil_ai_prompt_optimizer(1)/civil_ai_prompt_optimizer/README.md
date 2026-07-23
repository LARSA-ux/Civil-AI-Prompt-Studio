# CIVIL AI PROMPT OPTIMIZER

A portfolio-ready Streamlit application that converts vague civil-engineering questions into precise technical prompts, generates cautious answers, evaluates those answers adversarially, and revises them when problems are found.

The project is designed for a Civil Engineering graduate beginning an M.Tech in Structural Engineering and learning AI, prompt engineering, retrieval-augmented generation, hallucination detection, and technical evaluation.

> **Important:** This is an AI-assisted educational and drafting tool. It is not a substitute for design calculations, governing codes, checked drawings, laboratory results, or review by a qualified engineer.

---

## 1. Project overview

A user can enter a short question such as:

```text
Explain storey drift.
```

The application runs four separate AI stages:

1. **Prompt analyser** — identifies intent, audience, task type, missing information, assumptions, risk flags, clarification questions, and an improved prompt.
2. **Answer generator** — creates a cautious draft using the improved prompt and optional trusted context.
3. **Answer evaluator** — behaves like an adversarial technical reviewer and scores the answer for accuracy, relevance, completeness, clarity, terminology, and grounding.
4. **Answer revision** — corrects the issues identified by the evaluator when the verdict is `revise` and automatic revision is enabled.

The application uses the **OpenAI Responses API**, the official OpenAI Python SDK, Pydantic structured outputs, Streamlit, and `python-dotenv`.

---

## 2. Features

- Quick Mode with visible low-risk assumptions
- Strict Clarification Mode that stops when important information is missing
- Configurable model through `OPENAI_MODEL` and the Streamlit sidebar
- Nine engineering/general domain options
- Eight audience options
- Nine output styles
- Optional trusted-context evidence boundary
- Structured `PromptAnalysis` and `AnswerEvaluation` Pydantic models
- Separate prompt-analysis, generation, evaluation, and revision stages
- Adversarial review rather than simple confidence scoring
- Unsupported-claim and hallucination checks
- Missing-assumption, caveat, design-code, jurisdiction, and safety checks
- Prompt-injection resistance for pasted notes and future retrieved documents
- Clear error messages for API-key, connection, quota, model-access, malformed-output, and empty-response problems
- Local unit tests that do not require an API key
- Phase 2 PDF RAG and Phase 3 benchmark designs

---

## 3. Architecture

```text
User interface (app.py)
        |
        v
Configuration and validation
(config.py + utils.py)
        |
        v
Prompt analysis -- structured PromptAnalysis
        |
        +--> Strict mode may stop and ask questions
        |
        v
Draft answer generation -- free-form text
        |
        v
Adversarial evaluation -- structured AnswerEvaluation
        |
        +--> pass: display draft as final
        +--> insufficient_context: request evidence/data
        +--> revise: send feedback to revision stage
        |
        v
Revised final answer
```

### Evidence boundary

When trusted context is supplied, the generator and evaluator are instructed to treat it as the primary evidence boundary. The app must state when the material is insufficient and must not silently add unsupported code clauses, project facts, numerical values, citations, or test results.

When trusted context is absent, the app may provide cautious general background but cannot claim independent technical verification. Project-specific and code-specific conclusions require separate checking.

### Prompt security

The original question and trusted context are serialized and placed inside explicit untrusted-data delimiters. System prompts instruct every stage to ignore embedded requests that try to:

- override system instructions,
- expose secrets or API keys,
- bypass safety rules,
- invent sources,
- escape the evidence boundary,
- treat reference material as a new system prompt.

This is a strong application-level guardrail, but no prompt-only defence should be considered perfect. Phase 2 adds document-level screening and citation validation.

---

## 4. Folder structure

```text
civil_ai_prompt_optimizer/
|
|-- app.py
|-- ai_pipeline.py
|-- prompts.py
|-- schemas.py
|-- config.py
|-- utils.py
|-- requirements.txt
|-- .env.example
|-- .gitignore
|-- README.md
|
|-- tests/
|   |-- test_schemas.py
|   |-- test_prompt_rules.py
|   `-- test_utils.py
|
`-- sample_data/
    `-- sample_trusted_context.txt
```

### File responsibilities

- `app.py` — Streamlit layout, controls, status messages, and result rendering.
- `ai_pipeline.py` — OpenAI client calls and four-stage orchestration.
- `prompts.py` — guarded system prompts and safe request builders.
- `schemas.py` — Pydantic enums and structured-output models.
- `config.py` — `.env` loading and non-secret runtime configuration.
- `utils.py` — validation, formatting, context limits, risk detection, and deterministic strict-mode rules.
- `tests/` — schema, security-prompt, and utility tests.
- `sample_data/` — safe example context for a first run.

---

## 5. Why Python instead of C++ for the MVP?

Python is the better first choice because this version is mainly an **AI orchestration and RAG application**, not a high-performance finite-element solver.

Python provides mature libraries for:

- OpenAI API requests,
- Pydantic structured validation,
- Streamlit user interfaces,
- PDF extraction,
- embeddings and vector databases,
- data analysis and CSV export,
- rapid testing and iteration.

C++ would add more development complexity to API orchestration, document handling, UI work, and prompt management without improving the main MVP workflow.

A future C++ component can be added as a separate calculation engine for matrix operations, structural-analysis algorithms, finite-element calculations, numerical solvers, and optimisation. Python should continue to manage the interface, API calls, RAG, prompts, evaluation, and reporting.

---

## 6. Prerequisites

Install:

- Python 3.12 or newer
- Visual Studio Code
- The VS Code Python extension
- Git, recommended but optional
- An OpenAI API key with API billing enabled

### ChatGPT subscription versus API billing

A ChatGPT Free, Plus, Pro, or other ChatGPT subscription does **not** automatically include OpenAI API credits. ChatGPT subscriptions and OpenAI API billing are separate products. The application uses the API and therefore requires an API key and an API project with billing or available credits.

Never publish your API key in GitHub, screenshots, source code, a resume, or a portfolio demonstration.

---

## 7. Windows installation

### Step 1 — Open the project in VS Code

1. Extract or clone the `civil_ai_prompt_optimizer` folder.
2. Open VS Code.
3. Select **File > Open Folder**.
4. Choose the `civil_ai_prompt_optimizer` folder.
5. Open a new terminal with **Terminal > New Terminal**.

### Step 2 — Create a virtual environment

```powershell
python -m venv .venv
```

### Step 3 — Activate it in PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks script activation, open PowerShell as your normal user and run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate the environment again.

### Command Prompt activation

```bat
.venv\Scripts\activate.bat
```

### Step 4 — Select the VS Code interpreter

1. Press `Ctrl+Shift+P`.
2. Search for **Python: Select Interpreter**.
3. Select the interpreter inside `.venv`.

### Step 5 — Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 8. Environment setup

Copy `.env.example` to a new file named `.env`.

PowerShell:

```powershell
Copy-Item .env.example .env
```

Command Prompt:

```bat
copy .env.example .env
```

Open `.env` and replace the placeholder:

```dotenv
OPENAI_API_KEY=your_real_api_key_here
OPENAI_MODEL=gpt-4.1-mini
OPENAI_TIMEOUT_SECONDS=90
```

The model shown is only a configurable default. Model access varies by API project and may change over time. Enter another Responses-API model in the Streamlit sidebar when necessary.

The `.env` file is included in `.gitignore`. Do not remove that rule.

---

## 9. Run the application

From the project folder with the virtual environment active:

```powershell
python -m streamlit run app.py
```

Streamlit should open a local browser page, usually at:

```text
http://localhost:8501
```

Stop the server with `Ctrl+C` in the terminal.

---

## 10. Run the tests

The local tests do not call the OpenAI API and do not spend API credits.

```powershell
python -m pytest -q
```

The tests check:

- Pydantic score limits and enum values,
- prompt-analysis and evaluation schema validity,
- system-prompt safety rules,
- untrusted-data delimiters,
- strict-mode clarification backstops,
- input validation and context truncation.

---

## 11. First test

1. Launch the app.
2. Confirm the sidebar says `OPENAI_API_KEY detected`.
3. Keep **Quick Mode** by leaving Strict Clarification Mode off.
4. Keep Automatic Revision on.
5. Select:
   - Domain: `Civil / Structural Engineering`
   - Audience: `First-year engineering student`
   - Output style: `Clear explanation`
6. Enter:

```text
Explain storey drift.
```

7. Leave trusted context empty for the first run.
8. Select **Run Civil AI Pipeline**.

### Expected first-run behaviour

The interface should display:

1. the original prompt,
2. detected intent, audience, and task type,
3. missing information, assumptions, and risk flags,
4. an improved technical prompt,
5. clarification questions,
6. a cautious draft answer,
7. six evaluation metric cards,
8. unsupported claims,
9. technical issues,
10. missing caveats and assumptions,
11. revision instructions,
12. a revised answer when the verdict is `revise`.

Because no trusted source was supplied, the answer and evaluation should state that project-specific and code-specific conclusions were not independently verified.

### Trusted-context test

Copy the contents of:

```text
sample_data/sample_trusted_context.txt
```

into the Trusted Context box and run the same question again. The answer should ground itself in the supplied note and should not invent a design-code limit or claim that a real building is safe.

### Strict-mode test

Enable Strict Clarification Mode and ask:

```text
Is this building safe under seismic loads?
```

The app should stop before answer generation and request important missing information such as the governing design code/jurisdiction, project data, and intended audience.

---

## 12. Sample questions

1. Explain storey drift.
2. Is maximum storey drift always found at the top floor?
3. Is saturated unit weight equal to submerged unit weight?
4. Does active remote sensing use sunlight?
5. Is a high slump always desirable?
6. Is Boussinesq theory used for seepage?
7. Is punching shear the same as one-way shear?
8. Can the Equivalent Static Method be used for every irregular high-rise building?
9. Does increasing concrete grade automatically make a structure earthquake-proof?
10. Analyse this storey-drift table.

For question 10, paste the complete table, units, storey heights, model description, load case/combination, analysis method, and governing code into the Trusted Context box.

---

## 13. Application modes

### Quick Mode

Quick Mode may make low-risk assumptions so that conceptual questions can be answered without unnecessary interruption. Every assumption is displayed.

It must not silently invent:

- design-code clauses,
- project conditions,
- numerical values,
- material properties,
- loads,
- dimensions,
- safety conclusions,
- citations,
- test results.

### Strict Clarification Mode

Strict mode stops before answer generation when important information is missing. In addition to model-based analysis, deterministic local rules force clarification for common high-risk situations.

Examples include:

- unknown audience,
- numerical solution without data or units,
- safety/design-sensitive request without a governing code and jurisdiction.

---

## 14. Troubleshooting

### `OPENAI_API_KEY not found`

- Confirm the file is named exactly `.env`, not `.env.txt`.
- Confirm `.env` is in the project root beside `app.py`.
- Restart Streamlit after editing `.env`.
- Do not add quotes unless the key itself requires them.

### Invalid API key

- Create or copy a valid key from the OpenAI API platform.
- Make sure no spaces were added before or after the key.
- Replace a revoked or expired key.

### Rate limit or quota error

- Check API billing and available credits.
- Reduce repeated runs.
- Wait before retrying when a rate limit is temporary.
- Remember that ChatGPT subscription billing is separate.

### Model unavailable or access denied

- Enter a model available to your API project in the sidebar.
- Use a model that supports the Responses API and structured outputs.
- Do not assume that a model visible in ChatGPT is automatically available through your API project.

### Structured-output error

- Try the request again.
- Use a model that supports structured outputs.
- Keep the OpenAI SDK and Pydantic within the versions allowed by `requirements.txt`.
- Check whether an extremely long input caused an incomplete response.

### Connection failure

- Check internet access, proxy, firewall, and VPN settings.
- Confirm that corporate or university networks are not blocking API traffic.

### PowerShell will not activate `.venv`

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then:

```powershell
.venv\Scripts\Activate.ps1
```

### `streamlit` is not recognised

Use the module command so the selected Python interpreter is explicit:

```powershell
python -m streamlit run app.py
```

### Imports fail in VS Code but work in the terminal

Select the `.venv` interpreter through **Python: Select Interpreter**, then reload the VS Code window.

---

## 15. Current limitations

- The model is still an AI system and can make technical mistakes.
- The evaluator is another AI stage, not an independent licensed engineer or deterministic proof system.
- No PDF upload or vector database is implemented in the MVP.
- Pasted trusted context is limited to a configurable character count and is not token-optimised.
- The MVP does not independently fetch or verify design codes.
- It does not perform structural calculations or finite-element analysis.
- It does not validate drawings, ETABS models, laboratory certificates, or site conditions.
- It does not guarantee prompt-injection immunity.
- A `pass` verdict means the answer passed the configured AI review, not that an engineering design is approved.
- API cost and latency increase because one user run may require three or four model calls.
- Model availability and structured-output support depend on the user's API project.

---

## 16. Portfolio skills demonstrated

This project can demonstrate:

- Python application development
- Streamlit UI development
- OpenAI Responses API integration
- Pydantic structured outputs
- prompt decomposition and orchestration
- prompt-injection-aware system design
- evidence-bound technical generation
- hallucination and unsupported-claim review
- adversarial LLM evaluation
- human-in-the-loop clarification workflows
- civil-engineering domain guardrails
- error handling and secret management
- automated tests
- RAG architecture planning
- benchmark and CSV-reporting design
- software documentation and technical writing

A useful portfolio description is:

> Built a multi-stage civil-engineering AI assistant that optimises vague prompts, generates evidence-bounded technical answers, adversarially evaluates hallucinations and safety gaps, and automatically revises failed responses using the OpenAI Responses API, Pydantic, and Streamlit.

---

## 17. Phase 2 — PDF-based RAG plan

Phase 2 should add genuine document retrieval, not decorative or fabricated citations.

### Proposed components

- **PDF extraction:** PyMuPDF (`fitz`) as the first option; `pypdf` as a fallback.
- **Page-aware document model:** retain filename, page number, extraction status, and character offsets.
- **Chunking:** headings/paragraph-aware chunks with controlled overlap.
- **Embeddings:** an OpenAI embedding model selected through configuration.
- **Local vector store:** FAISS for a lightweight local MVP or Chroma for richer metadata management.
- **Upload UI:** Streamlit `file_uploader` with multiple PDF support.
- **Retrieval:** similarity search plus optional lexical filtering and reranking.
- **Generation:** answer only from genuinely retrieved chunks when document-grounded mode is enabled.
- **Citations:** filename, page number when extractable, and stable chunk identifier.
- **Refusal:** state that evidence is insufficient when retrieved chunks do not support the requested claim.

### Suggested Phase 2 data models

```text
DocumentRecord
- document_id
- filename
- sha256
- page_count
- extraction_errors

DocumentChunk
- chunk_id
- document_id
- filename
- page_start
- page_end
- text
- character_start
- character_end

RetrievedChunk
- chunk_id
- score
- text
- filename
- page_start
- page_end
```

### PDF workflow

1. Accept one or more PDFs.
2. Validate file type and size.
3. Save to a local temporary/session directory.
4. Calculate a SHA-256 hash to detect duplicate uploads.
5. Extract text page by page.
6. Report encrypted, damaged, empty, image-only, or partially extracted files.
7. Normalise whitespace without deleting page metadata.
8. Split text into meaningful overlapping chunks.
9. Create embeddings in batches.
10. Store vectors and metadata locally.
11. Embed the user's question.
12. Retrieve the most relevant chunks.
13. Apply a relevance threshold and optional reranker.
14. Send only retrieved evidence to the generator.
15. Generate citations from the actual retrieved metadata.
16. Validate that every displayed citation refers to a retrieved chunk.
17. Refuse unsupported answers.

### Citation integrity

A citation should never be generated from model memory. The application should build citations programmatically from retrieved chunk metadata. The model can place citation markers such as `[S1]`, but local code must map `[S1]` to a real retrieved chunk and reject unknown markers.

Example:

```text
[S1] lecture_notes.pdf, page 12, chunk lecture_notes-p12-c03
```

### Local storage and privacy

- Keep uploaded PDFs and vector indexes local by default.
- Provide a clear delete-session-data button.
- Avoid logging document contents.
- Store only the minimum metadata needed.
- Explain that text sent for embeddings or generation is transmitted to the selected API unless a fully local model is added later.

### Phase 2 prompt-injection defence

- Continue treating retrieved text as untrusted reference material.
- Detect instruction-like phrases in chunks and label them as document content.
- Never allow document text to alter system policies.
- Retrieve only relevant chunks rather than sending entire documents.
- Keep citations separate from model-authored prose.
- Add a post-generation citation validator.

### Phase 2 testing

- extraction from normal PDFs,
- damaged/encrypted PDF handling,
- empty and image-only pages,
- page-number preservation,
- deterministic chunk identifiers,
- retrieval relevance,
- no-evidence refusal,
- citation-to-chunk integrity,
- multiple-document source separation,
- prompt injection embedded in a PDF.

---

## 18. Phase 3 — Evaluation benchmark plan

Create a benchmark dataset containing 20–50 reviewed civil-engineering questions. Each record should include:

```text
question_id
civil_domain
vague_prompt
optimised_prompt_reference
expected_answer_points
known_common_errors
required_caveats
trusted_source_ids
risk_level
numerical_or_conceptual
```

### Candidate benchmark categories

- structural analysis and mechanics,
- earthquake engineering,
- reinforced concrete,
- steel structures,
- geotechnical engineering,
- surveying and remote sensing,
- transportation engineering,
- environmental engineering,
- hydrology and water resources,
- construction materials and management.

### Compared systems

**A. Baseline**

- Generate an answer directly from the vague prompt.
- No prompt optimisation.
- No trusted retrieval.

**B. Optimised + retrieval**

- Run prompt analysis.
- Use the improved prompt.
- Retrieve trusted chunks.
- Generate an evidence-bounded answer.
- Evaluate and revise.

### Metrics

- technical accuracy,
- completeness,
- unsupported claims,
- clarity,
- terminology,
- response consistency across repeated runs,
- evidence grounding,
- citation precision,
- citation recall,
- citation correctness,
- appropriate refusal rate,
- missing safety caveats.

### Evaluation method

Use a combination of:

1. deterministic checks for required keywords, units, formulas, and citation IDs;
2. expected-answer-point matching;
3. adversarial AI review using a fixed rubric;
4. manual review by a civil-engineering subject expert for a smaller gold set.

Do not treat AI evaluator scores alone as ground truth.

### Consistency testing

Run each benchmark question multiple times with a controlled configuration. Record score mean, standard deviation, unsupported-claim frequency, and citation consistency.

### CSV export

Suggested output columns:

```text
run_id
question_id
system_variant
model
prompt_version
accuracy_score
completeness_score
clarity_score
grounding_score
unsupported_claim_count
missing_caveat_count
citation_precision
citation_recall
citation_correctness
verdict
latency_seconds
estimated_input_tokens
estimated_output_tokens
raw_answer_path
```

Streamlit should provide a download button for the CSV and a summary dashboard comparing A and B.

---

## 19. Future C++ calculation engine

After the Python MVP and RAG pipeline are complete, add C++ only where it creates measurable value.

Possible C++ responsibilities:

- matrix assembly and factorisation,
- structural-analysis algorithms,
- finite-element element routines,
- eigenvalue calculations,
- nonlinear numerical solvers,
- optimisation routines,
- high-volume parametric calculations.

Possible integration approaches:

- `pybind11` Python bindings,
- a local C++ command-line executable with JSON input/output,
- a separate local service with a documented API.

Recommended separation:

```text
Python
- Streamlit interface
- OpenAI API
- RAG
- prompts
- validation
- evaluation
- reporting

C++
- deterministic calculation kernels
- numerical methods
- performance-sensitive structural routines
```

Every C++ calculation module should have independently verified unit tests and should return calculation metadata, units, assumptions, and error states. The AI layer should explain verified results; it should not replace the deterministic solver.

---

## 20. Responsible-use checklist

Before using an output for a report, thesis, calculation, design, or construction decision:

- verify the governing code and edition,
- verify jurisdiction and project requirements,
- verify all dimensions, loads, units, and material properties,
- check assumptions and boundary conditions,
- reproduce numerical calculations independently,
- inspect the trusted source text,
- confirm that every citation is genuine,
- obtain review from a qualified engineer where safety or compliance is involved.
