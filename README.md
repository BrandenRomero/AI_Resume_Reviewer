# ResumeLens — AI Resume Reviewer

CS4680 Prompt Engineering, Assignment 3. A working Streamlit MVP that compares a resume with a target job posting and returns actionable, grounded feedback.

## The problem and product

Applicants often submit one general resume to different jobs. ResumeLens identifies where the resume explicitly supports the posting, where evidence is partial or absent, and how to make existing experience clearer. It does **not** predict hiring outcomes or claim to reproduce an applicant tracking system.

The user uploads a resume (text-based PDF, DOCX, or TXT) or pastes text, pastes a job description, selects a review focus, and clicks **Review my resume**. The app sends these texts to an OpenAI model, validates the returned JSON and quoted evidence, then displays strengths, requirement comparisons, priorities, draft bullet revisions, and interview talking points. A Markdown export is available. Files are parsed in memory; no database or logging of document content is implemented.

## Run in VS Code

1. Open this folder in VS Code. Install Python 3.10+.
2. In the VS Code terminal:

   ```bash
   python -m venv .venv
   ```

3. Activate the environment:
   - Windows PowerShell: `.venv\Scripts\Activate.ps1`
   - macOS/Linux: `source .venv/bin/activate`
4. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

5. Copy `.env.example` to `.env` and replace `replace_with_your_api_key` with your own OpenAI API key. The API may incur charges. Do not commit `.env` to GitHub.
6. Start the app:

   ```bash
   streamlit run app.py
   ```

   Open the local URL printed by Streamlit (usually `http://localhost:8501`). Click **Load sample resume + job** for a fictional demo, then **Review my resume**. A live model request requires the API key and network access.

## Prompt engineering used

The prompts are inspectable in `reviewer.py`.

1. **Persona:** The system instructs the model to act as a constructive recruiter and writing coach, prioritizing specific, kind recommendations with job related evidence.
2. **Few-shot:** Two miniature examples distinguish supported API experience from an unsupported Kubernetes claim, plus a good and bad bullet revision. This teaches the expected evidence boundary and format.
3. **Iterative refinement:** If JSON is incomplete or a supposed quote is absent from the submitted text, the app gives the model the precise validation error and retries once. It then fails clearly instead of showing an invented citation.

The prompt also treats resume and job text as untrusted data, limits the output to JSON, and explicitly forbids invented skills or numerical outcomes. Suggested bullet text remains a draft that the user must check.

## Limits and testing

- Scanned/image-only PDFs need OCR; DOCX text in shapes or images may be missed.
- The AI may make a judgment mistake even when quotes are verified. Review the draft edits against the source documents.
- The live call requires a valid OpenAI key, model access, account credit, and internet connectivity.
- The app caps extracted resume and job text at 18,000 characters each; only the first part is analyzed if longer.

Run focused checks from the repository root with `python -m pip install pytest` followed by `python -m pytest -q`. GitHub Actions runs the same checks on each push and pull request. These checks mock the model, so they do not spend API credit. A real model call must be tested with your key before recording.

## Project files

- `app.py`: Streamlit UI and report display
- `reviewer.py`: parsing, prompts, API request, validation, export
- `sample_data/`: fictional inputs for the demonstration
- `tests/`: extraction, evidence, retry, and error tests
- `.env.example`: key configuration template
