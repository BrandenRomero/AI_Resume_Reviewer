import io
import json
from types import SimpleNamespace

import pytest
from docx import Document
from google.genai import errors

from reviewer import ReviewError, analyze, extract_resume, validate_review


RESUME = "Jordan Lee\nBuilt a Flask API for searching records with Python and SQLite.\nWrote tests in pytest."
JOB = "Junior engineer needed to build Python APIs, React interfaces, SQL queries, automated tests, and Docker deployments."


def review_data():
    return {
        "summary": "Experience aligns with Python APIs and testing; Docker is not shown.",
        "strengths": ["API project"],
        "requirements": [
            {"requirement": "Python APIs", "status": "evidenced", "job_evidence": "Python APIs",
             "resume_evidence": "Built a Flask API for searching records with Python and SQLite.", "note": "Relevant API work."},
            {"requirement": "Docker", "status": "not shown", "job_evidence": "Docker deployments",
             "resume_evidence": "", "note": "Mention if genuinely used."},
        ],
        "priorities": ["Clarify project impact"],
        "bullet_rewrites": [{"original": "Wrote tests in pytest.",
                             "suggestion": "Added pytest tests to the API project.",
                             "why": "Connects testing to the project."}],
        "interview_talking_points": ["Discuss API tradeoffs"],
    }


def test_text_and_docx_extraction():
    assert extract_resume("r.txt", RESUME.encode()) == RESUME
    doc = Document()
    doc.add_paragraph(RESUME)
    data = io.BytesIO()
    doc.save(data)
    assert "Flask API" in extract_resume("r.docx", data.getvalue())


def test_invalid_or_scanned_input():
    with pytest.raises(ReviewError, match="Little text"):
        extract_resume("r.txt", b"hello")
    with pytest.raises(ReviewError, match="Use a PDF"):
        extract_resume("r.exe", b"A" * 100)


def test_evidence_must_be_exact():
    bad = review_data()
    bad["requirements"][0]["resume_evidence"] = "Invented project result"
    with pytest.raises(ReviewError, match="resume evidence"):
        validate_review(bad, RESUME, JOB)


def test_retry_after_invalid_model_quote():
    bad = review_data()
    bad["requirements"][0]["resume_evidence"] = "Invented project result"

    class FakeModels:
        def __init__(self):
            self.calls = 0
            self.prompts = []

        def generate_content(self, **kwargs):
            self.calls += 1
            self.prompts.append(kwargs["contents"])
            assert kwargs["config"].response_mime_type == "application/json"
            body = bad if self.calls == 1 else review_data()
            return SimpleNamespace(text=json.dumps(body))

    models = FakeModels()
    fake = SimpleNamespace(models=models)
    output = analyze(RESUME, JOB, "Balanced review", "test-key", client_factory=lambda **kwargs: fake)
    assert models.calls == 2
    assert "Validation failed" in models.prompts[1]
    assert output["requirements"][0]["status"] == "evidenced"


def test_missing_key_prevents_request():
    with pytest.raises(ReviewError, match="GEMINI_API_KEY"):
        analyze(RESUME, JOB, "Balanced review", "")


def test_gemini_quota_has_actionable_error():
    class FailingModels:
        def generate_content(self, **kwargs):
            raise errors.APIError(429, {"message": "Quota exhausted"})

    fake = SimpleNamespace(models=FailingModels())
    with pytest.raises(ReviewError, match="Gemini quota or rate limit"):
        analyze(RESUME, JOB, "Balanced review", "test-key", client_factory=lambda **kwargs: fake)
