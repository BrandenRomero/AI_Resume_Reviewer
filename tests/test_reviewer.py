import io
import json

import pytest
from docx import Document
from httpx import Request, Response
from openai import RateLimitError

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

    class FakeCompletions:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            body = bad if self.calls == 1 else review_data()
            return type("Response", (), {"choices": [type("Choice", (), {
                "message": type("Message", (), {"content": json.dumps(body)})()})()]})()

    completions = FakeCompletions()
    fake = type("Client", (), {"chat": type("Chat", (), {"completions": completions})()})()
    output = analyze(RESUME, JOB, "Balanced review", "test-key", client_factory=lambda **kwargs: fake)
    assert completions.calls == 2
    assert output["requirements"][0]["status"] == "evidenced"


def test_missing_key_prevents_request():
    with pytest.raises(ReviewError, match="OPENAI_API_KEY"):
        analyze(RESUME, JOB, "Balanced review", "")


def test_exhausted_api_credit_has_actionable_error():
    class FailingCompletions:
        def create(self, **kwargs):
            response = Response(429, request=Request("POST", "https://api.openai.com/v1/chat/completions"))
            raise RateLimitError(
                "Quota exhausted", response=response,
                body={"code": "credit_balance_exhausted"},
            )

    fake = type("Client", (), {"chat": type("Chat", (), {"completions": FailingCompletions()})()})()
    with pytest.raises(ReviewError, match="credits are exhausted"):
        analyze(RESUME, JOB, "Balanced review", "test-key", client_factory=lambda **kwargs: fake)
