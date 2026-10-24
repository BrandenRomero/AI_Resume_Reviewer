"""Streamlit entry point. Run: streamlit run app.py"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import dotenv_values

from reviewer import ReviewError, analyze, export_markdown, extract_resume

ROOT = Path(__file__).parent


def setting(name: str, default: str = "") -> str:
    """Read local settings on each review so newly saved keys work immediately."""
    return (dotenv_values(ROOT / ".env").get(name) or os.getenv(name) or default).strip()


st.set_page_config(page_title="ResumeLens | AI Resume Reviewer", page_icon="📄", layout="wide")
st.markdown("""
<style>
  .block-container {max-width: 1120px; padding-top: 2.3rem}
  h1 {letter-spacing: -.04em}
  [data-testid="stMetric"] {background: #f5f7fb; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 1rem}
  .subtle {color: #526078; font-size: 1.02rem}
</style>
""", unsafe_allow_html=True)

st.caption("CS4680 • Prompt Engineering MVP")
st.title("📄 ResumeLens")
st.markdown('<p class="subtle">Get a practical, evidence based review tailored to a job posting.</p>', unsafe_allow_html=True)

with st.sidebar:
    st.header("How it works")
    st.write("1. Upload or paste a resume.\n\n2. Paste the job description.\n\n3. Review matches, gaps, and draft edits.")
    st.divider()
    st.info("Your resume and job description are sent to the Gemini API when you click Review. This app does not store them in a database. Google's free tier may use submitted data to improve its products; avoid sharing sensitive details you do not want to send.")
    st.caption("This is coaching feedback, not a hiring decision or an ATS score.")

sample = st.button("Load sample resume + job", help="Use fictional data for the live demo")
if sample:
    st.session_state.pop("review", None)
    st.session_state["resume_text"] = (ROOT / "sample_data" / "resume.txt").read_text()
    st.session_state["job_text"] = (ROOT / "sample_data" / "job.txt").read_text()

left, right = st.columns(2, gap="large")
with left:
    st.subheader("1 · Your resume")
    uploaded = st.file_uploader("Upload a PDF, DOCX, or TXT", type=["pdf", "docx", "txt"], max_upload_size=5)
    resume_text = st.text_area("Or paste resume text", key="resume_text", height=260,
                               placeholder="Paste experience, skills, projects, and education here…")
with right:
    st.subheader("2 · Target job")
    job_text = st.text_area("Paste the job description", key="job_text", height=330,
                            placeholder="Paste the responsibilities and requirements here…")

focus = st.selectbox("Review focus", ["Balanced review", "Technical skills", "Impact and clarity", "Entry level applicant"])
ready = bool((uploaded or resume_text.strip()) and job_text.strip())
run = st.button("Review my resume →", type="primary", disabled=not ready)

if run:
    st.session_state.pop("review", None)
    try:
        if uploaded is not None:
            resume_text = extract_resume(uploaded.name, uploaded.getvalue())
        key = setting("GEMINI_API_KEY")
        with st.spinner("Comparing your experience with the posting…"):
            review = analyze(resume_text, job_text, focus, key,
                             model=setting("GEMINI_MODEL", "gemini-3.8-flash"))
        st.session_state["review"] = review
    except ReviewError as exc:
        st.error(str(exc))

if "review" in st.session_state:
    review = st.session_state["review"]
    st.divider()
    st.subheader("Your review")
    st.write(review["summary"])
    evidenced = sum(row["status"] == "evidenced" for row in review["requirements"])
    partial = sum(row["status"] == "partial" for row in review["requirements"])
    missing = sum(row["status"] == "not shown" for row in review["requirements"])
    a, b, c = st.columns(3)
    a.metric("Evidence found", evidenced)
    b.metric("Partial evidence", partial)
    c.metric("Not shown", missing)
    st.caption("Counts describe the selected requirements, not an ATS or hiring score.")

    strengths, alignment, edits = st.tabs(["Strengths + next steps", "Requirement evidence", "Bullet edits"])
    with strengths:
        st.markdown("#### What already works")
        for item in review["strengths"]:
            st.write("✓", item)
        st.markdown("#### Your next three actions")
        for n, item in enumerate(review["priorities"], 1):
            st.write(f"{n}. {item}")
        st.markdown("#### Interview talking points")
        for item in review["interview_talking_points"]:
            st.write("•", item)
    with alignment:
        for row in review["requirements"]:
            icon = {"evidenced": "✅", "partial": "◐", "not shown": "○"}[row["status"]]
            with st.expander(f"{icon} {row['requirement']} — {row['status']}"):
                st.write(row["note"])
                st.caption("JOB POSTING")
                st.write(row["job_evidence"] or "No verified quote returned")
                st.caption("RESUME")
                st.write(row["resume_evidence"] or "No matching evidence shown")
    with edits:
        if not review["bullet_rewrites"]:
            st.info("No safely grounded bullet edits were returned.")
        for n, row in enumerate(review["bullet_rewrites"], 1):
            st.markdown(f"#### Draft {n}")
            st.caption("ORIGINAL")
            st.write(row["original"])
            st.caption("SUGGESTED")
            st.write(row["suggestion"])
            st.caption(row["why"])
        st.warning("Check every suggestion before using it. Never add experience or metrics you cannot support.")

    st.download_button("Download review as Markdown", export_markdown(review),
                       file_name="resume_review.md", mime="text/markdown")
