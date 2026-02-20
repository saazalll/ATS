"""Streamlit application entrypoint for AI-powered Resume Analyzer."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from modules.cleaner import clean_text
from modules.parser import parse_resume
from modules.report_generator import generate_pdf_report
from modules.scorer import ATSResult, evaluate_resume
from modules.skill_extractor import extract_skills, flatten_skills, load_skill_database


st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄", layout="wide")


@st.cache_data
def get_skill_pool() -> set[str]:
    """Cache flattened skill database for app sessions."""
    skill_db = load_skill_database()
    return flatten_skills(skill_db)


def analyze_resume(file_obj: Any, job_description: str, skill_pool: set[str]) -> tuple[str, ATSResult]:
    """Run end-to-end resume analysis and return candidate name + result object."""
    file_bytes = file_obj.getvalue()
    extracted_text = parse_resume(file_obj.name, file_bytes)

    cleaned_resume = clean_text(extracted_text)
    cleaned_jd = clean_text(job_description)

    resume_skills = extract_skills(extracted_text, skill_pool)
    jd_skills = extract_skills(job_description, skill_pool)

    result = evaluate_resume(cleaned_resume, cleaned_jd, resume_skills, jd_skills)
    candidate_name = file_obj.name.rsplit(".", maxsplit=1)[0]
    return candidate_name, result


def render_result(candidate_name: str, result: ATSResult, job_title: str) -> None:
    """Render result cards, skills, charts, and report download for one candidate."""
    st.subheader(f"Results for: {candidate_name}")

    col1, col2, col3 = st.columns(3)
    col1.metric("ATS Match Score", f"{result.score:.2f}%")
    col2.metric("Matched Skills", len(result.matched_skills))
    col3.metric("Missing Skills", len(result.missing_skills))

    st.markdown("#### Strength Analysis")
    st.info(result.strength_analysis)

    st.markdown("#### Weakness Areas")
    st.warning(result.weakness_analysis)

    skills_col1, skills_col2 = st.columns(2)
    with skills_col1:
        st.markdown("#### ✅ Matched Skills")
        st.write(result.matched_skills if result.matched_skills else ["No matched skills found"])

    with skills_col2:
        st.markdown("#### ❌ Missing Skills")
        st.write(result.missing_skills if result.missing_skills else ["No missing skills"])

    chart_data = pd.DataFrame(
        {
            "Category": ["Matched Skills", "Missing Skills"],
            "Count": [len(result.matched_skills), len(result.missing_skills)],
        }
    )
    bar_chart = px.bar(chart_data, x="Category", y="Count", color="Category", title="Skill Coverage")
    st.plotly_chart(bar_chart, use_container_width=True)

    score_chart = px.bar(
        pd.DataFrame({"Metric": ["Similarity Score"], "Value": [result.score]}),
        x="Metric",
        y="Value",
        range_y=[0, 100],
        text="Value",
        title="ATS Similarity Score",
    )
    st.plotly_chart(score_chart, use_container_width=True)

    pdf_bytes = generate_pdf_report(candidate_name, result, job_title=job_title)
    st.download_button(
        label="Download PDF Report",
        data=pdf_bytes,
        file_name=f"{candidate_name}_ats_report.pdf",
        mime="application/pdf",
    )


def main() -> None:
    """Render Streamlit app with single and bulk resume analysis options."""
    st.title("📄 AI-Powered Resume Analyzer")
    st.caption("Open-source ATS analyzer using NLP, TF-IDF, and cosine similarity.")

    skill_pool = get_skill_pool()

    with st.sidebar:
        st.header("Navigation")
        mode = st.radio("Choose Mode", ["Single Resume Analysis", "Bulk Resume Screening"])
        st.markdown("---")
        st.write("Supported file types: PDF, DOCX, TXT")

    job_title = st.text_input("Job Title (optional)", placeholder="e.g., Senior Python Developer")
    job_description = st.text_area("Paste Job Description", height=220)

    if not job_description.strip():
        st.info("Paste a job description to begin analysis.")
        return

    if mode == "Single Resume Analysis":
        resume_file = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"], accept_multiple_files=False)
        if resume_file and st.button("Analyze Resume", type="primary"):
            with st.spinner("Analyzing resume..."):
                try:
                    candidate_name, result = analyze_resume(resume_file, job_description, skill_pool)
                    render_result(candidate_name, result, job_title)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Failed to analyze resume: {exc}")

    else:
        resume_files = st.file_uploader(
            "Upload Multiple Resumes",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
        )

        if resume_files and st.button("Run Bulk Screening", type="primary"):
            records: list[dict[str, Any]] = []
            with st.spinner("Screening resumes..."):
                for file_obj in resume_files:
                    try:
                        candidate_name, result = analyze_resume(file_obj, job_description, skill_pool)
                        records.append(
                            {
                                "Candidate": candidate_name,
                                "Score": result.score,
                                "Matched Skills": len(result.matched_skills),
                                "Missing Skills": len(result.missing_skills),
                                "Result": result,
                            }
                        )
                    except Exception as exc:  # noqa: BLE001
                        st.warning(f"Skipped {file_obj.name}: {exc}")

            if not records:
                st.error("No resumes were successfully analyzed.")
                return

            ranked_df = pd.DataFrame(records).sort_values(by="Score", ascending=False).reset_index(drop=True)
            ranked_df.index = ranked_df.index + 1
            st.subheader("Ranked Candidates")
            st.dataframe(ranked_df[["Candidate", "Score", "Matched Skills", "Missing Skills"]], use_container_width=True)

            ranking_chart = px.bar(
                ranked_df,
                x="Candidate",
                y="Score",
                color="Score",
                title="Candidate Ranking by ATS Score",
                text="Score",
            )
            st.plotly_chart(ranking_chart, use_container_width=True)

            selected_candidate = st.selectbox(
                "View Detailed Report",
                options=ranked_df["Candidate"].tolist(),
            )

            selected_row = ranked_df[ranked_df["Candidate"] == selected_candidate].iloc[0]
            render_result(selected_candidate, selected_row["Result"], job_title)


if __name__ == "__main__":
    main()
