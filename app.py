"""Streamlit application entrypoint for AI-powered Resume Analyzer."""

from __future__ import annotations

from io import StringIO
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modules.advanced_analysis import (
    build_score_breakdown,
    detect_sections,
    estimate_experience_level,
    flesch_readability_score,
    highlight_keywords,
    keyword_density_analysis,
    suggest_improvements,
    top_keyword_matches,
)
from modules.cleaner import clean_text
from modules.parser import parse_resume
from modules.report_generator import generate_pdf_report
from modules.scorer import ATSResult, evaluate_resume
from modules.skill_extractor import extract_skills, flatten_skills, load_skill_database


st.set_page_config(page_title="ATS Resume Analyzer", page_icon="🧠", layout="wide")


def apply_custom_css() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background: radial-gradient(circle at top left, #1f2a60, #0b1020 48%, #111827);
                color: #f8fafc;
            }
            .block-container { max-width: 1500px; padding-top: 1rem; }
            .glass { background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,.16); border-radius: 18px; padding: 1rem; backdrop-filter: blur(10px); }
            .kpi { background: linear-gradient(120deg, rgba(124,92,255,.35), rgba(34,197,94,.15)); border: 1px solid rgba(255,255,255,.2); border-radius: 16px; padding: .9rem 1rem; }
            .kpi-title { font-size: .85rem; opacity: .85; }
            .kpi-value { font-size: 2rem; font-weight: 800; }
            .uploader-wrap { border: 2px dashed rgba(255,255,255,.25); border-radius: 14px; padding: .5rem; background: rgba(255,255,255,.03); }
            mark.kw { background: rgba(99,102,241,.45); border-radius: 4px; color: #fff; padding: 0 2px; }
            @keyframes fadeIn { from {opacity:0; transform: translateY(8px);} to {opacity:1; transform: translateY(0);} }
            .fade { animation: fadeIn .5s ease; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def get_skill_pool() -> set[str]:
    return flatten_skills(load_skill_database())


def animate_progress(label: str, value: float) -> None:
    st.markdown(f"**{label}: {value:.2f}%**")
    bar = st.progress(0)
    for i in range(0, int(value) + 1, 4):
        bar.progress(min(100, i))


def analyze_resume_text(candidate_name: str, extracted_text: str, job_description: str, skill_pool: set[str]) -> tuple[ATSResult, dict[str, Any]]:
    cleaned_resume = clean_text(extracted_text)
    cleaned_jd = clean_text(job_description)

    resume_skills = extract_skills(extracted_text, skill_pool)
    jd_skills = extract_skills(job_description, skill_pool)

    result = evaluate_resume(cleaned_resume, cleaned_jd, resume_skills, jd_skills)

    readability = flesch_readability_score(extracted_text)
    level, years = estimate_experience_level(extracted_text)
    sections = detect_sections(extracted_text)
    keyword_density = keyword_density_analysis(extracted_text, jd_skills)
    score_breakdown = build_score_breakdown(
        similarity_score=result.score,
        matched_skill_count=len(result.matched_skills),
        missing_skill_count=len(result.missing_skills),
        readability=readability,
    )
    score_breakdown["Confidence"] = result.confidence_score
    suggestions = suggest_improvements(result.missing_skills, sections, readability)
    overlaps = top_keyword_matches(extracted_text, job_description)

    insights = {
        "candidate_name": candidate_name,
        "readability": readability,
        "experience_level": level,
        "experience_years": years,
        "sections": sections,
        "keyword_density": keyword_density,
        "score_breakdown": score_breakdown,
        "suggestions": suggestions,
        "preview_html": highlight_keywords(extracted_text, overlaps),
    }
    return result, insights


def analyze_uploaded_file(file_obj: Any, job_description: str, skill_pool: set[str]) -> tuple[str, ATSResult, dict[str, Any]]:
    text = parse_resume(file_obj.name, file_obj.getvalue())
    name = file_obj.name.rsplit(".", maxsplit=1)[0]
    result, insights = analyze_resume_text(name, text, job_description, skill_pool)
    return name, result, insights


def render_kpis(result: ATSResult) -> None:
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='kpi fade'><div class='kpi-title'>🎯 ATS Score</div><div class='kpi-value'>{result.score:.2f}%</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='kpi fade'><div class='kpi-title'>🤖 SVM Confidence</div><div class='kpi-value'>{result.confidence_score:.2f}%</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='kpi fade'><div class='kpi-title'>❌ Missing Skills</div><div class='kpi-value'>{len(result.missing_skills)}</div></div>", unsafe_allow_html=True)


def render_detailed(candidate_name: str, result: ATSResult, insights: dict[str, Any], job_title: str) -> None:
    st.markdown(f"### 👤 {candidate_name}")
    render_kpis(result)
    animate_progress("ATS Progress", result.score)

    tabs = st.tabs(["Overview", "Heatmap", "Preview", "Download"])
    with tabs[0]:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        col1.metric("Readability", insights["readability"])
        col2.metric("Experience", f"{insights['experience_level']} ({insights['experience_years']}y)")

        st.info(result.strength_analysis)
        st.warning(result.weakness_analysis)
        st.write("**Matched Skills:**", result.matched_skills or ["None"])
        st.write("**Missing Skills:**", result.missing_skills or ["None"])
        st.write("**Suggestions:**")
        for tip in insights["suggestions"]:
            st.markdown(f"- {tip}")

        breakdown_df = pd.DataFrame({"Metric": list(insights["score_breakdown"].keys()), "Score": list(insights["score_breakdown"].values())})
        st.plotly_chart(px.bar(breakdown_df, x="Metric", y="Score", text="Score", title="Ranking Breakdown"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[1]:
        density_df = pd.DataFrame(insights["keyword_density"]).sort_values("density", ascending=False)
        if density_df.empty:
            st.info("No skill keywords from JD were detected.")
        else:
            fig = go.Figure(data=go.Heatmap(z=[density_df["density"].tolist()], x=density_df["keyword"].tolist(), y=["Density %"], colorscale="Plasma"))
            fig.update_layout(title="Skill Heatmap")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(density_df, use_container_width=True)

    with tabs[2]:
        st.markdown(f"<div class='glass'>{insights['preview_html']}</div>", unsafe_allow_html=True)

    with tabs[3]:
        pdf_bytes = generate_pdf_report(candidate_name, result, job_title)
        st.download_button("Download PDF Report", data=pdf_bytes, file_name=f"{candidate_name}_report.pdf", mime="application/pdf")


def bulk_mode(job_title: str, jd: str, skill_pool: set[str]) -> None:
    st.markdown("### 📂 Bulk Resume Screening (max 500 files)")
    st.markdown("<div class='uploader-wrap'>", unsafe_allow_html=True)
    files = st.file_uploader("Upload multiple resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    st.markdown("</div>", unsafe_allow_html=True)

    min_ats = st.slider("Shortlist ATS threshold", min_value=0, max_value=100, value=60)

    if files and st.button("Run Bulk Analysis", type="primary"):
        if len(files) > 500:
            st.error("You can upload up to 500 resumes only.")
            return

        records: list[dict[str, Any]] = []
        for f in files:
            try:
                name, result, insights = analyze_uploaded_file(f, jd, skill_pool)
                records.append(
                    {
                        "File": f.name,
                        "Candidate": name,
                        "ATS Score": result.score,
                        "Confidence Score": result.confidence_score,
                        "Matched Skills": len(result.matched_skills),
                        "Missing Skills": len(result.missing_skills),
                        "Shortlisted": "Yes" if result.score >= min_ats else "No",
                        "result": result,
                        "insights": insights,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Skipped {f.name}: {exc}")

        if not records:
            st.error("No files analyzed.")
            return

        df = pd.DataFrame(records).sort_values(by=["ATS Score", "Confidence Score"], ascending=False)
        shortlisted = df[df["Shortlisted"] == "Yes"].copy()

        st.markdown("### 🏆 Shortlisted Results")
        st.dataframe(shortlisted[["File", "Candidate", "ATS Score", "Confidence Score", "Matched Skills", "Missing Skills"]], use_container_width=True)

        st.plotly_chart(px.bar(shortlisted, x="File", y="ATS Score", title="ATS Score vs Resume File", color="ATS Score"), use_container_width=True)

        if not shortlisted.empty:
            ats_conf = shortlisted[["File", "ATS Score", "Confidence Score"]].melt(id_vars="File", var_name="Metric", value_name="Score")
            st.plotly_chart(px.bar(ats_conf, x="File", y="Score", color="Metric", barmode="group", title="ATS Score and Confidence"), use_container_width=True)

        csv_bytes = shortlisted[["File", "Candidate", "ATS Score", "Confidence Score", "Matched Skills", "Missing Skills"]].to_csv(index=False).encode("utf-8")
        st.download_button("Download Shortlisted CSV", data=csv_bytes, file_name="shortlisted_resumes.csv", mime="text/csv")

        if not shortlisted.empty:
            selected = st.selectbox("Inspect shortlisted candidate", shortlisted["Candidate"].tolist())
            row = shortlisted[shortlisted["Candidate"] == selected].iloc[0]
            render_detailed(selected, row["result"], row["insights"], job_title)


def video_resume_mode(job_title: str, jd: str, skill_pool: set[str]) -> None:
    st.markdown("### 🎥 Video Resume Analyzer")
    st.markdown("Upload video + paste transcript/summary for ATS evaluation.")
    video_file = st.file_uploader("Upload video resume", type=["mp4", "mov", "avi", "mkv"], accept_multiple_files=False)
    transcript = st.text_area("Paste video transcript/summary", height=220)

    if video_file:
        st.video(video_file)

    if video_file and transcript.strip() and st.button("Analyze Video Resume", type="primary"):
        candidate_name = video_file.name.rsplit(".", maxsplit=1)[0]
        result, insights = analyze_resume_text(candidate_name, transcript, jd, skill_pool)

        pie_df = pd.DataFrame({"Type": ["Matched", "Missing"], "Count": [len(result.matched_skills), len(result.missing_skills)]})
        st.plotly_chart(px.pie(pie_df, names="Type", values="Count", title="Video Resume Skill Split"), use_container_width=True)
        render_detailed(candidate_name, result, insights, job_title)


def resume_builder() -> None:
    st.markdown("### 🛠️ Resume Builder")
    with st.form("resume_builder_form"):
        name = st.text_input("Full Name")
        email = st.text_input("Email")
        phone = st.text_input("Phone")
        summary = st.text_area("Professional Summary")
        skills = st.text_area("Skills (comma separated)")
        experience = st.text_area("Experience (bullet points)")
        education = st.text_area("Education")
        projects = st.text_area("Projects")
        submitted = st.form_submit_button("Generate Resume Draft")

    if submitted:
        skill_list = [s.strip() for s in skills.split(",") if s.strip()]
        exp_lines = [f"- {line.strip()}" for line in experience.splitlines() if line.strip()]

        draft = f"""# {name}\n\nEmail: {email} | Phone: {phone}\n\n## Summary\n{summary}\n\n## Skills\n{', '.join(skill_list)}\n\n## Experience\n{chr(10).join(exp_lines)}\n\n## Education\n{education}\n\n## Projects\n{projects}\n"""
        st.markdown("#### Generated Resume Draft")
        st.text_area("Preview", value=draft, height=360)
        st.download_button("Download Resume Draft (.txt)", data=draft.encode("utf-8"), file_name="resume_draft.txt", mime="text/plain")


def main() -> None:
    apply_custom_css()
    st.title("🧠 ATS Resume Analyzer")
    st.caption("Modern ATS checker with SVM confidence, advanced analytics, bulk shortlisting, and resume builder.")

    skill_pool = get_skill_pool()

    with st.sidebar:
        st.markdown("### Job Requirement Input")
        job_title = st.text_input("Job Title", placeholder="e.g. Data Scientist")
        jd = st.text_area("Paste Job Description", height=220)

    if not jd.strip():
        st.info("Paste a job description to begin.")
        return

    tab_single, tab_bulk, tab_video, tab_builder = st.tabs([
        "📄 Single Resume",
        "📂 Bulk Resumes (500)",
        "🎥 Video Resume",
        "🛠️ Resume Builder",
    ])

    with tab_single:
        st.markdown("<div class='uploader-wrap'>", unsafe_allow_html=True)
        file_obj = st.file_uploader("Upload one resume", type=["pdf", "docx", "txt"], accept_multiple_files=False)
        st.markdown("</div>", unsafe_allow_html=True)

        if file_obj and st.button("Analyze Single Resume", type="primary"):
            try:
                name, result, insights = analyze_uploaded_file(file_obj, jd, skill_pool)
                render_detailed(name, result, insights, job_title)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Failed to analyze resume: {exc}")

    with tab_bulk:
        bulk_mode(job_title, jd, skill_pool)

    with tab_video:
        video_resume_mode(job_title, jd, skill_pool)

    with tab_builder:
        resume_builder()


if __name__ == "__main__":
    main()
