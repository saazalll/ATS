"""Streamlit application entrypoint for AI-powered Resume Analyzer."""

from __future__ import annotations

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


st.set_page_config(page_title="AI Resume Analyzer", page_icon="🧠", layout="wide")


def apply_custom_css() -> None:
    """Apply modern SaaS-like dark glassmorphism styles."""
    st.markdown(
        """
        <style>
            :root {
                --bg1: #0b1020;
                --bg2: #151f3a;
                --glass: rgba(255, 255, 255, 0.08);
                --border: rgba(255, 255, 255, 0.18);
                --text: #edf3ff;
                --muted: #adc2e8;
                --accent: #7c5cff;
                --ok: #22c55e;
                --warn: #f59e0b;
                --bad: #ef4444;
            }
            .stApp {
                background: radial-gradient(circle at 10% 20%, #1a2454 0%, #0b1020 45%), linear-gradient(120deg, var(--bg1), var(--bg2));
                color: var(--text);
            }
            .block-container { padding-top: 1.3rem; padding-bottom: 2rem; max-width: 1450px; }
            h1, h2, h3 { letter-spacing: 0.2px; }
            .glass {
                background: var(--glass);
                border: 1px solid var(--border);
                border-radius: 18px;
                padding: 1rem 1.2rem;
                backdrop-filter: blur(10px);
                animation: floatIn .55s ease;
            }
            .kpi {
                background: linear-gradient(135deg, rgba(255,255,255,0.12), rgba(255,255,255,0.04));
                border: 1px solid var(--border);
                border-radius: 16px;
                padding: 1rem;
                min-height: 112px;
            }
            .kpi-label { color: var(--muted); font-size: 0.88rem; }
            .kpi-value { font-size: 2rem; font-weight: 700; margin-top: .2rem; }
            .uploader-wrap {
                border: 2px dashed rgba(255,255,255,0.25);
                border-radius: 16px;
                padding: .6rem;
                background: rgba(255,255,255,0.03);
            }
            mark.kw {
                background: rgba(124,92,255,0.35);
                border: 1px solid rgba(124,92,255,0.8);
                color: #f5f7ff;
                border-radius: 4px;
                padding: 0 2px;
            }
            @keyframes floatIn { from {opacity:0; transform: translateY(10px);} to {opacity:1; transform: translateY(0);} }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def get_skill_pool() -> set[str]:
    """Cache flattened skill database for app sessions."""
    skill_db = load_skill_database()
    return flatten_skills(skill_db)


def animate_progress(label: str, value: float) -> None:
    """Render animated progress bar simulation."""
    st.markdown(f"**{label}: {value:.2f}%**")
    bar = st.progress(0)
    target = int(max(0, min(100, value)))
    for i in range(0, target + 1, 5):
        bar.progress(i)


def analyze_resume(file_obj: Any, job_description: str, skill_pool: set[str]) -> tuple[str, str, ATSResult, dict[str, Any]]:
    """Run end-to-end resume analysis and return all insights."""
    file_bytes = file_obj.getvalue()
    extracted_text = parse_resume(file_obj.name, file_bytes)

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
    suggestions = suggest_improvements(result.missing_skills, sections, readability)
    overlaps = top_keyword_matches(extracted_text, job_description)
    preview_html = highlight_keywords(extracted_text, overlaps)

    candidate_name = file_obj.name.rsplit(".", maxsplit=1)[0]
    insights = {
        "readability": readability,
        "experience_level": level,
        "experience_years": years,
        "sections": sections,
        "keyword_density": keyword_density,
        "score_breakdown": score_breakdown,
        "suggestions": suggestions,
        "preview_html": preview_html,
    }
    return candidate_name, extracted_text, result, insights


def render_kpi_cards(result: ATSResult) -> None:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='kpi'><div class='kpi-label'>🎯 ATS Score</div><div class='kpi-value'>%.2f%%</div></div>" % result.score, unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='kpi'><div class='kpi-label'>✅ Matched Skills</div><div class='kpi-value'>%d</div></div>" % len(result.matched_skills), unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='kpi'><div class='kpi-label'>⚠️ Missing Skills</div><div class='kpi-value'>%d</div></div>" % len(result.missing_skills), unsafe_allow_html=True)


def render_result(candidate_name: str, result: ATSResult, insights: dict[str, Any], job_title: str) -> None:
    st.markdown(f"### 👤 Candidate Dashboard — {candidate_name}")
    render_kpi_cards(result)

    c1, c2, c3 = st.columns(3)
    c1.metric("Readability (Flesch)", insights["readability"])
    c2.metric("Experience Level", insights["experience_level"])
    c3.metric("Detected Years", insights["experience_years"])

    animate_progress("Similarity Progress", result.score)

    tabs = st.tabs([
        "📈 Analytics",
        "🧠 ATS Insights",
        "🗺️ Skill Heatmap",
        "🔍 Keyword Preview",
        "📄 Download Report",
    ])

    with tabs[0]:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        skills_df = pd.DataFrame(
            {
                "Category": ["Matched Skills", "Missing Skills"],
                "Count": [len(result.matched_skills), len(result.missing_skills)],
            }
        )
        st.plotly_chart(px.bar(skills_df, x="Category", y="Count", color="Category", title="Skills Coverage"), use_container_width=True)

        breakdown = pd.DataFrame(
            {"Metric": list(insights["score_breakdown"].keys()), "Score": list(insights["score_breakdown"].values())}
        )
        st.plotly_chart(px.bar(breakdown, x="Metric", y="Score", title="Ranking Score Breakdown", range_y=[0, 100], text="Score"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[1]:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("#### 💪 Strength Analysis")
        st.info(result.strength_analysis)
        st.markdown("#### 🧩 Weakness Areas")
        st.warning(result.weakness_analysis)

        s_col1, s_col2 = st.columns(2)
        with s_col1:
            st.markdown("#### ✅ Matched Skills")
            st.write(result.matched_skills or ["No matched skills found"])
        with s_col2:
            st.markdown("#### ❌ Missing Skills")
            st.write(result.missing_skills or ["No missing skills"])

        st.markdown("#### 📌 Section Detection")
        section_df = pd.DataFrame(
            {"Section": list(insights["sections"].keys()), "Detected": ["Yes" if v else "No" for v in insights["sections"].values()]}
        )
        st.dataframe(section_df, use_container_width=True)

        st.markdown("#### 💡 Smart Improvement Suggestions")
        for tip in insights["suggestions"]:
            st.markdown(f"- {tip}")
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[2]:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        density_df = pd.DataFrame(insights["keyword_density"])
        if not density_df.empty:
            density_df = density_df.sort_values("density", ascending=False)
            fig = go.Figure(
                data=go.Heatmap(
                    z=[density_df["density"].tolist()],
                    x=density_df["keyword"].tolist(),
                    y=["Keyword Density %"],
                    colorscale="Viridis",
                )
            )
            fig.update_layout(title="Skill Heatmap (JD keywords in resume)")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(density_df, use_container_width=True)
        else:
            st.info("No JD keywords detected for density analysis.")
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[3]:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("#### 🎯 Matched Keywords Highlighted in Resume Preview")
        st.markdown(f"<div style='line-height:1.7'>{insights['preview_html']}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[4]:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        pdf_bytes = generate_pdf_report(candidate_name, result, job_title=job_title)
        st.download_button(
            label="⬇️ Download Detailed PDF Report",
            data=pdf_bytes,
            file_name=f"{candidate_name}_ats_report.pdf",
            mime="application/pdf",
        )
        st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    apply_custom_css()

    st.markdown("# 🧠 AI Resume Analyzer")
    st.caption("Modern ATS dashboard — TF-IDF + NLP + explainable insights (100% open source)")

    skill_pool = get_skill_pool()

    with st.sidebar:
        st.markdown("### ⚙️ Control Center")
        job_title = st.text_input("Job Title (optional)", placeholder="e.g., Senior Python Developer")
        job_description = st.text_area("📋 Paste Job Description", height=240)

    if not job_description.strip():
        st.info("Paste a job description in the sidebar to begin analysis.")
        return

    single_tab, bulk_tab = st.tabs(["📄 Single Resume", "🏆 Bulk Screening Leaderboard"])

    with single_tab:
        st.markdown("<div class='uploader-wrap'>", unsafe_allow_html=True)
        file_obj = st.file_uploader("Drag & drop resume here", type=["pdf", "docx", "txt"], accept_multiple_files=False)
        st.markdown("</div>", unsafe_allow_html=True)

        if file_obj and st.button("🚀 Analyze Resume", type="primary"):
            with st.spinner("Analyzing resume with advanced ATS engine..."):
                try:
                    candidate_name, _raw, result, insights = analyze_resume(file_obj, job_description, skill_pool)
                    render_result(candidate_name, result, insights, job_title)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Failed to analyze resume: {exc}")

    with bulk_tab:
        st.markdown("<div class='uploader-wrap'>", unsafe_allow_html=True)
        files = st.file_uploader("Drag & drop multiple resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if files and st.button("⚡ Run Bulk ATS Screening", type="primary"):
            records: list[dict[str, Any]] = []
            with st.spinner("Evaluating resumes and building leaderboard..."):
                for file_obj in files:
                    try:
                        candidate, _raw, result, insights = analyze_resume(file_obj, job_description, skill_pool)
                        records.append(
                            {
                                "Candidate": candidate,
                                "ATS Score": result.score,
                                "Matched Skills": len(result.matched_skills),
                                "Missing Skills": len(result.missing_skills),
                                "Readability": insights["readability"],
                                "Experience": insights["experience_level"],
                                "Final Rank Score": insights["score_breakdown"]["Final Ranking"],
                                "result": result,
                                "insights": insights,
                            }
                        )
                    except Exception as exc:  # noqa: BLE001
                        st.warning(f"Skipped {file_obj.name}: {exc}")

            if not records:
                st.error("No resumes were successfully analyzed.")
                return

            df = pd.DataFrame(records).sort_values(by="Final Rank Score", ascending=False).reset_index(drop=True)
            df.index = df.index + 1

            st.markdown("### 🏆 Leaderboard")
            st.dataframe(
                df[["Candidate", "ATS Score", "Final Rank Score", "Matched Skills", "Missing Skills", "Readability", "Experience"]],
                use_container_width=True,
            )

            leaderboard_fig = px.bar(df, x="Candidate", y="Final Rank Score", color="Final Rank Score", title="Bulk Resume Ranking")
            st.plotly_chart(leaderboard_fig, use_container_width=True)

            selected = st.selectbox("Inspect Candidate Details", df["Candidate"].tolist())
            row = df[df["Candidate"] == selected].iloc[0]
            render_result(selected, row["result"], row["insights"], job_title)


if __name__ == "__main__":
    main()
