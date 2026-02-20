"""Scoring utilities for ATS matching and strength analysis."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class ATSResult:
    """Structured ATS scoring output."""

    score: float
    matched_skills: list[str]
    missing_skills: list[str]
    strength_analysis: str
    weakness_analysis: str


def calculate_similarity_score(resume_text: str, job_description: str) -> float:
    """Compute cosine similarity of resume and JD text with TF-IDF features."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=4000)
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(float(similarity) * 100, 2)


def build_analysis(score: float, matched_count: int, missing_count: int) -> tuple[str, str]:
    """Generate high-level strengths and weaknesses summary."""
    if score >= 80:
        strength = "Excellent alignment with the job description and strong keyword overlap."
    elif score >= 60:
        strength = "Good alignment with relevant content, but there is room to improve targeting."
    else:
        strength = "Limited alignment. The resume should be tailored further to this specific role."

    if missing_count == 0:
        weakness = "No major skill gaps detected from the configured database."
    elif missing_count <= 5:
        weakness = "A few key skills are missing. Consider adding project evidence and keywords."
    else:
        weakness = "Multiple relevant skills are missing. Focus on upskilling and emphasizing role-fit."

    if matched_count == 0:
        weakness += " Also, none of the tracked skills were identified in the resume."

    return strength, weakness


def evaluate_resume(
    cleaned_resume_text: str,
    cleaned_job_description: str,
    resume_skills: set[str],
    jd_skills: set[str],
) -> ATSResult:
    """Evaluate ATS score and skill gaps for one resume."""
    score = calculate_similarity_score(cleaned_resume_text, cleaned_job_description)

    matched_skills = sorted(resume_skills.intersection(jd_skills))
    missing_skills = sorted(jd_skills.difference(resume_skills))
    strength, weakness = build_analysis(score, len(matched_skills), len(missing_skills))

    return ATSResult(
        score=score,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        strength_analysis=strength,
        weakness_analysis=weakness,
    )
