"""Advanced ATS analytics for richer resume insights."""

from __future__ import annotations

import re
from collections import Counter

from modules.cleaner import tokenize_text


SECTION_PATTERNS = {
    "skills": r"\b(skills?|technical skills|core competencies)\b",
    "education": r"\b(education|academic|qualification)\b",
    "projects": r"\b(projects?|portfolio|case studies)\b",
    "experience": r"\b(experience|employment|work history|professional experience)\b",
}


def keyword_density_analysis(text: str, jd_skills: set[str]) -> list[dict[str, float | str]]:
    """Calculate normalized keyword density for skills from JD."""
    lowered = text.lower()
    total_words = max(1, len(tokenize_text(lowered)))
    rows: list[dict[str, float | str]] = []
    for skill in sorted(jd_skills):
        count = len(re.findall(rf"\b{re.escape(skill)}\b", lowered))
        rows.append({"keyword": skill, "count": count, "density": round((count / total_words) * 100, 3)})
    return rows


def flesch_readability_score(text: str) -> float:
    """Estimate Flesch reading ease score."""

    sentences = max(1, len(re.findall(r"[.!?]+", text)) or 1)
    words = [w for w in tokenize_text(text) if w.isalpha()]
    word_count = max(1, len(words))

    def syllables(word: str) -> int:
        word = word.lower().strip()
        if not word:
            return 1
        groups = re.findall(r"[aeiouy]+", word)
        count = len(groups)
        if word.endswith("e") and count > 1:
            count -= 1
        return max(1, count)

    syllable_count = sum(syllables(w) for w in words)
    score = 206.835 - 1.015 * (word_count / sentences) - 84.6 * (syllable_count / word_count)
    return round(score, 2)


def estimate_experience_level(text: str) -> tuple[str, int]:
    """Estimate seniority from year mentions and keywords."""
    lowered = text.lower()
    years = [int(y) for y in re.findall(r"(\d{1,2})\+?\s*(?:years|yrs)", lowered)]
    max_years = max(years) if years else 0

    if any(k in lowered for k in ["principal", "staff", "architect", "lead"]):
        max_years = max(max_years, 8)
    elif "senior" in lowered:
        max_years = max(max_years, 5)

    if max_years >= 8:
        return "Senior", max_years
    if max_years >= 3:
        return "Mid", max_years
    return "Entry", max_years


def detect_sections(text: str) -> dict[str, bool]:
    """Detect common resume sections."""
    lowered = text.lower()
    return {section: bool(re.search(pattern, lowered)) for section, pattern in SECTION_PATTERNS.items()}


def build_score_breakdown(
    similarity_score: float,
    matched_skill_count: int,
    missing_skill_count: int,
    readability: float,
) -> dict[str, float]:
    """Build weighted ATS ranking component breakdown."""
    total_skills = max(1, matched_skill_count + missing_skill_count)
    skill_coverage = (matched_skill_count / total_skills) * 100
    readability_norm = max(0, min(100, readability))

    return {
        "Similarity": round(similarity_score, 2),
        "Skill Coverage": round(skill_coverage, 2),
        "Readability": round(readability_norm, 2),
        "Final Ranking": round(0.6 * similarity_score + 0.3 * skill_coverage + 0.1 * readability_norm, 2),
    }


def suggest_improvements(missing_skills: list[str], sections: dict[str, bool], readability: float) -> list[str]:
    """Generate targeted, data-driven resume improvement suggestions."""
    suggestions: list[str] = []

    if missing_skills:
        top_missing = ", ".join(missing_skills[:8])
        suggestions.append(f"Add evidence-backed mentions for missing skills: {top_missing}.")

    for section in ("skills", "projects", "education"):
        if not sections.get(section, False):
            suggestions.append(f"Add a dedicated '{section.title()}' section for ATS-friendly parsing.")

    if readability < 45:
        suggestions.append("Improve readability using shorter bullet points and clearer action-result statements.")

    if not suggestions:
        suggestions.append("Resume is well-aligned. Add quantified achievements to further strengthen impact.")

    return suggestions


def top_keyword_matches(resume_text: str, jd_text: str, limit: int = 25) -> list[str]:
    """Get top overlapping keywords between JD and resume for highlighting."""
    resume_tokens = [t for t in tokenize_text(resume_text.lower()) if len(t) > 2]
    jd_tokens = [t for t in tokenize_text(jd_text.lower()) if len(t) > 2]

    resume_counter = Counter(resume_tokens)
    overlap = {token: resume_counter[token] for token in set(jd_tokens) if token in resume_counter}
    ranked = sorted(overlap.items(), key=lambda x: x[1], reverse=True)
    return [k for k, _ in ranked[:limit]]


def highlight_keywords(text: str, keywords: list[str]) -> str:
    """Return HTML-highlighted preview with matched keywords emphasized."""
    snippet = text[:5000]
    highlighted = snippet
    for kw in sorted(set(keywords), key=len, reverse=True):
        if not kw.strip():
            continue
        highlighted = re.sub(
            rf"\b({re.escape(kw)})\b",
            r"<mark class='kw'>\1</mark>",
            highlighted,
            flags=re.IGNORECASE,
        )
    return highlighted.replace("\n", "<br>")
