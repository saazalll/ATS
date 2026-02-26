"""Skill extraction helpers using a predefined skills database."""

from __future__ import annotations

import json
import re
from pathlib import Path


DEFAULT_DB_PATH = Path("data/skill_database.json")

SKILL_ALIASES = {
    "scikit learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "machine-learning": "machine learning",
    "ml": "machine learning",
    "deep-learning": "deep learning",
    "nlp": "nlp",
    "node js": "nodejs",
}


def load_skill_database(path: Path = DEFAULT_DB_PATH) -> dict[str, list[str]]:
    """Load skill categories and entries from JSON file."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def normalize_skill(skill: str) -> str:
    """Normalize skill token and resolve aliases."""
    normalized = re.sub(r"\s+", " ", skill.strip().lower())
    return SKILL_ALIASES.get(normalized, normalized)


def flatten_skills(skill_db: dict[str, list[str]]) -> set[str]:
    """Flatten category-based skill DB into a normalized unique set."""
    return {normalize_skill(skill) for skills in skill_db.values() for skill in skills}


def _normalized_text(text: str) -> str:
    base = text.lower().replace("-", " ")
    for alias, canonical in SKILL_ALIASES.items():
        base = re.sub(rf"\b{re.escape(alias)}\b", canonical, base)
    return base


def extract_skills(text: str, known_skills: set[str]) -> set[str]:
    """Extract known skills from input text using normalized phrase matching."""
    lowered_text = _normalized_text(text)
    found_skills: set[str] = set()

    for skill in known_skills:
        pattern = rf"\b{re.escape(skill)}\b"
        if re.search(pattern, lowered_text):
            found_skills.add(skill)

    return found_skills
