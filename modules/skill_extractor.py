"""Skill extraction helpers using a predefined skills database."""

from __future__ import annotations

import json
import re
from pathlib import Path


DEFAULT_DB_PATH = Path("data/skill_database.json")


def load_skill_database(path: Path = DEFAULT_DB_PATH) -> dict[str, list[str]]:
    """Load skill categories and entries from JSON file."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def flatten_skills(skill_db: dict[str, list[str]]) -> set[str]:
    """Flatten category-based skill DB into a lowercase unique set."""
    return {skill.strip().lower() for skills in skill_db.values() for skill in skills}


def extract_skills(text: str, known_skills: set[str]) -> set[str]:
    """Extract known skills from input text using exact phrase matching with word boundaries."""
    lowered_text = text.lower()
    found_skills: set[str] = set()

    for skill in known_skills:
        pattern = rf"\b{re.escape(skill)}\b"
        if re.search(pattern, lowered_text):
            found_skills.add(skill)

    return found_skills
