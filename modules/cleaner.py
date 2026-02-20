"""Text cleaning and preprocessing helpers."""

from __future__ import annotations

import re
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


@lru_cache(maxsize=1)
def _download_nltk_assets() -> None:
    """Ensure required NLTK assets are available locally.

    Newer NLTK versions may require `punkt_tab` in addition to `punkt`.
    """
    for package in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
        nltk.download(package, quiet=True)


@lru_cache(maxsize=1)
def _stop_words() -> set[str]:
    _download_nltk_assets()
    try:
        return set(stopwords.words("english"))
    except LookupError:
        return {
            "a",
            "an",
            "the",
            "and",
            "or",
            "of",
            "in",
            "to",
            "for",
            "with",
            "on",
            "at",
            "is",
            "are",
            "was",
            "were",
            "be",
            "by",
            "as",
            "from",
            "that",
            "this",
            "it",
        }


def tokenize_text(text: str) -> list[str]:
    """Tokenize safely with fallback when punkt resources are unavailable."""
    _download_nltk_assets()
    normalized = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    try:
        return word_tokenize(normalized)
    except LookupError:
        return re.findall(r"\b\w+\b", normalized)


def clean_text(text: str) -> str:
    """Normalize text by lowercasing, tokenizing, removing stopwords, and lemmatizing."""
    tokens = tokenize_text(text)
    stop_words = _stop_words()

    try:
        lemmatizer = WordNetLemmatizer()
        cleaned_tokens = [
            lemmatizer.lemmatize(token)
            for token in tokens
            if token.isalnum() and token not in stop_words and len(token) > 1
        ]
    except LookupError:
        cleaned_tokens = [
            token for token in tokens if token.isalnum() and token not in stop_words and len(token) > 1
        ]

    return " ".join(cleaned_tokens)
