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
    """Ensure required NLTK assets are available locally."""
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)


def clean_text(text: str) -> str:
    """Normalize text by lowercasing, tokenizing, removing stopwords, and lemmatizing."""
    _download_nltk_assets()
    normalized = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    tokens = word_tokenize(normalized)

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    cleaned_tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token.isalnum() and token not in stop_words and len(token) > 1
    ]

    return " ".join(cleaned_tokens)
