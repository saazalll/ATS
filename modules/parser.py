"""File parsing utilities for resume files."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import docx
import pdfplumber


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def parse_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF byte stream using pdfplumber."""
    text_chunks: list[str] = []
    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_chunks.append(page_text)
    return "\n".join(text_chunks)


def parse_docx(file_bytes: bytes) -> str:
    """Extract text from DOCX byte stream using python-docx."""
    document = docx.Document(BytesIO(file_bytes))
    return "\n".join(paragraph.text for paragraph in document.paragraphs if paragraph.text.strip())


def parse_txt(file_bytes: bytes) -> str:
    """Decode text bytes safely for TXT files."""
    return file_bytes.decode("utf-8", errors="ignore")


def parse_resume(file_name: str, file_bytes: bytes) -> str:
    """Parse a resume file based on its extension and return extracted plain text."""
    extension = Path(file_name).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file format: {extension}. Allowed: {sorted(SUPPORTED_EXTENSIONS)}")

    if extension == ".pdf":
        return parse_pdf(file_bytes)
    if extension == ".docx":
        return parse_docx(file_bytes)
    return parse_txt(file_bytes)
