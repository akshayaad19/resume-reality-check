"""PDF/text extraction for resumes."""
from __future__ import annotations

import io

from pypdf import PdfReader


def extract_text(file_bytes: bytes, filename: str) -> str:
    """Extract plain text from an uploaded resume file.

    Supports PDF (.pdf) and plain text (.txt, or anything else, as a fallback).
    """
    if filename.lower().endswith(".pdf"):
        return _extract_pdf_text(file_bytes)
    return _extract_plain_text(file_bytes)


def _extract_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def _extract_plain_text(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="replace").strip()
