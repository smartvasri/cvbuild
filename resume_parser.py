"""
resume_parser.py
Extracts plain text from PDF and DOCX resume files.
"""

import io
from pdfminer.high_level import extract_text as _pdf_extract
from docx import Document


def extract_text_from_pdf(content: bytes) -> str:
    """Return all text from a PDF given its raw bytes."""
    text = _pdf_extract(io.BytesIO(content))
    return (text or "").strip()


def extract_text_from_docx(content: bytes) -> str:
    """Return all text from a DOCX given its raw bytes (paragraphs + tables)."""
    doc   = Document(io.BytesIO(content))
    lines = []

    for para in doc.paragraphs:
        t = para.text.strip()
        if t:
            lines.append(t)

    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                t = cell.text.strip()
                if t and t not in lines:
                    lines.append(t)

    return "\n".join(lines)


def extract_text(content: bytes, filename: str) -> str:
    """
    Dispatch to the correct extractor based on file extension.
    `filename` must be the original uploaded filename (lowercased).
    """
    if filename.endswith(".pdf"):
        return extract_text_from_pdf(content)
    if filename.endswith(".docx"):
        return extract_text_from_docx(content)
    raise ValueError(f"Unsupported file type: {filename}")
