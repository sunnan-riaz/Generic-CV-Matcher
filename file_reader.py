"""
file_reader.py
==============
Reads CV text from uploaded PDF, DOCX, or TXT files.

Returns a (text, error) tuple so the caller can display a user-friendly
message instead of a raw Python traceback.

Supported formats
-----------------
  • PDF  – uses pdfplumber (pure-Python, no system dependencies)
  • DOCX – uses python-docx
  • TXT  – decoded as UTF-8 with a latin-1 fallback

Not supported
-------------
  • Scanned PDFs (images of text) – these require OCR (e.g. pytesseract),
    which is outside the scope of this project.  The app will return a
    helpful message explaining this.
"""

from __future__ import annotations

import io
from typing import Tuple

# Maximum file size we accept (5 MB).  Larger files are likely not plain CVs
# and would cause excessive memory use on Streamlit Community Cloud.
MAX_FILE_BYTES = 5 * 1024 * 1024  # 5 MB


def read_uploaded_file(uploaded_file) -> Tuple[str, str]:
    """
    Extract plain text from a Streamlit UploadedFile object.

    Parameters
    ----------
    uploaded_file : streamlit.runtime.uploaded_file_manager.UploadedFile

    Returns
    -------
    (text, error)
        text  – extracted string (empty string on failure)
        error – human-readable error message (empty string on success)
    """
    # ── Size guard ────────────────────────────────────────────────────────────
    raw_bytes = uploaded_file.read()
    if len(raw_bytes) > MAX_FILE_BYTES:
        size_mb = len(raw_bytes) / (1024 * 1024)
        return "", (
            f"File is {size_mb:.1f} MB — larger than the 5 MB limit. "
            "Please paste your CV text directly into the text area instead."
        )

    filename = uploaded_file.name.lower()

    # ── Route by extension ────────────────────────────────────────────────────
    if filename.endswith(".pdf"):
        return _read_pdf(raw_bytes)
    elif filename.endswith(".docx"):
        return _read_docx(raw_bytes)
    elif filename.endswith(".txt"):
        return _read_txt(raw_bytes)
    else:
        return "", (
            f"Unsupported file type: '{uploaded_file.name}'. "
            "Please upload a PDF, DOCX, or TXT file."
        )


# ══════════════════════════════════════════════════════════════════════════════
# Format-specific readers
# ══════════════════════════════════════════════════════════════════════════════

def _read_pdf(raw_bytes: bytes) -> Tuple[str, str]:
    """
    Extract text from a PDF using pdfplumber.

    pdfplumber works on text-based PDFs.  It will return empty text for
    scanned PDFs (which are images, not searchable text).  We detect this
    and return a clear explanation.
    """
    try:
        import pdfplumber  # type: ignore
    except ImportError:
        return "", "pdfplumber is not installed. Run: pip install pdfplumber"

    try:
        pages_text = []
        with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    pages_text.append(page_text)

        full_text = "\n\n".join(pages_text).strip()

        if not full_text:
            return "", (
                "No text could be extracted from this PDF.  It may be a scanned "
                "(image-only) PDF, which requires OCR software to read. "
                "Please export your CV as a text-based PDF, or paste the text manually."
            )

        return full_text, ""

    except Exception as exc:
        return "", f"Could not read PDF: {exc}"


def _read_docx(raw_bytes: bytes) -> Tuple[str, str]:
    """Extract text from a DOCX file using python-docx."""
    try:
        from docx import Document  # type: ignore
    except ImportError:
        return "", "python-docx is not installed. Run: pip install python-docx"

    try:
        doc = Document(io.BytesIO(raw_bytes))
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        full_text = "\n\n".join(paragraphs).strip()

        if not full_text:
            return "", "The DOCX file appears to be empty or contains only images."

        return full_text, ""

    except Exception as exc:
        return "", f"Could not read DOCX: {exc}"


def _read_txt(raw_bytes: bytes) -> Tuple[str, str]:
    """Decode a plain-text file as UTF-8 (with latin-1 fallback)."""
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = raw_bytes.decode("latin-1")
        except Exception as exc:
            return "", f"Could not decode text file: {exc}"

    text = text.strip()
    if not text:
        return "", "The TXT file is empty."

    return text, ""
