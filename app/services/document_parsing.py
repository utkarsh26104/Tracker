"""Extracts plain text from an uploaded client-dossier file so it can be
chunked and embedded by app/memory/client_uploads.py. The output direction
(markdown report -> PDF) lives in report.py; this is the input direction."""

import io

from pypdf import PdfReader

_TEXT_EXTENSIONS = (".txt", ".md")


def extract_text(filename: str, content: bytes) -> str:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(content))
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
        text = "\n\n".join(p for p in pages if p)
    elif lower.endswith(_TEXT_EXTENSIONS):
        text = content.decode("utf-8", errors="replace")
    else:
        raise ValueError(f"Unsupported file type: {filename} - only .pdf, .txt, .md are supported")

    text = text.strip()
    if not text:
        raise ValueError(f"No extractable text found in {filename}")
    return text
