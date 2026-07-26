import io

import pytest
from pypdf import PdfWriter

from app.services.document_parsing import extract_text


def test_extract_text_from_txt():
    text = extract_text("dossier.txt", b"E-Vitamins sells supplements online.\n\nFounded 2019.")
    assert "E-Vitamins" in text


def test_extract_text_from_md():
    text = extract_text("dossier.md", "# E-Vitamins\n\nKey competitor: HealthKart".encode("utf-8"))
    assert "HealthKart" in text


def test_extract_text_rejects_unsupported_extension():
    with pytest.raises(ValueError, match="Unsupported file type"):
        extract_text("dossier.docx", b"whatever")


def test_extract_text_rejects_empty_content():
    with pytest.raises(ValueError, match="No extractable text"):
        extract_text("dossier.txt", b"   \n\n  ")


def test_extract_text_from_pdf_with_no_text_raises_value_error():
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    with pytest.raises(ValueError, match="No extractable text"):
        extract_text("dossier.pdf", buf.getvalue())
