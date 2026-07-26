import io

import markdown as markdown_lib
from xhtml2pdf import pisa

# xhtml2pdf's default font (Helvetica) only has Latin-1 glyphs. LLM output
# regularly includes "smart" typographic punctuation outside that range -
# non-breaking hyphens, en/em dashes, curly quotes - which render as black
# missing-glyph boxes instead of raising an error. Normalize to plain ASCII
# equivalents rather than bundling/embedding a Unicode font, which would tie
# PDF generation to a specific font file being present wherever this deploys.
_UNICODE_REPLACEMENTS = {
    "‐": "-",  # hyphen
    "‑": "-",  # non-breaking hyphen
    "‒": "-",  # figure dash
    "–": "-",  # en dash
    "—": "--",  # em dash
    "‘": "'",  # left single quote
    "’": "'",  # right single quote
    "“": '"',  # left double quote
    "”": '"',  # right double quote
    "…": "...",  # ellipsis
    " ": " ",  # non-breaking space
    " ": " ",  # narrow no-break space
    "•": "*",  # bullet
    "→": "->",  # rightwards arrow
    "←": "<-",  # leftwards arrow
}


def _sanitize_for_pdf(text: str) -> str:
    for char, replacement in _UNICODE_REPLACEMENTS.items():
        text = text.replace(char, replacement)
    # Safety net for anything else outside Helvetica's Latin-1 support - a
    # plain "?" is a far better failure mode than a black box.
    return text.encode("latin-1", errors="replace").decode("latin-1")


_PDF_STYLE = """
<style>
    @page { size: letter; margin: 2cm; }
    body { font-family: Helvetica, Arial, sans-serif; font-size: 10pt; line-height: 1.5; color: #1a1a1a; }
    h1 { font-size: 20pt; margin-top: 0; }
    h2 { font-size: 15pt; margin-top: 18pt; border-bottom: 1px solid #ccc; padding-bottom: 4px; }
    h3 { font-size: 12pt; margin-top: 14pt; }
    table { border-collapse: collapse; width: 100%; margin: 10px 0; }
    th, td { border: 1px solid #999; padding: 6px 8px; text-align: left; font-size: 9pt; }
    th { background-color: #eee; }
    code { background-color: #f0f0f0; padding: 1px 4px; }
    a { color: #b03a2e; }
</style>
"""


def markdown_to_pdf(markdown_text: str) -> bytes:
    """Renders a Markdown report (tables, headings, etc.) to PDF bytes."""
    markdown_text = _sanitize_for_pdf(markdown_text)
    html_body = markdown_lib.markdown(markdown_text, extensions=["tables", "fenced_code"])
    html_doc = f"<html><head>{_PDF_STYLE}</head><body>{html_body}</body></html>"

    buffer = io.BytesIO()
    result = pisa.CreatePDF(io.StringIO(html_doc), dest=buffer)
    if result.err:
        raise RuntimeError(f"PDF generation failed with {result.err} error(s)")

    return buffer.getvalue()
