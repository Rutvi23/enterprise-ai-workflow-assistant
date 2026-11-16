"""Document ingestion: text extraction and page-aware chunking."""
from __future__ import annotations

import io
import re
import uuid

from pypdf import PdfReader

SUPPORTED = {".pdf", ".txt", ".md"}


class IngestionError(ValueError):
    pass


def extract_pages(filename: str, content: bytes) -> list[str]:
    """Return a list of page texts (plain-text files are treated as one page)."""
    name = filename.lower()
    if name.endswith(".pdf"):
        try:
            reader = PdfReader(io.BytesIO(content))
            pages = [(p.extract_text() or "") for p in reader.pages]
        except Exception as exc:  # corrupt / encrypted PDFs
            raise IngestionError(f"Could not read PDF: {exc}") from exc
    elif name.endswith((".txt", ".md")):
        pages = [content.decode("utf-8", errors="replace")]
    else:
        raise IngestionError("Unsupported file type. Use .pdf, .txt or .md")

    if not any(p.strip() for p in pages):
        raise IngestionError(
            "No extractable text found (scanned PDF? OCR is not supported yet)."
        )
    return pages


def split_text(text: str, size: int, overlap: int) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(". ", start + size // 2, end)  # prefer sentence boundary
            if cut != -1:
                end = cut + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_pages(doc_id: str, pages: list[str], size: int, overlap: int) -> list[dict]:
    chunks = []
    for page_no, page_text in enumerate(pages, start=1):
        for piece in split_text(page_text, size, overlap):
            chunks.append(
                {
                    "id": f"{doc_id}-{uuid.uuid4().hex[:8]}",
                    "document_id": doc_id,
                    "page": page_no,
                    "text": piece,
                }
            )
    return chunks
