"""Local PDF extraction and page-aware chunking; no network calls."""

from pathlib import PurePosixPath

import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

MAX_PDF_BYTES = 20 * 1024 * 1024
MAX_PAGES = 300


class PolicyError(Exception):
    """A safe, user-facing processing error (never contains API details)."""


def extract_pdf(pdf_bytes: bytes, filename: str) -> tuple[list[Document], int]:
    """Extract selectable text, keeping physical PDF page numbers (1-based)."""
    source = PurePosixPath(filename.replace("\\", "/")).name
    if not source.lower().endswith(".pdf"):
        raise PolicyError("Please select a PDF file.")
    if not pdf_bytes or len(pdf_bytes) > MAX_PDF_BYTES:
        raise PolicyError("The PDF must be non-empty and no larger than 20 MB.")
    try:
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as pdf:
            if not pdf.is_pdf or pdf.needs_pass:
                raise PolicyError("Use a valid, unencrypted PDF without a password.")
            page_count = len(pdf)
            if not 1 <= page_count <= MAX_PAGES:
                raise PolicyError("Use a PDF with between 1 and 300 pages.")
            pages = []
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text", sort=True).replace("\x00", "").strip()
                if text:
                    pages.append(Document(page_content=text, metadata={
                        "source": source, "page": page_number,
                    }))
    except PolicyError:
        raise
    except Exception:
        raise PolicyError("Unable to read this PDF. It may be invalid or damaged.") from None
    if not pages:
        raise PolicyError("No selectable text was found. Scanned/image-only PDFs require OCR, which is not supported yet.")
    return pages, page_count


def chunk_pages(pages: list[Document]) -> list[Document]:
    """Split each page separately into <=1,000 characters with up to 200 overlap."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(pages)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index
    return chunks
