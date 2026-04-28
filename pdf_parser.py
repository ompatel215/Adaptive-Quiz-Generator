# Extract and chunk PDF text page-by-page

import fitz
import re


def extract_pages(pdf_bytes):
    # Return one cleaned text string per non-blank page.
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page in doc:
        text = page.get_text()
        cleaned = re.sub(r"\n{3,}", "\n\n", text).strip()
        if len(cleaned) > 40:   # skip blank / image-only pages
            pages.append(cleaned)
    doc.close()
    return pages


def chunk_pages(pages, target_size=800):
    # Group consecutive pages into ~target_size-char chunks.
    # Slide pages are short so several merge; dense textbook pages fill a chunk alone.
    chunks = []
    current = ""
    chunk_id = 0

    for page in pages:
        if current and len(current) + len(page) > target_size:
            chunks.append({"id": chunk_id, "text": current.strip()})
            chunk_id += 1
            current = page
        else:
            current = current + "\n\n" + page if current else page

    if current.strip():
        chunks.append({"id": chunk_id, "text": current.strip()})

    return chunks


def parse_pdf(pdf_bytes):
    # Parse raw PDF bytes and return a list of text chunks.
    pages = extract_pages(pdf_bytes)
    if not pages:
        return []
    return chunk_pages(pages)
