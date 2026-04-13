# Extract text from pdf
import fitz

# Opens pdf and extracts all text and joins it together into one string
def extract_text(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            pages.append(text)
    doc.close()
    return "\n\n".join(pages)

# Splits data into overlapping chunks of 1000 characters and stores them in a dictionary so its easier to retrieve
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[dict]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_chunk = ""
    chunk_id = 0

    for para in paragraphs:
        if len(current_chunk) + len(para) > chunk_size and current_chunk:
            chunks.append({"id": chunk_id, "text": current_chunk.strip()})
            chunk_id += 1
            # Keep overlap from end of previous chunk
            words = current_chunk.split()
            overlap_words = words[-overlap // 5 :] if len(words) > overlap // 5 else words
            current_chunk = " ".join(overlap_words) + "\n\n" + para
        else:
            current_chunk = current_chunk + "\n\n" + para if current_chunk else para

    if current_chunk.strip():
        chunks.append({"id": chunk_id, "text": current_chunk.strip()})

    return chunks

# Main function that will return chunks if the pdf has readable data
def parse_pdf(pdf_bytes: bytes, chunk_size: int = 1000) -> list[dict]:
    text = extract_text(pdf_bytes)
    if not text.strip():
        return []
    return chunk_text(text, chunk_size=chunk_size)
