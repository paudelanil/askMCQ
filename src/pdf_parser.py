import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter


def parse_pdf(file_path):
    """
    Extracts text from a PDF file.
    """
    doc = fitz.open(file_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text


def split_text(text, chunk_size=500, chunk_overlap=50):
    """
    Splits text into smaller chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks