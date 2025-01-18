import os
import sys

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from src.pdf_parser import parse_pdf, split_text
from src.vector_db import initialize_chroma, upsert_documents
import yaml

def load_config():
    """
    Loads configuration from config.yaml.
    """
    with open(os.path.join(project_root, "config/config.yaml"), "r") as file:
        config = yaml.safe_load(file)
    return config

def preprocess_and_store_pdfs(pdf_directory: str):
    """
    Preprocesses PDFs in the given directory and stores their embeddings in Chroma.
    """
    config = load_config()
    vector_store = initialize_chroma()

    # Iterate over all PDFs in the directory
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_directory, filename)
            print(f"Processing: {filename}")

            # Parse PDF
            text = parse_pdf(file_path)

            # Split text into chunks
            chunks = split_text(text)

            # Upsert documents into Chroma
            upsert_documents(vector_store, chunks)

    print("All PDFs processed and embeddings stored in Chroma.")

if __name__ == "__main__":
    # Directory containing PDFs
    pdf_directory = os.path.join(project_root, "data/raw")

    # Preprocess and store PDFs
    preprocess_and_store_pdfs(pdf_directory)